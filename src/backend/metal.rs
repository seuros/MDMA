//! Native Metal backend for unified-memory Macs.
//!
//! Shaders are embedded and compiled once at startup. Public operations are
//! synchronous. Decode executors own private command batches and exclusively
//! borrow request buffers until GPU completion; sessions never share a batch.
//! Activations and KV are f32; weights remain in their GGUF representation.

use super::{AttnSpec, Backend, Error, KvView, Operation, Pool, QWeight, Result, RopeSpec};
use super::{dtype_code, shape as geometry};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary,
    MTLResource, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::ptr::NonNull;

// objc2-metal requires CoreGraphics to initialize the default Metal device.
// SAFETY: the block only links the framework; it declares no foreign items.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

mod abi;
mod batch;
mod hybrid;
const KERNELS: &[&str] = &[
    "copy_bytes",
    "add_f32",
    "silu_mul_f32",
    "rmsnorm_f32",
    "add_rmsnorm_f32",
    "rope_f32",
    "qmatmul",
    "qmatmul_tile",
    "ffn_gate_up",
    "embed_rows",
    "attn_f32",
    "attn_simd",
    "scale_f32",
    "softcap_f32",
    "gelu_tanh_mul_f32",
    "rope_split_half",
    "layernorm_f32",
    "gelu_f32",
    "add_bias_f32",
    "argmax_f32",
];

/// Scalar/elementwise kernels count individual threads; cooperative kernels
/// assign one complete SIMD group per row, with independent batches on y.
#[derive(Clone, Copy)]
enum Dispatch {
    Threads([usize; 3]),
    SimdRows { rows: usize, batches: usize },
    Tiles { rows: usize, tokens: usize },
}

impl Dispatch {
    fn is_empty(&self) -> bool {
        match self {
            Self::Threads(grid) => grid.contains(&0),
            Self::SimdRows { rows, batches } => *rows == 0 || *batches == 0,
            Self::Tiles { rows, tokens } => *rows == 0 || *tokens == 0,
        }
    }

    fn sizes(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    ) -> Result<(MTLSize, MTLSize)> {
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        let (groups, width) = match *self {
            Self::Threads(grid) => {
                let width = max_threads.min(abi::ELEMENTWISE_THREADS);
                if width == 0 {
                    return Err(Error::Backend("Metal pipeline supports no threads".into()));
                }
                // Do not let padded uint thread coordinates wrap.
                dim(product(grid[0].div_ceil(width), width)?)?;
                dim(grid[1])?;
                dim(grid[2])?;
                ([grid[0].div_ceil(width), grid[1], grid[2]], width)
            }
            Self::SimdRows { rows, batches } => {
                dim(rows)?;
                dim(batches)?;
                let lanes = pipeline.threadExecutionWidth();
                if lanes == 0 || lanes > max_threads {
                    return Err(Error::Backend(
                        "Metal pipeline has invalid SIMD width".into(),
                    ));
                }
                let simd_groups = (max_threads / lanes).min(abi::SIMD_GROUPS);
                (
                    [rows.div_ceil(simd_groups), batches, 1],
                    product(lanes, simd_groups)?,
                )
            }
            Self::Tiles { rows, tokens } => {
                dim(rows)?;
                dim(tokens)?;
                if max_threads < abi::TILE_THREADS
                    || pipeline.threadExecutionWidth() != abi::TILE_SIMD_WIDTH
                {
                    return Err(Error::Backend(
                        "Metal matrix tile requires four 32-lane SIMD groups".into(),
                    ));
                }
                (
                    [
                        rows.div_ceil(abi::TILE_ROWS),
                        tokens.div_ceil(abi::TILE_TOKENS),
                        1,
                    ],
                    abi::TILE_THREADS,
                )
            }
        };
        Ok((
            MTLSize {
                width: groups[0],
                height: groups[1],
                depth: groups[2],
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        ))
    }
}

/// Owns an allocation, not a clonable view: mutable access cannot alias.
/// The Metal object retains its device and may outlive the backend.
pub struct MetalBuf {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

// SAFETY: Metal resources can move between threads, but objc2 deliberately
// leaves synchronization to wrappers. This allocation is private and not
// clonable, its mapped pointer never escapes, and every write requires
// &mut MetalBuf. Public calls finish before returning that borrow. The private
// batched executor retains resources and exclusively borrows request buffers
// until completion. Retaining/releasing the Metal object is thread-safe.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "justified by the SAFETY comment above"
)]
unsafe impl Send for MetalBuf {}
unsafe impl Sync for MetalBuf {}

/// Host-rounded RoPE frequency tables keyed by (rotary width, theta bits).
type RopeFreqCache = HashMap<(usize, u32), std::sync::Arc<MetalBuf>>;

pub struct MetalBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    // Immutable model constants, not activation computation on the CPU.
    // Host powf matches the CPU/reference frequency rounding at large positions.
    rope_freqs: std::sync::Arc<std::sync::Mutex<RopeFreqCache>>,
    // None for public eager operations; private to a request-owned executor.
    pending: Option<batch::Pending>,
    tile_prefill: bool,
    batch_decode: bool,
    shader_id: String,
    health: crate::federation::profile::gpu::Health,
    identity: std::sync::Arc<()>,
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        // Request-owned copies share health but must not retire the master.
        if self.pending.is_none() {
            self.health.withdraw();
        }
    }
}

fn shape(message: &str) -> Error {
    Error::ShapeMismatch(format!("metal: {message}"))
}

fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| shape("size overflow"))
}

fn dim(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| shape("dimension exceeds u32"))
}

fn floats(buf: &MetalBuf) -> Result<usize> {
    geometry::elems("metal", buf.len)
}

fn expect_floats(buf: &MetalBuf, count: usize) -> Result<()> {
    if floats(buf)? != count {
        return Err(shape("incorrect f32 buffer length"));
    }
    Ok(())
}

fn weight_dims(w: &QWeight<MetalBuf>) -> Result<(usize, usize, u32)> {
    let code = dtype_code::decoder_code(w.dtype)?.cast_unsigned();
    let &[rows, cols] = w.shape.as_slice() else {
        return Err(shape("weight must be rank 2"));
    };
    if rows == 0 || cols == 0 || cols % w.dtype.block_size() != 0 {
        return Err(shape(
            "weight dimensions must be nonzero and rows block-aligned",
        ));
    }
    dim(rows)?;
    dim(cols)?;
    let row_bytes = product(cols / w.dtype.block_size(), w.dtype.type_size())?;
    if w.buf.len != product(rows, row_bytes)? {
        return Err(shape("weight buffer length does not match shape and dtype"));
    }
    Ok((rows, cols, code))
}

fn matmul_params(w: &QWeight<MetalBuf>, x: &MetalBuf, y: &MetalBuf) -> Result<[u32; 4]> {
    let (rows, cols, code) = weight_dims(w)?;
    let n = floats(x)?;
    if n == 0 || n % cols != 0 {
        return Err(shape(
            "qmatmul input must contain complete, nonempty token rows",
        ));
    }
    let tokens = n / cols;
    expect_floats(y, product(tokens, rows)?)?;
    Ok([dim(rows)?, dim(cols)?, dim(tokens)?, code])
}

fn norm_params(w: &MetalBuf, x: &MetalBuf, y: &MetalBuf, eps: f32) -> Result<[u32; 3]> {
    let width = floats(w)?;
    let n = floats(x)?;
    if width == 0 || n % width != 0 || !eps.is_finite() || eps <= 0.0 {
        return Err(shape(
            "rmsnorm requires complete rows and positive finite epsilon",
        ));
    }
    expect_floats(y, n)?;
    Ok([dim(width)?, dim(n / width)?, eps.to_bits()])
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        autoreleasepool(|_| {
            let device = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| Error::Backend("no Metal GPU available".into()))?;
            if !device.hasUnifiedMemory() {
                return Err(Error::Backend(
                    "metal backend requires a unified-memory GPU".into(),
                ));
            }
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| Error::Backend("Metal command queue allocation failed".into()))?;
            let options = MTLCompileOptions::new();
            // Requires macOS 12+.
            options.setLanguageVersion(MTLLanguageVersion::Version2_4);
            #[allow(deprecated)]
            options.setFastMathEnabled(false);
            let matrix_tile = device.supportsFamily(MTLGPUFamily::Apple7);
            let source = abi::source(matrix_tile);
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
                .map_err(|e| Error::Backend(format!("Metal shader compilation: {e}")))?;
            let mut pipelines = HashMap::new();
            for &name in KERNELS {
                if name == "qmatmul_tile" && !matrix_tile {
                    continue;
                }
                let function = library
                    .newFunctionWithName(&NSString::from_str(name))
                    .ok_or_else(|| Error::Backend(format!("missing Metal kernel {name}")))?;
                let pipeline = device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| Error::Backend(format!("Metal pipeline {name}: {e}")))?;
                pipelines.insert(name, pipeline);
            }
            let tile_prefill =
                matrix_tile && std::env::var("MDMA_METAL_TILE").as_deref() != Ok("0");
            Ok(Self {
                device,
                queue,
                pipelines,
                rope_freqs: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
                pending: None,
                tile_prefill,
                batch_decode: std::env::var("MDMA_METAL_BATCH").as_deref() != Ok("0"),
                shader_id: {
                    use sha2::{Digest, Sha256};
                    format!("sha256:{}", hex::encode(Sha256::digest(source.as_bytes())))
                },
                health: crate::federation::profile::gpu::Health::default(),
                identity: std::sync::Arc::new(()),
            })
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub(crate) fn federation_profile(
        &self,
    ) -> std::io::Result<crate::federation::profile::gpu::Gpu> {
        use crate::federation::profile::gpu::{Gpu, fields};
        let os = sysinfo::System::os_version()
            .ok_or_else(|| std::io::Error::other("missing Metal OS runtime"))?;
        let kernel = sysinfo::System::kernel_version()
            .ok_or_else(|| std::io::Error::other("missing Metal kernel runtime"))?;
        let mut pipelines: Vec<_> = self
            .pipelines
            .iter()
            .map(|(name, pipeline)| {
                format!(
                    "{name}:{}:{}",
                    pipeline.threadExecutionWidth(),
                    pipeline.maxTotalThreadsPerThreadgroup()
                )
            })
            .collect();
        pipelines.sort();
        Ok(Gpu::immutable(
            "metal",
            "f32",
            "f32",
            fields([
                ("vocabulary", "mdma.metal.v1".into()),
                ("workload", "decoder".into()),
                ("device_name", self.device_name()),
                ("device_registry_id", self.device.registryID().to_string()),
                ("unified_memory", self.device.hasUnifiedMemory().to_string()),
                ("runtime_os", os),
                ("runtime_kernel", kernel),
                ("kernel_source_id", self.shader_id.clone()),
                ("pipelines", pipelines.join(",")),
                ("language", "metal-2.4-fast-math-disabled".into()),
                ("weights", "gguf-raw-matrices-f32-norms".into()),
                (
                    "execution",
                    if self.batch_decode {
                        "request-owned-command-batches"
                    } else {
                        "eager-joined"
                    }
                    .into(),
                ),
                ("tile_prefill", self.tile_prefill.to_string()),
                ("attention", "online-causal-f32".into()),
                ("rope", "host-f32-frequencies-metal-f32".into()),
                ("target_arch", std::env::consts::ARCH.into()),
                ("target_os", std::env::consts::OS.into()),
            ]),
            self.health.clone(),
        ))
    }

    fn pipeline(&self, name: &str) -> Result<&ProtocolObject<dyn MTLComputePipelineState>> {
        self.pipelines
            .get(name)
            .map(|p| &**p)
            .ok_or_else(|| Error::Backend(format!("unknown Metal kernel {name}")))
    }

    fn use_tile(&self, params: &[u32; 4]) -> bool {
        self.tile_prefill
            && params[0] as usize >= abi::TILE_ROWS
            && params[1] as usize >= abi::TILE_K
            && params[2] as usize >= abi::TILE_TOKENS
            && self.pipelines.get("qmatmul_tile").is_some_and(|pipeline| {
                pipeline.threadExecutionWidth() == abi::TILE_SIMD_WIDTH
                    && pipeline.maxTotalThreadsPerThreadgroup() >= abi::TILE_THREADS
            })
    }

    /// # Safety
    ///
    /// All buffer lengths, shapes, indices and scalar ABI layouts must be
    /// checked by the caller against the named kernel. Outputs must be
    /// exclusively borrowed until completion (the whole private request when batching).
    unsafe fn launch<P: bytemuck::Pod>(
        &self,
        name: &str,
        buffers: &[&MetalBuf],
        params: &P,
        dispatch: Dispatch,
    ) -> Result<()> {
        if dispatch.is_empty() {
            return Ok(());
        }
        let pipeline = self.pipeline(name)?;
        let (groups, threads) = dispatch.sizes(pipeline)?;
        for buf in buffers {
            if buf.buffer.device().registryID() != self.device.registryID() {
                return Err(Error::Backend(
                    "Metal buffer belongs to another device".into(),
                ));
            }
        }
        autoreleasepool(|_| {
            if let Some(pending) = &self.pending {
                let mut pending = pending
                    .lock()
                    .map_err(|_| Error::Backend("Metal batch lock poisoned".into()))?;
                if pending.is_none() {
                    *pending = Some(batch::Batch::new(&self.queue)?);
                }
                // SAFETY: the caller's checked contract extends through the
                // enclosing request; encode retains every referenced resource.
                let batch = pending.as_mut().expect("batch created above");
                unsafe {
                    batch.encode(pipeline, buffers, params, groups, threads);
                }
                Ok(())
            } else {
                let mut batch = batch::Batch::new(&self.queue)?;
                // SAFETY: checked shader contract; eager completion below.
                unsafe {
                    batch.encode(pipeline, buffers, params, groups, threads);
                }
                self.health.check(batch.finish())
            }
        })
    }

    fn elementwise(&self, name: &str, a: &MetalBuf, b: &MetalBuf, y: &mut MetalBuf) -> Result<()> {
        let n = geometry::elementwise(name, &[floats(a)?, floats(b)?], &[floats(y)?])?;
        // SAFETY: inputs contain n floats, output contains at least n,
        // and dispatch bounds preserve any unused output tail.
        unsafe { self.launch(name, &[a, b, y], &[dim(n)?], Dispatch::Threads([n, 1, 1])) }
    }
}

impl Backend for MetalBackend {
    type Buf = MetalBuf;

    fn join_execution(&self) -> Result<()> {
        self.finish()
    }
    fn execution_identity(&self) -> Option<std::sync::Arc<()>> {
        Some(self.identity.clone())
    }

    fn device_label(&self) -> Option<String> {
        Some(self.device_name())
    }
    fn unified_memory(&self) -> bool {
        self.device.hasUnifiedMemory()
    }

    /// What this device can use with good performance, less what it holds now.
    fn available_device_bytes(&self) -> Result<Option<usize>> {
        Ok(self
            .total_device_bytes()
            .map(|total| total.saturating_sub(self.device.currentAllocatedSize())))
    }

    fn total_device_bytes(&self) -> Option<usize> {
        usize::try_from(self.device.recommendedMaxWorkingSetSize()).ok()
    }

    fn decoder<'a>(
        &'a self,
        model: &'a crate::model::Model<Self>,
        kv: &'a mut crate::kv_cache::KvCache<Self>,
        ws: &'a mut crate::model::Workspace<Self>,
    ) -> Result<Box<dyn crate::model::decoder::Decoder + 'a>> {
        if !self.batch_decode {
            return Ok(Box::new(crate::model::decoder::EagerDecoder {
                backend: self,
                model,
                kv,
                ws,
            }));
        }
        Ok(Box::new(batch::BatchedDecoder::new(self, model, kv, ws)?))
    }

    fn supports_weight_dtype(&self, dtype: crate::gguf::GgmlDType) -> bool {
        dtype_code::decoder_code(dtype).is_ok()
    }

    fn alloc(&self, bytes: usize, _pool: Pool) -> Result<MetalBuf> {
        if bytes.max(4) > self.device.maxBufferLength() {
            return Err(Error::Backend(format!(
                "Metal allocation of {bytes} bytes exceeds device limit"
            )));
        }
        let buffer = self
            .device
            .newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| Error::Backend(format!("Metal allocation of {bytes} bytes failed")))?;
        // SAFETY: fresh shared allocation, writable for its entire length,
        // not yet exposed to either the GPU or another host thread.
        unsafe { std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, bytes.max(4)) };
        Ok(MetalBuf { buffer, len: bytes })
    }

    fn upload(&self, dst: &mut MetalBuf, offset: usize, src: &[u8]) -> Result<()> {
        geometry::range("metal upload", offset, src.len(), dst.len)?;
        self.before_upload(dst)?;
        // SAFETY: exclusive destination, bounded range, pending GPU accesses
        // to this allocation have completed. The mapped pointer never escapes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.buffer.contents().as_ptr().cast::<u8>().add(offset),
                src.len(),
            );
        }
        Ok(())
    }

    fn download(&self, src: &MetalBuf, offset: usize, dst: &mut [u8]) -> Result<()> {
        geometry::range("metal download", offset, dst.len(), src.len)?;
        self.finish()?;
        // SAFETY: bounded shared allocation; GPU writes completed before
        // their exclusive borrow ended. Concurrent readers are harmless.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.buffer.contents().as_ptr().cast::<u8>().add(offset),
                dst.as_mut_ptr(),
                dst.len(),
            );
        }
        Ok(())
    }

    fn copy(
        &self,
        src: &MetalBuf,
        src_off: usize,
        dst: &mut MetalBuf,
        dst_off: usize,
        len: usize,
    ) -> Result<()> {
        geometry::range("metal copy source", src_off, len, src.len)?;
        geometry::range("metal copy destination", dst_off, len, dst.len)?;
        // SAFETY: bounded, distinct allocations, byte offsets (including
        // unaligned offsets) are handled by the byte-copy kernel.
        unsafe {
            self.launch(
                "copy_bytes",
                &[src, dst],
                &[src_off as u64, dst_off as u64, len as u64],
                Dispatch::Threads([len, 1, 1]),
            )
        }
    }

    fn qmatmul(&self, w: &QWeight<MetalBuf>, x: &MetalBuf, y: &mut MetalBuf) -> Result<()> {
        let params = matmul_params(w, x, y)?;
        if self.use_tile(&params) {
            // SAFETY: exact storage/shapes checked; staging is f32, padded
            // rows/tokens/K are zero-filled, and final stores are guarded.
            return unsafe {
                self.launch(
                    "qmatmul_tile",
                    &[&w.buf, x, y],
                    &params,
                    Dispatch::Tiles {
                        rows: params[0] as usize,
                        tokens: params[2] as usize,
                    },
                )
            };
        }
        let dispatch = Dispatch::SimdRows {
            rows: params[0] as usize,
            batches: params[2] as usize,
        };
        // SAFETY: exact weight storage, input/output sizes and block
        // alignment checked above. Shader guards padded SIMD groups, and
        // every lane participates in the row reduction even for narrow cols.
        unsafe { self.launch("qmatmul", &[&w.buf, x, y], &params, dispatch) }
    }

    fn ffn_gate_up(
        &self,
        gate: &QWeight<MetalBuf>,
        up: &QWeight<MetalBuf>,
        x: &MetalBuf,
        gate_scratch: &mut MetalBuf,
        up_scratch: &mut MetalBuf,
        y: &mut MetalBuf,
    ) -> Result<()> {
        if gate.shape != up.shape {
            return Err(shape("ffn gate/up weight shapes must match"));
        }
        let g = matmul_params(gate, x, gate_scratch)?;
        let u = matmul_params(up, x, up_scratch)?;
        expect_floats(y, floats(gate_scratch)?)?;
        if self.use_tile(&g) {
            // Large prefill reuses each dequantized tile across eight tokens.
            // Scratch contents are unspecified by Backend; storing them is valid.
            self.qmatmul(gate, x, gate_scratch)?;
            self.qmatmul(up, x, up_scratch)?;
            return self.silu_mul(gate_scratch, up_scratch, y);
        }
        let params = [g[0], g[1], g[2], g[3], u[3]];
        let dispatch = Dispatch::SimdRows {
            rows: g[0] as usize,
            batches: g[2] as usize,
        };
        // SAFETY: both packed matrices and all shapes validated, including
        // mixed dtypes. Scratch contents are unspecified by Backend and are
        // not accessed: the two dot products stay in registers until SiLU.
        // Each active SIMD group writes exactly one exclusive output.
        unsafe {
            self.launch(
                "ffn_gate_up",
                &[&gate.buf, &up.buf, x, y],
                &params,
                dispatch,
            )
        }
    }

    fn embed(
        &self,
        table: &QWeight<MetalBuf>,
        token_ids: &[u32],
        out: &mut MetalBuf,
    ) -> Result<()> {
        let (vocab, cols, code) = weight_dims(table)?;
        expect_floats(out, product(token_ids.len(), cols)?)?;
        if token_ids.iter().any(|&id| id as usize >= vocab) {
            return Err(shape("embedding token is outside the vocabulary"));
        }
        let params = [dim(cols)?, dim(token_ids.len())?, code];
        if token_ids.is_empty() {
            return Ok(());
        }
        let mut ids = self.alloc(std::mem::size_of_val(token_ids), Pool::Scratch)?;
        self.upload(&mut ids, 0, bytemuck::cast_slice(token_ids))?;
        // SAFETY: ids are in range, table format/length and output checked;
        // command encoding retains temporary IDs through completion.
        unsafe {
            self.launch(
                "embed_rows",
                &[&table.buf, &ids, out],
                &params,
                Dispatch::Threads([cols, token_ids.len(), 1]),
            )
        }
    }

    fn rmsnorm(&self, w: &MetalBuf, x: &MetalBuf, y: &mut MetalBuf, eps: f32) -> Result<()> {
        let params = norm_params(w, x, y, eps)?;
        let dispatch = Dispatch::SimdRows {
            rows: params[1] as usize,
            batches: 1,
        };
        // SAFETY: one complete SIMD group per token row, output exclusive.
        unsafe { self.launch("rmsnorm_f32", &[w, x, y], &params, dispatch) }
    }

    fn add_rmsnorm(
        &self,
        w: &MetalBuf,
        a: &MetalBuf,
        b: &MetalBuf,
        sum: &mut MetalBuf,
        y: &mut MetalBuf,
        eps: f32,
    ) -> Result<()> {
        let params = norm_params(w, a, y, eps)?;
        let n = floats(a)?;
        expect_floats(b, n)?;
        expect_floats(sum, n)?;
        let dispatch = Dispatch::SimdRows {
            rows: params[1] as usize,
            batches: 1,
        };
        // SAFETY: both exclusive outputs and inputs have complete rows;
        // each SIMD group owns one row. The residual is preserved in sum
        // for the next model layer, not discarded after normalization.
        unsafe { self.launch("add_rmsnorm_f32", &[w, a, b, sum, y], &params, dispatch) }
    }

    fn add(&self, a: &MetalBuf, b: &MetalBuf, y: &mut MetalBuf) -> Result<()> {
        self.elementwise("add_f32", a, b, y)
    }

    fn silu_mul(&self, gate: &MetalBuf, up: &MetalBuf, y: &mut MetalBuf) -> Result<()> {
        self.elementwise("silu_mul_f32", gate, up, y)
    }

    fn rope(
        &self,
        q: &mut MetalBuf,
        k: &mut MetalBuf,
        pos: u32,
        n_tokens: usize,
        spec: &RopeSpec,
    ) -> Result<()> {
        let q_len = floats(q)?;
        let k_len = floats(k)?;
        if n_tokens == 0
            || spec.head_dim == 0
            || !spec.head_dim.is_multiple_of(2)
            || !spec.theta.is_finite()
            || spec.theta <= 0.0
        {
            return Err(shape(
                "rope requires tokens, even positive head_dim and positive finite theta",
            ));
        }
        let row = product(n_tokens, spec.head_dim)?;
        if q_len == 0 || k_len == 0 || q_len % row != 0 || k_len % row != 0 {
            return Err(shape("rope buffers must contain complete token/head rows"));
        }
        let q_heads = q_len / row;
        let k_heads = k_len / row;
        let heads = q_heads
            .checked_add(k_heads)
            .ok_or_else(|| shape("head count overflow"))?;
        dim(heads)?;
        let tokens = dim(n_tokens)?;
        pos.checked_add(tokens - 1)
            .ok_or_else(|| shape("rope position overflow"))?;
        let params = [
            pos,
            dim(spec.head_dim)?,
            dim(q_heads)?,
            dim(k_heads)?,
            tokens,
            spec.theta.to_bits(),
        ];
        // SAFETY: complete, nonaliasing q/k rows; every grid point touches
        // a distinct interleaved pair, positions cannot wrap.
        let frequencies = self.rope_frequencies(spec)?;
        unsafe {
            self.launch(
                "rope_f32",
                &[q, k, &frequencies],
                &params,
                Dispatch::Threads([spec.head_dim / 2, heads, n_tokens]),
            )
        }
    }

    fn supports(&self, operation: Operation) -> bool {
        match operation {
            Operation::Scale
            | Operation::GeluTanhMul
            | Operation::Softcap
            | Operation::RopeSplitHalf
            | Operation::WindowedAttention
            | Operation::AddBias
            | Operation::LayerNorm
            | Operation::Gelu
            | Operation::Argmax => true,
            Operation::RopePartialInterleaved
            | Operation::RopePartialSplitHalf
            | Operation::SigmoidMul
            | Operation::GatedDeltaNet => false,
        }
    }

    fn layernorm(&self, w: &MetalBuf, bias: Option<&MetalBuf>, x: &MetalBuf, y: &mut MetalBuf, eps: f32) -> Result<()> {
        self.hybrid_layernorm(w, bias, x, y, eps)
    }

    fn gelu(&self, x: &MetalBuf, y: &mut MetalBuf) -> Result<()> {
        self.hybrid_gelu(x, y)
    }

    fn add_bias(&self, bias: &MetalBuf, y: &mut MetalBuf) -> Result<()> {
        self.hybrid_add_bias(bias, y)
    }

    fn argmax(&self, x: &MetalBuf, index: &mut MetalBuf) -> Result<()> {
        self.hybrid_argmax(x, index)
    }

    fn scale(&self, x: &mut MetalBuf, factor: f32) -> Result<()> {
        if !factor.is_finite() {
            return Err(shape("nonfinite scale"));
        }
        let n = floats(x)?;
        // SAFETY: each thread owns one checked f32 element.
        unsafe {
            self.launch(
                "scale_f32",
                &[x],
                &[dim(n)?, factor.to_bits()],
                Dispatch::Threads([n, 1, 1]),
            )
        }
    }

    fn softcap(&self, x: &mut MetalBuf, cap: f32) -> Result<()> {
        geometry::softcap(cap)?;
        let n = floats(x)?;
        // SAFETY: each thread owns one checked f32 element.
        unsafe {
            self.launch(
                "softcap_f32",
                &[x],
                &[dim(n)?, cap.to_bits()],
                Dispatch::Threads([n, 1, 1]),
            )
        }
    }

    fn gelu_tanh_mul(&self, a: &MetalBuf, b: &MetalBuf, y: &mut MetalBuf) -> Result<()> {
        self.elementwise("gelu_tanh_mul_f32", a, b, y)
    }

    fn rope_split_half(
        &self,
        x: &mut MetalBuf,
        pos: u32,
        spec: &RopeSpec,
        factors: &MetalBuf,
    ) -> Result<()> {
        let n = floats(x)?;
        geometry::rope_split_half(n, floats(factors)?, spec)?;
        let heads = n / spec.head_dim;
        let p = [dim(spec.head_dim)?, dim(heads)?, pos, spec.theta.to_bits()];
        // SAFETY: complete heads and factors checked; pairs are disjoint.
        let frequencies = self.rope_frequencies(spec)?;
        unsafe {
            self.launch(
                "rope_split_half",
                &[x, factors, &frequencies],
                &p,
                Dispatch::Threads([spec.head_dim / 2, heads, 1]),
            )
        }
    }

    fn attn_windowed(
        &self,
        q: &MetalBuf,
        kv: &KvView<MetalBuf>,
        spec: &AttnSpec,
        scale: f32,
        window: usize,
        o: &mut MetalBuf,
    ) -> Result<()> {
        geometry::attention_scale(scale)?;
        self.attention(q, kv, spec, window, scale, o)
    }

    fn attn(
        &self,
        q: &MetalBuf,
        kv: &KvView<MetalBuf>,
        spec: &AttnSpec,
        o: &mut MetalBuf,
    ) -> Result<()> {
        self.attention(q, kv, spec, 0, 1.0 / (spec.head_dim as f32).sqrt(), o)
    }
}

impl MetalBackend {
    fn rope_frequencies(&self, spec: &RopeSpec) -> Result<std::sync::Arc<MetalBuf>> {
        let mut cache = self
            .rope_freqs
            .lock()
            .map_err(|_| Error::Backend("Metal RoPE constants lock poisoned".into()))?;
        let key = (spec.head_dim, spec.theta.to_bits());
        if let Some(buffer) = cache.get(&key) {
            return Ok(buffer.clone());
        }
        let frequencies: Vec<f32> = (0..spec.head_dim / 2)
            .map(|i| spec.theta.powf(-2.0 * i as f32 / spec.head_dim as f32))
            .collect();
        let mut buffer = self.alloc(frequencies.len() * 4, Pool::Scratch)?;
        self.upload(&mut buffer, 0, bytemuck::cast_slice(&frequencies))?;
        let buffer = std::sync::Arc::new(buffer);
        // Bound backend-private constants for callers varying RoPE parameters.
        // Outstanding launches retain an immutable Arc even across eviction.
        if cache.len() == abi::ROPE_CACHE_KEYS {
            cache.clear();
        }
        cache.insert(key, buffer.clone());
        Ok(buffer)
    }

    fn attention(
        &self,
        q: &MetalBuf,
        kv: &KvView<MetalBuf>,
        spec: &AttnSpec,
        window: usize,
        scale: f32,
        o: &mut MetalBuf,
    ) -> Result<()> {
        if spec.head_dim == 0
            || spec.n_q_heads == 0
            || spec.n_kv_heads == 0
            || !spec.n_q_heads.is_multiple_of(spec.n_kv_heads)
        {
            return Err(shape(
                "attention requires positive dimensions and divisible GQA heads",
            ));
        }
        let row = product(spec.n_q_heads, spec.head_dim)?;
        let n = floats(q)?;
        if n == 0 || n % row != 0 {
            return Err(shape("attention requires complete, nonempty query rows"));
        }
        let tokens = n / row;
        if tokens > kv.len {
            return Err(shape("attention queries exceed initialized KV length"));
        }
        expect_floats(o, n)?;
        let kv_elems = product(kv.len, product(spec.n_kv_heads, spec.head_dim)?)?;
        if floats(kv.k)? < kv_elems || floats(kv.v)? < kv_elems {
            return Err(shape(
                "attention KV buffers are shorter than initialized length",
            ));
        }
        dim(kv.len)?;
        let params = [
            dim(spec.head_dim)?,
            dim(spec.n_q_heads)?,
            dim(spec.n_kv_heads)?,
            dim(tokens)?,
            dim(kv.len - tokens)?,
            scale.to_bits(),
            dim(window)?,
        ];
        let lanes = self.pipeline("attn_simd")?.threadExecutionWidth();
        let (kernel, dispatch) = if spec.head_dim <= product(lanes, abi::ATTN_ELEMENTS_PER_LANE)? {
            (
                "attn_simd",
                Dispatch::SimdRows {
                    rows: spec.n_q_heads,
                    batches: tokens,
                },
            )
        } else {
            // The cooperative shader holds sixteen dimensions per lane in
            // registers. Preserve arbitrary head sizes via the scalar path.
            ("attn_f32", Dispatch::Threads([spec.n_q_heads, tokens, 1]))
        };
        // SAFETY: query/output lengths and visible KV prefix checked; GQA
        // maps each query to a valid KV head. Cooperative register capacity
        // is checked against this pipeline's SIMD width; each output has
        // exactly one writer in either dispatch layout.
        unsafe { self.launch(kernel, &[q, kv.k, kv.v, o], &params, dispatch) }
    }
}

#[cfg(test)]
mod tests;
