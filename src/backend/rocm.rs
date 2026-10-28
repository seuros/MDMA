// HIP backend: RDNA4 uses packed-integer decode and int8 WMMA prefill;
// other build targets use portable GEMV and f32 activations/KV.
use super::{
    AttnSpec, Backend, Error, KvView, Operation, Pool, QWeight, Result, RopeSpec, dtype_code, shape,
};
use crate::gguf::GgmlDType;
use crate::memory::{BufferBudget, product, sum};
use memory::DeviceMemory;
use rocm_rs::hip::{Device, Dim3, Module, Stream, calculate_grid_1d, ffi};
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

const HSACO: &[u8] = include_bytes!(env!("MDMA_HIP_KERNELS_HSACO"));

fn has_rdna4_wmma(arch: &str) -> bool {
    matches!(arch.split(':').next(), Some("gfx1200" | "gfx1201"))
}

// Launch geometry mirrored from kernels.hip's #defines and tile shapes.
// The kernel side is the source of truth; a drift here under-sizes
// dynamic LDS or mis-tiles the grid with no compile-time error.
const WARP: u32 = 32;
const MAX_BLOCK_THREADS: u32 = 1024;
const QK_K: u32 = 256;
const Q8_1_BLOCK: u32 = 32;
const Q8_1_LANES: u32 = 8; // quantize_q8_1_kernel threads per 32-float block
const M_TILE: u32 = 64; // output rows per GEMM block: 4 warps x one 16-row fragment
const TOKEN_SUBTILE: u32 = 16; // one WMMA fragment along the token axis
const GEMM_THREADS: u32 = 4 * WARP;
const I9_AS_STRIDE: u32 = 272; // padded LDS row of one Q4_K/Q6_K superblock
const I9_BS_STRIDE: u32 = 144; // padded LDS row of one 128-K half of Q8_1 values
const ATT_KT: u32 = 16; // keys per staged K/V tile
const ATT_QT: u32 = 8; // max queries per attention block
const ATT_MAX_HEAD_DIM: usize = 4 * WARP as usize; // tiled kernel holds <= 4 dims per lane
const ATT_WMMA_HEAD_DIM: usize = 128; // the one instantiated attn_prefill_wmma_hd<N>
const ATT_DECODE_HEAD_DIM: usize = 128; // attn_decode_hd128, also used by graph replay
const ATT_DECODE_MAX_KV: usize = 4096; // validated cooperative-decode LDS limit
const ATT_DECODE_THREADS: u32 = 128;
const ATT_GENERIC_THREADS: u32 = 64;
const ATT_DC: u32 = 32; // flash-decoding keys per block
const ATT_DPART: u32 = 132; // floats per (head, chunk) partial
const ATT_SPLIT_GROUP: usize = 4; // the instantiated attn_decode_split_hd128_g<N>
const ATT_DMAX_KV: usize = 128 * ATT_DC as usize; // chunk weights the fused combine holds in LDS
const ATT_COUNTERS_BYTES: usize = 256; // arrival counters (one u32 per KV head) ahead of the partials
const TARGET_BLOCKS: u32 = 256; // ~4 blocks/WGP on a 64-CU gfx1201
const F32_BYTES: u32 = size_of::<f32>() as u32;
const F16_BYTES: u32 = size_of::<half::f16>() as u32;

fn gemm_token_tile(kernel: &str, requested: u32, tokens: u32) -> u32 {
    if kernel.starts_with("qmatmul_gemm_wmma_i8_k256_nt")
        || kernel.starts_with("qmatmul_gemm_wmma_i8_k256_packed_nt")
        || kernel.starts_with("qmatmul_gemm_wmma_i8_q6k_k256_nt")
    {
        // These kernels lay out LDS using compile-time NT, including tail tiles.
        requested
    } else {
        requested
            .min(tokens.div_ceil(TOKEN_SUBTILE) * TOKEN_SUBTILE)
            .max(TOKEN_SUBTILE)
    }
}

fn hip_err(e: rocm_rs::hip::error::Error) -> Error {
    if e.is_out_of_memory() {
        Error::OutOfMemory(e.to_string())
    } else {
        Error::Backend(e.to_string())
    }
}

pub struct HipBuf {
    mem: DeviceMemory<u8>,
    len: usize,
    packed_q4: bool,
}

// SAFETY: HIP permits device pointers to cross threads. Writes require
// exclusive buffer borrows, and operations synchronize through the backend.
unsafe impl Send for HipBuf {}
unsafe impl Sync for HipBuf {}

impl HipBuf {
    fn as_device_ptr(&self) -> *mut c_void {
        self.mem.as_ptr()
    }
}

pub struct HipBackend {
    device_name: String,
    unified_memory: bool,
    module: Module,
    // Eager operations share this backend's stream; graphs own separate backends.
    stream: Option<Stream>,
    replay: Option<graph::ReplayParams>,
    // Serializes access to the module, stream and shared scratch across sessions.
    lock: Mutex<()>,
    // Q8_1-quantized activations for decode and prefill: q8[T][n_in]
    // followed by per-(token, 32-block) {d, sum} float pairs. Grown on
    // demand. Lock order is always `lock` then `q8_scratch`, and the
    // outer `lock` is held across quantize + consume so no other
    // launch can interleave between them on this backend's stream.
    q8_scratch: Mutex<Q8Scratch>,
    // Flash-decoding partials: [n_q_heads][chunks][ATT_DPART] f32. Grown
    // on demand by eager decode; a graph backend reserves it for the
    // capacity before capture (no allocation inside a capture).
    attn_scratch: Mutex<AttnScratch>,
    buffers: BufferBudget,
    f32_compute: bool,
    graphs_enabled: bool,
    device_identity: String,
    health: crate::federation::profile::gpu::Health,
    identity: Arc<()>,
    profile_owner: bool,
}

struct AttnScratch {
    mem: Option<DeviceMemory<u8>>,
    cap: usize,
}

struct Q8Scratch {
    mem: Option<DeviceMemory<u8>>,
    cap: usize,
    // (x pointer, n_tokens, n_in) whose Q8_1 quantization currently sits in
    // `mem`. Every device-buffer write invalidates this key; consecutive
    // matmuls may reuse it while the input is unchanged.
    cached: Option<(usize, i32, i32)>,
}

impl Q8Scratch {
    /// Caller holds the backend launch lock and the scratch guard. Sync
    /// before replacing memory, and invalidate the cached input only on
    /// successful growth. A failed sync/allocation leaves the old state live.
    fn reserve(&mut self, need: usize, budget: &BufferBudget) -> Result<()> {
        if self.cap < need {
            rocm_rs::hip::device_synchronize().map_err(hip_err)?;
            self.mem = Some(DeviceMemory::<u8>::new(need, budget)?);
            self.cap = need;
            self.cached = None;
        }
        Ok(())
    }
}

// SAFETY: the raw HIP handles inside (Module, Stream, DeviceMemory scratch/params) are not thread-affine: they
// live in device 0's process-wide primary context, and device 0 is HIP's default current device on every thread.
// Every launch, copy and scratch growth on `module`/`stream` runs under `lock` (scratch also behind its own Mutex,
// order `lock` -> `q8_scratch`); `replay` is fixed at capture and its params are written only by the graph decoder
// that exclusively owns that backend. BufferBudget, Health and Arc<()> are already Send + Sync.
unsafe impl Send for HipBackend {}
unsafe impl Sync for HipBackend {}

impl Drop for HipBackend {
    fn drop(&mut self) {
        if self.profile_owner {
            self.health.withdraw();
        }
        if let Some(stream) = &self.stream {
            let _ = self.health.check(stream.synchronize().map_err(hip_err));
        }
    }
}

impl HipBackend {
    pub fn new() -> Result<Self> {
        Self::create(false)
    }

    /// GPU-only reference mode: f32 activations and KV, no int8/WMMA or graphs.
    /// Also selected automatically on build targets without RDNA4 WMMA.
    pub fn new_f32() -> Result<Self> {
        Self::create(true)
    }

    fn create(f32_compute: bool) -> Result<Self> {
        let f32_compute = f32_compute || !has_rdna4_wmma(env!("MDMA_HIP_ARCH_BUILT"));
        let device = Device::new(0).map_err(hip_err)?;
        device.set_current().map_err(hip_err)?;
        let properties = device.properties().map_err(hip_err)?;
        // SAFETY: hipDeviceProp_tR0600 holds only integers, fixed arrays, hipUUID, the
        // hipDeviceArch_t bitfield and two raw pointers, so all-zero is a valid value.
        let mut raw = unsafe { std::mem::zeroed::<ffi::hipDeviceProp_tR0600>() };
        // SAFETY: `raw` is a live, aligned hipDeviceProp_tR0600 out-param, and the R0600
        // entry point is the one whose layout these bindings mirror; device 0 exists
        // (Device::new(0) succeeded above). The status is checked before `raw` is read.
        let status = unsafe { ffi::hipGetDevicePropertiesR0600(&mut raw, 0) };
        rocm_rs::hip::error::Error::from_hip_error::<()>(status).map_err(hip_err)?;
        // SAFETY: only reached on success. HIP writes gcnArchName (char[256]) NUL-terminated,
        // and the zeroed init leaves a terminator in the array for any shorter write; the
        // CStr borrows `raw`, which outlives its use in this function.
        let arch = unsafe { std::ffi::CStr::from_ptr(raw.gcnArchName.as_ptr()) }.to_string_lossy();
        let device_identity = format!(
            "{arch}/pci-{}:{}:{}/uuid-{:?}",
            raw.pciDomainID, raw.pciBusID, raw.pciDeviceID, raw.uuid.bytes
        );
        let module = Module::load_data(HSACO).map_err(hip_err)?;
        let stream = Some(
            device
                .get_stream_with_flags(rocm_rs::hip::bindings::hipStreamNonBlocking)
                .map_err(hip_err)?,
        );
        Ok(Self {
            device_name: properties.name,
            unified_memory: properties.integrated != 0,
            module,
            stream,
            replay: None,
            lock: Mutex::new(()),
            q8_scratch: Mutex::new(Q8Scratch {
                mem: None,
                cap: 0,
                cached: None,
            }),
            attn_scratch: Mutex::new(AttnScratch { mem: None, cap: 0 }),
            buffers: BufferBudget::default(),
            f32_compute,
            graphs_enabled: std::env::var("MDMA_HIP_GRAPHS").as_deref() != Ok("0"),
            device_identity,
            health: Default::default(),
            identity: Arc::new(()),
            profile_owner: true,
        })
    }

    pub(crate) fn federation_profile(
        &self,
    ) -> std::io::Result<crate::federation::profile::gpu::Gpu> {
        use crate::federation::profile::gpu::{Gpu, fields};
        use sha2::{Digest, Sha256};
        let mut runtime = 0;
        let mut driver = 0;
        for status in [
            // SAFETY: both calls only write one c_int through a pointer to a live local and
            // need no device, stream or context.
            unsafe { ffi::hipRuntimeGetVersion(&mut runtime) },
            unsafe { ffi::hipDriverGetVersion(&mut driver) },
        ] {
            rocm_rs::hip::error::Error::from_hip_error::<()>(status)
                .map_err(std::io::Error::other)?;
        }
        if runtime <= 0 || driver <= 0 {
            return Err(std::io::Error::other("unverifiable HIP runtime"));
        }
        Ok(Gpu::immutable(
            "rocm",
            "f32",
            if self.f32_compute { "f32" } else { "f16" },
            fields([
                ("vocabulary", "mdma.rocm.v1".into()),
                ("workload", "decoder".into()),
                ("device_name", self.device_name.clone()),
                ("device_identity", self.device_identity.clone()),
                ("runtime_version", runtime.to_string()),
                ("driver_version", driver.to_string()),
                ("kernel_target", env!("MDMA_HIP_ARCH_BUILT").into()),
                (
                    "kernel_id",
                    format!("sha256:{}", hex::encode(Sha256::digest(HSACO))),
                ),
                ("weights", "gguf-raw-matrices-f32-norms".into()),
                (
                    "compute",
                    if self.f32_compute {
                        "f32-reference"
                    } else {
                        "rdna4-wmma-q8_1-f16"
                    }
                    .into(),
                ),
                (
                    "execution",
                    if self.graphs_enabled && !self.f32_compute {
                        "eligible-request-graph-with-eager-fallback"
                    } else {
                        "eager"
                    }
                    .into(),
                ),
                ("attention", "geometry-selected-flash-or-generic".into()),
                ("target_arch", std::env::consts::ARCH.into()),
                ("target_os", std::env::consts::OS.into()),
            ]),
            self.health.clone(),
        ))
    }

    /// Bytes of flash-decoding partials for `n_q_heads` heads over a KV
    /// length of `kv_len`.
    fn attn_partials_bytes(n_q_heads: usize, kv_len: usize) -> Result<usize> {
        product(&[
            n_q_heads,
            kv_len.max(1).div_ceil(ATT_DC as usize),
            ATT_DPART as usize,
            F32_BYTES as usize,
        ])
    }

    /// Ensure the partials buffer holds `bytes`; caller holds `self.lock`
    /// (or is single-threaded setup before a graph capture).
    pub(super) fn reserve_attn_scratch(
        &self,
        partial_bytes: usize,
        n_kv_heads: usize,
    ) -> Result<*mut c_void> {
        if n_kv_heads > ATT_COUNTERS_BYTES / F32_BYTES as usize {
            return Err(Error::ShapeMismatch(format!(
                "attn: {n_kv_heads} KV heads exceed the arrival-counter block"
            )));
        }
        let bytes = sum([ATT_COUNTERS_BYTES, partial_bytes])?;
        let mut scratch = self.attn_scratch.lock().unwrap();
        if scratch.cap < bytes {
            rocm_rs::hip::device_synchronize().map_err(hip_err)?;
            // Arrival counters must be zero before the first split kernel.
            // The memset uses the null stream; nonblocking streams need an
            // explicit sync before reading its result.
            let mut mem = DeviceMemory::<u8>::new(bytes, &self.buffers)?;
            mem.memset(0)?;
            rocm_rs::hip::device_synchronize().map_err(hip_err)?;
            scratch.mem = Some(mem);
            scratch.cap = bytes;
        }
        Ok(scratch.mem.as_ref().expect("reserved above").as_ptr())
    }

    /// Any op that writes a device buffer calls this first: the Q8_1
    /// scratch may hold that buffer's quantization. Never held together
    /// with `lock` from a caller that doesn't already hold `lock` (lock
    /// order is always `lock` then `q8_scratch`), and released before
    /// returning, so there is no cycle with qmatmul_i8.
    fn q8_invalidate(&self) {
        self.q8_scratch.lock().unwrap().cached = None;
    }

    fn launch1d(&self, name: &str, n: usize, args: &mut [*mut c_void]) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        self.q8_invalidate();
        let func = self.module.get_function(name).map_err(hip_err)?;
        let block = 256u32;
        let grid = calculate_grid_1d(n as u32, block);
        func.launch(grid, Dim3::new_1d(block), 0, self.stream.as_ref(), args)
            .map_err(hip_err)?;
        Ok(())
    }
}

macro_rules! arg {
    ($v:expr) => {
        &mut $v as *mut _ as *mut c_void
    };
}

mod decoder_ops;
mod hybrid;

impl Backend for HipBackend {
    type Buf = HipBuf;

    fn execution_identity(&self) -> Option<Arc<()>> {
        Some(self.identity.clone())
    }

    fn join_execution(&self) -> Result<()> {
        self.health.check((|| {
            let _lock = self
                .lock
                .lock()
                .map_err(|_| Error::Backend("HIP launch lock poisoned".into()))?;
            Device::new(0)
                .and_then(|device| device.synchronize())
                .map_err(hip_err)
        })())
    }

    fn device_label(&self) -> Option<String> {
        Some(self.device_name.clone())
    }
    fn unified_memory(&self) -> bool {
        self.unified_memory
    }

    fn available_device_bytes(&self) -> Result<Option<usize>> {
        memory_info().map(|(free, _)| Some(free))
    }

    fn total_device_bytes(&self) -> Option<usize> {
        memory_info().ok().map(|(_, total)| total)
    }

    fn weight_cache_profile(&self) -> Option<&str> {
        use sha2::{Digest, Sha256};
        static PROFILE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        Some(PROFILE.get_or_init(|| {
            format!(
                "rocm/{}/q4k-scales-v3/{}",
                env!("MDMA_HIP_ARCH_BUILT"),
                hex::encode(Sha256::digest(HSACO)),
            )
        }))
    }

    fn upload_weight(&self, bytes: &[u8], layout: crate::cache::WeightLayout) -> Result<Self::Buf> {
        let packed_q4 = layout == crate::cache::WeightLayout::RocmQ4kV3;
        if packed_q4
            && !bytes
                .len()
                .is_multiple_of(crate::cache::layout::Q4K_BLOCK_BYTES)
        {
            return Err(Error::ShapeMismatch("partial packed Q4_K block".into()));
        }
        let mut buf = self.alloc(bytes.len(), Pool::Weights)?;
        self.upload(&mut buf, 0, bytes)?;
        buf.packed_q4 = packed_q4;
        Ok(buf)
    }

    super::delegate_decoder!(graph::decoder);

    fn supports(&self, op: Operation) -> bool {
        matches!(
            op,
            Operation::Scale
                | Operation::GeluTanhMul
                | Operation::Softcap
                | Operation::RopeSplitHalf
                | Operation::WindowedAttention
                | Operation::AddBias
                | Operation::RopePartialInterleaved
                | Operation::LayerNorm
                | Operation::Gelu
                | Operation::Argmax
        )
    }

    fn supports_weight_dtype(&self, dtype: GgmlDType) -> bool {
        dtype_code::decoder_code(dtype).is_ok()
    }

    fn max_head_dim(&self) -> Option<usize> {
        Some(512)
    }

    fn layernorm(&self, w: &HipBuf, bias: Option<&HipBuf>, x: &HipBuf, y: &mut HipBuf, eps: f32) -> Result<()> {
        self.hybrid_layernorm(w, bias, x, y, eps)
    }

    fn gelu(&self, x: &HipBuf, y: &mut HipBuf) -> Result<()> { self.hybrid_gelu(x, y) }

    fn argmax(&self, x: &HipBuf, index: &mut HipBuf) -> Result<()> { self.hybrid_argmax(x, index) }

    fn scale(&self, x: &mut HipBuf, factor: f32) -> Result<()> {
        if !factor.is_finite() {
            return Err(Error::ShapeMismatch("scale: factor must be finite".into()));
        }
        self.unary(x, factor, 0)
    }

    fn softcap(&self, x: &mut HipBuf, cap: f32) -> Result<()> {
        shape::softcap(cap)?;
        self.unary(x, cap, 1)
    }

    fn gelu_tanh_mul(&self, a: &HipBuf, b: &HipBuf, y: &mut HipBuf) -> Result<()> {
        let n = shape::elementwise(
            "gelu_tanh_mul",
            &[shape::elems("gelu", a.len)?, shape::elems("gelu", b.len)?],
            &[shape::elems("gelu", y.len)?],
        )?;
        if n == 0 {
            return Ok(());
        }
        let mut n_i = n as i32;
        self.launch1d(
            "gelu_tanh_mul_kernel",
            n,
            &mut [
                arg!(a.as_device_ptr()),
                arg!(b.as_device_ptr()),
                arg!(y.as_device_ptr()),
                arg!(n_i),
            ],
        )
    }

    fn rope_split_half(
        &self,
        x: &mut HipBuf,
        pos: u32,
        spec: &RopeSpec,
        factors: &HipBuf,
    ) -> Result<()> {
        self.split_rope(x, pos, spec, factors)
    }

    fn rope_partial_interleaved(
        &self,
        x: &mut HipBuf,
        pos: u32,
        spec: &RopeSpec,
        rotary_dim: usize,
    ) -> Result<()> {
        self.partial_rope(x, pos, spec, rotary_dim)
    }

    fn add_bias(&self, b: &HipBuf, y: &mut HipBuf) -> Result<()> {
        let width = shape::elems("add_bias", b.len)?;
        let n = shape::elems("add_bias", y.len)?;
        shape::rows("add_bias", width, n, n)?;
        if n == 0 {
            return Ok(());
        }
        let mut width = i32::try_from(width).map_err(|_| Error::SizeOverflow)?;
        let mut n_i = i32::try_from(n).map_err(|_| Error::SizeOverflow)?;
        self.launch1d(
            "add_bias_kernel",
            n,
            &mut [
                arg!(b.as_device_ptr()),
                arg!(y.as_device_ptr()),
                arg!(width),
                arg!(n_i),
            ],
        )
    }

    fn attn_windowed(
        &self,
        q: &HipBuf,
        kv: &KvView<HipBuf>,
        spec: &AttnSpec,
        scale: f32,
        window: usize,
        o: &mut HipBuf,
    ) -> Result<()> {
        self.attention_generic(q, kv, spec, scale, window, o)
    }

    fn set_buffer_budget(&mut self, budget: BufferBudget) -> Result<()> {
        self.buffers.replace(budget)
    }

    fn buffer_budget(&self) -> Option<&BufferBudget> {
        Some(&self.buffers)
    }

    fn alloc(&self, bytes: usize, _pool: Pool) -> Result<Self::Buf> {
        let mem = DeviceMemory::<u8>::new(bytes.max(1), &self.buffers)?;
        Ok(HipBuf {
            mem,
            len: bytes,
            packed_q4: false,
        })
    }

    fn upload(&self, dst: &mut Self::Buf, offset: usize, src: &[u8]) -> Result<()> {
        self.q8_invalidate();
        shape::range("upload", offset, src.len(), dst.len)?;
        let _guard = self.lock.lock().unwrap();
        // SAFETY: shape::range proved offset + src.len() <= dst.len, and dst owns at least
        // dst.len bytes (alloc uses len.max(1)), so the add and the device write stay in
        // dst's hipMalloc range. `src` is a live host slice of src.len() bytes, and the
        // stream is synchronized below before returning, so the copy is done reading it
        // while it is still borrowed (only a failing sync, i.e. a faulted stream, returns
        // first). The stream is always Some (create() sets it); `lock` orders this launch.
        let status = unsafe {
            ffi::hipMemcpyAsync(
                dst.as_device_ptr().add(offset),
                src.as_ptr() as *const c_void,
                src.len(),
                ffi::hipMemcpyKind_hipMemcpyHostToDevice,
                self.stream.as_ref().unwrap().as_raw(),
            )
        };
        if status != ffi::hipError_t_hipSuccess {
            return Err(Error::Backend(format!("hipMemcpy H2D failed: {status:?}")));
        }
        self.stream
            .as_ref()
            .unwrap()
            .synchronize()
            .map_err(hip_err)?;
        Ok(())
    }

    fn download(&self, src: &Self::Buf, offset: usize, dst: &mut [u8]) -> Result<()> {
        shape::range("download", offset, dst.len(), src.len)?;
        let _guard = self.lock.lock().unwrap();
        // SAFETY: shape::range proved offset + dst.len() <= src.len, which src's allocation
        // covers, so the add and the device read stay in range. `dst` is an exclusively
        // borrowed host slice of dst.len() bytes, and the stream sync below completes the
        // async write into it before the borrow ends (only a failing sync returns first).
        // Stream order puts the copy after every earlier kernel that wrote `src`.
        let status = unsafe {
            ffi::hipMemcpyAsync(
                dst.as_mut_ptr() as *mut c_void,
                src.as_device_ptr().add(offset),
                dst.len(),
                ffi::hipMemcpyKind_hipMemcpyDeviceToHost,
                self.stream.as_ref().unwrap().as_raw(),
            )
        };
        if status != ffi::hipError_t_hipSuccess {
            return Err(Error::Backend(format!("hipMemcpy D2H failed: {status:?}")));
        }
        self.stream
            .as_ref()
            .unwrap()
            .synchronize()
            .map_err(hip_err)?;
        Ok(())
    }

    fn copy(
        &self,
        src: &Self::Buf,
        src_off: usize,
        dst: &mut Self::Buf,
        dst_off: usize,
        len: usize,
    ) -> Result<()> {
        self.q8_invalidate();
        shape::range("copy", src_off, len, src.len)?;
        shape::range("copy", dst_off, len, dst.len)?;
        let _guard = self.lock.lock().unwrap();
        // SAFETY: the two shape::range checks keep [src_off, +len) and [dst_off, +len)
        // inside their buffers' allocations. `src` and `&mut dst` are distinct HipBufs,
        // hence distinct hipMalloc allocations, so the regions cannot overlap. The copy is
        // device-to-device, so no host memory must outlive it, and a later hipFree of
        // either buffer implicitly synchronizes the device first.
        let status = unsafe {
            ffi::hipMemcpyAsync(
                dst.as_device_ptr().add(dst_off),
                src.as_device_ptr().add(src_off),
                len,
                ffi::hipMemcpyKind_hipMemcpyDeviceToDevice,
                self.stream.as_ref().unwrap().as_raw(),
            )
        };
        if status != ffi::hipError_t_hipSuccess {
            return Err(Error::Backend(format!("hipMemcpy D2D failed: {status:?}")));
        }
        Ok(())
    }

    /// Decode fuses the three projections into one packed-int8 launch.
    fn qmatmul_qkv(
        &self,
        wq: &QWeight<Self::Buf>,
        wk: &QWeight<Self::Buf>,
        wv: &QWeight<Self::Buf>,
        x: &Self::Buf,
        q: &mut Self::Buf,
        k: &mut Self::Buf,
        v: &mut Self::Buf,
    ) -> Result<()> {
        let kquant = |w: &QWeight<Self::Buf>| {
            matches!(w.dtype, GgmlDType::Q4K | GgmlDType::Q6K) && w.shape.len() == 2
        };
        let one_token = wq.shape.len() == 2 && x.len == wq.shape[1] * F32_BYTES as usize;
        if !self.f32_compute && one_token && kquant(wq) && kquant(wk) && kquant(wv) {
            return self.launch_gemv_i8_x3([wq, wk, wv], x, [q, k, v]);
        }
        self.qmatmul(wq, x, q)?;
        self.qmatmul(wk, x, k)?;
        self.qmatmul(wv, x, v)
    }

    #[allow(clippy::too_many_arguments)]
    fn rope_kv_write(
        &self,
        q: &mut Self::Buf,
        k: &mut Self::Buf,
        v: &Self::Buf,
        dst_k: &mut Self::Buf,
        dst_v: &mut Self::Buf,
        elem_offset: usize,
        pos: u32,
        n_tokens: usize,
        spec: &RopeSpec,
        n_elems: usize,
    ) -> Result<()> {
        if self.f32_compute {
            self.rope(q, k, pos, n_tokens, spec)?;
            return self.kv_write(k, v, dst_k, dst_v, elem_offset, n_elems);
        }
        self.q8_invalidate();
        let half = spec.head_dim / 2;
        let q_elems = q.len / F32_BYTES as usize;
        let k_elems = k.len / F32_BYTES as usize;
        let dst_bytes = (elem_offset + n_elems) * F16_BYTES as usize;
        if n_tokens == 0
            || q_elems % n_tokens != 0
            || k_elems % n_tokens != 0
            || k_elems != n_elems
            || v.len != k.len
            || dst_bytes > dst_k.len
            || dst_bytes > dst_v.len
            || spec.head_dim == 0
            || spec.head_dim % 2 != 0
            || (k_elems / n_tokens) % spec.head_dim != 0
            || (q_elems / n_tokens) % spec.head_dim != 0
        {
            return Err(Error::ShapeMismatch(
                "rope_kv_write: q/k/v/cache shapes disagree".into(),
            ));
        }
        let q_stride = q_elems / n_tokens;
        let k_stride = k_elems / n_tokens;
        let n_q_heads = q_stride / spec.head_dim;
        let n_kv_heads = k_stride / spec.head_dim;
        if let Some(replay) = &self.replay {
            if k_stride * replay.capacity * F16_BYTES as usize > dst_k.len {
                return Err(Error::ShapeMismatch(
                    "graph KV buffers do not match capacity".into(),
                ));
            }
        }
        let total = (n_q_heads + n_kv_heads) * half;
        let _guard = self.lock.lock().unwrap();
        let graph = self.replay.is_some();
        let func = self
            .module
            .get_function(if graph {
                "rope_kv_graph"
            } else {
                "rope_kv_kernel"
            })
            .map_err(hip_err)?;
        let mut off_i = elem_offset as i32;
        let mut pos_mut = pos;
        let mut head_dim_i = spec.head_dim as i32;
        let mut n_q_i = n_q_heads as i32;
        let mut n_kv_i = n_kv_heads as i32;
        let mut theta = spec.theta;
        let mut q_stride_i = q_stride as i32;
        let mut k_stride_i = k_stride as i32;
        let mut params = self
            .replay
            .as_ref()
            .map_or(std::ptr::null_mut(), |p| p.mem.as_ptr());
        let mut args = [
            arg!(q.as_device_ptr()),
            arg!(k.as_device_ptr()),
            arg!(v.as_device_ptr()),
            arg!(dst_k.as_device_ptr()),
            arg!(dst_v.as_device_ptr()),
            arg!(off_i),
            arg!(pos_mut),
            arg!(head_dim_i),
            arg!(n_q_i),
            arg!(n_kv_i),
            arg!(theta),
            arg!(q_stride_i),
            arg!(k_stride_i),
            arg!(params),
        ];
        let n_args = if graph { 14 } else { 13 };
        let block = 256u32;
        func.launch(
            Dim3::new_2d((total as u32).div_ceil(block), n_tokens as u32),
            Dim3::new_1d(block),
            0,
            self.stream.as_ref(),
            &mut args[..n_args],
        )
        .map_err(hip_err)?;
        Ok(())
    }

    /// The fast path rounds KV to f16; reference mode keeps f32.
    fn kv_elem_bytes(&self) -> usize {
        if self.f32_compute {
            F32_BYTES as usize
        } else {
            F16_BYTES as usize
        }
    }

    fn kv_write(
        &self,
        k: &Self::Buf,
        v: &Self::Buf,
        dst_k: &mut Self::Buf,
        dst_v: &mut Self::Buf,
        elem_offset: usize,
        n_elems: usize,
    ) -> Result<()> {
        let src_bytes = n_elems * F32_BYTES as usize;
        let dst_bytes = (elem_offset + n_elems) * self.kv_elem_bytes();
        if src_bytes > k.len || src_bytes > v.len || dst_bytes > dst_k.len || dst_bytes > dst_v.len
        {
            return Err(Error::ShapeMismatch(
                "kv_write: rows exceed the source or cache buffers".into(),
            ));
        }
        if self.f32_compute {
            self.copy(k, 0, dst_k, elem_offset * 4, src_bytes)?;
            return self.copy(v, 0, dst_v, elem_offset * 4, src_bytes);
        }
        let mut n = n_elems as i32;
        if let Some(replay) = &self.replay {
            if n_elems * replay.capacity * F16_BYTES as usize > dst_k.len {
                return Err(Error::ShapeMismatch(
                    "graph KV buffers do not match capacity".into(),
                ));
            }
            let mut params = replay.mem.as_ptr();
            self.launch1d(
                "kv_write_graph",
                n_elems,
                &mut [
                    arg!(k.as_device_ptr()),
                    arg!(v.as_device_ptr()),
                    arg!(dst_k.as_device_ptr()),
                    arg!(dst_v.as_device_ptr()),
                    arg!(n),
                    arg!(params),
                ],
            )
        } else {
            let mut offset = elem_offset as i32;
            self.launch1d(
                "kv_write_f16",
                n_elems,
                &mut [
                    arg!(k.as_device_ptr()),
                    arg!(v.as_device_ptr()),
                    arg!(dst_k.as_device_ptr()),
                    arg!(dst_v.as_device_ptr()),
                    arg!(n),
                    arg!(offset),
                ],
            )
        }
    }

    /// K-quants use packed int8 GEMV for decode and int8 WMMA for prefill.
    fn qmatmul(&self, w: &QWeight<Self::Buf>, x: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        let shape::Matmul {
            n_out,
            n_in,
            n_tokens,
        } = shape::matmul(&w.shape, x.len / 4, y.len / 4)?;
        shape::kquant_aligned("qmatmul", w.dtype, n_in)?;
        let dtype = prepared::dtype(w)?;
        let n_out_i = n_out as i32;
        let n_in_i = n_in as i32;
        if self.f32_compute || matches!(w.dtype, GgmlDType::BF16 | GgmlDType::Q5K) {
            self.q8_invalidate();
            for t in 0..n_tokens {
                // Shape validation above bounds both token slices.
                // SAFETY: shape::matmul gives x.len / 4 == n_tokens * n_in and
                // y.len / 4 >= n_tokens * n_out; with t < n_tokens both byte offsets stay
                // inside their buffers' hipMalloc ranges.
                let xp = unsafe { x.as_device_ptr().add(t * n_in * 4) };
                let yp = unsafe { y.as_device_ptr().add(t * n_out * 4) };
                self.launch_gemv(w.buf.as_device_ptr(), dtype, n_out_i, n_in_i, xp, yp)?;
            }
            return Ok(());
        }
        match (w.dtype, n_tokens) {
            (GgmlDType::Q4K | GgmlDType::Q6K, 1) => self.launch_gemv_i8(w, x, y),
            (_, 1) => {
                self.q8_invalidate();
                self.launch_gemv(
                    w.buf.as_device_ptr(),
                    dtype,
                    n_out_i,
                    n_in_i,
                    x.as_device_ptr(),
                    y.as_device_ptr(),
                )
            }
            (GgmlDType::Q4K | GgmlDType::Q6K, _) => {
                let nt = match (n_tokens as u32).div_ceil(TOKEN_SUBTILE) {
                    0..=1 => 1,
                    2 => 2,
                    3..=4 => 4,
                    _ => 8,
                };
                let n_tile = nt * TOKEN_SUBTILE;
                let kernel = match (w.dtype, nt) {
                    (GgmlDType::Q4K, 1) if w.buf.packed_q4 => {
                        "qmatmul_gemm_wmma_i8_k256_packed_nt1"
                    }
                    (GgmlDType::Q4K, 2) if w.buf.packed_q4 => {
                        "qmatmul_gemm_wmma_i8_k256_packed_nt2"
                    }
                    (GgmlDType::Q4K, 4) if w.buf.packed_q4 => {
                        "qmatmul_gemm_wmma_i8_k256_packed_nt4"
                    }
                    (GgmlDType::Q4K, _) if w.buf.packed_q4 => {
                        "qmatmul_gemm_wmma_i8_k256_packed_nt8"
                    }
                    (GgmlDType::Q4K, 1) => "qmatmul_gemm_wmma_i8_k256_nt1",
                    (GgmlDType::Q4K, 2) => "qmatmul_gemm_wmma_i8_k256_nt2",
                    (GgmlDType::Q4K, 4) => "qmatmul_gemm_wmma_i8_k256_nt4",
                    (GgmlDType::Q4K, _) => "qmatmul_gemm_wmma_i8_k256_nt8",
                    (_, 1) => "qmatmul_gemm_wmma_i8_q6k_k256_nt1",
                    (_, 2) => "qmatmul_gemm_wmma_i8_q6k_k256_nt2",
                    (_, 4) => "qmatmul_gemm_wmma_i8_q6k_k256_nt4",
                    (_, _) => "qmatmul_gemm_wmma_i8_q6k_k256_nt8",
                };
                self.qmatmul_i8(
                    kernel,
                    n_tile,
                    w.buf.as_device_ptr(),
                    n_out_i,
                    n_in_i,
                    n_tokens as i32,
                    x.as_device_ptr(),
                    y.as_device_ptr(),
                )
            }
            _ => self.qmatmul_wmma(
                w.buf.as_device_ptr(),
                dtype,
                n_out_i,
                n_in_i,
                n_tokens as i32,
                x.as_device_ptr(),
                y.as_device_ptr(),
            ),
        }
    }

    fn ffn_gate_up(
        &self,
        gate: &QWeight<Self::Buf>,
        up: &QWeight<Self::Buf>,
        x: &Self::Buf,
        gate_scratch: &mut Self::Buf,
        up_scratch: &mut Self::Buf,
        y: &mut Self::Buf,
    ) -> Result<()> {
        if gate.shape != up.shape {
            return Err(Error::ShapeMismatch(
                "ffn gate/up weight shapes must match".into(),
            ));
        }
        if !self.f32_compute
            && matches!(gate.dtype, GgmlDType::Q4K | GgmlDType::Q6K)
            && gate.dtype == up.dtype
            && gate.buf.packed_q4 == up.buf.packed_q4
            && let [_, n_in] = gate.shape[..]
            && x.len == n_in * 4
        {
            return self.launch_gated_gemv_i8(gate, up, x, y);
        }
        self.qmatmul(gate, x, gate_scratch)?;
        self.qmatmul(up, x, up_scratch)?;
        self.silu_mul(gate_scratch, up_scratch, y)
    }

    fn embed(
        &self,
        table: &QWeight<Self::Buf>,
        token_ids: &[u32],
        out: &mut Self::Buf,
    ) -> Result<()> {
        let shape::Embed { n_embd, .. } = shape::embed(&table.shape, token_ids, out.len / 4)?;
        shape::kquant_aligned("embed", table.dtype, n_embd)?;
        let mut dtype = prepared::dtype(table)?;
        let mut n_embd_i = n_embd as i32;
        for (t, &token_id) in token_ids.iter().enumerate() {
            let mut token_id_mut = token_id;
            // SAFETY: shape::embed gives out.len / 4 >= token_ids.len() * n_embd and
            // t < token_ids.len(), so the offset stays inside out's hipMalloc range.
            let mut out_ptr = unsafe { out.as_device_ptr().add(t * n_embd * 4) };
            let mut params = self
                .replay
                .as_ref()
                .map_or(std::ptr::null_mut(), |p| p.mem.as_ptr());
            let mut args = [
                arg!(table.buf.as_device_ptr()),
                arg!(dtype),
                arg!(n_embd_i),
                arg!(token_id_mut),
                arg!(out_ptr),
                arg!(params),
            ];
            self.launch1d(
                if self.replay.is_some() {
                    "embed_gather_graph"
                } else {
                    "embed_gather"
                },
                n_embd,
                &mut args[..if self.replay.is_some() { 6 } else { 5 }],
            )?;
        }
        Ok(())
    }

    /// One block per token; full waves also cover tiny and non-aligned rows.
    fn rmsnorm(&self, w: &Self::Buf, x: &Self::Buf, y: &mut Self::Buf, eps: f32) -> Result<()> {
        self.q8_invalidate();
        let dim = w.len / 4;
        let n_tokens = shape::rows("rmsnorm", dim, x.len / 4, y.len / 4)?;
        let block = (dim.min(256) as u32).div_ceil(WARP) * WARP;
        let mut dim_i = dim as i32;
        let mut eps_mut = eps;
        let _guard = self.lock.lock().unwrap();
        if !self.f32_compute && n_tokens == 1 && dim % QK_K as usize == 0 {
            // Decode: emit the row's Q8_1 too, so the matvecs on `y` skip
            // their quantize launch.
            let mut scratch = self.q8_scratch.lock().unwrap();
            let (mut q8_ptr, mut ys_ptr) = self.q8_row_ptrs(&mut scratch, dim)?;
            let func = self
                .module
                .get_function("rmsnorm_q8_kernel")
                .map_err(hip_err)?;
            let mut args = [
                arg!(w.as_device_ptr()),
                arg!(x.as_device_ptr()),
                arg!(y.as_device_ptr()),
                arg!(dim_i),
                arg!(eps_mut),
                arg!(q8_ptr),
                arg!(ys_ptr),
            ];
            func.launch(
                Dim3::new_1d(1),
                Dim3::new_1d(block),
                0,
                self.stream.as_ref(),
                &mut args,
            )
            .map_err(hip_err)?;
            scratch.cached = Some((y.as_device_ptr() as usize, 1, dim as i32));
            return Ok(());
        }
        let func = self
            .module
            .get_function("rmsnorm_kernel")
            .map_err(hip_err)?;
        let mut args = [
            arg!(w.as_device_ptr()),
            arg!(x.as_device_ptr()),
            arg!(y.as_device_ptr()),
            arg!(dim_i),
            arg!(eps_mut),
        ];
        func.launch(
            Dim3::new_1d(n_tokens as u32),
            Dim3::new_1d(block),
            0,
            self.stream.as_ref(),
            &mut args,
        )
        .map_err(hip_err)?;
        Ok(())
    }

    fn add_rmsnorm(
        &self,
        w: &Self::Buf,
        a: &Self::Buf,
        b: &Self::Buf,
        sum: &mut Self::Buf,
        y: &mut Self::Buf,
        eps: f32,
    ) -> Result<()> {
        let dim = w.len / F32_BYTES as usize;
        let n = shape::elementwise("add_rmsnorm", &[a.len, b.len], &[sum.len, y.len])?
            / F32_BYTES as usize;
        if self.f32_compute || dim == 0 || dim % 4 != 0 || n % dim != 0 {
            self.add(a, b, sum)?;
            return self.rmsnorm(w, sum, y, eps);
        }
        self.q8_invalidate();
        let n_tokens = n / dim;
        let block = (dim.min(256) as u32).div_ceil(WARP) * WARP;
        let mut dim_i = dim as i32;
        let mut eps_mut = eps;
        let _guard = self.lock.lock().unwrap();
        if n_tokens == 1 && dim % QK_K as usize == 0 {
            let mut scratch = self.q8_scratch.lock().unwrap();
            let (mut q8_ptr, mut ys_ptr) = self.q8_row_ptrs(&mut scratch, dim)?;
            let func = self
                .module
                .get_function("add_rmsnorm_q8_kernel")
                .map_err(hip_err)?;
            let mut args = [
                arg!(w.as_device_ptr()),
                arg!(a.as_device_ptr()),
                arg!(b.as_device_ptr()),
                arg!(sum.as_device_ptr()),
                arg!(y.as_device_ptr()),
                arg!(dim_i),
                arg!(eps_mut),
                arg!(q8_ptr),
                arg!(ys_ptr),
            ];
            func.launch(
                Dim3::new_1d(1),
                Dim3::new_1d(block),
                0,
                self.stream.as_ref(),
                &mut args,
            )
            .map_err(hip_err)?;
            scratch.cached = Some((y.as_device_ptr() as usize, 1, dim as i32));
            return Ok(());
        }
        let func = self
            .module
            .get_function("add_rmsnorm_kernel")
            .map_err(hip_err)?;
        let mut args = [
            arg!(w.as_device_ptr()),
            arg!(a.as_device_ptr()),
            arg!(b.as_device_ptr()),
            arg!(sum.as_device_ptr()),
            arg!(y.as_device_ptr()),
            arg!(dim_i),
            arg!(eps_mut),
        ];
        func.launch(
            Dim3::new_1d(n_tokens as u32),
            Dim3::new_1d(block),
            0,
            self.stream.as_ref(),
            &mut args,
        )
        .map_err(hip_err)?;
        Ok(())
    }

    fn add(&self, a: &Self::Buf, b: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        let n = shape::elementwise("add", &[a.len, b.len], &[y.len])? / 4;
        let mut n_i = n as i32;
        self.launch1d(
            "add_kernel",
            n,
            &mut [
                arg!(a.as_device_ptr()),
                arg!(b.as_device_ptr()),
                arg!(y.as_device_ptr()),
                arg!(n_i),
            ],
        )
    }

    fn rope(
        &self,
        q: &mut Self::Buf,
        k: &mut Self::Buf,
        pos: u32,
        n_tokens: usize,
        spec: &RopeSpec,
    ) -> Result<()> {
        self.q8_invalidate();
        let half = spec.head_dim / 2;
        let shape::Rope {
            n_q_heads,
            n_kv_heads,
        } = shape::rope(spec, q.len / 4, k.len / 4, n_tokens)?;
        let q_stride = n_q_heads * spec.head_dim;
        let k_stride = n_kv_heads * spec.head_dim;
        let total = n_q_heads * half + n_kv_heads * half;

        // x selects (head, pair); y selects the token and its RoPE position.
        let _guard = self.lock.lock().unwrap();
        let func = self
            .module
            .get_function(if self.replay.is_some() {
                "rope_graph"
            } else {
                "rope_kernel"
            })
            .map_err(hip_err)?;
        let mut pos_mut = pos;
        let mut head_dim_i = spec.head_dim as i32;
        let mut n_q_i = n_q_heads as i32;
        let mut n_kv_i = n_kv_heads as i32;
        let mut theta = spec.theta;
        let mut q_stride_i = q_stride as i32;
        let mut k_stride_i = k_stride as i32;
        let mut params = self
            .replay
            .as_ref()
            .map_or(std::ptr::null_mut(), |p| p.mem.as_ptr());
        let mut args = [
            arg!(q.as_device_ptr()),
            arg!(k.as_device_ptr()),
            arg!(pos_mut),
            arg!(head_dim_i),
            arg!(n_q_i),
            arg!(n_kv_i),
            arg!(theta),
            arg!(q_stride_i),
            arg!(k_stride_i),
            arg!(params),
        ];
        let block = 256u32;
        let grid_x = (total as u32).div_ceil(block);
        let n_args = if self.replay.is_some() { 10 } else { 9 };
        func.launch(
            Dim3::new_2d(grid_x, n_tokens as u32),
            Dim3::new_1d(block),
            0,
            self.stream.as_ref(),
            &mut args[..n_args],
        )
        .map_err(hip_err)?;
        Ok(())
    }

    fn attn(
        &self,
        q: &Self::Buf,
        kv: &KvView<Self::Buf>,
        spec: &AttnSpec,
        o: &mut Self::Buf,
    ) -> Result<()> {
        if self.f32_compute || spec.head_dim > ATT_MAX_HEAD_DIM {
            return self.attention_generic(q, kv, spec, 1.0 / (spec.head_dim as f32).sqrt(), 0, o);
        }
        self.q8_invalidate();
        let kv_elems = |b: &HipBuf| b.len / self.kv_elem_bytes();
        let shape::Attn { n_q, .. } = shape::attn(
            spec,
            q.len / 4,
            kv.len,
            kv_elems(kv.k),
            kv_elems(kv.v),
            o.len / 4,
        )?;
        let per_token = spec.n_q_heads * spec.head_dim;

        let _guard = self.lock.lock().unwrap();
        let group = spec.n_q_heads / spec.n_kv_heads;
        let split_ok = spec.head_dim == ATT_WMMA_HEAD_DIM
            && group == ATT_SPLIT_GROUP
            && self.replay.as_ref().map_or(kv.len, |p| p.capacity) <= ATT_DMAX_KV;
        // shape::attn validated GQA; hd128/g4 also guarantees Q8 row alignment.
        match n_q {
            1 if split_ok => {
                // Flash-decoding: block per (KV head, 32-key chunk) shared by the
                // group's heads, then a fixed-order combine per head. Under
                // graph replay the chunk count is the capacity's; blocks past
                // the live length exit immediately.
                let max_len = self.replay.as_ref().map_or(kv.len, |p| p.capacity).max(1);
                let chunks = (max_len as u32).div_ceil(ATT_DC);
                let partial_bytes = Self::attn_partials_bytes(spec.n_q_heads, max_len)?;
                let mut counters = self.reserve_attn_scratch(partial_bytes, spec.n_kv_heads)?;
                // SAFETY: reserve_attn_scratch lays the allocation out as the
                // zeroed arrival counters (a fixed ATT_COUNTERS_BYTES) followed
                // by the partials -- fixed offsets, so a shorter call after a
                // longer one never reads stale partial data as a counter.
                let mut part = unsafe {
                    counters
                        .cast::<u8>()
                        .add(ATT_COUNTERS_BYTES)
                        .cast::<c_void>()
                };
                let mut n_kv_i = spec.n_kv_heads as i32;
                let mut kv_len_i = kv.len as i32;
                let mut params = self
                    .replay
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |p| p.mem.as_ptr());
                let graph = self.replay.is_some();
                let split = self
                    .module
                    .get_function(if graph {
                        "attn_decode_split_hd128_g4_graph"
                    } else {
                        "attn_decode_split_hd128_g4"
                    })
                    .map_err(hip_err)?;
                // The merge also emits the output row's Q8_1 so the output
                // projection skips its quantize launch.
                let mut scratch = self.q8_scratch.lock().unwrap();
                let (mut q8_ptr, mut ys_ptr) = self.q8_row_ptrs(&mut scratch, per_token)?;
                let mut sargs = [
                    arg!(q.as_device_ptr()),
                    arg!(kv.k.as_device_ptr()),
                    arg!(kv.v.as_device_ptr()),
                    arg!(part),
                    arg!(counters),
                    arg!(o.as_device_ptr()),
                    arg!(q8_ptr),
                    arg!(ys_ptr),
                    arg!(n_kv_i),
                    arg!(kv_len_i),
                    arg!(params),
                ];
                let n_sargs = if graph { 11 } else { 10 };
                split
                    .launch(
                        Dim3::new_2d(spec.n_kv_heads as u32, chunks),
                        Dim3::new_1d(group as u32 * WARP),
                        0,
                        self.stream.as_ref(),
                        &mut sargs[..n_sargs],
                    )
                    .map_err(hip_err)?;
                scratch.cached = Some((o.as_device_ptr() as usize, 1, per_token as i32));
            }
            1 => {
                let cooperative =
                    spec.head_dim == ATT_DECODE_HEAD_DIM && kv.len <= ATT_DECODE_MAX_KV;
                let kernel = match (cooperative, self.replay.is_some()) {
                    (true, true) => "attn_decode_hd128_graph",
                    (true, false) => "attn_decode_hd128",
                    (false, true) => "attn_decode_graph",
                    (false, false) => "attn_decode_kernel",
                };
                let func = self.module.get_function(kernel).map_err(hip_err)?;
                let mut head_dim_i = spec.head_dim as i32;
                let mut n_q_i = spec.n_q_heads as i32;
                let mut n_kv_i = spec.n_kv_heads as i32;
                let mut kv_len_i = kv.len as i32;
                let mut params = self
                    .replay
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |p| p.mem.as_ptr());
                let mut args = [
                    arg!(q.as_device_ptr()),
                    arg!(kv.k.as_device_ptr()),
                    arg!(kv.v.as_device_ptr()),
                    arg!(o.as_device_ptr()),
                    arg!(head_dim_i),
                    arg!(n_q_i),
                    arg!(n_kv_i),
                    arg!(kv_len_i),
                    arg!(params),
                ];
                let block = if cooperative {
                    ATT_DECODE_THREADS
                } else {
                    ATT_GENERIC_THREADS.min((spec.head_dim.max(kv.len).max(1)) as u32)
                };
                let shared =
                    (self.replay.as_ref().map_or(kv.len.max(1), |p| p.capacity) * 4) as u32;
                let n_args = if self.replay.is_some() { 9 } else { 8 };
                func.launch(
                    Dim3::new_1d(spec.n_q_heads as u32),
                    Dim3::new_1d(block),
                    shared,
                    self.stream.as_ref(),
                    &mut args[..n_args],
                )
                .map_err(hip_err)?;
            }
            _ => {
                let prefix = kv.len - n_q;
                let mut head_dim_i = spec.head_dim as i32;
                let mut n_q_heads_i = spec.n_q_heads as i32;
                let mut n_kv_heads_i = spec.n_kv_heads as i32;
                let mut n_q_i = n_q as i32;
                let mut prefix_i = prefix as i32;
                let mut args = [
                    arg!(q.as_device_ptr()),
                    arg!(kv.k.as_device_ptr()),
                    arg!(kv.v.as_device_ptr()),
                    arg!(o.as_device_ptr()),
                    arg!(head_dim_i),
                    arg!(n_q_heads_i),
                    arg!(n_kv_heads_i),
                    arg!(n_q_i),
                    arg!(prefix_i),
                ];
                let block_ok = group as u32 * WARP <= MAX_BLOCK_THREADS;
                match (spec.head_dim, block_ok) {
                    (ATT_WMMA_HEAD_DIM, true) => {
                        // Flash attention on f16 WMMA, block per (KV head,
                        // 16 queries), one warp per q-head of the group. Static LDS.
                        let func = self
                            .module
                            .get_function("attn_prefill_wmma_hd128")
                            .map_err(hip_err)?;
                        let grid = Dim3::new_2d(
                            spec.n_kv_heads as u32,
                            (n_q as u32).div_ceil(TOKEN_SUBTILE),
                        );
                        let mut wargs = [
                            args[0], args[1], args[2], args[3], args[5], args[6], args[7], args[8],
                        ];
                        func.launch(
                            grid,
                            Dim3::new_1d(group as u32 * WARP),
                            0,
                            self.stream.as_ref(),
                            &mut wargs,
                        )
                        .map_err(hip_err)?;
                    }
                    (_, true) => {
                        // Larger heads already use attention_generic above.
                        // K/V are staged per 16-key tile; aim for ~256 blocks.
                        let mut qt = ((n_q * spec.n_kv_heads) as u32 / TARGET_BLOCKS)
                            .clamp(1, ATT_QT) as i32;
                        let func = self
                            .module
                            .get_function("attn_prefill_tiled_kernel")
                            .map_err(hip_err)?;
                        let grid =
                            Dim3::new_2d(spec.n_kv_heads as u32, (n_q as u32).div_ceil(qt as u32));
                        let shared = 2 * ATT_KT * spec.head_dim as u32 * F32_BYTES; // Ks + Vs
                        let mut targs = [
                            args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[5],
                            args[6],
                            args[7],
                            args[8],
                            arg!(qt),
                        ];
                        func.launch(
                            grid,
                            Dim3::new_1d(group as u32 * WARP),
                            shared,
                            self.stream.as_ref(),
                            &mut targs,
                        )
                        .map_err(hip_err)?;
                    }
                    (_, false) => {
                        let func = self
                            .module
                            .get_function("attn_prefill_kernel")
                            .map_err(hip_err)?;
                        let max_visible = kv.len;
                        let block =
                            ATT_GENERIC_THREADS.min((spec.head_dim.max(max_visible).max(1)) as u32);
                        let shared = (max_visible.max(1) * 4) as u32;
                        let grid = (n_q * spec.n_q_heads) as u32;
                        func.launch(
                            Dim3::new_1d(grid),
                            Dim3::new_1d(block),
                            shared,
                            self.stream.as_ref(),
                            &mut args,
                        )
                        .map_err(hip_err)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn silu_mul(&self, gate: &Self::Buf, up: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        let n = shape::elementwise("silu_mul", &[gate.len, up.len], &[y.len])? / 4;
        let mut n_i = n as i32;
        self.launch1d(
            "silu_mul_kernel",
            n,
            &mut [
                arg!(gate.as_device_ptr()),
                arg!(up.as_device_ptr()),
                arg!(y.as_device_ptr()),
                arg!(n_i),
            ],
        )
    }
}

impl HipBackend {
    /// Multiply one 16x16 tile to test WMMA fragment packing.
    #[cfg(test)]
    fn wmma_probe(&self, a: &HipBuf, b: &HipBuf, d: &mut HipBuf) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        let func = self
            .module
            .get_function("wmma_probe_kernel")
            .map_err(hip_err)?;
        let mut args = [
            arg!(a.as_device_ptr()),
            arg!(b.as_device_ptr()),
            arg!(d.as_device_ptr()),
        ];
        func.launch(
            Dim3::new_1d(1),
            Dim3::new_1d(32),
            0,
            self.stream.as_ref(),
            &mut args,
        )
        .map_err(hip_err)?;
        rocm_rs::hip::device_synchronize().map_err(hip_err)?;
        Ok(())
    }

    /// GEMV memory-traffic probe with reduced dequantization arithmetic.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn launch_gemv_membound_probe(
        &self,
        mut w_ptr: *mut c_void,
        mut n_out: i32,
        mut n_in: i32,
        mut x_ptr: *mut c_void,
        mut y_ptr: *mut c_void,
    ) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        let func = self
            .module
            .get_function("qmatmul_gemv_membound_probe")
            .map_err(hip_err)?;
        let mut args = [
            arg!(w_ptr),
            arg!(n_out),
            arg!(n_in),
            arg!(x_ptr),
            arg!(y_ptr),
        ];
        const ROWS_PER_BLOCK: u32 = 8;
        const THREADS_PER_ROW: u32 = 32;
        let grid = (n_out as u32).div_ceil(ROWS_PER_BLOCK);
        func.launch(
            Dim3::new_1d(grid),
            Dim3::new_1d(ROWS_PER_BLOCK * THREADS_PER_ROW),
            0,
            self.stream.as_ref(),
            &mut args,
        )
        .map_err(hip_err)?;
        rocm_rs::hip::device_synchronize().map_err(hip_err)?;
        Ok(())
    }

    /// Launch GEMV with a caller-selected number of rows per block.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn launch_gemv_rows_per_block(
        &self,
        mut w_ptr: *mut c_void,
        mut dtype: i32,
        mut n_out: i32,
        mut n_in: i32,
        mut x_ptr: *mut c_void,
        mut y_ptr: *mut c_void,
        rows_per_block: u32,
    ) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        let func = self.module.get_function("qmatmul_gemv").map_err(hip_err)?;
        let mut args = [
            arg!(w_ptr),
            arg!(dtype),
            arg!(n_out),
            arg!(n_in),
            arg!(x_ptr),
            arg!(y_ptr),
        ];
        const THREADS_PER_ROW: u32 = 32;
        let grid = (n_out as u32).div_ceil(rows_per_block);
        func.launch(
            Dim3::new_1d(grid),
            Dim3::new_1d(rows_per_block * THREADS_PER_ROW),
            0,
            self.stream.as_ref(),
            &mut args,
        )
        .map_err(hip_err)?;
        Ok(())
    }

    /// One warp per output row, eight rows per block.
    #[allow(clippy::too_many_arguments)]
    fn launch_gemv(
        &self,
        mut w_ptr: *mut c_void,
        mut dtype: i32,
        mut n_out: i32,
        mut n_in: i32,
        mut x_ptr: *mut c_void,
        mut y_ptr: *mut c_void,
    ) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        let func = self.module.get_function("qmatmul_gemv").map_err(hip_err)?;
        let mut args = [
            arg!(w_ptr),
            arg!(dtype),
            arg!(n_out),
            arg!(n_in),
            arg!(x_ptr),
            arg!(y_ptr),
        ];
        const ROWS_PER_BLOCK: u32 = 8;
        const THREADS_PER_ROW: u32 = 32;
        let grid = (n_out as u32).div_ceil(ROWS_PER_BLOCK);
        func.launch(
            Dim3::new_1d(grid),
            Dim3::new_1d(ROWS_PER_BLOCK * THREADS_PER_ROW),
            0,
            self.stream.as_ref(),
            &mut args,
        )
        .map_err(hip_err)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn qmatmul_wmma(
        &self,
        mut w_ptr: *mut c_void,
        mut dtype: i32,
        mut n_out: i32,
        mut n_in: i32,
        mut n_tokens: i32,
        mut x_ptr: *mut c_void,
        mut y_ptr: *mut c_void,
    ) -> Result<()> {
        // Writes y, which may be a cached Q8 input (a norm output reused
        // as a matmul target): a stale hit here would multiply old data.
        self.q8_invalidate();
        let _guard = self.lock.lock().unwrap();
        let func = self
            .module
            .get_function("qmatmul_gemm_wmma")
            .map_err(hip_err)?;
        let mut args = [
            arg!(w_ptr),
            arg!(dtype),
            arg!(n_out),
            arg!(n_in),
            arg!(n_tokens),
            arg!(x_ptr),
            arg!(y_ptr),
        ];
        // All LDS is static; the dynamic shared-memory argument must be zero.
        let grid_x = (n_out as u32).div_ceil(M_TILE);
        let grid_y = (n_tokens as u32).div_ceil(TOKEN_SUBTILE);
        func.launch(
            Dim3::new_2d(grid_x, grid_y),
            Dim3::new_1d(GEMM_THREADS),
            0,
            self.stream.as_ref(),
            &mut args,
        )
        .map_err(hip_err)?;
        Ok(())
    }

    /// WMMA memory-traffic probe with reduced dequantization arithmetic.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn qmatmul_wmma_membound_probe(
        &self,
        mut w_ptr: *mut c_void,
        mut n_out: i32,
        mut n_in: i32,
        mut n_tokens: i32,
        mut x_ptr: *mut c_void,
        mut y_ptr: *mut c_void,
    ) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        let func = self
            .module
            .get_function("qmatmul_gemm_wmma_membound_probe")
            .map_err(hip_err)?;
        let mut args = [
            arg!(w_ptr),
            arg!(n_out),
            arg!(n_in),
            arg!(n_tokens),
            arg!(x_ptr),
            arg!(y_ptr),
        ];
        let grid_x = (n_out as u32).div_ceil(M_TILE);
        let grid_y = (n_tokens as u32).div_ceil(TOKEN_SUBTILE);
        let shared = (M_TILE + TOKEN_SUBTILE) * TOKEN_SUBTILE * F16_BYTES; // As[64][16] + Bs[16][16]
        func.launch(
            Dim3::new_2d(grid_x, grid_y),
            Dim3::new_1d(GEMM_THREADS),
            shared,
            self.stream.as_ref(),
            &mut args,
        )
        .map_err(hip_err)?;
        Ok(())
    }

    /// Multiply one integer tile to test iu8 fragment packing.
    #[cfg(test)]
    fn wmma_i8_probe(&self, a: &HipBuf, b: &HipBuf, c: &HipBuf, d: &mut HipBuf) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        let func = self
            .module
            .get_function("wmma_i8_probe_kernel")
            .map_err(hip_err)?;
        let mut args = [
            arg!(a.as_device_ptr()),
            arg!(b.as_device_ptr()),
            arg!(c.as_device_ptr()),
            arg!(d.as_device_ptr()),
        ];
        func.launch(
            Dim3::new_1d(1),
            Dim3::new_1d(32),
            0,
            self.stream.as_ref(),
            &mut args,
        )
        .map_err(hip_err)?;
        rocm_rs::hip::device_synchronize().map_err(hip_err)?;
        Ok(())
    }

    /// Q8_1 activation quantization into caller-provided buffers -- see
    /// kernels.hip's quantize_q8_1_kernel for the exact representation.
    /// Test-only entry point; the production int8 path quantizes into the
    /// backend-private scratch via qmatmul_i8.
    #[cfg(test)]
    fn quantize_q8_1(
        &self,
        x: &HipBuf,
        n_in: i32,
        n_tokens: i32,
        q8: &mut HipBuf,
        ys: &mut HipBuf,
    ) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        self.launch_quantize_q8_1_unlocked(
            x.as_device_ptr(),
            n_in,
            n_tokens,
            q8.as_device_ptr(),
            ys.as_device_ptr(),
        )?;
        rocm_rs::hip::device_synchronize().map_err(hip_err)?;
        Ok(())
    }

    /// Caller holds `self.lock`.
    fn launch_quantize_q8_1_unlocked(
        &self,
        mut x_ptr: *mut c_void,
        mut n_in: i32,
        mut n_tokens: i32,
        mut q8_ptr: *mut c_void,
        mut ys_ptr: *mut c_void,
    ) -> Result<()> {
        let func = self
            .module
            .get_function("quantize_q8_1_kernel")
            .map_err(hip_err)?;
        let mut args = [
            arg!(x_ptr),
            arg!(n_in),
            arg!(n_tokens),
            arg!(q8_ptr),
            arg!(ys_ptr),
        ];
        let n = n_tokens as u32 * (n_in as u32 / Q8_1_BLOCK) * Q8_1_LANES;
        let block = 256u32;
        func.launch(
            calculate_grid_1d(n, block),
            Dim3::new_1d(block),
            0,
            self.stream.as_ref(),
            &mut args,
        )
        .map_err(hip_err)?;
        Ok(())
    }

    /// int8-WMMA prefill matmul for Q4_K weights (see kernels.hip's
    /// qmatmul_gemm_wmma_i8). Holds `lock` for the whole quantize ->
    /// consume sequence so the scratch buffer is never in use by two
    /// overlapping matmuls, and grows the scratch only after a device
    /// sync so no in-flight kernel can still be reading the old one.
    #[allow(clippy::too_many_arguments)]
    fn qmatmul_i8(
        &self,
        kernel: &str,
        n_tile: u32,
        mut w_ptr: *mut c_void,
        mut n_out: i32,
        mut n_in: i32,
        mut n_tokens: i32,
        x_ptr: *mut c_void,
        mut y_ptr: *mut c_void,
    ) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        let mut scratch = self.q8_scratch.lock().unwrap();
        let q8_bytes = product(&[n_tokens as usize, n_in as usize])?;
        let ys_bytes = product(&[
            n_tokens as usize,
            n_in as usize / Q8_1_BLOCK as usize,
            size_of::<[f32; 2]>(),
        ])?;
        let n_tile = gemm_token_tile(kernel, n_tile, n_tokens as u32);
        // _nt8 is the 128-row x 128-token, 256-thread geometry; _j128 is
        // 8 warps as 4 along rows x 2 along tokens on 64 rows.
        let wide_rows = kernel.ends_with("_nt8");
        let m_tile = if wide_rows { 2 * M_TILE } else { M_TILE };
        let threads = if wide_rows || kernel.ends_with("_j128") {
            2 * GEMM_THREADS
        } else {
            GEMM_THREADS
        };
        let grid_x = (n_out as u32).div_ceil(m_tile);
        let grid_y = (n_tokens as u32).div_ceil(n_tile);
        // Splits must align to superblocks. Target 256 blocks, at most
        // eight splits, with partial storage capped at half the weight estimate.
        let superblocks = n_in as u32 / QK_K;
        // Four bits per weight underestimates both formats to limit scratch use.
        let weight_bytes = product(&[n_out as usize, n_in as usize])? / 2;
        let mut splits = 1u32;
        for cand in [2u32, 4, 8] {
            let partial = product(&[cand as usize, n_tokens as usize, n_out as usize, 4])?;
            if u64::from(grid_x) * u64::from(grid_y) * u64::from(splits) >= u64::from(TARGET_BLOCKS)
                || superblocks % cand != 0
                || partial > weight_bytes / 2
            {
                break;
            }
            splits = cand;
        }
        let partial_bytes = if splits > 1 {
            product(&[splits as usize, n_tokens as usize, n_out as usize, 4])?
        } else {
            0
        };
        let need = sum([q8_bytes, ys_bytes, partial_bytes])?;
        scratch.reserve(need, &self.buffers)?;
        let key = (x_ptr as usize, n_tokens, n_in);
        let reuse = scratch.cached == Some(key);
        let base = scratch
            .mem
            .as_ref()
            .expect("scratch allocated above")
            .as_ptr();
        let mut q8_ptr = base;
        // SAFETY: all three regions lie inside the live allocation of `need`
        // bytes; q8_bytes is a multiple of 32 and ys_bytes a multiple of 8,
        // so the float2 region and the f32 partials stay aligned.
        let mut ys_ptr = unsafe { (base as *mut u8).add(q8_bytes) as *mut c_void };
        let mut partial_ptr = unsafe { (base as *mut u8).add(q8_bytes + ys_bytes) as *mut c_void };

        if !reuse {
            self.launch_quantize_q8_1_unlocked(x_ptr, n_in, n_tokens, q8_ptr, ys_ptr)?;
        }
        // Valid until any op writes a buffer -- or this matmul writes into
        // the cached input itself.
        scratch.cached = if y_ptr as usize == x_ptr as usize {
            None
        } else {
            Some(key)
        };

        let func = self.module.get_function(kernel).map_err(hip_err)?;
        let mut k_split = n_in / splits as i32;
        let mut out_ptr = if splits > 1 { partial_ptr } else { y_ptr };
        let mut n_tile_i = n_tile as i32;
        let mut args = [
            arg!(w_ptr),
            arg!(n_out),
            arg!(n_in),
            arg!(n_tokens),
            arg!(q8_ptr),
            arg!(ys_ptr),
            arg!(out_ptr),
            arg!(k_split),
            arg!(n_tile_i),
        ];
        // The k256 kernel carves its LDS from a dynamic allocation sized to
        // the live token tile (see its i9_smem comment); the others declare
        // theirs statically.
        // Q4_K _nt kernels stage a token scale as 4 bytes ({half d,
        // int16 sum q}) for the 4 blocks of the current 128-K half, and a
        // coefficient pair as float2 (8 sub-blocks x rows); the others
        // keep float2 token scales for all 8 blocks and 16 x 64 float coefs.
        let per_sb = QK_K / Q8_1_BLOCK;
        let shared = if kernel.starts_with("qmatmul_gemm_wmma_i8_k256_nt")
            || kernel.starts_with("qmatmul_gemm_wmma_i8_k256_packed_nt")
        {
            // As, Bs, ysh (d per token per block of the half, f16), shl (s
            // hi/lo per token per block, f16), coef (d*sc), dmin per row,
            // m per row per sub-block (f16) -- see i9_gemm_q4k_nt_body.
            m_tile * I9_AS_STRIDE
                + n_tile * I9_BS_STRIDE
                + n_tile * (per_sb / 2) * F16_BYTES
                + n_tile * 2 * per_sb * F16_BYTES
                + per_sb * m_tile * F32_BYTES
                + m_tile * F32_BYTES
                + m_tile * per_sb * F16_BYTES
        } else if kernel.starts_with("qmatmul_gemm_wmma_i8_q6k_k256_nt") {
            // As, Bs, ysh (d per token per block of the half, f16), coef (16
            // per row: one per 16 K).
            m_tile * I9_AS_STRIDE
                + n_tile * I9_BS_STRIDE
                + n_tile * (per_sb / 2) * F16_BYTES
                + 2 * per_sb * m_tile * F32_BYTES
        } else if kernel.contains("_k256") {
            M_TILE * I9_AS_STRIDE
                + n_tile * I9_BS_STRIDE
                + n_tile * per_sb * 8
                + 2 * per_sb * M_TILE * F32_BYTES
        } else {
            0
        };
        func.launch(
            Dim3::new_3d(grid_x, grid_y, splits),
            Dim3::new_1d(threads),
            shared,
            self.stream.as_ref(),
            &mut args,
        )
        .map_err(hip_err)?;

        if splits > 1 {
            let reduce = self
                .module
                .get_function("splitk_reduce_kernel")
                .map_err(hip_err)?;
            let mut splits_i = splits as i32;
            let mut n = n_tokens * n_out;
            let mut rargs = [arg!(partial_ptr), arg!(splits_i), arg!(n), arg!(y_ptr)];
            let block = 256u32;
            reduce
                .launch(
                    calculate_grid_1d(n as u32, block),
                    Dim3::new_1d(block),
                    0,
                    self.stream.as_ref(),
                    &mut rargs,
                )
                .map_err(hip_err)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod attention;
mod decode;
mod graph;
mod memory;
mod prepared;

#[cfg(test)]
pub(crate) mod tests;

/// Free and total bytes of the current device.
fn memory_info() -> Result<(usize, usize)> {
    let mut free = 0;
    let mut total = 0;
    // SAFETY: `free`/`total` are live usize (size_t) out-params. The query targets the
    // calling thread's current device, which is device 0 here (set in create(), and
    // HIP's per-thread default), and touches no stream or backend state.
    let status = unsafe { ffi::hipMemGetInfo(&mut free, &mut total) };
    rocm_rs::hip::error::Error::from_hip_error_with_value(status, (free, total)).map_err(hip_err)
}
