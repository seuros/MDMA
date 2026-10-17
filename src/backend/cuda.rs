// NVIDIA backend through the CUDA driver API. CUDA C in kernels/kernels.cu
// is compiled to PTX by clang at build time (no toolkit) and JIT-compiled
// by libcuda at load. Buffers live in device memory; every eager op is
// enqueued on the legacy default stream, so ops are ordered and download
// (a synchronous DtoH copy) observes all of them. Graph decode (graph.rs)
// runs on a backend twin with its own stream.

use super::{
    AttnSpec, Backend, DeltaInputs, DeltaSpec, Error, KvView, Operation, Pool, QWeight, Result,
    RopeSpec, dtype_code, shape,
};
use crate::gguf::GgmlDType;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const CUDA_SUCCESS: c_int = 0;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_INTEGRATED: c_int = 18;
const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;

/// Rows per 128-thread matvec/attention block: one warp each.
const WARPS: u32 = 4;
/// Rows per warp in gemv_q (Q_ROWS in kernels.cu).
const Q_ROWS: u32 = 4;
/// Warps splitting one (head, query, chunk)'s keys in attn_f16kv
/// (ATTN_WARPS in kernels.cu).
const ATTN_WARPS: u32 = 8;
/// Fewest keys per attention chunk: 32 per warp.
const ATTN_MIN_CHUNK: usize = 256;
/// Most chunks per (head, query); bounds the partials and the merge loop.
const ATTN_MAX_SPLITS: usize = 64;
/// Attention blocks per SM that splitting keys aims for.
const ATTN_BLOCKS_PER_SM: usize = 4;
/// head_dim ceiling of attn_f16kv (8 elements per lane) and of graph
/// decode. Wider heads (to WINDOWED_MAX_HEAD_DIM) take attn_windowed_f16kv,
/// eagerly.
const MAX_HEAD_DIM: usize = 256;
/// Entries in the by-value RoPE frequency table (`struct Freqs`), one per
/// pair: head_dim <= 512.
const ROPE_PAIRS: usize = 256;
/// Output rows x tokens per gemm_q4k / gemm_q6k block (GM, GN in kernels.cu).
const GEMM_M: u32 = 64;
const GEMM_N: u32 = 32;
/// K-quant matmuls with at least this many tokens take the int8 GEMM
/// (quantized activations).
pub const GEMM_MIN_TOKENS: usize = 2;
/// Threads per quantize_q8_1 block (8 lanes per 32-element Q8_1 block).
const Q8_THREADS: u32 = 256;

const KERNELS: &[&str] = &[
    "add_f32",
    "silu_mul_f32",
    "rmsnorm_f32",
    "add_rmsnorm_f32",
    "rope_f32",
    "rope_kv_f16",
    "kv_write_f16",
    "attn_f16kv_g1",
    "attn_f16kv_g2",
    "attn_f16kv_g4",
    "attn_f16kv_g8",
    "attn_merge",
    "gemv_f",
    "gemv_q",
    "gemv_q3",
    "gemv_q_swiglu",
    "embed_rows",
    "quantize_q8_1",
    "gemm_q4k",
    "gemm_q6k",
    "gemv_q8",
    "unary_f32",
    "gelu_tanh_mul_f32",
    "rope_split_half_f32",
    "rope_partial_split_half_f32",
    "sigmoid_mul_f32",
    "ssm_conv_silu_f32",
    "gated_delta_f32",
    "gemv_q8_0",
    "attn_windowed_f16kv",
    "gemv_q5k",
];

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevicePtr = u64;
type CuStream = *mut c_void;

/// The kernels, JIT-compiled for the device at startup.
const PTX: &str = include_str!(env!("MDMA_CUDA_PTX"));

/// cuStreamCreate flag: no implicit sync with the legacy default stream.
const CU_STREAM_NON_BLOCKING: c_uint = 1;

// SAFETY: prototypes match cuda.h for libcuda's exports (build.rs links
// `cuda`): CUresult, CUdevice and the CUjit_option / CUdevice_attribute enums
// are C ints, CUdeviceptr is the _v2 API's 64-bit integer, size_t is usize,
// and context/module/function/stream handles are opaque pointers. The _v2
// names are what cuda.h's macros resolve to; none are per-thread-stream
// (_ptsz) variants, so a null stream means the legacy stream.
unsafe extern "C" {
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CuResult;
    fn cuDriverGetVersion(version: *mut c_int) -> CuResult;
    /// `uuid` is a CUuuid, 16 bytes.
    fn cuDeviceGetUuid_v2(uuid: *mut u8, dev: CuDevice) -> CuResult;
    fn cuInit(flags: c_uint) -> CuResult;
    fn cuDeviceGetCount(count: *mut c_int) -> CuResult;
    fn cuDeviceGet(device: *mut CuDevice, ordinal: c_int) -> CuResult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CuDevice) -> CuResult;
    fn cuDeviceGetPCIBusId(bus: *mut c_char, len: c_int, dev: CuDevice) -> CuResult;
    fn cuDeviceGetAttribute(pi: *mut c_int, attrib: c_int, dev: CuDevice) -> CuResult;
    fn cuDevicePrimaryCtxRetain(ctx: *mut CuContext, dev: CuDevice) -> CuResult;
    fn cuDevicePrimaryCtxRelease_v2(dev: CuDevice) -> CuResult;
    fn cuCtxSetCurrent(ctx: CuContext) -> CuResult;
    fn cuCtxSynchronize() -> CuResult;
    fn cuModuleLoadDataEx(
        module: *mut CuModule,
        image: *const c_void,
        n: c_uint,
        opts: *mut c_int,
        vals: *mut *mut c_void,
    ) -> CuResult;
    fn cuModuleUnload(module: CuModule) -> CuResult;
    fn cuModuleGetFunction(f: *mut CuFunction, module: CuModule, name: *const c_char) -> CuResult;
    fn cuMemAlloc_v2(dptr: *mut CuDevicePtr, bytes: usize) -> CuResult;
    fn cuMemFree_v2(dptr: CuDevicePtr) -> CuResult;
    fn cuMemsetD8_v2(dptr: CuDevicePtr, value: u8, n: usize) -> CuResult;
    fn cuMemcpyHtoD_v2(dst: CuDevicePtr, src: *const c_void, bytes: usize) -> CuResult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CuDevicePtr, bytes: usize) -> CuResult;
    fn cuMemcpyDtoD_v2(dst: CuDevicePtr, src: CuDevicePtr, bytes: usize) -> CuResult;
    fn cuStreamCreate(stream: *mut CuStream, flags: c_uint) -> CuResult;
    fn cuStreamDestroy_v2(stream: CuStream) -> CuResult;
    fn cuStreamSynchronize(stream: CuStream) -> CuResult;
    #[allow(clippy::too_many_arguments)]
    fn cuLaunchKernel(
        f: CuFunction,
        gx: c_uint,
        gy: c_uint,
        gz: c_uint,
        bx: c_uint,
        by: c_uint,
        bz: c_uint,
        shared: c_uint,
        stream: *mut c_void,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CuResult;
    fn cuGetErrorName(err: CuResult, s: *mut *const c_char) -> CuResult;
}

/// A size as a kernel's `int` argument.
fn int(n: impl TryInto<c_int>) -> Result<c_int> {
    n.try_into().map_err(|_| Error::SizeOverflow)
}

/// An element offset as a kernel's `long long` argument.
fn long(n: usize) -> Result<i64> {
    i64::try_from(n).map_err(|_| Error::SizeOverflow)
}

fn cu(what: &str, r: CuResult) -> Result<()> {
    if r == CUDA_SUCCESS {
        return Ok(());
    }
    let mut name: *const c_char = std::ptr::null();
    // SAFETY: cuGetErrorName takes any CUresult by value and only writes
    // through `&mut name`, a live local initialised to null. Any failure
    // (unknown code, or cuInit itself having failed) leaves or stores NULL,
    // and both the status and the null are checked before `name` is read.
    let name = if unsafe { cuGetErrorName(r, &raw mut name) } == CUDA_SUCCESS && !name.is_null() {
        // SAFETY: on success `name` is non-null (checked above) and points to
        // a static NUL-terminated string owned by libcuda for the life of the
        // process; into_owned copies it before returning.
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned()
    } else {
        "unknown".to_string()
    };
    Err(Error::Backend(format!("{what} failed: {name} ({r})")))
}

pub struct CudaBuf {
    ctx: CuContext,
    ptr: CuDevicePtr,
    len: usize,
    charge: crate::memory::BufferCharge,
    // Buffers may outlive the backend which allocated them.
    _kernels: Arc<Kernels>,
}

// SAFETY: the only !Send field is `ctx`, the device's primary context, which
// the driver lets any thread make current (Drop does so before freeing).
// `ptr` is a plain device address, `charge` is Send, and `_kernels` keeps the
// context retained wherever the buffer moves.
unsafe impl Send for CudaBuf {}
// SAFETY: nothing mutates a CudaBuf through `&self`: shared access only reads
// `ptr`/`len` to hand to the thread-safe driver API, and the device memory
// behind `ptr` is not Rust-visible memory that `&` could alias.
unsafe impl Sync for CudaBuf {}

impl Drop for CudaBuf {
    fn drop(&mut self) {
        // Any thread may drop a buffer; the primary context must be
        // current for the free. Failed frees remain charged.
        // SAFETY: `self.ctx`, still retained by `self._kernels` (fields drop
        // after this body), is made current first. `self.ptr` came from
        // cuMemAlloc_v2 in it (CudaBuf is built only in `alloc`, not Clone)
        // and is freed at most once: Drop runs once and `release` calls this
        // FnOnce only while the charge is live. cuMemFree, like cudaFree,
        // synchronises the device first, so queued kernels on `ptr` finish.
        self.charge.release(|| unsafe {
            let status = cuCtxSetCurrent(self.ctx);
            let status = if status == 0 {
                cuMemFree_v2(self.ptr)
            } else {
                status
            };
            if status != 0 {
                eprintln!("mdma: CUDA buffer release failed ({status}); buffer remains charged");
            }
            status == 0
        });
    }
}

/// The JITed module and the primary context it lives in, shared by a
/// backend and its graph twins; the last owner unloads and releases.
struct Kernels {
    device: CuDevice,
    ctx: CuContext,
    module: CuModule,
    functions: HashMap<&'static str, CuFunction>,
}

// SAFETY: the context, module and function handles are driver objects usable
// from any thread once the context is current (every user binds first, and
// Drop binds itself); `device` is a plain ordinal.
unsafe impl Send for Kernels {}
// SAFETY: shared access only reads the handles and the function map, neither
// of which is mutated after `new`; the driver API they feed is thread-safe.
unsafe impl Sync for Kernels {}

impl Drop for Kernels {
    fn drop(&mut self) {
        // SAFETY: this runs once, when the last Arc<Kernels> goes, so every
        // CudaBuf, backend and twin (and with it every graph executable) built
        // on this module is already gone. The context is made current,
        // cuCtxSynchronize drains any kernel of `self.module` still running
        // before the unload, and the release balances the single
        // cuDevicePrimaryCtxRetain in `new`. Failures only leak.
        unsafe {
            cuCtxSetCurrent(self.ctx);
            cuCtxSynchronize();
            cuModuleUnload(self.module);
            cuDevicePrimaryCtxRelease_v2(self.device);
        }
    }
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "independent device capabilities and modes"
)]
pub struct CudaBackend {
    device: CuDevice,
    ctx: CuContext,
    name: String,
    unified_memory: bool,
    /// Device and PTX are both sm_80+: the GEMMs' mma.sync is available.
    int8_mma: bool,
    /// Device and PTX are both sm_61+: gemv_q8's dp4a is available.
    dp4a: bool,
    /// Device and PTX are both sm_61+, whether or not decode uses it.
    dp4a_capable: bool,
    /// Q8_1 activations for the GEMMs and gemv_q8. Held across the quantize and
    /// matmul enqueues, so sessions sharing the default stream never
    /// interleave on it; grows, never shrinks.
    q8_scratch: Mutex<Option<Q8Scratch>>,
    /// Bumped by every op that may write device memory other than the
    /// int8 matmuls themselves, so a cached quantization of x is reused
    /// only while nothing can have changed x.
    writes: AtomicU64,
    /// Streaming multiprocessors, for sizing the attention key split.
    sms: usize,
    /// attn_f16kv's (m, l, acc) per chunk, held across the attention and
    /// merge enqueues; grows, never shrinks.
    attn_scratch: Mutex<Option<CudaBuf>>,
    /// Fixed keys per attention chunk instead of the SM-count heuristic.
    attn_chunk: Option<usize>,
    /// Null (the legacy default stream) for eager backends; a graph twin's
    /// own non-blocking stream otherwise.
    stream: CuStream,
    /// Set on a graph twin: token and position come from device memory.
    replay: Option<graph::Replay>,
    /// Withdraws the federation profile once device work cannot be joined.
    health: crate::federation::profile::gpu::Health,
    /// Decode may capture a graph (MDMA_CUDA_GRAPHS unset or not 0).
    graphs: bool,
    buffers: crate::memory::BufferBudget,
    /// Declared last so the buffers above are freed before the context
    /// can be released.
    kernels: Arc<Kernels>,
}

// SAFETY: the raw fields are the primary context (current-able from any
// thread; every entry point calls `bind` first) and a stream that is either
// null (the legacy stream) or owned by this twin until Drop. Every other
// field is Send.
unsafe impl Send for CudaBackend {}
// SAFETY: `&self` methods only read the handles and settings; the mutable
// state (Q8 and attention scratch, write epoch, buffer budget) sits behind a
// Mutex or atomics, and concurrent enqueues on one stream are allowed by the
// thread-safe driver API.
unsafe impl Sync for CudaBackend {}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // SAFETY: `self.ctx` is still retained by `self.kernels` (fields drop
        // after this body). A non-null `self.stream` was created by
        // `graph_twin` and is owned only by this twin, so it is synchronised
        // (replays still reading the scratch freed next finish) and destroyed
        // exactly once; the null legacy stream is never destroyed.
        unsafe {
            cuCtxSetCurrent(self.ctx);
            if !self.stream.is_null() {
                if cuStreamSynchronize(self.stream) != CUDA_SUCCESS {
                    self.health.withdraw();
                }
                cuStreamDestroy_v2(self.stream);
            }
        }
        if self.replay.is_none() {
            self.health.withdraw();
        }
    }
}

macro_rules! params {
    ($($v:ident),* $(,)?) => {
        [$(std::ptr::from_mut(&mut $v).cast::<c_void>()),*]
    };
}

/// quantize_q8_1's output and which input it holds: q, k and v (and gate
/// and up) read the same x, so decode quantizes it once for all of them.
struct Q8Scratch {
    q8: CudaBuf,
    ys: CudaBuf,
    /// (x ptr, x bytes, write epoch) of the activations in q8/ys.
    holds: Option<(u64, usize, u64)>,
}

/// `sm_86` -> 86; anything unparseable counts as pre-Ampere.
fn arch_number(arch: &str) -> u32 {
    arch.strip_prefix("sm_")
        .and_then(|n| {
            n.trim_end_matches(|c: char| c.is_ascii_alphabetic())
                .parse()
                .ok()
        })
        .unwrap_or(0)
}

/// `struct Gemv3` in kernels.cu, passed by value.
#[repr(C)]
struct Gemv3 {
    w: [CuDevicePtr; 3],
    y: [CuDevicePtr; 3],
    n_out: [i32; 3],
    dtype: [i32; 3],
    block_bytes: [i32; 3],
    blocks: [i32; 3],
}

/// `struct Freqs` in kernels.cu, passed by value: the driver copies all
/// ROPE_PAIRS entries, so the sizes must match.
#[repr(C)]
struct Freqs([f32; ROPE_PAIRS]);

/// theta^(-2i/head_dim) per pair, with the CPU backend's f32 powf.
fn rope_freqs(op: &str, spec: &RopeSpec) -> Result<Freqs> {
    let pairs = spec.head_dim / 2;
    if pairs > ROPE_PAIRS {
        return Err(Error::ShapeMismatch(format!(
            "{op}: head_dim {} exceeds {}",
            spec.head_dim,
            2 * ROPE_PAIRS
        )));
    }
    let mut freqs = Freqs([0.0; ROPE_PAIRS]);
    for (i, f) in freqs.0.iter_mut().take(pairs).enumerate() {
        *f = spec.theta.powf(-2.0 * i as f32 / spec.head_dim as f32);
    }
    Ok(freqs)
}

/// Shape checks shared by rope and rope_kv_write, and the frequency
/// table: (freqs, n_q_heads, n_kv_heads).
fn rope_setup(
    q: &CudaBuf,
    k: &CudaBuf,
    n_tokens: usize,
    spec: &RopeSpec,
) -> Result<(Freqs, i32, i32)> {
    let shape::Rope {
        n_q_heads,
        n_kv_heads,
    } = shape::rope(spec, q.len / 4, k.len / 4, n_tokens)?;
    Ok((rope_freqs("rope", spec)?, int(n_q_heads)?, int(n_kv_heads)?))
}

impl CudaBackend {
    /// Free and total device bytes.
    fn memory_info(&self) -> Result<(usize, usize)> {
        self.bind()?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: `bind` made the context current, as cuMemGetInfo requires;
        // `free` and `total` are live usize locals matching `size_t *`.
        cu("cuMemGetInfo", unsafe {
            cuMemGetInfo_v2(&raw mut free, &raw mut total)
        })?;
        Ok((free, total))
    }
    /// Opens device `MDMA_CUDA_DEVICE` (default 0) on its primary context
    /// and JITs the embedded PTX for it.
    pub fn new() -> Result<Self> {
        let ordinal = match std::env::var("MDMA_CUDA_DEVICE") {
            Ok(v) => v.parse::<c_int>().map_err(|_| {
                Error::Backend(format!("MDMA_CUDA_DEVICE='{v}' is not a device index"))
            })?,
            Err(_) => 0,
        };
        // SAFETY: cuInit runs first; every out-pointer is a live local of the
        // prototype's type; `ordinal` is checked against cuDeviceGetCount
        // before cuDeviceGet. The retained primary context is made current
        // before the JIT and released via `release` on the error paths after
        // it, except the MDMA_CUDA_INT8 rejection (leaks module and retain;
        // not UB). Pointer-specific calls are noted where they happen.
        unsafe {
            cu("cuInit", cuInit(0))?;
            let mut count = 0;
            cu("cuDeviceGetCount", cuDeviceGetCount(&raw mut count))?;
            if ordinal < 0 || ordinal >= count {
                return Err(Error::Backend(format!(
                    "CUDA device {ordinal} requested, {count} present"
                )));
            }
            let mut device = 0;
            cu("cuDeviceGet", cuDeviceGet(&raw mut device, ordinal))?;
            let mut raw = [0 as c_char; 256];
            cu(
                "cuDeviceGetName",
                cuDeviceGetName(raw.as_mut_ptr(), int(raw.len())?, device),
            )?;
            let (mut major, mut minor) = (0, 0);
            cu(
                "cuDeviceGetAttribute",
                cuDeviceGetAttribute(
                    &raw mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                ),
            )?;
            cu(
                "cuDeviceGetAttribute",
                cuDeviceGetAttribute(
                    &raw mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                ),
            )?;
            let mut sms = 0;
            cu(
                "cuDeviceGetAttribute",
                cuDeviceGetAttribute(
                    &raw mut sms,
                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    device,
                ),
            )?;
            let mut integrated = 0;
            cu(
                "cuDeviceGetAttribute",
                cuDeviceGetAttribute(&raw mut integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device),
            )?;
            // SAFETY: cuDeviceGetName succeeded with `raw`'s real length
            // (256), and the driver returns the name NUL-terminated within
            // `len` bytes, so CStr::from_ptr stays inside `raw`, which
            // outlives the borrow (format! copies it).
            let name = format!(
                "{} (sm_{major}{minor})",
                CStr::from_ptr(raw.as_ptr()).to_string_lossy()
            );

            let mut ctx = std::ptr::null_mut();
            cu(
                "cuDevicePrimaryCtxRetain",
                cuDevicePrimaryCtxRetain(&raw mut ctx, device),
            )?;
            let release = |e: Error| {
                cuDevicePrimaryCtxRelease_v2(device);
                e
            };
            cu("cuCtxSetCurrent", cuCtxSetCurrent(ctx)).map_err(release)?;

            // PTX must be NUL-terminated for the JIT.
            let ptx = CString::new(PTX).expect("PTX has no interior NUL");
            let mut log = vec![0u8; 8192];
            let mut opts = [CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
            let mut vals = [log.as_mut_ptr().cast::<c_void>(), log.len() as *mut c_void];
            let mut module = std::ptr::null_mut();
            // SAFETY: context current; `ptx` is the NUL-terminated image the
            // JIT needs; n = 2 matches `opts`/`vals`. vals[0] is `log` (8192
            // bytes, live across the call), vals[1] its size by value, as the
            // ERROR_LOG_BUFFER(_SIZE_BYTES) options specify; the driver caps
            // the log there and may write the used size into vals[1] (ours).
            // Kernel lookups below pass CStrings alive across each call.
            let r = cuModuleLoadDataEx(
                &raw mut module,
                ptx.as_ptr().cast(),
                2,
                opts.as_mut_ptr(),
                vals.as_mut_ptr(),
            );
            if r != CUDA_SUCCESS {
                let msg = CStr::from_bytes_until_nul(&log)
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                return Err(release(Error::Backend(format!(
                    "cuModuleLoadDataEx ({r}) JITing PTX for {name}: {msg}"
                ))));
            }
            let mut functions = HashMap::new();
            for &k in KERNELS {
                let c_name = CString::new(k).expect("kernel name");
                let mut f = std::ptr::null_mut();
                let r = cuModuleGetFunction(&raw mut f, module, c_name.as_ptr());
                if r != CUDA_SUCCESS {
                    cuModuleUnload(module);
                    return Err(release(
                        cu(&format!("cuModuleGetFunction({k})"), r).unwrap_err(),
                    ));
                }
                functions.insert(k, f);
            }
            let built = arch_number(env!("MDMA_CUDA_ARCH_BUILT"));
            // MDMA_CUDA_INT8: prefill (default) = GEMM only; all = GEMM and
            // dp4a decode; off = K-quant matmuls on f32 activations only.
            let (gemm, decode) = match std::env::var("MDMA_CUDA_INT8").as_deref() {
                Err(_) | Ok("prefill") => (true, false),
                Ok("all") => (true, true),
                Ok("off") => (false, false),
                Ok(v) => {
                    return Err(Error::Backend(format!(
                        "MDMA_CUDA_INT8={v:?}: expected prefill, all or off"
                    )));
                }
            };
            let dp4a_capable = (major, minor) >= (6, 1) && built >= 61;
            Ok(Self {
                device,
                ctx,
                name,
                unified_memory: integrated != 0,
                int8_mma: gemm && major >= 8 && built >= 80,
                dp4a: decode && dp4a_capable,
                dp4a_capable,
                q8_scratch: Mutex::new(None),
                writes: AtomicU64::new(0),
                sms: sms.max(1).unsigned_abs() as usize,
                attn_scratch: Mutex::new(None),
                attn_chunk: None,
                stream: std::ptr::null_mut(),
                replay: None,
                health: crate::federation::profile::gpu::Health::default(),
                graphs: std::env::var("MDMA_CUDA_GRAPHS").as_deref() != Ok("0"),
                buffers: crate::memory::BufferBudget::default(),
                kernels: Arc::new(Kernels {
                    device,
                    ctx,
                    module,
                    functions,
                }),
            })
        }
    }

    /// A twin sharing this backend's module and settings, with its own
    /// non-blocking stream and scratch, whose token and position come from
    /// device memory: what graph capture records and replay reruns.
    fn graph_twin(&self, capacity: usize) -> Result<Self> {
        self.bind()?;
        let mut stream = std::ptr::null_mut();
        // SAFETY: `bind` above made `self.ctx` current; `stream` is a live
        // local out-pointer. The new stream moves into `twin`, whose Drop
        // destroys it, including when the alloc below fails.
        cu("cuStreamCreate", unsafe {
            cuStreamCreate(&raw mut stream, CU_STREAM_NON_BLOCKING)
        })?;
        let mut twin = Self {
            device: self.device,
            ctx: self.ctx,
            name: self.name.clone(),
            unified_memory: self.unified_memory,
            int8_mma: self.int8_mma,
            dp4a: self.dp4a,
            dp4a_capable: self.dp4a_capable,
            q8_scratch: Mutex::new(None),
            writes: AtomicU64::new(0),
            sms: self.sms,
            attn_scratch: Mutex::new(None),
            attn_chunk: self.attn_chunk,
            stream,
            replay: None,
            health: self.health.clone(),
            graphs: self.graphs,
            buffers: self.buffers.clone(),
            kernels: Arc::clone(&self.kernels),
        };
        twin.replay = Some(graph::Replay {
            params: twin.alloc(8, Pool::Scratch)?,
            capacity,
        });
        Ok(twin)
    }

    /// Waits for everything queued on this backend's stream.
    fn sync(&self) -> Result<()> {
        self.bind()?;
        // SAFETY: `bind` made the context current; `self.stream` is either
        // null (the legacy stream) or this twin's own stream, live until Drop.
        self.health.check(cu("cuStreamSynchronize", unsafe {
            cuStreamSynchronize(self.stream)
        }))
    }

    pub(crate) fn federation_profile(
        &self,
    ) -> std::io::Result<crate::federation::profile::gpu::Gpu> {
        use crate::federation::profile::gpu::{Gpu, fields};
        use sha2::{Digest, Sha256};
        let (mut driver, mut uuid) = (0, [0u8; 16]);
        // SAFETY: both write through pointers to live locals of the
        // prototypes' types (an int, a 16-byte CUuuid); `self.device` came
        // from cuDeviceGet after cuInit, and neither call needs a context.
        cu("cuDriverGetVersion", unsafe {
            cuDriverGetVersion(&raw mut driver)
        })
        .map_err(std::io::Error::other)?;
        cu("cuDeviceGetUuid", unsafe {
            cuDeviceGetUuid_v2(uuid.as_mut_ptr(), self.device)
        })
        .map_err(std::io::Error::other)?;
        if driver <= 0 || uuid == [0; 16] {
            return Err(std::io::Error::other("unverifiable CUDA driver or device"));
        }
        let pci = self.pci_bus_id().map_err(std::io::Error::other)?;
        Ok(Gpu::immutable(
            "cuda",
            "f32",
            "f16",
            fields([
                ("vocabulary", "mdma.cuda.v1".into()),
                ("workload", "decoder".into()),
                ("device_name", self.name.clone()),
                (
                    "device_identity",
                    format!("pci-{pci}/uuid-{}", hex::encode(uuid)),
                ),
                ("driver_version", driver.to_string()),
                ("kernel_target", env!("MDMA_CUDA_ARCH_BUILT").into()),
                (
                    "kernel_id",
                    format!("sha256:{}", hex::encode(Sha256::digest(PTX))),
                ),
                ("weights", "gguf-raw-matrices-f32-norms".into()),
                (
                    "int8_prefill",
                    if self.int8_mma { "mma-q8_1" } else { "off" }.into(),
                ),
                (
                    "int8_decode",
                    if self.dp4a { "dp4a-q8_1" } else { "off" }.into(),
                ),
                (
                    "execution",
                    if self.graphs {
                        "eligible-request-graph-with-eager-fallback"
                    } else {
                        "eager"
                    }
                    .into(),
                ),
                (
                    "attention",
                    match self.attn_chunk {
                        Some(keys) => format!("f16kv-chunks-of-{keys}"),
                        None => format!("f16kv-split-over-{}-sms", self.sms),
                    },
                ),
                ("target_arch", std::env::consts::ARCH.into()),
                ("target_os", std::env::consts::OS.into()),
            ]),
            self.health.clone(),
        ))
    }

    /// Device name and compute capability, e.g. for the serve banner.
    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// PCI address, `dddd:bb:dd.f`, to find the same device in NVML.
    pub fn pci_bus_id(&self) -> Result<String> {
        let mut raw = [0 as c_char; 32];
        // SAFETY: `self.device` came from cuDeviceGet after cuInit in `new`,
        // and this device query needs no current context. `raw` is 32 bytes
        // and that length is passed; the `dddd:bb:dd.f` id needs 13 with NUL.
        cu("cuDeviceGetPCIBusId", unsafe {
            cuDeviceGetPCIBusId(raw.as_mut_ptr(), int(raw.len())?, self.device)
        })?;
        // SAFETY: the call above succeeded, so `raw` holds a NUL-terminated
        // id within its 32 bytes; into_owned copies it before `raw` drops.
        Ok(unsafe { CStr::from_ptr(raw.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// Whether K-quant prefill takes the int8 tensor-core GEMM.
    pub fn int8_gemm(&self) -> bool {
        self.int8_mma
    }

    /// Whether K-quant decode takes the dp4a matvec on Q8_1 activations.
    pub fn int8_gemv(&self) -> bool {
        self.dp4a
    }

    /// Adds the dp4a decode matvec where the device has it, as with
    /// MDMA_CUDA_INT8=all.
    #[must_use]
    pub fn with_int8_decode(mut self) -> Self {
        self.dp4a = self.dp4a_capable;
        self
    }

    /// Attention in chunks of `keys` keys (at least 1), merged by
    /// attn_merge whenever that makes more than one chunk.
    #[must_use]
    pub fn with_attn_chunk(mut self, keys: usize) -> Self {
        self.attn_chunk = Some(keys.max(1));
        self
    }

    /// Keys per attention chunk for n_q queries over kv_len keys, with
    /// head_blocks blocks per query (q heads / G): one chunk while those
    /// blocks already fill the SMs, else enough chunks of at least
    /// ATTN_MIN_CHUNK keys to reach ATTN_BLOCKS_PER_SM.
    fn attn_chunk(&self, head_blocks: usize, n_q: usize, kv_len: usize) -> usize {
        if let Some(keys) = self.attn_chunk {
            return keys;
        }
        let blocks = head_blocks * n_q;
        let want = (self.sms * ATTN_BLOCKS_PER_SM).div_ceil(blocks.max(1));
        let splits = want
            .min(kv_len.div_ceil(ATTN_MIN_CHUNK))
            .clamp(1, ATTN_MAX_SPLITS);
        kv_len.div_ceil(splits)
    }

    /// (G, keys per chunk, chunks) for attn_f16kv over kv_len keys. G is
    /// the largest query-head group the kernel is built for that divides
    /// the GQA group, so a block reads its kv head's rows once. A graph
    /// twin's kv_len is the capacity, most of it past pos: it keeps the
    /// minimum chunk up to ATTN_MAX_SPLITS chunks, so the chunks a short
    /// context fills match eager decode's and the rest exit at once.
    fn attn_layout(&self, spec: &AttnSpec, n_q: usize, kv_len: usize) -> (usize, usize, usize) {
        let group = spec.n_q_heads / spec.n_kv_heads;
        let g = [8, 4, 2, 1]
            .into_iter()
            .find(|g| group.is_multiple_of(*g))
            .expect("1 divides everything");
        let chunk = match (self.attn_chunk, &self.replay) {
            (None, Some(_)) => ATTN_MIN_CHUNK.max(kv_len.div_ceil(ATTN_MAX_SPLITS)),
            _ => self.attn_chunk(spec.n_q_heads / g, n_q, kv_len),
        };
        (g, chunk, kv_len.div_ceil(chunk))
    }

    fn attn_partials_bytes(spec: &AttnSpec, n_q: usize, splits: usize) -> usize {
        n_q * spec.n_q_heads * splits * (spec.head_dim + 2) * 4
    }

    /// Ops whose position or token would be frozen into a graph.
    fn eager_only(&self, op: &str) -> Result<()> {
        match self.replay {
            Some(_) => Err(Error::Backend(format!(
                "cuda: {op} is not graph-replayable"
            ))),
            None => Ok(()),
        }
    }

    /// K-quant matmuls on f32 activations only (gemv_q), as with
    /// MDMA_CUDA_INT8=off.
    #[must_use]
    pub fn without_int8(mut self) -> Self {
        self.int8_mma = false;
        self.dp4a = false;
        self
    }

    /// (n_in, n_tokens) when every weight is a rank-2 K-quant of the same
    /// n_in and x is one token, which qmatmul would run on gemv_q (f32
    /// activations): the fused decode launches below replace exactly those.
    /// More tokens take the GEMM, where fusing only halves the grid. None
    /// sends the caller to the unfused trait default.
    fn gemv_q_fusable(&self, ws: &[&QWeight<CudaBuf>], x: &CudaBuf) -> Option<(usize, usize)> {
        let n_in = *ws.first()?.shape.get(1)?;
        let fits = |w: &&QWeight<CudaBuf>| {
            w.shape.len() == 2
                && w.shape[1] == n_in
                && matches!(w.dtype, GgmlDType::Q4K | GgmlDType::Q6K)
        };
        if n_in == 0 || !n_in.is_multiple_of(256) || !ws.iter().all(fits) {
            return None;
        }
        let x_elems = x.len / 4;
        if !x_elems.is_multiple_of(n_in) {
            return None;
        }
        let n_tokens = x_elems / n_in;
        (n_tokens == 1 && !self.dp4a).then_some((n_in, n_tokens))
    }

    /// Quantizes x ([n_tokens][n_in] f32) to Q8_1 in the scratch buffers
    /// and runs gemm_q4k / gemm_q6k (prefill) or gemv_q8 (decode) into y, under one
    /// scratch lock.
    #[allow(clippy::too_many_arguments)]
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "the kernel writes y on the device"
    )]
    fn qmatmul_q8(
        &self,
        w: &QWeight<CudaBuf>,
        x: &CudaBuf,
        y: &mut CudaBuf,
        n_out: usize,
        n_in: usize,
        n_tokens: usize,
        gemm: bool,
    ) -> Result<()> {
        let n_blocks = n_tokens * n_in / 32;
        let (q_bytes, ys_bytes) = (n_tokens * n_in, n_blocks * 8);
        let mut scratch = self
            .q8_scratch
            .lock()
            .map_err(|_| Error::Backend("cuda: q8 scratch lock poisoned".into()))?;
        if scratch
            .as_ref()
            .is_none_or(|s| s.q8.len < q_bytes || s.ys.len < ys_bytes)
        {
            // Queued kernels may still read the old buffers.
            if scratch.is_some() {
                self.sync()?;
            }
            *scratch = None;
            let q8 = self.alloc(q_bytes, Pool::Scratch)?;
            let ys = self.alloc(ys_bytes, Pool::Scratch)?;
            *scratch = Some(Q8Scratch {
                q8,
                ys,
                holds: None,
            });
        }
        let s = scratch.as_mut().expect("scratch allocated above");
        let (mut q_ptr, mut ys_ptr) = (s.q8.ptr, s.ys.ptr);
        let key = (x.ptr, x.len, self.writes.load(Ordering::Acquire));
        if s.holds != Some(key) {
            let mut x_ptr = x.ptr;
            let mut n_blocks_i = int(n_blocks)?;
            self.launch_quiet(
                "quantize_q8_1",
                [(n_blocks as u32 * 8).div_ceil(Q8_THREADS), 1, 1],
                [Q8_THREADS, 1, 1],
                &mut params!(x_ptr, q_ptr, ys_ptr, n_blocks_i),
            )?;
            s.holds = Some(key);
        }
        // y aliasing x would change x under the cached quantization.
        if y.ptr < x.ptr + x.len as u64 && x.ptr < y.ptr + y.len as u64 {
            s.holds = None;
        }
        let (mut w_ptr, mut y_ptr) = (w.buf.ptr, y.ptr);
        let (mut n_out_i, mut n_in_i) = (int(n_out)?, int(n_in)?);
        let mut code = dtype_code::code(w.dtype)?;
        if gemm {
            let mut n_tok_i = int(n_tokens)?;
            let grid = [
                (n_out as u32).div_ceil(GEMM_M),
                (n_tokens as u32).div_ceil(GEMM_N),
                1,
            ];
            // Q4_K blocks are 16-byte aligned, so gemm_q4k stages them
            // with cp.async; Q6_K (210-byte blocks) stays synchronous.
            if w.dtype == GgmlDType::Q4K {
                self.launch_quiet(
                    "gemm_q4k",
                    grid,
                    [128, 1, 1],
                    &mut params!(w_ptr, q_ptr, ys_ptr, y_ptr, n_out_i, n_in_i, n_tok_i),
                )
            } else {
                self.launch_quiet(
                    "gemm_q6k",
                    grid,
                    [128, 1, 1],
                    &mut params!(w_ptr, q_ptr, ys_ptr, y_ptr, n_out_i, n_in_i, n_tok_i),
                )
            }
        } else {
            let mut bb = dtype_code::block_bytes(w.dtype);
            self.launch_quiet(
                "gemv_q8",
                [(n_out as u32).div_ceil(WARPS * Q_ROWS), n_tokens as u32, 1],
                [32, WARPS, 1],
                &mut params!(w_ptr, q_ptr, ys_ptr, y_ptr, n_out_i, n_in_i, code, bb),
            )
        }
    }

    /// Tokio may call from any worker thread; the context is per-thread state.
    fn bind(&self) -> Result<()> {
        // SAFETY: `self.ctx` is the primary context retained by `self.kernels`
        // for as long as `self` exists; making it current only changes this
        // thread's driver state.
        cu("cuCtxSetCurrent", unsafe { cuCtxSetCurrent(self.ctx) })
    }

    /// Marks device memory as possibly written: cached Q8_1 activations
    /// are stale from here on.
    fn wrote(&self) {
        self.writes.fetch_add(1, Ordering::Release);
    }

    fn launch(
        &self,
        name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        params: &mut [*mut c_void],
    ) -> Result<()> {
        self.wrote();
        self.launch_quiet(name, grid, block, params)
    }

    // Arguments stay live through launch, including its write-epoch update.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "the kernel writes y on the device"
    )]
    fn launch_binary_f32(
        &self,
        op: &str,
        kernel: &str,
        a: &CudaBuf,
        b: &CudaBuf,
        y: &mut CudaBuf,
    ) -> Result<()> {
        let n = shape::elementwise(op, &[a.len, b.len], &[y.len])? / 4;
        let mut a_ptr = a.ptr;
        let mut b_ptr = b.ptr;
        let mut y_ptr = y.ptr;
        let mut n_i = int(n)?;
        self.launch(
            kernel,
            [(n as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
            &mut params!(a_ptr, b_ptr, y_ptr, n_i),
        )
    }

    /// launch without bumping the write epoch: only for the int8 matmul
    /// path, whose writes (scratch, and y, checked for aliasing x) the Q8
    /// cache accounts for itself.
    fn launch_quiet(
        &self,
        name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        params: &mut [*mut c_void],
    ) -> Result<()> {
        self.bind()?;
        let f = *self
            .kernels
            .functions
            .get(name)
            .ok_or_else(|| Error::Backend(format!("cuda: no kernel {name}")))?;
        // SAFETY: context bound; `f` is from the module `self.kernels` keeps
        // loaded; `self.stream` is null (legacy) or this twin's live stream.
        // The driver reads one pointer per PTX parameter from `params`, so
        // every caller (this is a safe fn) must pass a `params!` array of live
        // locals in the kernel's .cu order and types (Gemv3, Freqs repr(C));
        // values are copied at launch. Device extents: callers' shape checks.
        let r = unsafe {
            cuLaunchKernel(
                f,
                grid[0].max(1),
                grid[1].max(1),
                grid[2].max(1),
                block[0],
                block[1],
                block[2],
                0,
                self.stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        cu(name, r)
    }
}

impl Backend for CudaBackend {
    type Buf = CudaBuf;

    /// Joins this backend's own stream. Graph twins join theirs when
    /// dropped; synchronizing the whole context would invalidate another
    /// thread's graph capture (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED).
    fn join_execution(&self) -> Result<()> {
        self.sync()
    }

    fn device_label(&self) -> Option<String> {
        Some(self.device_name().into())
    }
    fn unified_memory(&self) -> bool {
        self.unified_memory
    }

    fn available_device_bytes(&self) -> Result<Option<usize>> {
        self.memory_info().map(|(free, _)| Some(free))
    }

    fn total_device_bytes(&self) -> Option<usize> {
        self.memory_info().ok().map(|(_, total)| total)
    }

    super::delegate_decoder!(graph::decoder);

    fn set_buffer_budget(&mut self, budget: crate::memory::BufferBudget) -> Result<()> {
        self.buffers.replace(budget)
    }

    fn buffer_budget(&self) -> Option<&crate::memory::BufferBudget> {
        Some(&self.buffers)
    }

    fn max_head_dim(&self) -> Option<usize> {
        Some(decoder_ops::WINDOWED_MAX_HEAD_DIM)
    }

    fn supports(&self, op: Operation) -> bool {
        matches!(
            op,
            Operation::Scale
                | Operation::GeluTanhMul
                | Operation::Softcap
                | Operation::RopeSplitHalf
                | Operation::RopePartialSplitHalf
                | Operation::SigmoidMul
                | Operation::GatedDeltaNet
                | Operation::WindowedAttention
        )
    }

    fn supports_weight_dtype(&self, dtype: GgmlDType) -> bool {
        dtype_code::decoder_code(dtype).is_ok()
    }

    fn scale(&self, x: &mut Self::Buf, factor: f32) -> Result<()> {
        if !factor.is_finite() {
            return Err(Error::ShapeMismatch("scale: factor must be finite".into()));
        }
        self.unary(x, factor, 0)
    }

    fn softcap(&self, x: &mut Self::Buf, cap: f32) -> Result<()> {
        shape::softcap(cap)?;
        self.unary(x, cap, 1)
    }

    fn gelu_tanh_mul(&self, a: &Self::Buf, b: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        self.gelu(a, b, y)
    }

    fn rope_split_half(
        &self,
        x: &mut Self::Buf,
        pos: u32,
        spec: &RopeSpec,
        factors: &Self::Buf,
    ) -> Result<()> {
        self.split_rope(x, pos, spec, factors)
    }

    fn rope_partial_split_half(
        &self,
        x: &mut Self::Buf,
        pos: u32,
        spec: &RopeSpec,
        rotary_dim: usize,
    ) -> Result<()> {
        self.split_rope_partial(x, pos, spec, rotary_dim)
    }

    fn sigmoid_mul(&self, gate: &Self::Buf, x: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        self.sigmoid_mul(gate, x, y)
    }

    fn ssm_conv_silu(
        &self,
        w: &Self::Buf,
        state: &mut Self::Buf,
        x: &Self::Buf,
        y: &mut Self::Buf,
    ) -> Result<()> {
        self.ssm_conv_silu(w, state, x, y)
    }

    fn gated_delta_step(
        &self,
        spec: &DeltaSpec,
        inputs: &DeltaInputs<Self::Buf>,
        state: &mut Self::Buf,
        o: &mut Self::Buf,
    ) -> Result<()> {
        self.gated_delta_step(spec, inputs, state, o)
    }

    fn attn_windowed(
        &self,
        q: &Self::Buf,
        kv: &KvView<Self::Buf>,
        spec: &AttnSpec,
        scale: f32,
        window: usize,
        o: &mut Self::Buf,
    ) -> Result<()> {
        self.attention_windowed(q, kv, spec, scale, window, o)
    }

    fn alloc(&self, bytes: usize, _pool: Pool) -> Result<Self::Buf> {
        // A fresh buffer may reuse a freed x's address.
        self.wrote();
        self.bind()?;
        // cuMemAlloc rejects 0 bytes; a zero-length buffer still needs a
        // distinct, freeable pointer.
        let (ptr, charge) =
            self.buffers
                .allocate(crate::memory::MemoryDomain::Device, bytes.max(4), || {
                    let mut ptr = 0;
                    // SAFETY: `bind` above made `self.ctx` current on this thread and
                    // `allocate` runs this closure synchronously; `ptr` is a live u64
                    // out-pointer and the size is non-zero, which cuMemAlloc requires.
                    cu("cuMemAlloc", unsafe {
                        cuMemAlloc_v2(&raw mut ptr, bytes.max(4))
                    })?;
                    Ok(ptr)
                })?;
        let buf = CudaBuf {
            ctx: self.ctx,
            ptr,
            len: bytes,
            charge,
            _kernels: self.kernels.clone(),
        };
        // SAFETY: `ptr` was just allocated in the current context with
        // exactly bytes.max(4) bytes, the length cleared here; `buf` already
        // owns it, so a failed memset still frees it.
        cu("cuMemsetD8", unsafe { cuMemsetD8_v2(ptr, 0, bytes.max(4)) })?;
        Ok(buf)
    }

    fn upload(&self, dst: &mut Self::Buf, offset: usize, src: &[u8]) -> Result<()> {
        shape::range("upload", offset, src.len(), dst.len)?;
        if src.is_empty() {
            return Ok(());
        }
        self.wrote();
        self.bind()?;
        // SAFETY: shape::range above bounds offset + src.len() by dst.len,
        // which never exceeds dst's allocation (bytes.max(4)). `src` is valid
        // for src.len() reads, and the synchronous HtoD copy has consumed or
        // staged it before returning, so the slice is not read afterwards.
        cu("cuMemcpyHtoD", unsafe {
            cuMemcpyHtoD_v2(dst.ptr + offset as u64, src.as_ptr().cast(), src.len())
        })
    }

    fn download(&self, src: &Self::Buf, offset: usize, dst: &mut [u8]) -> Result<()> {
        shape::range("download", offset, dst.len(), src.len)?;
        if dst.is_empty() {
            return Ok(());
        }
        // The legacy stream's copy does not wait for a twin's own stream.
        if !self.stream.is_null() {
            self.sync()?;
        }
        self.bind()?;
        // SAFETY: shape::range above bounds offset + dst.len() by src.len.
        // `dst` is an exclusive borrow valid for dst.len() writes, and a DtoH
        // copy into host memory returns only once it is complete, so nothing
        // lands after the borrow ends.
        cu("cuMemcpyDtoH", unsafe {
            cuMemcpyDtoH_v2(dst.as_mut_ptr().cast(), src.ptr + offset as u64, dst.len())
        })
    }

    fn copy(
        &self,
        src: &Self::Buf,
        src_off: usize,
        dst: &mut Self::Buf,
        dst_off: usize,
        len: usize,
    ) -> Result<()> {
        shape::range("copy", src_off, len, src.len)?;
        shape::range("copy", dst_off, len, dst.len)?;
        if len == 0 {
            return Ok(());
        }
        self.wrote();
        self.bind()?;
        // SAFETY: both ranges are bounded by the shape::range checks above,
        // and `src` and `dst` are distinct CudaBufs, each owning its own
        // allocation, so the ranges cannot overlap. Device to device only: no
        // host memory is touched, and freeing either buffer while the copy is
        // queued goes through cuMemFree, which synchronises first.
        cu("cuMemcpyDtoD", unsafe {
            cuMemcpyDtoD_v2(dst.ptr + dst_off as u64, src.ptr + src_off as u64, len)
        })
    }

    /// Decode on gemv_q: q, k and v in one gemv_q3 launch.
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
        let Some((n_in, n_tokens)) = self.gemv_q_fusable(&[wq, wk, wv], x) else {
            self.qmatmul(wq, x, q)?;
            self.qmatmul(wk, x, k)?;
            return self.qmatmul(wv, x, v);
        };
        let mut m = Gemv3 {
            w: [wq.buf.ptr, wk.buf.ptr, wv.buf.ptr],
            y: [q.ptr, k.ptr, v.ptr],
            n_out: [0; 3],
            dtype: [0; 3],
            block_bytes: [0; 3],
            blocks: [0; 3],
        };
        for (i, (w, y)) in [(wq, &*q), (wk, &*k), (wv, &*v)].into_iter().enumerate() {
            let n_out = shape::matmul(&w.shape, x.len / 4, y.len / 4)?.n_out;
            m.n_out[i] = int(n_out)?;
            m.dtype[i] = dtype_code::code(w.dtype)?;
            m.block_bytes[i] = dtype_code::block_bytes(w.dtype);
            m.blocks[i] = int(n_out.div_ceil((WARPS * Q_ROWS) as usize))?;
        }
        let total = u32::try_from(m.blocks.iter().sum::<i32>()).map_err(|_| Error::SizeOverflow)?;
        let (mut x_ptr, mut n_in_i) = (x.ptr, int(n_in)?);
        self.launch(
            "gemv_q3",
            [total, n_tokens as u32, 1],
            [32, WARPS, 1],
            &mut params!(m, x_ptr, n_in_i),
        )
    }

    /// Decode on gemv_q: gate, up and silu_mul in one gemv_q_swiglu launch.
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
        let Some((n_in, n_tokens)) = self.gemv_q_fusable(&[gate, up], x) else {
            self.qmatmul(gate, x, gate_scratch)?;
            self.qmatmul(up, x, up_scratch)?;
            return self.silu_mul(gate_scratch, up_scratch, y);
        };
        let n_out = shape::matmul(&gate.shape, x.len / 4, y.len / 4)?.n_out;
        let (mut wg, mut wu, mut x_ptr, mut y_ptr) = (gate.buf.ptr, up.buf.ptr, x.ptr, y.ptr);
        let (mut n_out_i, mut n_in_i) = (int(n_out)?, int(n_in)?);
        let (mut dt_g, mut bb_g) = (
            dtype_code::code(gate.dtype)?,
            dtype_code::block_bytes(gate.dtype),
        );
        let (mut dt_u, mut bb_u) = (
            dtype_code::code(up.dtype)?,
            dtype_code::block_bytes(up.dtype),
        );
        self.launch(
            "gemv_q_swiglu",
            [(n_out as u32).div_ceil(WARPS * Q_ROWS), n_tokens as u32, 1],
            [32, WARPS, 1],
            &mut params!(
                wg, wu, x_ptr, y_ptr, n_out_i, n_in_i, dt_g, bb_g, dt_u, bb_u
            ),
        )
    }

    fn qmatmul(&self, w: &QWeight<Self::Buf>, x: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        let shape::Matmul {
            n_out,
            n_in,
            n_tokens,
        } = shape::matmul(&w.shape, x.len / 4, y.len / 4)?;
        let mut code = dtype_code::decoder_code(w.dtype)?;
        if w.dtype == GgmlDType::Q5K {
            return self.qmatmul_q5k(w, x, y, n_out, n_in, n_tokens);
        }
        if w.dtype == GgmlDType::Q8_0 {
            return self.qmatmul_q8_0(w, x, y, n_out, n_in, n_tokens);
        }
        let mut w_ptr = w.buf.ptr;
        let mut x_ptr = x.ptr;
        let mut y_ptr = y.ptr;
        let mut n_out_i = int(n_out)?;
        let mut n_in_i = int(n_in)?;
        let grid = [(n_out as u32).div_ceil(WARPS), n_tokens as u32, 1];
        let block = [32, WARPS, 1];
        if dtype_code::is_kquant(code) {
            shape::kquant_aligned("qmatmul", w.dtype, n_in)?;
            if self.int8_mma && n_tokens >= GEMM_MIN_TOKENS {
                return self.qmatmul_q8(w, x, y, n_out, n_in, n_tokens, true);
            }
            if self.dp4a {
                return self.qmatmul_q8(w, x, y, n_out, n_in, n_tokens, false);
            }
            let mut bb = dtype_code::block_bytes(w.dtype);
            let grid = [(n_out as u32).div_ceil(WARPS * Q_ROWS), n_tokens as u32, 1];
            self.launch(
                "gemv_q",
                grid,
                block,
                &mut params!(w_ptr, x_ptr, y_ptr, n_out_i, n_in_i, code, bb),
            )
        } else {
            self.launch(
                "gemv_f",
                grid,
                block,
                &mut params!(w_ptr, x_ptr, y_ptr, n_out_i, n_in_i, code),
            )
        }
    }

    fn embed(
        &self,
        table: &QWeight<Self::Buf>,
        token_ids: &[u32],
        out: &mut Self::Buf,
    ) -> Result<()> {
        let shape::Embed { n_embd, .. } = shape::embed(&table.shape, token_ids, out.len / 4)?;
        shape::kquant_aligned("embed", table.dtype, n_embd)?;
        let mut code = dtype_code::decoder_code(table.dtype)?;
        let mut table_ptr = table.buf.ptr;
        let mut out_ptr = out.ptr;
        let mut n_embd_i = int(n_embd)?;
        let mut bb = dtype_code::block_bytes(table.dtype);
        let grid = [(n_embd as u32).div_ceil(256), token_ids.len() as u32, 1];
        if let Some(replay) = &self.replay {
            // The token is params[0]; the recorded id is only a placeholder.
            if token_ids.len() != 1 {
                return Err(Error::ShapeMismatch(format!(
                    "embed: graph replay takes 1 token, got {}",
                    token_ids.len()
                )));
            }
            let mut ids_ptr = replay.params.ptr;
            return self.launch(
                "embed_rows",
                grid,
                [256, 1, 1],
                &mut params!(table_ptr, ids_ptr, out_ptr, n_embd_i, code, bb),
            );
        }
        let mut ids = self.alloc(token_ids.len() * 4, Pool::Scratch)?;
        self.upload(&mut ids, 0, bytemuck::cast_slice(token_ids))?;
        let mut ids_ptr = ids.ptr;
        self.launch(
            "embed_rows",
            grid,
            [256, 1, 1],
            &mut params!(table_ptr, ids_ptr, out_ptr, n_embd_i, code, bb),
        )?;
        // The id buffer is freed on return; the kernel must be done with it.
        self.sync()
    }

    fn rmsnorm(&self, w: &Self::Buf, x: &Self::Buf, y: &mut Self::Buf, eps: f32) -> Result<()> {
        let dim = w.len / 4;
        let n_tokens = shape::rows("rmsnorm", dim, x.len / 4, y.len / 4)? as u32;
        let mut w_ptr = w.ptr;
        let mut x_ptr = x.ptr;
        let mut y_ptr = y.ptr;
        let mut dim_i = int(dim)?;
        let mut eps = eps;
        self.launch(
            "rmsnorm_f32",
            [n_tokens, 1, 1],
            [256, 1, 1],
            &mut params!(w_ptr, x_ptr, y_ptr, dim_i, eps),
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
        self.eager_only("rope")?;
        let (mut freqs, mut n_q, mut n_kv) = rope_setup(q, k, n_tokens, spec)?;
        let pairs = spec.head_dim / 2;
        let mut q_ptr = q.ptr;
        let mut k_ptr = k.ptr;
        let mut pos_i = int(pos)?;
        let mut head_dim = int(spec.head_dim)?;
        let grid = [1, (n_q + n_kv).cast_unsigned(), n_tokens as u32];
        self.launch(
            "rope_f32",
            grid,
            [pairs as u32, 1, 1],
            &mut params!(q_ptr, k_ptr, pos_i, head_dim, n_q, n_kv, freqs),
        )
    }

    fn add(&self, a: &Self::Buf, b: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        self.launch_binary_f32("add", "add_f32", a, b, y)
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
        let dim = w.len / 4;
        shape::elementwise("add_rmsnorm", &[a.len, b.len], &[sum.len, y.len])?;
        let n_tokens = shape::rows("add_rmsnorm", dim, a.len / 4, y.len / 4)? as u32;
        let (mut w_ptr, mut a_ptr, mut b_ptr) = (w.ptr, a.ptr, b.ptr);
        let (mut sum_ptr, mut y_ptr) = (sum.ptr, y.ptr);
        let mut dim_i = int(dim)?;
        let mut eps = eps;
        self.launch(
            "add_rmsnorm_f32",
            [n_tokens, 1, 1],
            [256, 1, 1],
            &mut params!(w_ptr, a_ptr, b_ptr, sum_ptr, y_ptr, dim_i, eps),
        )
    }

    /// The KV cache is f16: half the bytes per key for every attention
    /// read. Rows are rounded once on the way in by kv_write_f16.
    fn kv_elem_bytes(&self) -> usize {
        2
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
        let src_bytes = n_elems * 4;
        let dst_bytes = (elem_offset + n_elems) * 2;
        if src_bytes > k.len || src_bytes > v.len || dst_bytes > dst_k.len || dst_bytes > dst_v.len
        {
            return Err(Error::ShapeMismatch(format!(
                "kv_write: {n_elems} elems at {elem_offset} exceed sources {}/{} or caches {}/{}",
                k.len, v.len, dst_k.len, dst_v.len
            )));
        }
        if n_elems == 0 {
            return Ok(());
        }
        let (mut k_ptr, mut v_ptr) = (k.ptr, v.ptr);
        let (mut dk, mut dv) = (dst_k.ptr, dst_v.ptr);
        let mut n = int(n_elems)?;
        let mut offset = long(elem_offset)?;
        let mut pp = self.replay.as_ref().map_or(0, |replay| replay.params.ptr);
        self.launch(
            "kv_write_f16",
            [(n_elems as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
            &mut params!(k_ptr, v_ptr, dk, dv, n, offset, pp),
        )
    }

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
        let (mut freqs, mut n_q, mut n_kv) = rope_setup(q, k, n_tokens, spec)?;
        // Replay writes wherever the params' pos points: one token's row
        // anywhere below the capacity, at the offset kv_cache derives from pos.
        let (mut pp, dst_bytes) = if let Some(replay) = &self.replay {
            if n_tokens != 1 || elem_offset != pos as usize * n_elems {
                return Err(Error::ShapeMismatch(format!(
                    "rope_kv_write: graph replay takes 1 token at pos * row, got {n_tokens} at {elem_offset}"
                )));
            }
            (replay.params.ptr, replay.capacity * n_elems * 2)
        } else {
            (0u64, (elem_offset + n_elems) * 2)
        };
        if n_elems != k.len / 4 || v.len != k.len || dst_bytes > dst_k.len || dst_bytes > dst_v.len
        {
            return Err(Error::ShapeMismatch(format!(
                "rope_kv_write: {n_elems} elems at {elem_offset}, k {} v {} bytes, caches {}/{}",
                k.len, v.len, dst_k.len, dst_v.len
            )));
        }
        let (mut q_ptr, mut k_ptr, mut v_ptr) = (q.ptr, k.ptr, v.ptr);
        let (mut dk, mut dv) = (dst_k.ptr, dst_v.ptr);
        let mut pos_i = int(pos)?;
        let mut head_dim = int(spec.head_dim)?;
        let mut offset = long(elem_offset)?;
        self.launch(
            "rope_kv_f16",
            [1, (n_q + n_kv).cast_unsigned(), n_tokens as u32],
            [(spec.head_dim / 2) as u32, 1, 1],
            &mut params!(
                q_ptr, k_ptr, v_ptr, dk, dv, pos_i, head_dim, n_q, n_kv, offset, freqs, pp
            ),
        )
    }

    fn attn(
        &self,
        q: &Self::Buf,
        kv: &KvView<Self::Buf>,
        spec: &AttnSpec,
        o: &mut Self::Buf,
    ) -> Result<()> {
        let shape::Attn { n_q, prefix } = shape::attn(
            spec,
            q.len / 4,
            kv.len,
            kv.k.len / 2,
            kv.v.len / 2,
            o.len / 4,
        )?;
        if spec.head_dim > MAX_HEAD_DIM {
            // Eager only: graph capture refuses heads this wide.
            return self.attention_windowed(q, kv, spec, 1.0 / (spec.head_dim as f32).sqrt(), 0, o);
        }
        if !spec.head_dim.is_multiple_of(2) {
            return Err(Error::ShapeMismatch(format!(
                "attn: head_dim {} is odd (f16 pairs)",
                spec.head_dim
            )));
        }
        // Replay reads up to params' pos + 1 keys: the grid and the bounds
        // cover the capacity.
        let (keys, mut pp) = match &self.replay {
            Some(replay) if n_q == 1 => (replay.capacity, replay.params.ptr),
            Some(_) => {
                return Err(Error::ShapeMismatch(format!(
                    "attn: graph replay takes 1 query, got {n_q}"
                )));
            }
            None => (kv.len, 0),
        };
        let kv_bytes = keys * spec.n_kv_heads * spec.head_dim * 2;
        if kv.k.len < kv_bytes || kv.v.len < kv_bytes {
            return Err(Error::ShapeMismatch(format!(
                "attn: replay capacity {keys} needs {kv_bytes} bytes, cache has {}/{}",
                kv.k.len, kv.v.len
            )));
        }
        let mut prefix = int(prefix)?;
        let mut q_ptr = q.ptr;
        let mut k_ptr = kv.k.ptr;
        let mut v_ptr = kv.v.ptr;
        let mut o_ptr = o.ptr;
        let mut head_dim = int(spec.head_dim)?;
        let mut n_q_heads = int(spec.n_q_heads)?;
        let mut n_kv_heads = int(spec.n_kv_heads)?;
        // Query heads per block: the largest G the kernel is built for that
        // divides the GQA group, so a block reads its kv head's rows once.
        let group = spec.n_q_heads / spec.n_kv_heads;
        let mut group_i = int(group)?;
        let (g, chunk, splits) = self.attn_layout(spec, n_q, keys);
        let mut chunk_i = int(chunk)?;
        // Held across both enqueues, like the Q8_1 scratch.
        let mut scratch = self
            .attn_scratch
            .lock()
            .map_err(|_| Error::Backend("cuda: attention scratch lock poisoned".into()))?;
        let mut part_ptr = if splits > 1 {
            let bytes = Self::attn_partials_bytes(spec, n_q, splits);
            if scratch.as_ref().is_none_or(|b| b.len < bytes) {
                // Queued merges may still read the old partials.
                if scratch.is_some() {
                    self.sync()?;
                }
                *scratch = None;
                *scratch = Some(self.alloc(bytes, Pool::Scratch)?);
            }
            scratch.as_ref().expect("allocated above").ptr
        } else {
            0u64
        };
        self.launch(
            [
                "attn_f16kv_g1",
                "attn_f16kv_g2",
                "attn_f16kv_g4",
                "attn_f16kv_g8",
            ][g.trailing_zeros() as usize],
            [(spec.n_q_heads / g) as u32, n_q as u32, splits as u32],
            [32, ATTN_WARPS, 1],
            &mut params!(
                q_ptr, k_ptr, v_ptr, o_ptr, part_ptr, head_dim, n_q_heads, n_kv_heads, group_i,
                prefix, chunk_i, pp
            ),
        )?;
        if splits > 1 {
            let mut splits_i = int(splits)?;
            self.launch(
                "attn_merge",
                [spec.n_q_heads as u32, n_q as u32, 1],
                [(spec.head_dim as u32).div_ceil(32) * 32, 1, 1],
                &mut params!(part_ptr, o_ptr, head_dim, n_q_heads, splits_i),
            )?;
        }
        Ok(())
    }

    fn silu_mul(&self, gate: &Self::Buf, up: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        self.launch_binary_f32("silu_mul", "silu_mul_f32", gate, up, y)
    }
}

mod decoder_ops;
mod graph;

#[cfg(test)]
mod tests;
