// CUDA graph decode: one T=1 forward is captured once per decoder on a
// backend twin (own non-blocking stream, own scratch, token and position
// read from a device params buffer) and replayed per token, so the
// ~200 per-token launches cost one cuGraphLaunch. The ROCm backend's
// graph.rs is the same design.

use super::{
    CuResult, CuStream, CudaBackend, CudaBuf, Error, MAX_HEAD_DIM, Q8Scratch, Result, cu,
    cuStreamSynchronize,
};
use crate::backend::{AttnSpec, Backend, Pool};
use crate::kv_cache::KvCache;
use crate::model::decoder::{Decoder, EagerDecoder};
use crate::model::{Model, Workspace};
use std::ffi::{c_int, c_void};

type CuGraph = *mut c_void;
type CuGraphExec = *mut c_void;

/// Only this thread's capture-unsafe calls fail; other sessions keep
/// running eager ops on the legacy stream meanwhile.
const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;

// SAFETY: prototypes match cuda.h for libcuda's exports (linked by build.rs):
// CUresult and CUstreamCaptureMode are C ints, CUgraph, CUgraphExec and
// CUstream opaque pointers, the instantiate flags `unsigned long long`,
// CUdeviceptr u64 and size_t usize. The _v2 names are the entry points the
// plain names map to, and all are the legacy-stream (non-_ptsz) variants, as
// in cuda.rs.
unsafe extern "C" {
    fn cuStreamBeginCapture_v2(stream: CuStream, mode: c_int) -> CuResult;
    fn cuStreamEndCapture(stream: CuStream, graph: *mut CuGraph) -> CuResult;
    fn cuGraphInstantiateWithFlags(exec: *mut CuGraphExec, graph: CuGraph, flags: u64) -> CuResult;
    fn cuGraphLaunch(exec: CuGraphExec, stream: CuStream) -> CuResult;
    fn cuGraphDestroy(graph: CuGraph) -> CuResult;
    fn cuGraphExecDestroy(exec: CuGraphExec) -> CuResult;
    fn cuMemcpyHtoDAsync_v2(
        dst: u64,
        src: *const c_void,
        bytes: usize,
        stream: CuStream,
    ) -> CuResult;
}

/// A graph twin's (token, pos) and the KV capacity its grid covers.
pub(super) struct Replay {
    pub params: CudaBuf,
    pub capacity: usize,
}

struct Graph(CuGraph);

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: a non-null handle came from a successful
            // cuStreamEndCapture and is owned only by this Graph (not Clone),
            // so it is destroyed exactly once; an executable instantiated
            // from it is an independent copy and outlives it safely.
            unsafe { cuGraphDestroy(self.0) };
        }
    }
}

struct Executable(CuGraphExec);

impl Drop for Executable {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: a non-null handle came from cuGraphInstantiateWithFlags
            // and is owned only by this Executable, so it is destroyed once;
            // the driver defers the free if a launch is still in flight.
            // GraphState drops it before the twin holding context and module.
            unsafe { cuGraphExecDestroy(self.0) };
        }
    }
}

/// Ends a capture that errored midway, so the stream is usable (and
/// destroyable) again.
struct Capture {
    stream: CuStream,
    active: bool,
}

impl Capture {
    fn begin(stream: CuStream) -> Result<Self> {
        // SAFETY: the only caller (GraphState::capture) passes the graph
        // twin's own non-blocking stream, never the legacy null stream, and
        // the twin outlives this Capture: it is declared first there, so it
        // drops last. `begin` itself trusts that the handle is live.
        cu("cuStreamBeginCapture", unsafe {
            cuStreamBeginCapture_v2(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        })?;
        Ok(Self {
            stream,
            active: true,
        })
    }

    fn finish(mut self) -> Result<Graph> {
        let mut graph = Graph(std::ptr::null_mut());
        self.active = false;
        // SAFETY: `self.stream` is the live stream this Capture began on, and
        // this is the thread that began it (Capture holds a raw pointer, so it
        // is !Send), as THREAD_LOCAL mode requires. `graph.0` is a live
        // out-slot left null on failure, so Graph::drop skips it; `active` is
        // cleared first so Drop never ends the capture a second time.
        cu("cuStreamEndCapture", unsafe {
            cuStreamEndCapture(self.stream, &raw mut graph.0)
        })?;
        Ok(graph)
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if self.active {
            let mut graph = Graph(std::ptr::null_mut());
            // SAFETY: as in `finish`: the same live stream, on the thread that
            // began the capture (Capture is !Send), with a live out-slot;
            // whatever graph comes back is destroyed by `graph`'s Drop.
            unsafe { cuStreamEndCapture(self.stream, &raw mut graph.0) };
        }
    }
}

// Private: a captured state must immediately be wrapped in CudaDecoder's
// borrows, which pin the KV and workspace buffers the graph recorded.
struct GraphState {
    // Destroyed before the twin's stream.
    executable: Executable,
    backend: CudaBackend,
}

impl GraphState {
    fn capture(
        eager: &CudaBackend,
        model: &Model<CudaBackend>,
        kv: &mut KvCache<CudaBackend>,
        ws: &mut Workspace<CudaBackend>,
    ) -> Result<Self> {
        // Capture from the current prefix.  This matters for recurrent
        // decoders: position zero is no longer resumable after prefill, and
        // the captured state must start from the already-processed prefix.
        let capture_pos = kv.len() as u32;
        model.validate_decode(kv, ws, 0, capture_pos)?;
        let config = model.config();
        let backend = eager.graph_twin(kv.capacity())?;
        // Capture cannot allocate: reserve the widest decode's scratch.
        let spec = AttnSpec {
            head_dim: config.head_dim,
            n_q_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
        };
        if spec.n_kv_heads == 0 || !spec.n_q_heads.is_multiple_of(spec.n_kv_heads) {
            return Err(Error::ShapeMismatch(format!(
                "graph: {} q heads over {} kv heads",
                spec.n_q_heads, spec.n_kv_heads
            )));
        }
        let (_, _, splits) = backend.attn_layout(&spec, 1, kv.capacity());
        if splits > 1 {
            let bytes = CudaBackend::attn_partials_bytes(&spec, 1, splits);
            *backend.attn_scratch.lock().expect("fresh twin") =
                Some(backend.alloc(bytes, Pool::Scratch)?);
        }
        if backend.dp4a {
            let width = config
                .n_embd
                .max(config.n_ff)
                .max(config.n_heads * config.head_dim);
            *backend.q8_scratch.lock().expect("fresh twin") = Some(Q8Scratch {
                q8: backend.alloc(width, Pool::Scratch)?,
                ys: backend.alloc(width / 32 * 8, Pool::Scratch)?,
                holds: None,
            });
        }
        // Prefill and the allocations above ran on the legacy stream, which
        // the twin's stream does not wait for. Only that stream: synchronizing
        // the context would invalidate another thread's capture.
        backend.bind()?;
        // SAFETY: `bind` just made the context current on this thread; the
        // legacy stream is never captured, since twins capture their own
        // non-blocking streams.
        cu("cuStreamSynchronize", unsafe {
            cuStreamSynchronize(std::ptr::null_mut())
        })?;
        let saved_len = kv.len();
        let capture = Capture::begin(backend.stream)?;
        let recorded = model.forward_device(&backend, kv, ws, 0, capture_pos);
        let captured = capture.finish();
        kv.set_len(saved_len);
        recorded?;
        let graph = captured?;
        let mut executable = Executable(std::ptr::null_mut());
        // SAFETY: `graph.0` is the non-null graph of a successful capture
        // (failures returned above), live to the end of this fn; the context
        // is current on this thread (bound above, rebound by each captured
        // op). The executable bakes in pointers into the model, `kv`, `ws`
        // and the twin's scratch/params, which CudaDecoder's borrows and
        // GraphState's ownership keep alive as long as it exists.
        cu("cuGraphInstantiate", unsafe {
            cuGraphInstantiateWithFlags(&raw mut executable.0, graph.0, 0)
        })?;
        Ok(Self {
            executable,
            backend,
        })
    }
}

struct CudaDecoder<'a> {
    state: GraphState,
    model: &'a Model<CudaBackend>,
    kv: &'a mut KvCache<CudaBackend>,
    ws: &'a mut Workspace<CudaBackend>,
}

impl Decoder for CudaDecoder<'_> {
    fn forward(&mut self, token: u32, pos: u32) -> Result<Vec<f32>> {
        self.model.validate_decode(self.kv, self.ws, token, pos)?;
        let backend = &self.state.backend;
        let replay = backend.replay.as_ref().expect("graph twin");
        let params = [token, pos];
        backend.bind()?;
        // Pageable source: the call returns once params is staged.
        // SAFETY: `replay.params` is the twin's 8-byte buffer (graph_twin
        // allocates 8) and `params` is exactly 8 bytes; from pageable memory
        // the driver copies the source to staging before returning, so the
        // stack array may die afterwards. The context is bound above and
        // `backend.stream` is the twin's live stream.
        cu("cuMemcpyHtoDAsync", unsafe {
            cuMemcpyHtoDAsync_v2(
                replay.params.ptr,
                params.as_ptr().cast(),
                size_of_val(&params),
                backend.stream,
            )
        })?;
        // download syncs the twin's stream first; on an error the twin's
        // Drop does.
        // SAFETY: the executable is live in `self.state`, instantiated in this
        // context, launched on the twin stream it was captured from. Its baked
        // pointers stay valid: model (`&'a`), kv and ws (`&'a mut`: nothing
        // else can free or move them), twin scratch/params (`state.backend`).
        // validate_decode above bounds the token (< vocab rows) and pos
        // (< KV capacity) the replayed kernels read from params.
        cu("cuGraphLaunch", unsafe {
            cuGraphLaunch(self.state.executable.0, backend.stream)
        })?;
        let logits = self.model.download_logits(backend, self.ws)?;
        self.kv.set_len(pos as usize + 1);
        Ok(logits)
    }

    fn is_graph(&self) -> bool {
        true
    }
}

/// A graph decoder unless MDMA_CUDA_GRAPHS=0 at startup, heads are wider than
/// attn_f16kv takes (Gemma's 512-wide global heads run eagerly on the
/// windowed kernel), or the capture fails, which falls back to eager decode
/// with the reason on stderr.
#[expect(
    clippy::unnecessary_wraps,
    reason = "delegate_decoder! calls fallible factories"
)]
pub(super) fn decoder<'a>(
    backend: &'a CudaBackend,
    model: &'a Model<CudaBackend>,
    kv: &'a mut KvCache<CudaBackend>,
    ws: &'a mut Workspace<CudaBackend>,
) -> Result<Box<dyn Decoder + 'a>> {
    if backend.graphs && model.config().head_dim <= MAX_HEAD_DIM {
        match GraphState::capture(backend, model, kv, ws) {
            Ok(state) => {
                return Ok(Box::new(CudaDecoder {
                    state,
                    model,
                    kv,
                    ws,
                }));
            }
            Err(error) => {
                eprintln!("mdma: CUDA graph capture unavailable, using eager decode: {error}");
            }
        }
    }
    Ok(Box::new(EagerDecoder {
        backend,
        model,
        kv,
        ws,
    }))
}

#[cfg(test)]
mod tests;
