use crate::backend::{AttnSpec, Backend, Error, KvView, Operation, Pool, QWeight, Result, RopeSpec};
use crate::backend::cpu::{CpuBackend, CpuBuf};
use std::cell::{Cell, RefCell};

#[derive(Default)]
pub(super) struct Trace {
    pub allocations: Cell<usize>,
    pub uploads: Cell<usize>,
    pub downloads: RefCell<Vec<usize>>,
    pub fail_attention: Cell<bool>,
}

impl Backend for Trace {
    type Buf = CpuBuf;
    fn supports(&self, op: Operation) -> bool { CpuBackend.supports(op) }
    fn alloc(&self, bytes: usize, pool: Pool) -> Result<CpuBuf> {
        self.allocations.set(self.allocations.get() + 1);
        CpuBackend.alloc(bytes, pool)
    }
    fn upload(&self, dst: &mut CpuBuf, offset: usize, src: &[u8]) -> Result<()> {
        self.uploads.set(self.uploads.get() + 1);
        CpuBackend.upload(dst, offset, src)
    }
    fn download(&self, src: &CpuBuf, offset: usize, dst: &mut [u8]) -> Result<()> {
        self.downloads.borrow_mut().push(dst.len());
        CpuBackend.download(src, offset, dst)
    }
    fn copy(&self, src: &CpuBuf, so: usize, dst: &mut CpuBuf, doff: usize, n: usize) -> Result<()> {
        CpuBackend.copy(src, so, dst, doff, n)
    }
    fn qmatmul(&self, w: &QWeight<CpuBuf>, x: &CpuBuf, y: &mut CpuBuf) -> Result<()> {
        CpuBackend.qmatmul(w, x, y)
    }
    fn embed(&self, w: &QWeight<CpuBuf>, ids: &[u32], y: &mut CpuBuf) -> Result<()> {
        CpuBackend.embed(w, ids, y)
    }
    fn rmsnorm(&self, w: &CpuBuf, x: &CpuBuf, y: &mut CpuBuf, eps: f32) -> Result<()> {
        CpuBackend.rmsnorm(w, x, y, eps)
    }
    fn add(&self, a: &CpuBuf, b: &CpuBuf, y: &mut CpuBuf) -> Result<()> {
        CpuBackend.add(a, b, y)
    }
    fn rope(&self, q: &mut CpuBuf, k: &mut CpuBuf, pos: u32, n: usize, spec: &RopeSpec) -> Result<()> {
        CpuBackend.rope(q, k, pos, n, spec)
    }
    fn rope_split_half(&self, x: &mut CpuBuf, pos: u32, spec: &RopeSpec, factors: &CpuBuf) -> Result<()> {
        CpuBackend.rope_split_half(x, pos, spec, factors)
    }
    fn attn(&self, q: &CpuBuf, kv: &KvView<CpuBuf>, spec: &AttnSpec, y: &mut CpuBuf) -> Result<()> {
        if self.fail_attention.get() { return Err(Error::Backend("injected attention failure".into())); }
        CpuBackend.attn(q, kv, spec, y)
    }
    fn silu_mul(&self, a: &CpuBuf, b: &CpuBuf, y: &mut CpuBuf) -> Result<()> {
        CpuBackend.silu_mul(a, b, y)
    }
    fn layernorm(&self, w: &CpuBuf, bias: Option<&CpuBuf>, x: &CpuBuf, y: &mut CpuBuf, eps: f32) -> Result<()> {
        CpuBackend.layernorm(w, bias, x, y, eps)
    }
    fn add_bias(&self, bias: &CpuBuf, y: &mut CpuBuf) -> Result<()> {
        CpuBackend.add_bias(bias, y)
    }
    fn gelu(&self, x: &CpuBuf, y: &mut CpuBuf) -> Result<()> { CpuBackend.gelu(x, y) }
    fn argmax(&self, x: &CpuBuf, index: &mut CpuBuf) -> Result<()> { CpuBackend.argmax(x, index) }
}
