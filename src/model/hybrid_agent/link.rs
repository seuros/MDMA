use super::{Backend, Result, weights::Weights};
use crate::{backend::{Pool, QWeight}, gguf::GgmlDType, memory::product};

pub(super) struct HybridLink<B: Backend> {
    pub width: usize,
    proj1: QWeight<B::Buf>,
    bias1: B::Buf,
    proj2: QWeight<B::Buf>,
    bias2: B::Buf,
    pre: [B::Buf; 2],
    post: [B::Buf; 2],
}

pub(super) struct Workspace<B: Backend> {
    normalized: B::Buf,
    projected: B::Buf,
    activated: B::Buf,
}

impl<B: Backend> HybridLink<B> {
    pub fn upload(b: &B, weights: Weights) -> Result<Self> {
        weights.validate()?;
        let upload = |name: &str| {
            let values = &weights.tensors[name];
            let mut buf = b.alloc(product(&[values.len(), 4])?, Pool::Weights)?;
            b.upload(&mut buf, 0, bytemuck::cast_slice(values))?;
            Ok::<_, crate::backend::Error>(buf)
        };
        let matrix = |name| Ok::<_, crate::backend::Error>(QWeight {
            dtype: GgmlDType::F32, shape: vec![weights.width, weights.width], buf: upload(name)?,
        });
        Ok(Self {
            width: weights.width,
            proj1: matrix("proj1.weight")?, bias1: upload("proj1.bias")?,
            proj2: matrix("proj2.weight")?, bias2: upload("proj2.bias")?,
            pre: [upload("pre_ln.weight")?, upload("pre_ln.bias")?],
            post: [upload("post_ln.weight")?, upload("post_ln.bias")?],
        })
    }

    pub fn workspace(&self, b: &B) -> Result<Workspace<B>> {
        let bytes = product(&[self.width, 4])?;
        Ok(Workspace {
            normalized: b.alloc(bytes, Pool::Scratch)?,
            projected: b.alloc(bytes, Pool::Scratch)?,
            activated: b.alloc(bytes, Pool::Scratch)?,
        })
    }

    pub fn apply(&self, b: &B, hidden: &B::Buf, ws: &mut Workspace<B>, output: &mut B::Buf) -> Result<()> {
        b.layernorm(&self.pre[0], Some(&self.pre[1]), hidden, &mut ws.normalized, 1e-5)?;
        b.qmatmul(&self.proj1, &ws.normalized, &mut ws.projected)?;
        b.add_bias(&self.bias1, &mut ws.projected)?;
        b.gelu(&ws.projected, &mut ws.activated)?;
        b.qmatmul(&self.proj2, &ws.activated, &mut ws.normalized)?;
        b.add_bias(&self.bias2, &mut ws.normalized)?;
        b.add(hidden, &ws.normalized, &mut ws.projected)?;
        b.layernorm(&self.post[0], Some(&self.post[1]), &ws.projected, output, 1e-5)
    }
}
