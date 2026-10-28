//! Dense Qwen2/Qwen3 and the text-only Qwen2-VL decoder.
use super::Decoder;
use crate::backend::{AttnSpec, Backend, Pool, QWeight, Result, RopeSpec};
use crate::gguf::GgufModel;
use crate::kv_cache::KvCache;
use crate::memory::product;
use crate::model::{Model, ModelConfig, load_norm, load_output, load_weight};

pub(in crate::model) mod hf;
mod preflight;
mod yarn;
pub(in crate::model) use preflight::config as preflight_config;
#[cfg(test)]
pub(crate) mod tests;

pub(in crate::model) fn is_arch(arch: &str) -> bool {
    matches!(arch, "qwen2" | "qwen2vl" | "qwen3")
}

struct Layer<B: Backend> {
    attn_norm: B::Buf,
    q: QWeight<B::Buf>,
    k: QWeight<B::Buf>,
    v: QWeight<B::Buf>,
    out: QWeight<B::Buf>,
    biases: Option<[B::Buf; 3]>,
    head_norms: Option<[B::Buf; 2]>,
    ffn_norm: B::Buf,
    gate: QWeight<B::Buf>,
    up: QWeight<B::Buf>,
    down: QWeight<B::Buf>,
}

pub(in crate::model) struct Qwen<B: Backend> {
    layers: Vec<Layer<B>>,
    factors: B::Buf,
    rope_amplitude: f32,
}

pub(in crate::model) struct Workspace<B: Backend> {
    kn: B::Buf,
}

impl<B: Backend> Qwen<B> {
    pub fn load(b: &B, g: &mut GgufModel, c: ModelConfig) -> Result<Model<B>> {
        let arch = g
            .metadata_str("general.architecture")
            .expect("validated architecture");
        let qwen3 = arch == "qwen3";
        let scaling = yarn::Yarn::read(g, arch, &c)?;
        let token_embd = load_weight(b, g, "token_embd.weight")?;
        let output_norm = load_norm(b, g, "output_norm.weight")?;
        let output = load_output(b, g)?;
        let divisors = scaling
            .as_ref()
            .map_or_else(|| vec![1f32; c.head_dim / 2], |y| y.divisors(&c));
        let rope_amplitude = scaling.as_ref().map_or(1.0, yarn::Yarn::amplitude);
        let mut factors = b.alloc(product(&[c.head_dim / 2, 4])?, Pool::Weights)?;
        b.upload(&mut factors, 0, bytemuck::cast_slice(&divisors))?;
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            let name = |s: &str| format!("blk.{i}.{s}.weight");
            layers.push(Layer {
                attn_norm: load_norm(b, g, &name("attn_norm"))?,
                q: load_weight(b, g, &name("attn_q"))?,
                k: load_weight(b, g, &name("attn_k"))?,
                v: load_weight(b, g, &name("attn_v"))?,
                out: load_weight(b, g, &name("attn_output"))?,
                biases: if qwen3 {
                    None
                } else {
                    Some([
                        load_norm(b, g, &format!("blk.{i}.attn_q.bias"))?,
                        load_norm(b, g, &format!("blk.{i}.attn_k.bias"))?,
                        load_norm(b, g, &format!("blk.{i}.attn_v.bias"))?,
                    ])
                },
                head_norms: if qwen3 {
                    Some([
                        load_norm(b, g, &name("attn_q_norm"))?,
                        load_norm(b, g, &name("attn_k_norm"))?,
                    ])
                } else {
                    None
                },
                ffn_norm: load_norm(b, g, &name("ffn_norm"))?,
                gate: load_weight(b, g, &name("ffn_gate"))?,
                up: load_weight(b, g, &name("ffn_up"))?,
                down: load_weight(b, g, &name("ffn_down"))?,
            });
        }
        Ok(Model {
            config: c,
            token_embd,
            output_norm,
            output,
            architecture: Decoder::Qwen(Self {
                layers,
                factors,
                rope_amplitude,
            }),
            bos: None,
            memory: None,
        })
    }

    pub fn workspace_bytes(c: &ModelConfig) -> Result<usize> {
        product(&[c.n_kv_heads, c.head_dim, 4])
    }

    #[expect(
        clippy::unused_self,
        reason = "architectures share one workspace signature"
    )]
    pub fn workspace(&self, b: &B, c: &ModelConfig) -> Result<Workspace<B>> {
        Ok(Workspace {
            kn: b.alloc(Self::workspace_bytes(c)?, Pool::Scratch)?,
        })
    }

    pub fn run(
        &self,
        b: &B,
        model: &Model<B>,
        kv: &mut KvCache<B>,
        ws: &mut crate::model::Workspace<B>,
        token: u32,
        pos: u32,
    ) -> Result<()> {
        b.embed(&model.token_embd, &[token], &mut ws.x)?;
        self.run_embedded(b, model, kv, ws, pos)
    }

    // The caller fills ws.x with one embedding row, including continuous inputs.
    pub(in crate::model) fn run_embedded(
        &self,
        b: &B,
        model: &Model<B>,
        kv: &mut KvCache<B>,
        ws: &mut crate::model::Workspace<B>,
        pos: u32,
    ) -> Result<()> {
        let c = &model.config;
        let extra = ws.qwen.as_mut().expect("Qwen workspace was allocated");
        let rope = RopeSpec {
            head_dim: c.head_dim,
            theta: c.rope_theta,
        };
        let attn = AttnSpec {
            head_dim: c.head_dim,
            n_q_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
        };
        #[cfg(test)]
        tests::trace(b, &ws.x, c.n_embd, "inp_embd", 0, pos)?;
        for (i, l) in self.layers.iter().enumerate() {
            b.rmsnorm(&l.attn_norm, &ws.x, &mut ws.normed, c.rms_eps)?;
            #[cfg(test)]
            if i <= 1 {
                tests::trace(b, &ws.normed, c.n_embd, "attn_norm", i, pos)?;
            }
            b.qmatmul_qkv(
                &l.q, &l.k, &l.v, &ws.normed, &mut ws.q, &mut ws.k, &mut ws.v,
            )?;
            #[cfg(test)]
            if i <= 1 {
                tests::trace(b, &ws.q, c.n_heads * c.head_dim, "Qcur_raw", i, pos)?;
                tests::trace(b, &ws.k, c.n_kv_heads * c.head_dim, "Kcur_raw", i, pos)?;
                tests::trace(b, &ws.v, c.n_kv_heads * c.head_dim, "Vcur", i, pos)?;
            }
            if let Some([q, k, v]) = &l.biases {
                b.add_bias(q, &mut ws.q)?;
                b.add_bias(k, &mut ws.k)?;
                b.add_bias(v, &mut ws.v)?;
            }
            if let Some([q, k]) = &l.head_norms {
                b.rmsnorm(q, &ws.q, &mut ws.attn_out, c.rms_eps)?;
                std::mem::swap(&mut ws.q, &mut ws.attn_out);
                b.rmsnorm(k, &ws.k, &mut extra.kn, c.rms_eps)?;
                std::mem::swap(&mut ws.k, &mut extra.kn);
            }
            b.rope_split_half(&mut ws.q, pos, &rope, &self.factors)?;
            b.rope_split_half(&mut ws.k, pos, &rope, &self.factors)?;
            if self.rope_amplitude != 1.0 {
                b.scale(&mut ws.q, self.rope_amplitude)?;
                b.scale(&mut ws.k, self.rope_amplitude)?;
            }
            #[cfg(test)]
            if i <= 1 {
                tests::trace(b, &ws.q, c.n_heads * c.head_dim, "Qcur", i, pos)?;
                tests::trace(b, &ws.k, c.n_kv_heads * c.head_dim, "Kcur", i, pos)?;
            }
            kv.write(b, i, pos as usize, &ws.k, &ws.v, 1)?;
            kv.set_len(pos as usize + 1);
            b.attn(&ws.q, &kv.view(i), &attn, &mut ws.attn_out)?;
            b.qmatmul(&l.out, &ws.attn_out, &mut ws.proj)?;
            #[cfg(test)]
            if i <= 1 {
                tests::trace(b, &ws.proj, c.n_embd, "attn_out", i, pos)?;
            }
            b.add_rmsnorm(
                &l.ffn_norm,
                &ws.x,
                &ws.proj,
                &mut ws.x_alt,
                &mut ws.normed,
                c.rms_eps,
            )?;
            std::mem::swap(&mut ws.x, &mut ws.x_alt);
            #[cfg(test)]
            if i <= 1 {
                tests::trace(b, &ws.x, c.n_embd, "ffn_inp", i, pos)?;
                tests::trace(b, &ws.normed, c.n_embd, "ffn_norm", i, pos)?;
            }
            b.ffn_gate_up(
                &l.gate,
                &l.up,
                &ws.normed,
                &mut ws.gate,
                &mut ws.up,
                &mut ws.act,
            )?;
            #[cfg(test)]
            if i <= 1 {
                tests::trace(b, &ws.act, c.n_ff, "ffn_swiglu", i, pos)?;
            }
            b.qmatmul(&l.down, &ws.act, &mut ws.proj)?;
            #[cfg(test)]
            if i <= 1 {
                tests::trace(b, &ws.proj, c.n_embd, "ffn_out", i, pos)?;
            }
            b.add(&ws.x, &ws.proj, &mut ws.x_alt)?;
            std::mem::swap(&mut ws.x, &mut ws.x_alt);
            #[cfg(test)]
            tests::trace(b, &ws.x, c.n_embd, "l_out", i, pos)?;
        }
        b.rmsnorm(&model.output_norm, &ws.x, &mut ws.normed, c.rms_eps)?;
        #[cfg(test)]
        tests::trace(b, &ws.normed, c.n_embd, "result_norm", 0, pos)?;
        Ok(())
    }
}
