use super::*;
use crate::backend::{Pool, cpu::CpuBackend, test_support::{NoUpload, close}};
use crate::model::arch::qwen::tests::fixture;
use candle_core::{Device, Tensor};
use weights::{NAMES, Weights};

pub(super) fn seed_weights(width: usize) -> Weights {
    let tensors = NAMES.into_iter().enumerate().map(|(seed, name)| {
        let matrix = matches!(name, "proj1.weight" | "proj2.weight");
        let n = if matrix { width * width } else { width };
        let values = (0..n).map(|i| {
            if name.ends_with("ln.weight") { 0.7 + i as f32 * 0.01 }
            else { ((i + seed * 7) as f32 * 0.37).sin() * 0.2 }
        }).collect();
        (name.into(), values)
    }).collect();
    Weights { width, tensors }
}

pub(super) fn agent<B: Backend>(backend: &B) -> HybridAgent<B> {
    let file = fixture("qwen3", 12, 8).write();
    HybridAgent {
        model: Model::load(backend, &mut GgufModel::open(file.path()).unwrap()).unwrap(),
        link: link::HybridLink::upload(backend, seed_weights(12)).unwrap(),
        entry: 126, exit: 127,
    }
}

fn upload<B: Backend>(b: &B, values: &[f32]) -> B::Buf {
    let mut buf = b.alloc(values.len() * 4, Pool::Scratch).unwrap();
    b.upload(&mut buf, 0, bytemuck::cast_slice(values)).unwrap();
    buf
}

fn read<B: Backend>(b: &B, buf: &B::Buf, n: usize) -> Vec<f32> {
    let mut values = vec![0.0; n];
    b.download(buf, 0, bytemuck::cast_slice_mut(&mut values)).unwrap();
    values
}

// f64 scalar normalization/matmul and Candle's erf, independent of Backend ops.
fn reference(weights: &Weights, hidden: &[f32]) -> Vec<f32> {
    let get = |name: &str| weights.tensors[name].iter().map(|&x| x as f64).collect::<Vec<_>>();
    let norm = |x: &[f64], w: &[f64], bias: &[f64]| {
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
        x.iter().zip(w).zip(bias).map(|((x, w), b)| (x - mean) / (var + 1e-5).sqrt() * w + b).collect::<Vec<_>>()
    };
    let mat = |x: &[f64], name: &str, bias: &str| {
        get(name).chunks_exact(x.len()).zip(get(bias)).map(|(row, bias)| {
            row.iter().zip(x).map(|(a, b)| a * b).sum::<f64>() + bias
        }).collect::<Vec<_>>()
    };
    let h = hidden.iter().map(|&v| v as f64).collect::<Vec<_>>();
    let x = norm(&h, &get("pre_ln.weight"), &get("pre_ln.bias"));
    let x = mat(&x, "proj1.weight", "proj1.bias");
    let erf = Tensor::from_vec(x.iter().map(|v| v / std::f64::consts::SQRT_2).collect::<Vec<_>>(),
        x.len(), &Device::Cpu).unwrap().erf().unwrap().to_vec1::<f64>().unwrap();
    let x = x.iter().zip(erf).map(|(v, erf)| 0.5 * v * (1.0 + erf)).collect::<Vec<_>>();
    let x = mat(&x, "proj2.weight", "proj2.bias");
    let x = h.iter().zip(x).map(|(a, b)| a + b).collect::<Vec<_>>();
    norm(&x, &get("post_ln.weight"), &get("post_ln.bias")).into_iter().map(|v| v as f32).collect()
}

pub(crate) fn link_reference_parity<B: Backend>(b: &B) {
    for width in [1, 7, 31, 32, 33, 65] {
        let weights = seed_weights(width);
        let hidden = (0..width).map(|i| (i as f32 * 0.13).cos()).collect::<Vec<_>>();
        let want = reference(&weights, &hidden);
        let link = link::HybridLink::upload(b, weights).unwrap();
        let mut ws = link.workspace(b).unwrap();
        let mut output = b.alloc(width * 4, Pool::Scratch).unwrap();
        link.apply(b, &upload(b, &hidden), &mut ws, &mut output).unwrap();
        close(&read(b, &output, width), &want, 2e-5);
    }
}

#[test]
fn hybridlink_matches_independent_reference() { link_reference_parity(&CpuBackend); }

#[test]
fn malformed_adapter_rejects_before_upload() {
    let mut weights = seed_weights(12);
    weights.tensors.get_mut("proj2.bias").unwrap()[0] = f32::NAN;
    assert!(link::HybridLink::upload(&NoUpload, weights).is_err());
    let mut weights = seed_weights(12);
    weights.tensors.remove("post_ln.bias");
    assert!(link::HybridLink::upload(&NoUpload, weights).is_err());
    let mut weights = seed_weights(12);
    weights.tensors.get_mut("proj1.weight").unwrap().pop();
    assert!(link::HybridLink::upload(&NoUpload, weights).is_err());
}

#[test]
fn wrong_backbone_rejects_before_upload() {
    let file = fixture("qwen2", 12, 8).write();
    let mut g = GgufModel::open(file.path()).unwrap();
    assert!(HybridAgent::load(&NoUpload, &mut g, Path::new("/missing")).is_err());
}

#[test]
fn hf_bundle_never_bypasses_required_cache_policy() {
    use std::ffi::OsStr;
    assert!(raw_cache_policy(None).is_ok());
    for value in ["auto", "off"] { assert!(raw_cache_policy(Some(OsStr::new(value))).is_ok()); }
    for value in ["required", "", "bogus"] { assert!(raw_cache_policy(Some(OsStr::new(value))).is_err()); }
}

#[test]
fn cpu_argmax_ties_nonfinite_and_geometry_follow_the_device_contract() {
    let b = CpuBackend;
    let mut index = b.alloc(4, Pool::Scratch).unwrap();
    let mut raw = [0; 4];
    for (values, want) in [
        (vec![-5.0, -1.0, -1.0], 1), (vec![0.0, -0.0], 0),
        (vec![f32::NAN], u32::MAX), (vec![f32::INFINITY], u32::MAX),
        (vec![f32::NEG_INFINITY], u32::MAX),
    ] {
        b.argmax(&upload(&b, &values), &mut index).unwrap();
        b.download(&index, 0, &mut raw).unwrap();
        assert_eq!(u32::from_ne_bytes(raw), want);
    }
    assert!(b.argmax(&upload(&b, &[]), &mut index).is_err());
    let mut short = b.alloc(3, Pool::Scratch).unwrap();
    assert!(b.argmax(&upload(&b, &[1.0]), &mut short).is_err());
}

#[test]
fn explicit_embedding_matches_token_path_and_preserves_nonzero_kv_prefix() {
    let b = CpuBackend;
    let agent = agent(&b);
    let m = &agent.model;
    let mut a = m.new_kv_cache(&b, 8).unwrap();
    let mut c = m.new_kv_cache(&b, 8).unwrap();
    let mut wa = m.alloc_workspace(&b, 1).unwrap();
    let mut wc = m.alloc_workspace(&b, 1).unwrap();
    let Decoder::Qwen(qwen) = &m.architecture else { unreachable!() };
    for (pos, token) in [3, 5, 9, 13].into_iter().enumerate() {
        let want = m.forward(&b, &mut a, &mut wa, token, pos as u32).unwrap();
        b.embed(&m.token_embd, &[token], &mut wc.x).unwrap();
        qwen.run_embedded(&b, m, &mut c, &mut wc, pos as u32).unwrap();
        b.qmatmul(m.output_weight(), &wc.normed, &mut wc.logits).unwrap();
        close(&m.download_logits(&b, &wc).unwrap(), &want, 1e-6);
    }
    assert_eq!((a.len(), c.len()), (4, 4));
}

#[cfg(any(feature = "rocm", all(feature = "metal", target_os = "macos")))]
pub(crate) fn trajectory<B: Backend>(b: &B) -> Vec<f32> {
    let agent = agent(b);
    let mut d = agent.decoder(b, 32, LatentLimits { min_steps: 3, max_steps: 3 }).unwrap();
    d.feed_text(&[1, 3, 7]).unwrap();
    for _turn in 0..2 {
        d.token(agent.entry).unwrap();
        for steps in 1..=3 {
            assert_eq!(d.latent_step().unwrap(), LatentStep::Thought { steps });
        }
        assert_eq!(d.latent_step().unwrap(), LatentStep::Exit {
            token: agent.exit, steps: 3, reason: ExitReason::MaxSteps,
        });
        d.token(11).unwrap();
        d.feed_text(&[23, 29]).unwrap();
    }
    assert_eq!(d.position(), 19);
    d.logits().unwrap()
}

#[test]
#[ignore = "requires MDMA_TEST_HYBRID_AGENT_DIR (official 8B bundle)"]
fn released_hybridlink_checkpoint_matches_reference() {
    use sha2::{Digest, Sha256};
    let dir = std::env::var_os("MDMA_TEST_HYBRID_AGENT_DIR").expect("set MDMA_TEST_HYBRID_AGENT_DIR");
    let dir = Path::new(&dir);
    assert_eq!(
        hex::encode(Sha256::digest(std::fs::read(dir.join("adapter.pt")).unwrap())),
        "e5e91f8c61e68be9bd858d24816722eab110e3d1a9febed64481d7ed8c7ceca5",
        "expected the pinned official 8B adapter",
    );
    let weights = Weights::load(dir, 4096).unwrap();
    let hidden = (0..4096).map(|i| (i as f32 * 0.17).sin()).collect::<Vec<_>>();
    let want = reference(&weights, &hidden);
    let b = CpuBackend;
    let link = link::HybridLink::upload(&b, weights).unwrap();
    let mut ws = link.workspace(&b).unwrap();
    let mut output = b.alloc(4096 * 4, Pool::Scratch).unwrap();
    link.apply(&b, &upload(&b, &hidden), &mut ws, &mut output).unwrap();
    close(&read(&b, &output, 4096), &want, 2e-3);
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_hybridlink_and_trajectory_match_cpu() {
    let gpu = crate::backend::rocm::HipBackend::new_f32().unwrap();
    link_reference_parity(&gpu);
    close(&trajectory(&gpu), &trajectory(&CpuBackend), 3e-4);
    let mut index = gpu.alloc(4, Pool::Scratch).unwrap();
    for n in [1, 31, 32, 33, 151671] {
        let mut values = vec![-3.0; n];
        values[n - 1] = 1.0;
        values[n / 2] = 1.0;
        let mut raw = [0; 4];
        gpu.argmax(&upload(&gpu, &values), &mut index).unwrap();
        gpu.download(&index, 0, &mut raw).unwrap();
        assert_eq!(u32::from_ne_bytes(raw), (n / 2) as u32);
        values[n - 1] = f32::NAN;
        gpu.argmax(&upload(&gpu, &values), &mut index).unwrap();
        gpu.download(&index, 0, &mut raw).unwrap();
        assert_eq!(u32::from_ne_bytes(raw), u32::MAX);
    }
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires the official 8B bundle and a ROCm GPU with 18 GiB free"]
fn released_backbone_cpu_rocm_hybrid_trajectory() {
    let dir = std::env::var_os("MDMA_TEST_HYBRID_AGENT_DIR").expect("set MDMA_TEST_HYBRID_AGENT_DIR");
    let dir = Path::new(&dir);
    let cpu = CpuBackend;
    let gpu = crate::backend::rocm::HipBackend::new_f32().unwrap();
    let ca = HybridAgent::load_bundle(&cpu, dir).unwrap();
    let ga = HybridAgent::load_bundle(&gpu, dir).unwrap();
    let tokenizer = super::tokenizer::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let think = tokenizer.token_to_id("<think>").unwrap();
    let limits = LatentLimits { min_steps: 2, max_steps: 2 };
    let (mut cd, mut gd) = (ca.decoder(&cpu, 32, limits).unwrap(), ga.decoder(&gpu, 32, limits).unwrap());
    cd.feed_text(&[think]).unwrap();
    gd.feed_text(&[think]).unwrap();
    let compare = |c: &[f32], g: &[f32]| {
        let max = c.iter().zip(g).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        eprintln!("real 8B CPU/ROCm maximum logit error: {max}");
        assert_eq!(c.len(), g.len());
        for (c, g) in c.iter().zip(g) {
            assert!(c.is_finite() && g.is_finite());
            assert!((c-g).abs() <= 5e-3 + 5e-4 * c.abs(), "{c} vs {g}");
        }
    };
    compare(&cd.logits().unwrap(), &gd.logits().unwrap());
    for _ in 0..2 {
        cd.token(ca.entry).unwrap();
        gd.token(ga.entry).unwrap();
        for _ in 0..3 {
            assert_eq!(cd.latent_step().unwrap(), gd.latent_step().unwrap());
            compare(&cd.logits().unwrap(), &gd.logits().unwrap());
        }
        let logits = cd.logits().unwrap();
        let token = logits.iter().enumerate().filter(|(i,_)| *i as u32 != ca.exit && *i as u32 != ca.entry)
            .max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        cd.token(token).unwrap();
        gd.token(token).unwrap();
        cd.feed_text(&[think]).unwrap();
        gd.feed_text(&[think]).unwrap();
        compare(&cd.logits().unwrap(), &gd.logits().unwrap());
    }
    assert_eq!(cd.position(), gd.position());
    assert_eq!(cd.position(), 13);
}
