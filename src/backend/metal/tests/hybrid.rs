use super::*;

#[test]
fn hybridlink_and_persistent_trajectory_match_cpu() {
    use crate::model::hybrid_agent::tests::{link_reference_parity, trajectory};
    link_reference_parity(gpu());
    close(&trajectory(gpu()), &trajectory(&CpuBackend), 2e-4, 2e-4);
}

#[test]
fn layernorm_and_gelu_preserve_canaries_and_match_cpu() {
    let cpu = CpuBackend;
    let g = gpu();
    for width in [1, 7, 31, 32, 33, 257, 4096] {
        let x = data(width * 3, 5);
        let w = data(width, 3);
        let bias = data(width, 7);
        for use_bias in [false, true] {
            let mut want = fbuf(&cpu, &vec![12345.0; x.len() + 7]);
            let mut got = fbuf(g, &vec![12345.0; x.len() + 7]);
            cpu.layernorm(&fbuf(&cpu, &w), use_bias.then(|| fbuf(&cpu, &bias)).as_ref(),
                &fbuf(&cpu, &x), &mut want, 1e-5).unwrap();
            g.layernorm(&fbuf(g, &w), use_bias.then(|| fbuf(g, &bias)).as_ref(),
                &fbuf(g, &x), &mut got, 1e-5).unwrap();
            close(&read(g, &got, x.len() + 7), &read(&cpu, &want, x.len() + 7), 3e-6, 3e-6);
        }
        let x = x.iter().map(|v| v * 5.0).collect::<Vec<_>>();
        let mut want = fbuf(&cpu, &vec![12345.0; x.len() + 7]);
        let mut got = fbuf(g, &vec![12345.0; x.len() + 7]);
        cpu.gelu(&fbuf(&cpu, &x), &mut want).unwrap();
        g.gelu(&fbuf(g, &x), &mut got).unwrap();
        close(&read(g, &got, x.len() + 7), &read(&cpu, &want, x.len() + 7), 2e-6, 2e-6);
    }
}

#[test]
fn argmax_is_stable_at_ties_and_rejects_nonfinite_values() {
    let g = gpu();
    for n in [1, 31, 32, 33, 151671] {
        let mut x = vec![-3.0; n];
        x[n - 1] = 1.0;
        x[n / 2] = 1.0;
        let mut index = g.alloc(4, Pool::Scratch).unwrap();
        let mut bytes = [0; 4];
        g.argmax(&fbuf(g, &x), &mut index).unwrap();
        g.download(&index, 0, &mut bytes).unwrap();
        assert_eq!(u32::from_ne_bytes(bytes), (n / 2) as u32);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            x[n - 1] = bad;
            g.argmax(&fbuf(g, &x), &mut index).unwrap();
            g.download(&index, 0, &mut bytes).unwrap();
            assert_eq!(u32::from_ne_bytes(bytes), u32::MAX);
        }
    }
}

#[test]
fn hybrid_primitive_validation_rejects_bad_shapes_before_launch() {
    let g = gpu();
    let x = fbuf(g, &[1.0; 7]);
    let w = fbuf(g, &[1.0; 3]);
    let mut out = fbuf(g, &[0.0; 7]);
    assert!(g.layernorm(&w, None, &x, &mut out, 1e-5).is_err());
    for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(g.layernorm(&x, None, &x, &mut out, eps).is_err());
    }
    assert!(g.add_bias(&w, &mut out).is_err());
    let empty = g.alloc(0, Pool::Scratch).unwrap();
    let mut index = g.alloc(4, Pool::Scratch).unwrap();
    assert!(g.argmax(&empty, &mut index).is_err());
    assert!(g.argmax(&x, &mut out).is_err());
}
