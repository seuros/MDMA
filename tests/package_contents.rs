use std::{path::Path, process::Command};

#[test]
fn published_crate_keeps_runtime_files_without_test_payloads() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new(env!("CARGO"))
        .current_dir(root)
        .args(["package", "--list", "--allow-dirty", "--offline"])
        .output()
        .expect("run cargo package --list");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 package paths");
    let files: Vec<_> = stdout.lines().collect();
    for required in [
        "build.rs",
        "proto/mdma.proto",
        "src/lib.rs",
        "src/bin/mdma-serve.rs",
        "src/bin/mdma.rs",
        "src/bin/mdma/device.rs",
        "src/bin/mdma/doctor.rs",
        "src/bin/mdma/plan.rs",
        "src/bin/mdma/run.rs",
        "src/bin/mdma/bench.rs",
        "src/inference.rs",
        "src/gguf/dtype.rs",
        "src/gguf/header.rs",
        "src/model/arch/qwen/yarn.rs",
        "src/model/hybrid_agent/runtime.rs",
        "src/model/hybrid_agent/runtime/session.rs",
        "src/memory_bench.rs",
        "src/model/planning.rs",
        "src/model/hybrid.rs",
        "src/bin/mdma/info.rs",
        "src/bin/mdma/info/report.rs",
        "src/bin/mdma/info/render.rs",
        "src/bin/mdma/info/linux.rs",
        "src/bin/mdma/info/linux/cpu.rs",
        "src/bin/mdma/info/common.rs",
        "src/bin/mdma/info/dmi.rs",
        "src/bin/mdma/info/freebsd.rs",
        "src/bin/mdma/info/freebsd/pci.rs",
        "src/bin/mdma/info/macos.rs",
        "src/bin/mdma/info/macos/profiler.rs",
        "src/bin/mdma/info/linux/gpu.rs",
        "src/bin/mdma/info/linux/gpu/pci.rs",
        "src/backend/dtype_code/tags.rs",
        "src/backend/vulkan/abi.rs",
        "src/backend/metal/kernels.metal",
        "src/backend/metal/abi.rs",
        "src/backend/metal/batch.rs",
        "src/backend/rocm/kernels/kernels.hip",
        "src/backend/cuda/kernels/kernels.cu",
        "src/backend/cuda/kernels/decoder_ops.cu",
        "src/backend/level0/kernels/kernels.cl",
        "src/backend/vulkan/kernels/gemm_tile.glsl",
        "src/backend/vulkan/kernels/gemm_i8.glsl",
        "src/backend/vulkan/kernels/gemm_dot.glsl",
        "src/backend/vulkan/kernels/int8_weight_tile.glsl",
        "src/backend/vulkan/kernels/gemm_q4k_dot.comp",
        "src/backend/vulkan/kernels/gemm_q6k_dot.comp",
        "src/backend/vulkan/kernels/gemm_q4k_i8.comp",
        "src/backend/vulkan/kernels/gemm_q6k_i8.comp",
        "src/backend/vulkan/kernels/quantize_q8.comp",
        "src/backend/vulkan/kernels/add_rmsnorm.comp",
        "src/backend/vulkan/kernels/attn_tile.comp",
        "LICENSE-llama.cpp",
    ] {
        assert!(
            files.contains(&required),
            "missing runtime file: {required}"
        );
    }
    for file in files {
        assert!(
            !file.split('/').any(|part| matches!(
                part,
                ".tmp" | "tests" | "benches" | "fixtures" | "test_support" | "tools"
            )) && !file.ends_with("/tests.rs")
                && !file.ends_with("_tests.rs")
                && !file.ends_with("/test_support.rs")
                && !file.starts_with("doc/")
                && !file.starts_with("deploy/"),
            "test/development file packaged: {file}"
        );
        if Path::new(file)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("rs"))
        {
            let source = std::fs::read_to_string(root.join(file)).expect("read packaged source");
            assert!(
                !source.contains("#[test]") && !source.contains("#[tokio::test"),
                "inline tests packaged: {file}"
            );
        }
    }
}
