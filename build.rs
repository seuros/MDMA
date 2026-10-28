#[path = "src/backend/dtype_code/tags.rs"]
mod dtype_tags;
#[path = "src/backend/vulkan/abi.rs"]
mod vulkan_abi;

// Keep packed-weight ABI constants shared with the HIP compiler.
#[path = "src/cache/layout_abi.rs"]
mod packed_abi;

fn main() {
    // Changes to the generator/kernel build logic must invalidate cached output.
    println!("cargo::rerun-if-changed=build.rs");
    build_identity();
    #[cfg(feature = "service")]
    {
        rama::http::grpc::build::protobuf::configure()
            .with_type_attribute(".mdma", "#[derive(serde::Serialize)]")
            .compile_protos(&["proto/mdma.proto"], &["proto"])
            .unwrap_or_else(|e| panic!("failed to compile proto/mdma.proto: {e}"));
        for proto in ["proto/node.proto", "proto/federation.proto"] {
            rama::http::grpc::build::protobuf::compile_protos(proto)
                .unwrap_or_else(|e| panic!("failed to compile {proto}: {e}"));
        }
        println!("cargo::rerun-if-changed=proto");
    }
    if cfg!(feature = "service") && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        embed_info_plists();
    }

    if std::env::var("CARGO_FEATURE_ROCM").is_ok() {
        build_hip_kernels();
    }
    if std::env::var("CARGO_FEATURE_LEVEL0").is_ok() {
        build_level0_kernels();
    }
    if std::env::var("CARGO_FEATURE_VULKAN").is_ok() {
        build_vulkan_kernels();
    }
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda_kernels();
    }
}

/// macOS asks before a program reaches the local network and remembers the
/// answer by code signature; the embedded Info.plist names the program and
/// says why. `just sign` signs with these identifiers.
fn embed_info_plists() {
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let version = std::env::var("CARGO_PKG_VERSION").expect("cargo sets CARGO_PKG_VERSION");
    for (bin, why) in [
        (
            "mdma-serve",
            "mdma-serve reads the model catalogs of your enrolled federation peers and runs requests on them.",
        ),
        (
            "mdma-fed",
            "mdma-fed reads a federation peer's signed model catalog to set up routes to it.",
        ),
    ] {
        let plist = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>org.seuros.{bin}</string>
    <key>CFBundleName</key><string>{bin}</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleShortVersionString</key><string>{version}</string>
    <key>CFBundleVersion</key><string>{version}</string>
    <key>NSLocalNetworkUsageDescription</key><string>{why}</string>
</dict>
</plist>
"#
        );
        let path = out.join(format!("{bin}.Info.plist"));
        std::fs::write(&path, plist)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
        println!(
            "cargo::rustc-link-arg-bin={bin}=-Wl,-sectcreate,__TEXT,__info_plist,{}",
            path.display()
        );
    }
}

fn build_identity() {
    use std::process::Command;
    println!("cargo::rerun-if-env-changed=MDMA_BUILD_ID");
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed=Cargo.toml");
    println!("cargo::rerun-if-changed=Cargo.lock");
    let git = |args: &[&str]| -> Option<String> {
        let output = Command::new("git").args(args).output().ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    // Never attribute a source archive to an unrelated enclosing repository.
    let root =
        git(&["rev-parse", "--show-toplevel"]).and_then(|path| std::fs::canonicalize(path).ok());
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("Cargo sets CARGO_MANIFEST_DIR for build scripts");
    let manifest = std::fs::canonicalize(manifest_dir).ok();
    let id = if root.is_some() && root == manifest {
        let branch = git(&["symbolic-ref", "-q", "HEAD"]).unwrap_or_default();
        for path in ["HEAD", "index", "packed-refs", branch.as_str()] {
            if !path.is_empty()
                && let Some(path) = git(&["rev-parse", "--git-path", path])
                && std::path::Path::new(&path).exists()
            {
                println!("cargo::rerun-if-changed={path}");
            }
        }
        git(&["rev-parse", "HEAD"]).map(|sha| {
            if Command::new("git")
                .args(["diff-index", "--quiet", "HEAD", "--"])
                .status()
                .is_ok_and(|status| status.success())
            {
                sha
            } else {
                format!("{sha}-dirty")
            }
        })
    } else {
        None
    };
    let id = std::env::var("MDMA_BUILD_ID")
        .ok()
        .or(id)
        .unwrap_or_else(|| "unknown".into());
    assert!(
        !id.is_empty()
            && id.len() <= 128
            && id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"._+-".contains(&c)),
        "MDMA_BUILD_ID must be a 1..128 character build token"
    );
    println!("cargo::rustc-env=MDMA_BUILD_ID={id}");
}

/// Compiles the embedded HIP module for gfx1201 by default.
/// `MDMA_HIP_OFFLOAD_ARCH` selects a different target.
fn build_hip_kernels() {
    let src = "src/backend/rocm/kernels/kernels.hip";
    println!("cargo::rerun-if-changed={src}");
    println!("cargo::rerun-if-changed=src/backend/rocm/kernels/decode.hip");
    println!("cargo::rerun-if-changed=src/backend/rocm/kernels/attention.hip");
    println!("cargo::rerun-if-changed=src/backend/rocm/kernels/decoder_ops.hip");
    println!("cargo::rerun-if-changed=src/backend/rocm/kernels/hybrid.hip");
    println!("cargo::rerun-if-changed=src/cache/layout_abi.rs");
    println!("cargo::rerun-if-env-changed=MDMA_HIP_OFFLOAD_ARCH");

    let arch = std::env::var("MDMA_HIP_OFFLOAD_ARCH").unwrap_or_else(|_| "gfx1201".to_string());
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let out_path = format!("{out_dir}/kernels.hsaco");

    let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "/opt/rocm/bin/hipcc".to_string());
    let status = std::process::Command::new(&hipcc)
        .args(["-fno-gpu-rdc", "-fPIC", "--genco", "-O3"])
        .arg(format!("--offload-arch={arch}"))
        .arg(format!(
            "-DBLOCK_Q4K_PACKED_BYTES={}",
            packed_abi::Q4K_BLOCK_BYTES
        ))
        .arg(format!(
            "-DQ4K_PACKED_QUANTS_OFFSET={}",
            packed_abi::Q4K_QUANTS_OFFSET
        ))
        .arg(format!("-DDT_Q4K_PACKED={}", packed_abi::Q4K_DTYPE_TAG))
        .args(["-o", &out_path, src])
        .status()
        .unwrap_or_else(|e| panic!("failed to run {hipcc}: {e} - is ROCm installed at /opt/rocm?"));

    assert!(
        status.success(),
        "hipcc failed compiling {src} (arch {arch})"
    );

    println!("cargo::rustc-env=MDMA_HIP_KERNELS_HSACO={out_path}");
    println!("cargo::rustc-env=MDMA_HIP_ARCH_BUILT={arch}");
}

/// Compiles kernels.cl with ocloc for the Intel iGPU. MDMA_OCLOC_DEVICE
/// defaults to mtl (Meteor Lake); override for other Xe parts.
fn build_level0_kernels() {
    let src = "src/backend/level0/kernels/kernels.cl";
    println!("cargo::rerun-if-changed={src}");
    println!("cargo::rerun-if-env-changed=MDMA_OCLOC_DEVICE");
    println!("cargo::rerun-if-env-changed=OCLOC");
    let device = std::env::var("MDMA_OCLOC_DEVICE").unwrap_or_else(|_| "mtl".to_string());
    let ocloc = std::env::var("OCLOC").unwrap_or_else(|_| "ocloc".to_string());
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    let stem = format!("{out_dir}/level0");
    let status = std::process::Command::new(&ocloc)
        .args([
            "compile",
            "-file",
            src,
            "-device",
            &device,
            "-output",
            &stem,
            "-output_no_suffix",
            "-options",
            "-cl-std=CL2.0",
        ])
        .status()
        .unwrap_or_else(|e| panic!("failed to run {ocloc}: {e}"));
    assert!(
        status.success(),
        "ocloc failed compiling {src} for {device}"
    );
    let bin = format!("{stem}.bin");
    println!("cargo::rustc-env=MDMA_L0_ZEBIN={bin}");
    println!("cargo::rustc-link-lib=ze_loader");
}

/// Compiles every src/backend/vulkan/kernels/*.comp to SPIR-V with
/// glslangValidator (GLSLANG overrides the binary). The backend embeds
/// $OUT_DIR/vk/<name>.spv per kernel.
fn build_vulkan_kernels() {
    let dir = "src/backend/vulkan/kernels";
    println!("cargo::rerun-if-changed={dir}");
    println!("cargo::rerun-if-changed=src/backend/dtype_code/tags.rs");
    println!("cargo::rerun-if-changed=src/backend/vulkan/abi.rs");
    println!("cargo::rerun-if-env-changed=GLSLANG");
    let glslang = std::env::var("GLSLANG").unwrap_or_else(|_| "glslangValidator".to_string());
    let out_dir = format!("{}/vk", std::env::var("OUT_DIR").expect("OUT_DIR"));
    std::fs::create_dir_all(&out_dir).expect("create spv dir");
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("kernel dir")
        .flatten()
        .map(|e| e.path())
        .collect();
    entries.sort();
    let defines = [
        ("BINARY_THREADS", vulkan_abi::BINARY_THREADS),
        ("UNARY_THREADS", vulkan_abi::UNARY_THREADS),
        ("SPLIT_ROPE_THREADS", vulkan_abi::SPLIT_ROPE_THREADS),
        ("EMBED_DIM_THREADS", vulkan_abi::EMBED_DIM_THREADS),
        ("ROPE_PAIR_THREADS", vulkan_abi::ROPE_PAIR_THREADS),
        ("ATTN_QUERY_TILE", vulkan_abi::ATTN_QUERY_TILE),
        ("UNARY_SCALE", vulkan_abi::UNARY_SCALE),
        ("UNARY_SOFTCAP", vulkan_abi::UNARY_SOFTCAP),
        ("DTYPE_F32", dtype_tags::F32 as u32),
        ("DTYPE_F16", dtype_tags::F16 as u32),
        ("DTYPE_Q4K", dtype_tags::Q4K as u32),
        ("DTYPE_Q6K", dtype_tags::Q6K as u32),
        ("DTYPE_BF16", dtype_tags::BF16 as u32),
        ("DTYPE_Q5K", dtype_tags::Q5K as u32),
    ];
    for path in entries {
        if path.extension().and_then(|e| e.to_str()) != Some("comp") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_else(|| panic!("shader path {} has no UTF-8 file stem", path.display()));
        let out = format!("{out_dir}/{stem}.spv");
        let mut command = std::process::Command::new(&glslang);
        command.args([
            "-V",
            "--target-env",
            "vulkan1.2",
            &format!("-I{dir}"),
            "-o",
            &out,
        ]);
        for (name, value) in defines {
            command.arg(format!("-D{name}={value}u"));
        }
        let status = command.arg(&path).status().unwrap_or_else(|e| {
            panic!("failed to run {glslang}: {e} -- install glslang or set GLSLANG")
        });
        assert!(
            status.success(),
            "{glslang} failed compiling {}",
            path.display()
        );
    }
    println!("cargo::rustc-env=MDMA_VK_SPV_DIR={out_dir}");
}

/// Compiles src/backend/cuda/kernels/kernels.cu to PTX with clang's NVPTX
/// target -- no CUDA toolkit: the kernels use NVVM builtins, not
/// cuda_runtime.h, and the driver JITs the PTX at load. MDMA_CUDA_ARCH
/// defaults to sm_86 (Ampere consumer); PTX for an older arch still JITs
/// forward to newer GPUs. Links libcuda (the driver, from nvidia-utils).
fn build_cuda_kernels() {
    let src = "src/backend/cuda/kernels/kernels.cu";
    println!("cargo::rerun-if-changed={src}");
    println!("cargo::rerun-if-changed=src/backend/cuda/kernels/decoder_ops.cu");
    println!("cargo::rerun-if-env-changed=MDMA_CUDA_ARCH");
    println!("cargo::rerun-if-env-changed=CLANG");
    let arch = std::env::var("MDMA_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());
    let clang = std::env::var("CLANG").unwrap_or_else(|_| "clang".to_string());
    let out_path = format!("{}/kernels.ptx", std::env::var("OUT_DIR").expect("OUT_DIR"));
    let status = std::process::Command::new(&clang)
        .args([
            "--cuda-device-only",
            "-nocudainc",
            "-nocudalib",
            "--cuda-feature=+ptx80",
            "-O3",
            "-S",
        ])
        .arg(format!("--cuda-gpu-arch={arch}"))
        .args(["-o", &out_path, src])
        .status()
        .unwrap_or_else(|e| {
            panic!("failed to run {clang}: {e} -- needs clang with the NVPTX target")
        });
    assert!(
        status.success(),
        "{clang} failed compiling {src} for {arch}"
    );
    println!("cargo::rustc-env=MDMA_CUDA_PTX={out_path}");
    // The host gates sm_80-only paths (int8 mma.sync) on the built arch too.
    println!("cargo::rustc-env=MDMA_CUDA_ARCH_BUILT={arch}");
    println!("cargo::rustc-link-lib=cuda");
    // NVML, for the power watcher's GPU power limit; same driver package.
    println!("cargo::rustc-link-lib=nvidia-ml");
}
