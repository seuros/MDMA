//! Optional native CPU checkpoint gate. Token IDs come from the pinned Prism
//! CPU reference in doc/bonsai.md, not from MDMA's own output.
use crate::{artifact, backend::cpu::CpuBackend, gguf::GgufModel, inference};
use std::fs::File;

#[test]
#[ignore = "requires MDMA_TEST_BONSAI_GGUF; pinned Bonsai Q1_0 checkpoint"]
fn bonsai_cpu_checkpoint() {
    let path = std::env::var("MDMA_TEST_BONSAI_GGUF").expect("set MDMA_TEST_BONSAI_GGUF");
    let file = File::open(&path).unwrap();
    assert_eq!(
        artifact::hash_at(&file).unwrap().as_str(),
        "sha256:3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3",
        "this gate requires the exact pinned checkpoint"
    );
    let mut g = GgufModel::from_file(file).unwrap();
    assert_eq!(g.metadata_str("general.architecture"), Some("qwen3"));
    let generated =
        inference::generate_raw(&CpuBackend, &mut g, "The capital of France is", 8, None).unwrap();
    assert_eq!(generated.prompt_tokens, [785, 6722, 315, 9625, 374]);
    assert_eq!(
        generated.tokens,
        [12095, 11, 323, 279, 6722, 315, 17689, 374]
    );
    assert_eq!(generated.text, " Paris, and the capital of Spain is");
}
