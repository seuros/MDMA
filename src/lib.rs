//! Embeddable local inference with pluggable CPU/GPU backends.
//!
//! The default build is an in-process library. Enable `service` for the
//! optional mTLS gRPC service and `cli` for command-line frontends.

mod artifact;
pub mod backend;
pub mod cache;
#[cfg(feature = "service")]
pub mod decision;
pub mod federation;
pub mod gguf;
#[cfg(feature = "service")]
pub mod grpc;
#[cfg(feature = "service")]
pub mod host;
pub mod inference;
pub mod kv_cache;
pub mod laya;
pub mod memory;
pub mod memory_bench;
pub mod model;
#[cfg(feature = "service")]
pub mod power;
#[cfg(feature = "service")]
pub mod registry;
pub mod sampler;
#[cfg(feature = "service")]
pub mod session;
pub mod singleton;
pub mod tokenizer;

pub use backend::{Backend, Error, Result};
pub use model::hybrid_agent;
