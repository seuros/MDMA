//! Laya typed decisions: a ModernBERT encoder plus decision heads that
//! score `[MASK]` markers, one per option. No tokens are generated; a
//! request is a state and a set of questions, the answer a distribution
//! per question.
pub mod calibrate;
pub mod config;
pub mod model;
pub mod predictor;
mod sources;

/// The digest a frozen load of this checkpoint would report.
pub(crate) fn artifact_id(dir: &std::path::Path) -> std::io::Result<crate::artifact::ArtifactId> {
    sources::Sources::capture(dir).map(|sources| sources.id())
}
pub mod prompt;
pub mod pyjson;
pub mod tokenizer;
pub use crate::model::weights;

#[cfg(all(test, feature = "service"))]
pub(crate) mod compat_tests;
#[cfg(test)]
mod tests;

pub use config::{AgentConfig, EncoderConfig};
pub use model::DecisionModel;
pub use predictor::Predictor;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("laya config: {0}")]
    Config(String),
    #[error("laya checkpoint: {0}")]
    Checkpoint(String),
    #[error("laya tokenizer: {0}")]
    Tokenizer(String),
    #[error("laya question: {0}")]
    Question(String),
    #[error(transparent)]
    Backend(#[from] crate::backend::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<weights::Error> for Error {
    fn from(error: weights::Error) -> Self {
        match error {
            weights::Error::Checkpoint(message) => Self::Checkpoint(message),
            weights::Error::Backend(error) => Self::Backend(error),
        }
    }
}

impl From<crate::model::arch::modernbert::ConfigError> for Error {
    fn from(error: crate::model::arch::modernbert::ConfigError) -> Self {
        use crate::model::arch::modernbert::ConfigError;
        match error {
            ConfigError::Config(message) => Self::Config(message),
            ConfigError::Json(error) => Self::Json(error),
        }
    }
}
