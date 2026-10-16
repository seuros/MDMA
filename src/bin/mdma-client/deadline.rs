use rama::{error::BoxError, http::grpc::Request};
use std::{env::VarError, time::Duration};

/// The longest deadline a call may ask for, and the most a route runs: what
/// a routed call gets when MDMA_RPC_TIMEOUT_MS is unset.
pub(super) const MAX: Duration = Duration::from_mins(5);

pub(super) fn timeout(
    env: impl FnOnce(&str) -> Result<String, VarError>,
) -> Result<Option<Duration>, BoxError> {
    match env("MDMA_RPC_TIMEOUT_MS") {
        Err(VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("MDMA_RPC_TIMEOUT_MS: {error}").into()),
        Ok(value) => {
            let ms = value
                .parse::<u32>()
                .ok()
                .filter(|n| (1..=300_000).contains(n))
                .ok_or("MDMA_RPC_TIMEOUT_MS must be an integer between 1 and 300000")?;
            Ok(Some(Duration::from_millis(u64::from(ms))))
        }
    }
}

pub(super) fn request<T>(message: T, timeout: Option<Duration>) -> Result<Request<T>, BoxError> {
    let mut request = Request::new(message);
    if let Some(timeout) = timeout {
        request.try_set_timeout(timeout)?;
    }
    Ok(request)
}

#[cfg(test)]
#[path = "deadline/tests.rs"]
mod tests;
