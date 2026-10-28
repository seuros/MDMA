use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct Inner {
    limit: usize,
    used: AtomicUsize,
}

#[derive(Clone, Debug)]
pub(super) struct Counter(Arc<Inner>);

impl Counter {
    pub(super) fn new(limit: usize) -> Self {
        Self(Arc::new(Inner {
            limit,
            used: AtomicUsize::new(0),
        }))
    }

    pub(super) fn limit(&self) -> usize {
        self.0.limit
    }

    pub(super) fn used(&self) -> usize {
        self.0.used.load(Ordering::Relaxed)
    }

    pub(super) fn reserve(&self, bytes: usize) -> Result<Charge, usize> {
        self.0
            .used
            .try_update(Ordering::AcqRel, Ordering::Relaxed, |used| {
                used.checked_add(bytes).filter(|&next| next <= self.limit())
            })?;
        Ok(Charge {
            counter: self.clone(),
            bytes,
        })
    }
}

#[derive(Debug)]
pub(super) struct Charge {
    counter: Counter,
    bytes: usize,
}

impl Charge {
    #[cfg(any(feature = "service", test))]
    pub(super) fn split(&mut self, bytes: usize) -> Self {
        self.bytes = self
            .bytes
            .checked_sub(bytes)
            .expect("split within reserved bytes");
        Self {
            counter: self.counter.clone(),
            bytes,
        }
    }

    pub(super) const fn bytes(&self) -> usize {
        self.bytes
    }

    pub(super) fn covers(&self, counter: &Counter, bytes: usize) -> bool {
        Arc::ptr_eq(&self.counter.0, &counter.0) && self.bytes >= bytes
    }

    /// Keep unknown/unreleased driver memory charged without leaking the Arc.
    #[cfg_attr(
        not(any(test, feature = "rocm", feature = "cuda", feature = "vulkan")),
        allow(dead_code)
    )]
    pub(super) fn retain(mut self) {
        self.bytes = 0;
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.counter.0.used.fetch_sub(self.bytes, Ordering::Release);
    }
}
