//! Independent native tensor admission and backend buffer quotas.
use crate::backend::{Error, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg_attr(
    not(any(test, feature = "rocm", feature = "cuda", feature = "vulkan")),
    allow(dead_code)
)]
mod buffers;
mod counter;
#[cfg(any(feature = "rocm", feature = "cuda", feature = "vulkan"))]
pub(crate) use buffers::BufferCharge;
pub use buffers::{BufferBudget, MemoryDomain};

/// What a native reservation holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCategory {
    Weights,
    Kv,
    Workspace,
}

/// A point-in-time view of a [`NativeMemoryBudget`]; categories sum to `used`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BudgetSnapshot {
    pub limit: usize,
    pub used: usize,
    pub weights: usize,
    pub kv: usize,
    pub workspace: usize,
}

/// Share one budget across native models and their sessions. Counts weight
/// payloads, KV and workspace reservations only. Backend-private allocations,
/// driver overhead, staging and tokenizers are outside this budget. The registry
/// decoder budget excludes Laya; federated Laya uses a separate bounded logical
/// request-workspace instance, not shared decoder/model-weight accounting.
#[derive(Clone, Debug)]
pub struct NativeMemoryBudget {
    counter: counter::Counter,
    categories: Arc<[AtomicUsize; 3]>,
}

impl NativeMemoryBudget {
    pub fn new(limit: usize) -> Self {
        Self {
            counter: counter::Counter::new(limit),
            categories: Arc::default(),
        }
    }

    pub fn limit(&self) -> usize {
        self.counter.limit()
    }

    /// Live and reserved logical tensor bytes; not measured physical residency.
    pub fn used(&self) -> usize {
        self.counter.used()
    }

    pub fn snapshot(&self) -> BudgetSnapshot {
        let [weights, kv, workspace] = [
            MemoryCategory::Weights,
            MemoryCategory::Kv,
            MemoryCategory::Workspace,
        ]
        .map(|category| self.tally(category).load(Ordering::Relaxed));
        BudgetSnapshot {
            limit: self.limit(),
            used: self.used(),
            weights,
            kv,
            workspace,
        }
    }

    fn tally(&self, category: MemoryCategory) -> &AtomicUsize {
        &self.categories[category as usize]
    }

    pub(crate) fn reserve(&self, bytes: usize, category: MemoryCategory) -> Result<Reservation> {
        let charge = self
            .counter
            .reserve(bytes)
            .map_err(|used| Error::MemoryBudget {
                requested: bytes,
                used,
                limit: self.limit(),
            })?;
        self.tally(category).fetch_add(bytes, Ordering::Relaxed);
        Ok(Reservation {
            budget: self.clone(),
            charge,
            category,
        })
    }
}

#[derive(Debug)]
pub(crate) struct Reservation {
    pub(crate) budget: NativeMemoryBudget,
    charge: counter::Charge,
    category: MemoryCategory,
}

impl Reservation {
    /// Carve `bytes` out of this reservation and account them as `category`.
    #[cfg(any(feature = "service", test))]
    pub(crate) fn split_as(&mut self, bytes: usize, category: MemoryCategory) -> Self {
        let charge = self.charge.split(bytes);
        if category != self.category {
            self.budget
                .tally(self.category)
                .fetch_sub(bytes, Ordering::Relaxed);
            self.budget
                .tally(category)
                .fetch_add(bytes, Ordering::Relaxed);
        }
        Self {
            budget: self.budget.clone(),
            charge,
            category,
        }
    }

    pub(crate) fn covers(&self, budget: &NativeMemoryBudget, bytes: usize) -> bool {
        self.charge.covers(&budget.counter, bytes)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.charge.bytes()
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget
            .tally(self.category)
            .fetch_sub(self.charge.bytes(), Ordering::Relaxed);
    }
}

pub(crate) fn product(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |n, &value| n.checked_mul(value))
        .ok_or(Error::SizeOverflow)
}

pub(crate) fn sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(Error::SizeOverflow)
}

#[cfg(test)]
mod tests;
