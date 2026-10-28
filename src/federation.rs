//! MDMA federation (v1 draft): see `doc/federation.md` and
//! `doc/federation/*.md` for the full protocol design.
//!
//! **Disabled by default.** The opt-in server integration exposes authenticated
//! control RPCs and recipient-filtered signed native catalogs. Separate explicit
//! switches enable native single-hop execution and pinned origin routes:
//!
//! - [`server_id`]: the operator-chosen DNS-form peer identity.
//! - [`canonical_json`]: Matrix-style canonical JSON encoding/decoding,
//!   used as the catalog signing preimage.
//! - [`signing`]: Ed25519 catalog signing/verification, separate from TLS
//!   key material.
//! - [`catalog`]: the signed catalog envelope and model-descriptor data
//!   model from `doc/federation/identity-and-catalog-v1.md`.
//! - [`config`]: bounded, deny-by-default local peer policy.
//! - [`trust`]: durable enrollment, key history and revocations.
//! - [`tls`]: per-peer PKI, explicit certificate purpose/identity/SPKI checks
//!   and authenticated full-handshake bindings.
//! - [`control`]: authenticated Identify and signed catalog RPCs.
//! - `install`: models a peer with the install grant pushes here.
//! - [`catalog_cache`]: bounded challenges and verified expiring snapshots.
//! - [`runtime`]: opt-in listener, catalog polling and native/client integration.
//!
//! Native controllers require exact frozen artifacts and actual CPU profiles;
//! route aliases additionally require current imports and local caller consent.
//! No configuration automatically enrolls a peer or approves either permission.

#[cfg(feature = "service")]
pub mod admin;
pub mod canonical_json;
#[cfg(feature = "service")]
pub mod catalog;
#[cfg(feature = "service")]
pub mod catalog_cache;
#[cfg(feature = "service")]
pub mod config;
#[cfg(feature = "service")]
pub mod control;
#[cfg(feature = "service")]
pub(crate) mod execution;
#[cfg(feature = "service")]
pub(crate) mod install;
#[cfg(feature = "service")]
pub(crate) mod inventory;
pub mod profile;
#[cfg(feature = "service")]
pub(crate) mod push;
#[cfg(feature = "service")]
pub(crate) mod routing;
#[cfg(feature = "service")]
pub mod runtime;
#[cfg(feature = "service")]
pub mod server_id;
#[cfg(feature = "service")]
pub mod signing;
#[cfg(feature = "service")]
pub mod tls;
#[cfg(feature = "service")]
pub(crate) mod topology;
#[cfg(feature = "service")]
pub mod trust;

#[cfg(all(test, feature = "service"))]
mod tests;
