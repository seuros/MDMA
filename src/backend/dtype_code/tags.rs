//! Kernel ABI tags, also imported by build.rs for shader compilation.
pub const F32: i32 = 0;
pub const F16: i32 = 1;
pub const Q4K: i32 = 2;
pub const Q6K: i32 = 3;
pub const BF16: i32 = 4;
pub const Q5K: i32 = 5;
// `build.rs` includes this module for shader tags, while CUDA uses it in
// host-side decoder dispatch.
#[allow(dead_code)]
pub const Q8_0: i32 = 6;
