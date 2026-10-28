//! HybridLink primitives; activations never leave device buffers.
use super::*;

impl MetalBackend {
    pub(super) fn hybrid_layernorm(
        &self, w: &MetalBuf, bias: Option<&MetalBuf>, x: &MetalBuf, y: &mut MetalBuf, eps: f32,
    ) -> Result<()> {
        let width = floats(w)?;
        let rows = geometry::layernorm(width, bias.map(floats).transpose()?, floats(x)?, floats(y)?)?;
        geometry::norm_epsilon(eps)?;
        let params = [dim(width)?, dim(rows)?, eps.to_bits(), u32::from(bias.is_some())];
        // SAFETY: complete rows, optional bias and output capacity checked.
        // The fallback weight binding is never read as bias when p[3] == 0.
        unsafe {
            self.launch("layernorm_f32", &[w, bias.unwrap_or(w), x, y], &params,
                Dispatch::SimdRows { rows, batches: 1 })
        }
    }

    pub(super) fn hybrid_gelu(&self, x: &MetalBuf, y: &mut MetalBuf) -> Result<()> {
        let n = geometry::elementwise("gelu", &[floats(x)?], &[floats(y)?])?;
        // SAFETY: one writer per input element; an oversized output tail is untouched.
        unsafe { self.launch("gelu_f32", &[x, y], &[dim(n)?], Dispatch::Threads([n, 1, 1])) }
    }

    pub(super) fn hybrid_add_bias(&self, bias: &MetalBuf, y: &mut MetalBuf) -> Result<()> {
        let n = floats(y)?;
        let width = floats(bias)?;
        geometry::rows("add_bias", width, n, n)?;
        // SAFETY: complete nonempty rows, in-place disjoint element writes.
        unsafe { self.launch("add_bias_f32", &[bias, y], &[dim(n)?, dim(width)?], Dispatch::Threads([n, 1, 1])) }
    }

    pub(super) fn hybrid_argmax(&self, x: &MetalBuf, index: &mut MetalBuf) -> Result<()> {
        let n = floats(x)?;
        geometry::argmax(n, index.len)?;
        // SAFETY: one SIMD group reduces a nonempty, u32-indexable vector.
        // Only lane zero of the first SIMD group writes the one-u32 output.
        unsafe {
            self.launch("argmax_f32", &[x, index], &[dim(n)?],
                Dispatch::SimdRows { rows: 1, batches: 1 })
        }
    }
}
