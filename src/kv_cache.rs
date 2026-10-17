//! Preallocated per-layer KV buffers, updated in place.
//! The backend selects the element width through `Backend::kv_elem_bytes`
//! and converts f32 input rows on write.
//!
//! A sliding-window layer keeps only its window, as a ring: position `p`
//! lands in row `p % rows`. One query attends exactly the rows its window
//! covers, and softmax plus the weighted sum ignore row order (K is stored
//! rotated), so kernels read the ring as an ordinary `min(len, rows)` prefix.
//!
//! A recurrent layer keeps fixed-size state instead of rows. The state has
//! absorbed every position written, so such a cache resumes only where it
//! stopped.
use crate::backend::{Backend, Error, KvView, Pool, Result, RopeSpec};
use crate::memory::{Reservation, product, sum};

pub struct KvCache<B: Backend> {
    k: Vec<B::Buf>,
    v: Vec<B::Buf>,
    row_elems: Vec<usize>,
    /// Rows each layer stores; fewer than `n_ctx` makes the layer a ring.
    rows: Vec<usize>,
    n_ctx: usize,
    elem_bytes: usize,
    len: usize,
    /// One past the highest position written since the last reset.
    written: usize,
    /// Recurrent layer state, f32, covering exactly the `written` positions.
    states: Vec<B::Buf>,
    state_bytes: Vec<usize>,
    _memory: Option<Reservation>,
}

/// Rows a layer stores for `n_ctx` positions: its sliding window, or every
/// position when `window` is 0.
pub(crate) fn layer_rows(window: usize, n_ctx: usize) -> usize {
    if window == 0 {
        n_ctx
    } else {
        window.min(n_ctx)
    }
}

impl<B: Backend> KvCache<B> {
    pub fn new(
        backend: &B,
        n_layers: usize,
        n_kv_heads: usize,
        n_ctx: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let width = product(&[n_kv_heads, head_dim])?;
        Self::with_geometry(backend, &vec![width; n_layers], n_ctx)
    }

    /// Only producer layers need storage; callers map shared layers to them.
    /// Every layer keeps all `n_ctx` positions.
    pub fn with_geometry(backend: &B, widths: &[usize], n_ctx: usize) -> Result<Self> {
        Self::with_geometry_reserved(backend, widths, &vec![0; widths.len()], &[], n_ctx, None)
    }

    /// Bytes for rows of `widths` elements, each layer keeping `layer_rows`,
    /// plus `states` bytes of recurrent state.
    pub(crate) fn allocation_bytes(
        backend: &B,
        widths: &[usize],
        windows: &[usize],
        states: &[usize],
        n_ctx: usize,
    ) -> Result<usize> {
        if widths.len() != windows.len() {
            return Err(Error::ShapeMismatch(
                "KV widths and windows must cover the same layers".into(),
            ));
        }
        widths.iter().zip(windows).try_fold(
            sum(states.iter().copied())?,
            |bytes, (&width, &window)| {
                sum([
                    bytes,
                    product(&[width, backend.kv_elem_bytes(), layer_rows(window, n_ctx), 2])?,
                ])
            },
        )
    }

    pub(crate) fn with_geometry_reserved(
        backend: &B,
        widths: &[usize],
        windows: &[usize],
        states: &[usize],
        n_ctx: usize,
        memory: Option<Reservation>,
    ) -> Result<Self> {
        let elem_bytes = backend.kv_elem_bytes();
        let mut k = Vec::with_capacity(widths.len());
        let mut v = Vec::with_capacity(widths.len());
        if widths.is_empty() || widths.contains(&0) {
            return Err(Error::ShapeMismatch(
                "KV layer geometry must be nonempty".into(),
            ));
        }
        if states.contains(&0) {
            return Err(Error::ShapeMismatch(
                "recurrent state must be nonempty".into(),
            ));
        }
        Self::allocation_bytes(backend, widths, windows, states, n_ctx)?;
        let rows: Vec<usize> = windows.iter().map(|&w| layer_rows(w, n_ctx)).collect();
        for (&width, &rows) in widths.iter().zip(&rows) {
            let layer_bytes = product(&[width, elem_bytes, rows])?;
            k.push(backend.alloc(layer_bytes, Pool::Kv)?);
            v.push(backend.alloc(layer_bytes, Pool::Kv)?);
        }
        let buffers = states
            .iter()
            .map(|&bytes| backend.alloc(bytes, Pool::Kv))
            .collect::<Result<_>>()?;
        Ok(Self {
            k,
            v,
            row_elems: widths.to_vec(),
            rows,
            n_ctx,
            elem_bytes,
            len: 0,
            written: 0,
            states: buffers,
            state_bytes: states.to_vec(),
            _memory: memory,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.n_ctx
    }

    /// Bytes of one stored row in the first layer. For heterogeneous caches,
    /// use `layer_row_bytes` with the producer layer's index.
    pub fn row_bytes(&self) -> usize {
        self.layer_row_bytes(0)
    }

    pub fn layer_row_bytes(&self, layer: usize) -> usize {
        self.row_elems[layer] * self.elem_bytes
    }

    fn is_ring(&self, layer: usize) -> bool {
        self.rows[layer] < self.n_ctx
    }

    /// Storage row of `pos` in `layer`, for `n_rows` rows within capacity.
    fn row(&self, layer: usize, pos: usize, n_rows: usize) -> Result<usize> {
        if pos.checked_add(n_rows).is_none_or(|end| end > self.n_ctx) {
            return Err(Error::ShapeMismatch(format!(
                "KV write of {n_rows} rows at {pos} exceeds capacity {}",
                self.n_ctx
            )));
        }
        if !self.is_ring(layer) {
            return Ok(pos);
        }
        if n_rows != 1 {
            return Err(Error::ShapeMismatch(
                "a sliding-window KV ring takes one row per write".into(),
            ));
        }
        Ok(pos % self.rows[layer])
    }

    /// Write `n_rows` consecutive K/V rows starting at `pos`, straight
    /// from device buffers `k`/`v` holding f32 rows (at least
    /// `n_rows * n_kv_heads * head_dim` floats from their own offset 0),
    /// without a host round trip. Caller drives layer order and calls
    /// `set_len` once after every layer has been written for this
    /// position range. `n_rows == 1` is decode; prefill passes more, except
    /// into a ring.
    pub fn write(
        &mut self,
        backend: &B,
        layer: usize,
        pos: usize,
        k: &B::Buf,
        v: &B::Buf,
        n_rows: usize,
    ) -> Result<()> {
        let row = self.row(layer, pos, n_rows)?;
        let row_elems = self.row_elems[layer];
        backend.kv_write(
            k,
            v,
            &mut self.k[layer],
            &mut self.v[layer],
            row * row_elems,
            n_rows * row_elems,
        )?;
        self.written = self.written.max(pos + n_rows);
        Ok(())
    }

    /// `write` with the RoPE rotation of `q`/`k` folded in: rotates `q` in
    /// place and stores the rotated `k` and `v` rows at `pos`. `k` may not
    /// hold the rotated rows afterwards. Rings are not supported.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_write(
        &mut self,
        backend: &B,
        layer: usize,
        pos: usize,
        q: &mut B::Buf,
        k: &mut B::Buf,
        v: &B::Buf,
        n_rows: usize,
        spec: &RopeSpec,
    ) -> Result<()> {
        if self.is_ring(layer) {
            return Err(Error::ShapeMismatch(
                "RoPE-fused KV writes need a full-context layer".into(),
            ));
        }
        let row = self.row(layer, pos, n_rows)?;
        let row_elems = self.row_elems[layer];
        backend.rope_kv_write(
            q,
            k,
            v,
            &mut self.k[layer],
            &mut self.v[layer],
            row * row_elems,
            pos as u32,
            n_rows,
            spec,
            n_rows * row_elems,
        )?;
        self.written = self.written.max(pos + n_rows);
        Ok(())
    }

    /// Set the initialized prefix; 0 starts a new sequence. Going back to a
    /// nonzero length must pass `can_resume` first (or use `rewind`).
    pub fn set_len(&mut self, len: usize) {
        self.len = len;
        if len == 0 {
            self.written = 0;
        } else if !self.states.is_empty() {
            // Graph replay updates recurrent device state without going
            // through the ordinary layer writes, so keep the host cursor in
            // step with the replayed position.  Capture records ordinary
            // writes on the host while its kernels do not execute; assigning
            // here also restores the cursor to the captured prefix.
            self.written = len;
        }
    }

    /// Whether a query at `pos` sees only the current sequence: its prefix is
    /// initialized and no ring overwrote a row it attends. A ring holds the
    /// last `rows` positions written, so going back is only safe before any
    /// ring wrapped, or onto the last position written. Recurrent state
    /// cannot go back at all.
    pub fn can_resume(&self, pos: usize) -> bool {
        pos <= self.len
            && (pos == 0
                || if self.states.is_empty() {
                    pos + 1 >= self.written || self.rows.iter().all(|&rows| self.written <= rows)
                } else {
                    pos == self.written
                })
    }

    /// Truncate to `len` positions if decoding can resume there, otherwise
    /// start over. Returns the positions kept.
    pub fn rewind(&mut self, len: usize) -> usize {
        let len = if self.can_resume(len) { len } else { 0 };
        self.set_len(len);
        len
    }

    /// Recurrent state buffer `index`, in the order the cache was built with.
    pub(crate) fn state_mut(&mut self, index: usize) -> Result<&mut B::Buf> {
        self.states
            .get_mut(index)
            .ok_or_else(|| Error::ShapeMismatch(format!("KV cache has no recurrent state {index}")))
    }

    /// Zero every recurrent state, for a sequence starting at position 0.
    pub(crate) fn clear_states(&mut self, backend: &B) -> Result<()> {
        let zeros = vec![0; self.state_bytes.iter().copied().max().unwrap_or(0)];
        for (state, &bytes) in self.states.iter_mut().zip(&self.state_bytes) {
            backend.upload(state, 0, &zeros[..bytes])?;
        }
        Ok(())
    }

    /// A full ring exposes all its rows, in storage order.
    pub fn view(&self, layer: usize) -> KvView<'_, B::Buf> {
        KvView {
            k: &self.k[layer],
            v: &self.v[layer],
            len: self.len.min(self.rows[layer]),
        }
    }
}

#[cfg(test)]
mod tests;
