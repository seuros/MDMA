use super::*;

impl HipBackend {
    pub(super) fn hybrid_layernorm(
        &self, w: &HipBuf, bias: Option<&HipBuf>, x: &HipBuf, y: &mut HipBuf, mut eps: f32,
    ) -> Result<()> {
        let width = shape::elems("layernorm", w.len)?;
        let rows = shape::layernorm(width, bias.map(|b| shape::elems("layernorm", b.len)).transpose()?,
            shape::elems("layernorm", x.len)?, shape::elems("layernorm", y.len)?)?;
        shape::norm_epsilon(eps)?;
        let mut width = u32::try_from(width).map_err(|_| Error::SizeOverflow)?;
        let rows = u32::try_from(rows).map_err(|_| Error::SizeOverflow)?;
        let mut has_bias = u32::from(bias.is_some());
        let _guard = self.lock.lock().unwrap();
        self.q8_invalidate();
        self.module.get_function("hybrid_layernorm_kernel").map_err(hip_err)?
            .launch(Dim3::new_1d(rows), Dim3::new_1d(256), 0, self.stream.as_ref(), &mut [
                arg!(w.as_device_ptr()), arg!(bias.unwrap_or(w).as_device_ptr()), arg!(x.as_device_ptr()),
                arg!(y.as_device_ptr()), arg!(width), arg!(eps), arg!(has_bias),
            ]).map_err(hip_err)
    }

    pub(super) fn hybrid_gelu(&self, x: &HipBuf, y: &mut HipBuf) -> Result<()> {
        let n = shape::elementwise("gelu", &[shape::elems("gelu", x.len)?], &[shape::elems("gelu", y.len)?])?;
        if n == 0 { return Ok(()); }
        let mut count = u32::try_from(n).map_err(|_| Error::SizeOverflow)?;
        self.launch1d("hybrid_gelu_kernel", n, &mut [arg!(x.as_device_ptr()), arg!(y.as_device_ptr()), arg!(count)])
    }

    pub(super) fn hybrid_argmax(&self, x: &HipBuf, index: &mut HipBuf) -> Result<()> {
        let n = shape::elems("argmax", x.len)?;
        shape::argmax(n, index.len)?;
        let mut count = u32::try_from(n).map_err(|_| Error::SizeOverflow)?;
        let _guard = self.lock.lock().unwrap();
        self.q8_invalidate();
        self.module.get_function("hybrid_argmax_kernel").map_err(hip_err)?
            .launch(Dim3::new_1d(1), Dim3::new_1d(256), 0, self.stream.as_ref(), &mut [
                arg!(x.as_device_ptr()), arg!(index.as_device_ptr()), arg!(count),
            ]).map_err(hip_err)
    }
}
