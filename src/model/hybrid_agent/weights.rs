//! Strict, data-only loading through Candle's existing PyTorch reader.
use super::{Result, unsupported};
use candle_core::{DType, pickle::PthTensors};
use std::{collections::BTreeMap, path::Path};

pub(super) const NAMES: [&str; 8] = [
    "proj1.weight", "proj1.bias", "proj2.weight", "proj2.bias",
    "pre_ln.weight", "pre_ln.bias", "post_ln.weight", "post_ln.bias",
];

pub(super) struct Weights {
    pub width: usize,
    pub tensors: BTreeMap<String, Vec<f32>>,
}

fn expected_shape(name: &str, width: usize) -> Vec<usize> {
    if matches!(name, "proj1.weight" | "proj2.weight") { vec![width, width] } else { vec![width] }
}

impl Weights {
    pub fn load(directory: &Path, width: usize) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(directory.join("adapter_config.json"))?
        ).map_err(|e| unsupported(e.to_string()))?;
        if config.as_object().is_none_or(|v| v.len() != 1) || config["adapter_type"] != "ln_res_adapter" {
            return Err(unsupported("expected ln_res_adapter configuration"));
        }
        let path = directory.join("adapter.pt");
        let info = candle_core::pickle::read_pth_tensor_info(&path, false, None)
            .map_err(|e| unsupported(e.to_string()))?;
        if info.len() != NAMES.len() {
            return Err(unsupported("expected exactly eight adapter tensors"));
        }
        let mut names = std::collections::BTreeSet::new();
        for tensor in &info {
            if !NAMES.contains(&tensor.name.as_str()) || !names.insert(tensor.name.as_str())
                || tensor.layout.dims() != expected_shape(&tensor.name, width)
                || tensor.layout.start_offset() != 0 || !tensor.layout.is_contiguous()
                || !matches!(tensor.dtype, DType::F32 | DType::F16 | DType::BF16)
            {
                return Err(unsupported(format!("invalid adapter tensor {}", tensor.name)));
            }
        }
        let pth = PthTensors::new(path, None).map_err(|e| unsupported(e.to_string()))?;
        let mut tensors = BTreeMap::new();
        for name in NAMES {
            let tensor = pth.get(name).map_err(|e| unsupported(e.to_string()))?
                .ok_or_else(|| unsupported(format!("missing {name}")))?;
            let values = tensor.to_dtype(DType::F32).and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>()).map_err(|e| unsupported(e.to_string()))?;
            tensors.insert(name.into(), values);
        }
        let weights = Self { width, tensors };
        weights.validate()?;
        Ok(weights)
    }

    pub fn validate(&self) -> Result<()> {
        crate::memory::product(&[self.width, self.width, 4])?;
        if self.width == 0 || self.tensors.len() != NAMES.len() {
            return Err(unsupported("invalid adapter width or tensor set"));
        }
        for name in NAMES {
            let values = self.tensors.get(name).ok_or_else(|| unsupported(format!("missing {name}")))?;
            let count = expected_shape(name, self.width).iter().product::<usize>();
            if values.len() != count || values.iter().any(|v| !v.is_finite()) {
                return Err(unsupported(format!("wrong size or nonfinite values in {name}")));
            }
        }
        Ok(())
    }
}
