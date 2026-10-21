//! Candle adapter for the optional checked, persistently reordered F32 backend.
use aocl_kernels::PackedLinear;
use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape};
use std::sync::Arc;

pub(crate) struct Linear(pub(crate) Arc<PackedLinear>);

impl CustomOp1 for Linear {
    fn name(&self) -> &'static str {
        "aocl-f32-linear"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(values) = storage else {
            candle_core::bail!("F32 linear requires F32 activations")
        };
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("F32 linear requires contiguous activations")
        };
        let mut dims = layout.dims().to_vec();
        if dims.last() != Some(&self.0.input_size()) {
            candle_core::bail!("F32 linear input shape mismatch")
        }
        let rows = (end - start) / self.0.input_size();
        let output = self
            .0
            .forward(&values[start..end], rows)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        *dims.last_mut().unwrap() = self.0.output_size();
        Ok((CpuStorage::F32(output), Shape::from_dims(&dims)))
    }
}
