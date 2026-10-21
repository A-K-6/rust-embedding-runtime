use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

struct GeluMul;

impl CustomOp2 for GeluMul {
    fn name(&self) -> &'static str {
        "gelu-tanh-mul"
    }

    fn cpu_fwd(
        &self,
        a: &CpuStorage,
        al: &Layout,
        b: &CpuStorage,
        bl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(a), CpuStorage::F32(b)) = (a, b) else {
            candle_core::bail!("gelu-mul requires float32 inputs")
        };
        if al.shape() != bl.shape() {
            candle_core::bail!("gelu-mul requires identical shapes")
        }
        let Some((a0, a1)) = al.contiguous_offsets() else {
            candle_core::bail!("gelu-mul requires contiguous inputs")
        };
        let Some((b0, b1)) = bl.contiguous_offsets() else {
            candle_core::bail!("gelu-mul requires contiguous inputs")
        };
        let a = &a[a0..a1];
        let b = &b[b0..b1];
        let mut out = vec![0f32; a.len()];
        // Same tanh GELU as Candle/PyTorch. Parallelize independent chunks and
        // combine the multiply, avoiding an intermediate activation tensor.
        out.par_chunks_mut(8192)
            .zip(a.par_chunks(8192))
            .zip(b.par_chunks(8192))
            .for_each(|((out, a), b)| {
                for ((out, &x), &y) in out.iter_mut().zip(a).zip(b) {
                    *out = 0.5
                        * x
                        * (1.0
                            + (std::f32::consts::FRAC_2_PI.sqrt() * x * (1.0 + 0.044715 * x * x))
                                .tanh())
                        * y;
                }
            });
        Ok((CpuStorage::F32(out), al.shape().clone()))
    }
}

pub(crate) fn gelu_mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let _span = crate::profile::span("activation");
    // Accelerate already provides vectorized tanh on native macOS.
    if candle_core::utils::has_accelerate() {
        a.gelu()?.mul(b)
    } else {
        a.contiguous()?.apply_op2_no_bwd(&b.contiguous()?, &GeluMul)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn fused_gelu_multiply_matches_reference_across_chunks_and_offsets() {
        let a = Tensor::from_vec(
            (0..20001).map(|i| i as f32 * 0.002 - 20.0).collect(),
            20001,
            &Device::Cpu,
        )
        .unwrap();
        let b = (&a * -0.37).unwrap();
        let a = a.narrow(0, 1, 19999).unwrap();
        let b = b.narrow(0, 1, 19999).unwrap();
        let expected = a.gelu().unwrap().mul(&b).unwrap().to_vec1::<f32>().unwrap();
        let actual = a
            .apply_op2_no_bwd(&b, &GeluMul)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in actual.into_iter().zip(expected) {
            assert!((a - b).abs() < 2e-5, "activation drift: {a} vs {b}");
        }
    }
}
