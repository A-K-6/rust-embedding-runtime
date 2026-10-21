use super::*;
use candle_core::D;

#[test]
fn f32_projection_preserves_strided_batches_and_tail_dimensions() {
    let device = Device::Cpu;
    let weights = Tensor::from_vec(
        (0..17 * 19)
            .map(|i| (i as f32 * 0.13).sin())
            .collect::<Vec<_>>(),
        (17, 19),
        &device,
    )
    .unwrap();
    let input = Tensor::from_vec(
        (0..3 * 7 * 19)
            .map(|i| (i as f32 * 0.07).cos())
            .collect::<Vec<_>>(),
        (3, 7, 19),
        &device,
    )
    .unwrap()
    .transpose(0, 1)
    .unwrap();
    let projection = QuantProjection::new(&weights, QuantMode::F32).unwrap();
    let expected = input
        .contiguous()
        .unwrap()
        .reshape((21, 19))
        .unwrap()
        .matmul(&weights.t().unwrap())
        .unwrap()
        .reshape((7, 3, 17))
        .unwrap();
    let actual = projection.forward(&input).unwrap();
    assert_eq!(actual.dims(), &[7, 3, 17]);
    for (a, b) in actual
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .zip(expected.flatten_all().unwrap().to_vec1::<f32>().unwrap())
    {
        assert!((a - b).abs() < 2e-5, "projection drift: {a} vs {b}");
    }
}

#[test]
fn fused_rms_norm_preserves_unfused_math_on_strided_inputs() {
    let device = Device::Cpu;
    let input = Tensor::from_vec(
        (0..2 * 3 * 256)
            .map(|i| (i as f32 * 0.17).sin() * 3.0)
            .collect::<Vec<_>>(),
        (2, 3, 256),
        &device,
    )
    .unwrap();
    let strided = input.transpose(0, 1).unwrap();
    for x in [&input, &strided] {
        for scaled in [true, false] {
            let alpha = if scaled {
                Tensor::from_vec(
                    (0..256)
                        .map(|i| 0.5 + (i % 7) as f32 * 0.1)
                        .collect::<Vec<_>>(),
                    256,
                    &device,
                )
                .unwrap()
            } else {
                Tensor::ones(256, DType::F32, &device).unwrap()
            };
            let mut weights = HashMap::new();
            weights.insert("weight".to_owned(), alpha.clone());
            let norm = Norm::new(
                VarBuilder::from_tensors(weights, DType::F32, &device),
                256,
                scaled,
            )
            .unwrap();
            let expected = x
                .broadcast_div(
                    &(x.sqr().unwrap().mean_keepdim(D::Minus1).unwrap() + 1e-6f64)
                        .unwrap()
                        .sqrt()
                        .unwrap(),
                )
                .unwrap()
                .broadcast_mul(&alpha)
                .unwrap();
            let actual = norm.forward(x).unwrap();
            let max = actual
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .zip(expected.flatten_all().unwrap().to_vec1::<f32>().unwrap())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max < 2e-5, "RMS normalization drift: {max}");
        }
    }
}

#[test]
fn fused_rotary_preserves_split_half_rotation() {
    for hd in [256usize, 512] {
        let n = 7;
        let x = Tensor::from_vec(
            (0..4 * n * hd)
                .map(|i| (i as f32 * 0.031).sin())
                .collect::<Vec<_>>(),
            (1, 4, n, hd),
            &Device::Cpu,
        )
        .unwrap();
        let angles: Vec<f32> = (0..n)
            .flat_map(|p| {
                (0..hd / 2).map(move |d| p as f32 * 10000f32.powf(-((2 * d) as f32) / hd as f32))
            })
            .collect();
        let cos = Tensor::from_vec(
            angles.iter().map(|a| a.cos()).collect::<Vec<_>>(),
            (n, hd / 2),
            &Device::Cpu,
        )
        .unwrap();
        let sin = Tensor::from_vec(
            angles.iter().map(|a| a.sin()).collect::<Vec<_>>(),
            (n, hd / 2),
            &Device::Cpu,
        )
        .unwrap();
        let duplicated_cos = Tensor::cat(&[&cos, &cos], 1)
            .unwrap()
            .reshape((1, 1, n, hd))
            .unwrap();
        let duplicated_sin = Tensor::cat(&[&sin, &sin], 1)
            .unwrap()
            .reshape((1, 1, n, hd))
            .unwrap();
        let rotated = Tensor::cat(
            &[
                &x.narrow(3, hd / 2, hd / 2).unwrap().neg().unwrap(),
                &x.narrow(3, 0, hd / 2).unwrap(),
            ],
            3,
        )
        .unwrap();
        let expected = x
            .broadcast_mul(&duplicated_cos)
            .unwrap()
            .add(&rotated.broadcast_mul(&duplicated_sin).unwrap())
            .unwrap();
        let actual = candle_nn::rotary_emb::rope(&x, &cos, &sin).unwrap();
        let max = actual
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .zip(expected.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max < 2e-6, "Rotary drift: {max}");
    }
}
