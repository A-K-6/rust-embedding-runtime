// Regression contract for optional weight quantization.
//
// Proposed interface (explicit modes, no mutable process environment):
//   enum QuantMode { F32 (default), Q8_0, Q4_K }
//   QuantMode::parse(&str) -> anyhow::Result<QuantMode>  (exact, case-insensitive)
//   QuantMode::name(self) -> &'static str
//   struct QuantProjection
//   QuantProjection::new(&Tensor /* (out, in) f32 */, QuantMode) -> candle_core::Result<Self>
//   QuantProjection::{mode, is_quantized, storage_bytes, forward}
//
// `forward` must accept 2D (tokens, in) and 3D (batch, tokens, in) float32 inputs.
// Quantized modes must hold real Candle QTensor/QMatMul storage, not dequantized floats.
use super::*;
use candle_core::quantized::{GgmlDType, QTensor};

const IN: usize = 512;
const OUT: usize = 96;
const TOKENS: usize = 5;

// Deterministic fixture: no RNG dependency, stable across runs and platforms.
fn fixture(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..rows * cols)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 33) as u32) as f32 / (u32::MAX >> 1) as f32;
            unit * 2. - 1.
        })
        .collect()
}

fn weight_tensor() -> Tensor {
    Tensor::from_vec(fixture(OUT, IN, 0x51ed_2701), (OUT, IN), &Device::Cpu).unwrap()
}

fn input_2d() -> Tensor {
    Tensor::from_vec(fixture(TOKENS, IN, 0xc0ff_ee11), (TOKENS, IN), &Device::Cpu).unwrap()
}

fn input_3d() -> Tensor {
    input_2d().reshape((1, TOKENS, IN)).unwrap()
}

// Float reference: same math as a dequantized weight, computed by Candle matmul.
fn float_reference(x: &Tensor) -> Tensor {
    let w = weight_tensor();
    let mut output_shape = x.dims().to_vec();
    *output_shape.last_mut().unwrap() = OUT;
    x.reshape((x.elem_count() / IN, IN))
        .unwrap()
        .matmul(&w.t().unwrap().contiguous().unwrap())
        .unwrap()
        .reshape(output_shape)
        .unwrap()
}

fn relative_error(actual: &[f32], expected: &[f32]) -> f32 {
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e) * (a - e))
        .sum::<f32>()
        .sqrt();
    let reference = expected.iter().map(|e| e * e).sum::<f32>().sqrt();
    error / reference.max(1e-12)
}

#[test]
fn default_mode_is_float_and_keeps_exact_reference_behavior() {
    assert_eq!(QuantMode::default(), QuantMode::F32);
    assert_eq!(QuantMode::default().name(), "f32");
    assert_eq!(QuantMode::F32.name(), "f32");

    let weight = weight_tensor();
    let projection = QuantProjection::new(&weight, QuantMode::F32).unwrap();
    assert_eq!(projection.mode(), QuantMode::F32);
    assert!(
        !projection.is_quantized(),
        "F32 mode must keep an f32 weight tensor"
    );
    assert_eq!(
        projection.storage_bytes(),
        OUT * IN * 4,
        "F32 mode must store one f32 per weight"
    );

    let x = input_2d();
    let y = projection.forward(&x).unwrap();
    assert_eq!(y.dims(), &[TOKENS, OUT]);
    assert_eq!(y.dtype(), DType::F32);
    let delta = relative_error(
        &y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &float_reference(&x)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
    );
    assert!(
        delta < 1e-5,
        "default f32 projection must match the float reference, relative error {delta}"
    );
}

#[test]
fn unknown_cli_quant_modes_are_rejected() {
    assert_eq!(QuantMode::parse("f32").unwrap(), QuantMode::F32);
    assert_eq!(QuantMode::parse("q8_0").unwrap(), QuantMode::Q8_0);
    assert_eq!(QuantMode::parse("q4_k").unwrap(), QuantMode::Q4_K);
    // Case-insensitive, but no trimming and no silent default.
    assert_eq!(QuantMode::parse("Q8_0").unwrap(), QuantMode::Q8_0);
    assert_eq!(QuantMode::parse("Q4_K").unwrap(), QuantMode::Q4_K);
    assert_eq!(QuantMode::parse("F32").unwrap(), QuantMode::F32);
    for mode in [QuantMode::F32, QuantMode::Q8_0, QuantMode::Q4_K] {
        assert_eq!(QuantMode::parse(mode.name()).unwrap(), mode);
    }

    for rejected in [
        "", " f32", "f32 ", "bf16", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_1", "q2_k", "q6_k",
        "auto", "none", "true", "1",
    ] {
        assert!(
            QuantMode::parse(rejected).is_err(),
            "unknown quant mode {rejected:?} must be rejected"
        );
    }
    for hint in ["q4_0", "auto"] {
        let message = QuantMode::parse(hint).unwrap_err().to_string();
        assert!(
            message.contains(hint),
            "error for {hint:?} should name the rejected value, got {message:?}"
        );
    }
}

#[test]
fn quantized_modes_use_real_packed_qtensor_storage() {
    let weight = weight_tensor();
    let f32_bytes = OUT * IN * 4;

    // Independent oracle from the vendored Candle QTensor implementation.
    let oracle = |dtype: GgmlDType| {
        QTensor::quantize(&weight, dtype)
            .unwrap()
            .storage_size_in_bytes()
    };
    let q8_bytes = oracle(GgmlDType::Q8_0);
    let q4k_bytes = oracle(GgmlDType::Q4K);

    let q8 = QuantProjection::new(&weight, QuantMode::Q8_0).unwrap();
    assert_eq!(q8.mode(), QuantMode::Q8_0);
    assert!(q8.is_quantized(), "Q8_0 mode must hold a QTensor");
    assert_eq!(
        q8.storage_bytes(),
        q8_bytes,
        "Q8_0 mode must report the packed QTensor byte size"
    );
    assert!(
        q8.storage_bytes() * 3 < f32_bytes,
        "Q8_0 weights ({}) must be packed well below f32 ({f32_bytes})",
        q8.storage_bytes()
    );

    let q4k = QuantProjection::new(&weight, QuantMode::Q4_K).unwrap();
    assert_eq!(q4k.mode(), QuantMode::Q4_K);
    assert!(q4k.is_quantized(), "Q4_K mode must hold a QTensor");
    assert_eq!(
        q4k.storage_bytes(),
        q4k_bytes,
        "Q4_K mode must report the packed QTensor byte size"
    );
    assert!(
        q4k.storage_bytes() * 5 < f32_bytes,
        "Q4_K weights ({}) must be packed well below f32 ({f32_bytes})",
        q4k.storage_bytes()
    );
    assert!(
        q4k.storage_bytes() < q8.storage_bytes(),
        "Q4_K ({}) must be smaller than Q8_0 ({})",
        q4k.storage_bytes(),
        q8.storage_bytes()
    );
}

#[test]
fn batch_forward_accepts_2d_and_3d_and_tracks_float_reference() {
    let weight = weight_tensor();
    let x2 = input_2d();
    let x3 = input_3d();
    let reference_2d = float_reference(&x2)
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let reference_3d = float_reference(&x3)
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    for (mode, tolerance) in [
        (QuantMode::F32, 1e-5f32),
        (QuantMode::Q8_0, 0.03),
        (QuantMode::Q4_K, 0.10),
    ] {
        let projection = QuantProjection::new(&weight, mode).unwrap();

        let y2 = projection.forward(&x2).unwrap();
        assert_eq!(y2.dims(), &[TOKENS, OUT], "{mode:?} 2D output shape");
        assert_eq!(y2.dtype(), DType::F32, "{mode:?} activations stay float32");
        let values_2d = y2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            values_2d.iter().all(|v| v.is_finite()),
            "{mode:?} 2D output must stay finite"
        );
        let error_2d = relative_error(&values_2d, &reference_2d);
        assert!(
            error_2d < tolerance,
            "{mode:?} 2D relative error {error_2d} exceeds {tolerance}"
        );

        let y3 = projection.forward(&x3).unwrap();
        assert_eq!(y3.dims(), &[1, TOKENS, OUT], "{mode:?} 3D output shape");
        assert_eq!(y3.dtype(), DType::F32, "{mode:?} activations stay float32");
        let values_3d = y3.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            values_3d.iter().all(|v| v.is_finite()),
            "{mode:?} 3D output must stay finite"
        );
        let error_3d = relative_error(&values_3d, &reference_3d);
        assert!(
            error_3d < tolerance,
            "{mode:?} 3D relative error {error_3d} exceeds {tolerance}"
        );

        let batch_delta = relative_error(&values_3d, &values_2d);
        assert!(
            batch_delta < 1e-6,
            "{mode:?} batched and 2D forward must agree, relative error {batch_delta}"
        );
    }
}
