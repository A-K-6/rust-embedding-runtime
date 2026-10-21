use super::*;
use std::{sync::Arc, thread};

fn values(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 11 + salt) % 37) as f32 - 18.0) / 32.0)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite(), "non-finite output at {index}");
        assert!(
            (a - e).abs() <= 0.00002 + e.abs() * 0.00002,
            "output mismatch at {index}: {a} vs {e}"
        );
    }
}

fn reference(input: &[f32], weights: &[f32], rows: usize, n: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0; rows * n];
    for i in 0..rows {
        for j in 0..n {
            let mut sum = 0.0_f64;
            for p in 0..k {
                sum += f64::from(input[i * k + p]) * f64::from(weights[j * k + p]);
            }
            output[i * n + j] = sum as f32;
        }
    }
    output
}

fn require_backend() -> bool {
    if available() {
        true
    } else {
        eprintln!("AOCL numeric checks skipped: backend/AVX2/FMA unavailable");
        false
    }
}

#[test]
fn validates_shapes_lengths_threads_and_overflows_before_ffi() {
    for (weights, n, k, threads) in [
        (&[][..], 0, 19, 2),
        (&[][..], 23, 0, 2),
        (&[][..], usize::MAX, 19, 2),
        (&[][..], 23, usize::MAX, 2),
        (&[][..], 32_769, 1, 2),
        (&[][..], 1, 32_769, 2),
        (&[][..], 1, 1, 2),
        (&[0.0][..], 1, 1, 0),
        (&[0.0][..], 1, 1, 65),
        (&[0.0][..], 1, 1, usize::MAX),
    ] {
        assert!(matches!(
            PackedLinear::new(weights, n, k, threads),
            Err(Error::InvalidInput(_))
        ));
    }
    assert!(checked_elements(usize::MAX, 2).is_err());
    assert!(checked_elements(usize::MAX / 4, 2).is_err());
    assert!(checked_elements(isize::MAX as usize / 4 + 1, 1).is_err());
}

#[test]
fn persistent_reordered_weights_match_naive_for_all_supported_row_classes() {
    if !require_backend() {
        return;
    }
    // Neither dimension is a kernel packing width multiple.
    let (n, k) = (23, 19);
    let weights = values(n * k, 3);
    let linear = PackedLinear::new(&weights, n, k, 2).unwrap();
    assert_eq!(linear.input_size(), k);
    assert_eq!(linear.output_size(), n);
    assert!(linear.storage_bytes() >= weights.len() * size_of::<f32>());
    for rows in [1, 7, 16, 24, 128, 512, 1024] {
        let input = values(rows * k, 7);
        let expected = reference(&input, &weights, rows, n, k);
        let actual = linear.forward(&input, rows).unwrap();
        assert_close(&actual, &expected);
    }
}

#[test]
fn one_element_and_odd_dimensions_have_correct_orientation() {
    if !require_backend() {
        return;
    }
    for (n, k) in [(1, 1), (1, 7), (7, 1), (5, 3), (67, 13)] {
        let weights = values(n * k, 13);
        let linear = PackedLinear::new(&weights, n, k, 2).unwrap();
        for rows in [1, 7] {
            let input = values(rows * k, 17);
            assert_close(
                &linear.forward(&input, rows).unwrap(),
                &reference(&input, &weights, rows, n, k),
            );
        }
    }
}

#[test]
fn non_dyadic_values_cross_packing_panel_boundaries() {
    if !require_backend() {
        return;
    }
    let (n, k, rows) = (67, 513, 24);
    let weights: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32 * 0.13).sin() - 0.1) * 0.07)
        .collect();
    let input: Vec<f32> = (0..rows * k)
        .map(|i| ((i as f32 * 0.11).cos() + 0.2) * 0.09)
        .collect();
    let linear = PackedLinear::new(&weights, n, k, 2).unwrap();
    assert_close(
        &linear.forward(&input, rows).unwrap(),
        &reference(&input, &weights, rows, n, k),
    );
}

#[test]
fn source_weight_storage_can_be_dropped_after_construction() {
    if !require_backend() {
        return;
    }
    let linear = {
        let weights = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        PackedLinear::new(&weights, 2, 3, 2).unwrap()
    };
    assert_close(&linear.forward(&[1.0, 2.0, 3.0], 1).unwrap(), &[6.0, -12.0]);
}

#[test]
fn rejects_invalid_forward_without_disturbing_handle() {
    if !require_backend() {
        return;
    }
    let linear = PackedLinear::new(&[1.0, -2.0, 3.0], 1, 3, 2).unwrap();
    for (input, rows) in [
        (&[][..], 0),
        (&[][..], 1025),
        (&[][..], usize::MAX),
        (&[][..], 1),
        (&[1.0, 2.0][..], 1),
        (&[1.0, 2.0, 3.0, 4.0][..], 1),
    ] {
        assert!(matches!(
            linear.forward(input, rows),
            Err(Error::InvalidInput(_))
        ));
    }
    assert_close(&linear.forward(&[1.0, 2.0, 3.0], 1).unwrap(), &[6.0]);
}

#[test]
fn independent_handles_and_same_handle_work_from_different_calling_threads() {
    if !require_backend() {
        return;
    }
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PackedLinear>();
    let (n, k, rows) = (23, 19, 24);
    let weights = values(n * k, 3);
    let input = Arc::new(values(rows * k, 7));
    let expected = Arc::new(reference(&input, &weights, rows, n, k));
    let shared = Arc::new(PackedLinear::new(&weights, n, k, 2).unwrap());
    let mut jobs = Vec::new();
    for salt in 0..4 {
        let shared = Arc::clone(&shared);
        let input = Arc::clone(&input);
        let expected = Arc::clone(&expected);
        let weights = weights.clone();
        jobs.push(thread::spawn(move || {
            let local =
                PackedLinear::new(&weights, n, k, if salt % 2 == 0 { 1 } else { 2 }).unwrap();
            for _ in 0..3 {
                assert_close(&local.forward(&input, rows).unwrap(), &expected);
                assert_close(&shared.forward(&input, rows).unwrap(), &expected);
            }
        }));
    }
    for job in jobs {
        job.join().unwrap();
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
#[test]
fn other_targets_offer_explicit_unavailable_fallback() {
    assert!(!available());
    assert!(matches!(
        PackedLinear::new(&[1.0], 1, 1, 2),
        Err(Error::Unsupported)
    ));
}
