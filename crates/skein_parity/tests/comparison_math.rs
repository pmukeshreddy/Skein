//! Tests 1 + 2 — comparison math.

use skein_parity::ParityError;
use skein_parity::comparison::{kl_divergence, mse};

#[test]
fn mse_basic_correctness() {
    // Identical inputs → zero.
    assert_eq!(mse(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap(), 0.0);

    // Per-element diff of 1.0 each → MSE = 1.0.
    assert_eq!(mse(&[1.0, 2.0], &[2.0, 3.0]).unwrap(), 1.0);

    // Per-element diff of 2.0 each → MSE = 4.0.
    assert_eq!(mse(&[0.0, 0.0, 0.0], &[2.0, 2.0, 2.0]).unwrap(), 4.0);

    // Empty inputs trivially MSE = 0.
    assert_eq!(mse(&[], &[]).unwrap(), 0.0);

    // Shape mismatch errors out.
    match mse(&[1.0, 2.0], &[1.0]) {
        Err(ParityError::ShapeMismatch {
            expected: 2,
            got: 1,
        }) => {}
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn kl_divergence_basic_correctness() {
    let p = [0.5f32, -1.0, 3.0, 2.0, -0.5];

    // KL(p || p) = 0 exactly (modulo float roundoff).
    let kl_self = kl_divergence(&p, &p).unwrap();
    assert!(
        kl_self.abs() < 1e-10,
        "KL(p || p) should be ~0, got {kl_self}"
    );

    // KL(p || q) > 0 for distinct distributions.
    let q = [0.0f32, 0.0, 0.0, 0.0, 0.0];
    let kl_pq = kl_divergence(&p, &q).unwrap();
    assert!(kl_pq > 0.0, "KL(p || uniform) should be > 0, got {kl_pq}");
}

#[test]
fn kl_divergence_numerically_stable_at_extreme_logits() {
    // Logits at ±1e6 would overflow `exp` without the max-shift trick.
    let p = vec![1.0e6_f32, -1.0e6_f32, 0.0_f32, 0.0_f32];
    let q = vec![-1.0e6_f32, 1.0e6_f32, 0.0_f32, 0.0_f32];

    let kl = kl_divergence(&p, &q).unwrap();
    assert!(kl.is_finite(), "expected finite KL, got {kl}");
    assert!(kl > 0.0, "expected positive KL on extreme distinct logits");

    // Self-KL still zero at extreme magnitude.
    let kl_self = kl_divergence(&p, &p).unwrap();
    assert!(kl_self.abs() < 1e-9);
}

#[test]
fn kl_divergence_shape_mismatch_errors() {
    let r = kl_divergence(&[1.0, 2.0], &[1.0]);
    assert!(matches!(r, Err(ParityError::ShapeMismatch { .. })));
}
