//! Comparison math. Pure `f32`/`f64` Rust; no GPU, no FFI.
//!
//! `mse` is the standard mean-squared-error between two flat tensors of
//! equal length. Returns an `f64` so we never underflow on very small
//! activations.
//!
//! `kl_divergence` is numerically stable: both inputs are unnormalized
//! logits, we compute `log_softmax` of each (with the max-shift trick) and
//! sum `p * (log_p - log_q)`. Naive `log(p/q)` would produce `NaN` or
//! `±inf` on near-zero probabilities; this form does not.

use crate::error::ParityError;

/// Mean squared error between two tensors of equal length. Returns
/// `Err(ShapeMismatch)` if lengths differ.
pub fn mse(a: &[f32], b: &[f32]) -> Result<f64, ParityError> {
    if a.len() != b.len() {
        return Err(ParityError::ShapeMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Ok(0.0);
    }
    let sum_sq: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum();
    Ok(sum_sq / a.len() as f64)
}

/// KL divergence `KL(p || q)` between two distributions derived from
/// `p_logits` and `q_logits`. Both inputs are unnormalized logits; we
/// soft-max each (via the max-shift trick) and sum `p * (log_p - log_q)`.
///
/// Returns `0.0` for identical inputs and a strictly positive value when
/// the distributions differ. Stable for logits in `[-1e6, 1e6]`.
pub fn kl_divergence(p_logits: &[f32], q_logits: &[f32]) -> Result<f64, ParityError> {
    if p_logits.len() != q_logits.len() {
        return Err(ParityError::ShapeMismatch {
            expected: p_logits.len(),
            got: q_logits.len(),
        });
    }
    if p_logits.is_empty() {
        return Ok(0.0);
    }
    let p_log = log_softmax(p_logits);
    let q_log = log_softmax(q_logits);
    let kl: f64 = p_log
        .iter()
        .zip(q_log.iter())
        .map(|(lp, lq)| {
            // p = exp(lp) computed in f64 for accuracy.
            let p = (*lp as f64).exp();
            p * ((*lp as f64) - (*lq as f64))
        })
        .sum();
    // Tiny negative values can arise from float roundoff when `p ≈ q`.
    // Clamp at zero rather than letting a -1e-16 propagate.
    Ok(kl.max(0.0))
}

/// `log_softmax(x) = x - max(x) - log_sum_exp(x - max(x))`. Returns `f32`
/// so the downstream `kl_divergence` doesn't pay double the work on the
/// hot path; the f64 cast happens at the final dot product.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // All `-inf` or empty — return as-is so callers see the degenerate
        // case rather than silently fabricating a uniform distribution.
        return logits.to_vec();
    }
    let shifted: Vec<f32> = logits.iter().map(|x| x - max).collect();
    let log_sum_exp: f32 = shifted.iter().map(|x| x.exp()).sum::<f32>().ln();
    shifted.iter().map(|x| x - log_sum_exp).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_normalises_to_one() {
        let logits = [0.5f32, -1.0, 3.0, 2.0];
        let lp = log_softmax(&logits);
        let sum_p: f64 = lp.iter().map(|x| (*x as f64).exp()).sum();
        assert!((sum_p - 1.0).abs() < 1e-6, "softmax sum = {sum_p}");
    }
}
