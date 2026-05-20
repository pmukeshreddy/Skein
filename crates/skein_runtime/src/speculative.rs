//! Speculative-sampling verification (Leviathan et al., 2023 / Chen et al.).
//!
//! A draft model proposes `k` tokens, each with its draft distribution `q`; the
//! target model is then run once over all `k` positions (plus one) to give the
//! target distributions `p`. [`verify`] applies the exact speculative-sampling
//! accept/reject rule that makes the accepted output *distributionally
//! identical* to sampling from the target alone:
//!
//! - token `i` (drawn from `q_i`) is accepted with probability `min(1,
//!   p_i(x_i) / q_i(x_i))`;
//! - on the first rejection, one *correction* token is resampled from the
//!   residual `norm(max(0, p_i - q_i))` and the rest of the draft is discarded;
//! - if all `k` are accepted, one *bonus* token is sampled from `p_k`.
//!
//! So a step emits between 1 and `k+1` tokens. This is the pure verification
//! core: the draft/target forward passes (the GPU part) supply `draft_probs`
//! and `target_probs`; the sampler and the accept/reject coin flips are
//! injected so the rule is deterministic and unit-testable without a GPU.

/// Result of verifying one draft block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecOutcome {
    /// Draft tokens accepted, in order (length `n_accepted`).
    pub accepted: Vec<u32>,
    /// The correction token: resampled from the residual on a rejection, or the
    /// bonus token sampled from the target when all draft tokens were accepted.
    pub correction: u32,
    /// How many draft tokens were accepted (`0..=k`).
    pub n_accepted: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SpecError {
    #[error("draft has {tokens} tokens but {draft_probs} draft / {target_probs} target distributions (need k and k+1)")]
    ShapeMismatch {
        tokens: usize,
        draft_probs: usize,
        target_probs: usize,
    },
    #[error("need {tokens} accept coin flips, got {uniforms}")]
    NotEnoughUniforms { tokens: usize, uniforms: usize },
    #[error("token id {token} is out of range for a {vocab}-way distribution")]
    TokenOutOfRange { token: u32, vocab: usize },
}

/// Verify a draft block of `k` tokens.
///
/// - `draft_tokens`: the `k` proposed token ids.
/// - `draft_probs[i]`: the draft distribution token `i` was sampled from.
/// - `target_probs[i]`: the target distribution at position `i`, for
///   `i in 0..=k` (so `k+1` entries; `target_probs[k]` is the bonus position).
/// - `uniforms[i]`: a uniform sample in `[0,1)` for token `i`'s accept test.
/// - `sample`: draws a token id from a (normalized) distribution; injected so
///   tests are deterministic (e.g. argmax) and production can plug RNG sampling.
pub fn verify<F>(
    draft_tokens: &[u32],
    draft_probs: &[Vec<f32>],
    target_probs: &[Vec<f32>],
    uniforms: &[f32],
    sample: F,
) -> Result<SpecOutcome, SpecError>
where
    F: Fn(&[f32]) -> u32,
{
    let k = draft_tokens.len();
    if draft_probs.len() != k || target_probs.len() != k + 1 {
        return Err(SpecError::ShapeMismatch {
            tokens: k,
            draft_probs: draft_probs.len(),
            target_probs: target_probs.len(),
        });
    }
    if uniforms.len() < k {
        return Err(SpecError::NotEnoughUniforms {
            tokens: k,
            uniforms: uniforms.len(),
        });
    }

    let mut accepted = Vec::with_capacity(k);
    for i in 0..k {
        let x = draft_tokens[i] as usize;
        let p = &target_probs[i];
        let q = &draft_probs[i];
        if x >= p.len() || x >= q.len() {
            return Err(SpecError::TokenOutOfRange {
                token: draft_tokens[i],
                vocab: p.len().min(q.len()),
            });
        }
        // Accept with prob min(1, p(x)/q(x)). q(x)==0 (token the draft couldn't
        // have produced) → always accept (ratio treated as 1).
        let ratio = if q[x] > 0.0 {
            (p[x] / q[x]).min(1.0)
        } else {
            1.0
        };
        if uniforms[i] <= ratio {
            accepted.push(draft_tokens[i]);
        } else {
            // Reject: resample the correction from the residual max(0, p - q).
            let residual = residual_distribution(p, q);
            let correction = sample(&residual);
            return Ok(SpecOutcome {
                n_accepted: accepted.len(),
                accepted,
                correction,
            });
        }
    }
    // All k accepted: the bonus token comes from the target at position k.
    let correction = sample(&normalize(&target_probs[k]));
    Ok(SpecOutcome {
        n_accepted: k,
        accepted,
        correction,
    })
}

/// `norm(max(0, p - q))`. If the residual is all-zero (p == q), fall back to the
/// normalized target so a token is always producible.
fn residual_distribution(p: &[f32], q: &[f32]) -> Vec<f32> {
    let mut r: Vec<f32> = p
        .iter()
        .zip(q.iter())
        .map(|(pi, qi)| (pi - qi).max(0.0))
        .collect();
    let sum: f32 = r.iter().sum();
    if sum <= 0.0 {
        return normalize(p);
    }
    for v in &mut r {
        *v /= sum;
    }
    r
}

fn normalize(p: &[f32]) -> Vec<f32> {
    let sum: f32 = p.iter().sum();
    if sum <= 0.0 {
        return p.to_vec();
    }
    p.iter().map(|v| v / sum).collect()
}

/// argmax sampler — the deterministic (greedy) choice; used by tests and as a
/// default for greedy speculative decoding.
pub fn argmax(dist: &[f32]) -> u32 {
    dist.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Numerically-stable softmax (max-shift). Turns model logits into the
/// distributions [`verify`] consumes.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Full speculative-decode generation loop: a draft model proposes `k` tokens,
/// the target verifies them in one batch, and the accepted prefix + one
/// correction token advance the sequence — repeating until `max_new_tokens`.
///
/// `target_forward` / `draft_forward` map the running token sequence to that
/// model's next-token **logits** (the runtime supplies the real model forwards;
/// tests supply mocks). `sample` draws from a distribution (argmax for greedy);
/// `uniforms` supplies the accept/reject coin flips (injected for determinism).
/// Returns exactly `max_new_tokens` generated ids (it may overshoot internally
/// then truncate).
pub fn generate<TF, DF, S, U>(
    prompt: &[u32],
    max_new_tokens: usize,
    k: usize,
    target_forward: &mut TF,
    draft_forward: &mut DF,
    sample: S,
    uniforms: &mut U,
) -> Result<Vec<u32>, SpecError>
where
    TF: FnMut(&[u32]) -> Vec<f32>,
    DF: FnMut(&[u32]) -> Vec<f32>,
    S: Fn(&[f32]) -> u32 + Copy,
    U: FnMut() -> f32,
{
    let mut seq = prompt.to_vec();
    let mut out = Vec::with_capacity(max_new_tokens);

    while out.len() < max_new_tokens && k > 0 {
        // 1. Draft proposes `k` tokens, recording the distribution each was
        //    drawn from.
        let mut draft_tokens = Vec::with_capacity(k);
        let mut draft_probs = Vec::with_capacity(k);
        let mut scratch = seq.clone();
        for _ in 0..k {
            let probs = softmax(&draft_forward(&scratch));
            let tok = sample(&probs);
            draft_tokens.push(tok);
            draft_probs.push(probs);
            scratch.push(tok);
        }

        // 2. Target distributions at the `k + 1` positions over the proposed
        //    prefix.
        let mut target_probs = Vec::with_capacity(k + 1);
        let mut tseq = seq.clone();
        for i in 0..=k {
            target_probs.push(softmax(&target_forward(&tseq)));
            if i < k {
                tseq.push(draft_tokens[i]);
            }
        }

        // 3. Verify (exact speculative sampling).
        let unis: Vec<f32> = (0..k).map(|_| uniforms()).collect();
        let outcome = verify(&draft_tokens, &draft_probs, &target_probs, &unis, sample)?;

        // 4. Commit the accepted prefix + the one correction token.
        for &t in outcome.accepted.iter().chain(std::iter::once(&outcome.correction)) {
            seq.push(t);
            out.push(t);
            if out.len() >= max_new_tokens {
                break;
            }
        }
    }
    out.truncate(max_new_tokens);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_accepted_yields_bonus_token() {
        // Draft proposes [0, 1]; target strongly agrees, so both accept; the
        // bonus comes from target_probs[2].
        let draft_tokens = vec![0u32, 1];
        let draft_probs = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        let target_probs = vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.2, 0.8]];
        let uniforms = vec![0.0, 0.0]; // always accept
        let out = verify(&draft_tokens, &draft_probs, &target_probs, &uniforms, argmax).unwrap();
        assert_eq!(out.accepted, vec![0, 1]);
        assert_eq!(out.n_accepted, 2);
        assert_eq!(out.correction, 1); // argmax of [0.2,0.8]
    }

    #[test]
    fn rejection_resamples_from_residual_and_truncates() {
        // Token 0 proposed with high draft prob but low target prob → reject.
        let draft_tokens = vec![0u32, 0];
        let draft_probs = vec![vec![0.9, 0.1], vec![0.9, 0.1]];
        // target favors token 1 at position 0 → p(0)/q(0)=0.1/0.9 ≈ 0.11.
        let target_probs = vec![vec![0.1, 0.9], vec![0.5, 0.5], vec![0.5, 0.5]];
        let uniforms = vec![0.5, 0.0]; // 0.5 > 0.11 → reject at i=0
        let out = verify(&draft_tokens, &draft_probs, &target_probs, &uniforms, argmax).unwrap();
        assert_eq!(out.n_accepted, 0);
        assert!(out.accepted.is_empty());
        // residual = norm(max(0,[0.1,0.9]-[0.9,0.1])) = norm([0,0.8]) = [0,1] → argmax 1
        assert_eq!(out.correction, 1);
    }

    #[test]
    fn partial_acceptance() {
        // Accept first, reject second.
        let draft_tokens = vec![1u32, 0];
        let draft_probs = vec![vec![0.1, 0.9], vec![0.9, 0.1]];
        let target_probs = vec![vec![0.1, 0.9], vec![0.1, 0.9], vec![0.5, 0.5]];
        // i=0: p(1)/q(1)=0.9/0.9=1 → accept (u=0.3<=1).
        // i=1: p(0)/q(0)=0.1/0.9≈0.11 → reject (u=0.9>0.11).
        let uniforms = vec![0.3, 0.9];
        let out = verify(&draft_tokens, &draft_probs, &target_probs, &uniforms, argmax).unwrap();
        assert_eq!(out.accepted, vec![1]);
        assert_eq!(out.n_accepted, 1);
        // residual at i=1 = norm(max(0,[0.1,0.9]-[0.9,0.1])) = [0,1] → 1
        assert_eq!(out.correction, 1);
    }

    #[test]
    fn shape_mismatch_is_an_error() {
        let err = verify(&[0], &[vec![1.0]], &[vec![1.0]], &[0.0], argmax).unwrap_err();
        assert!(matches!(err, SpecError::ShapeMismatch { .. }));
    }

    #[test]
    fn softmax_normalizes() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn generate_with_agreeing_models_reaches_target_length() {
        // Both models favor token `(seq.len() % vocab)`. Draft == target, so
        // every proposed token is accepted and the loop advances by k+1 each
        // round; the result is exactly `max_new_tokens` tokens.
        let vocab = 4usize;
        let fwd = |seq: &[u32]| {
            let mut logits = vec![0.0f32; vocab];
            logits[seq.len() % vocab] = 10.0;
            logits
        };
        let mut target = fwd;
        let mut draft = fwd;
        let mut accept_all = || 0.0f32; // u <= ratio → always accept
        let out = generate(&[0], 5, 2, &mut target, &mut draft, argmax, &mut accept_all)
            .expect("generate");
        assert_eq!(out.len(), 5, "got {out:?}");
    }

    #[test]
    fn generate_handles_disagreement_without_panicking() {
        // Draft favors token 0; target favors token 1 → frequent rejection,
        // correction resampled from the target. Must still produce the
        // requested length.
        let mut draft = |_seq: &[u32]| vec![10.0f32, 0.0, 0.0];
        let mut target = |_seq: &[u32]| vec![0.0f32, 10.0, 0.0];
        let mut reject = || 1.0f32; // u=1 > ratio → reject, take correction
        let out =
            generate(&[0], 4, 3, &mut target, &mut draft, argmax, &mut reject).expect("generate");
        assert_eq!(out.len(), 4, "got {out:?}");
        // Target favors token 1, so the corrections are token 1.
        assert!(out.iter().all(|&t| t == 1), "got {out:?}");
    }
}
