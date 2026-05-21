#!/usr/bin/env python3
"""Decoupled final-KL gate: compare HF per-layer activations + final logits
(from verify_reference.py stdout JSONL) against the Skein candidate's
(hidden_after_block_*.dev0.f32 + final_logits.f32 dumped by `verify --dump-only`).

Uses the SAME math as crates/skein_parity/src/comparison.rs:
  - per-layer MSE on the last-token row,
  - final KL(p_HF || q_Skein) via log_softmax (max-shift, f64).

This produces the identical numbers `skein verify --hf-reference` would, but
without co-residing HF and the tp=2 Skein candidate on the same GPUs.
"""
import argparse
import json
import os
import numpy as np


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max()
    if not np.isfinite(m):
        return x
    s = x - m
    return s - np.log(np.exp(s).sum())


def kl(p_logits, q_logits):
    n = min(len(p_logits), len(q_logits))
    lp = log_softmax(p_logits[:n])
    lq = log_softmax(q_logits[:n])
    p = np.exp(lp)
    return max(0.0, float(np.sum(p * (lp - lq))))


def mse(a, b):
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    return float(np.mean((a - b) ** 2))


def last_row(flat, n):
    flat = np.asarray(flat, dtype=np.float64)
    if n > 0 and len(flat) >= n and len(flat) % n == 0:
        return flat[-n:]
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-jsonl", required=True, help="verify_reference.py stdout (one JSON line per prompt)")
    ap.add_argument("--skein-dir", required=True, help="dir with hidden_after_block_N.dev0.f32 + final_logits.f32")
    ap.add_argument("--max-drift", type=float, default=0.01)
    args = ap.parse_args()

    with open(args.hf_jsonl) as f:
        hf = json.loads([l for l in f if l.strip()][0])
    hf_layers = hf["per_layer_activations"]
    hf_logits = np.asarray(hf["final_logits"], dtype=np.float64)

    sk_logits = np.fromfile(os.path.join(args.skein_dir, "final_logits.f32"), dtype="<f4")
    nlayers = len(hf_layers)
    print(f"layers={nlayers}  hf_logits_len={len(hf_logits)}  sk_logits_len={len(sk_logits)}")
    print(f"{'layer':>5} {'mse':>14}")
    for i in range(nlayers):
        sk_path = os.path.join(args.skein_dir, f"hidden_after_block_{i}.dev0.f32")
        if not os.path.exists(sk_path):
            print(f"{i:>5}  (no skein dump)")
            continue
        sk = np.fromfile(sk_path, dtype="<f4")
        hf_l = last_row(hf_layers[i], len(sk))
        print(f"{i:>5} {mse(hf_l, sk):>14.4e}")

    final_kl = kl(last_row(hf_logits, len(sk_logits)), sk_logits)
    print(f"\nFINAL KL(HF||Skein) = {final_kl:.6f}   gate(max_drift={args.max_drift}) -> "
          f"{'PASS' if final_kl <= args.max_drift else 'FAIL'}")


if __name__ == "__main__":
    main()
