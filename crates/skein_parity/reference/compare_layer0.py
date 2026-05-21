#!/usr/bin/env python3
"""Compare HF layer-0 intermediates (.npy) against Skein op taps (.f32) to
find the first diverging op. Run after:
  1) python dump_layer0_intermediates.py --model <W> --out <HF_DIR>
  2) SKEIN_DEBUG_TAPS=1 SKEIN_DUMP_DIR=<SK_DIR> \
       skein verify --dump-only --artifact <ART> --tokens-file <HF_DIR>/input_ids.txt

Skein TP-shards q/k/v projections across devices; those are concatenated
dev0||dev1. Replicated tensors (norms, post-collective attn/moe, router) use
dev0 (dev1 is asserted identical).
"""
import argparse
import os
import sys

import numpy as np


def load_skein(sk_dir, name, sharded):
    d0 = os.path.join(sk_dir, f"{name}.dev0.f32")
    d1 = os.path.join(sk_dir, f"{name}.dev1.f32")
    if not os.path.exists(d0):
        return None
    a0 = np.fromfile(d0, dtype="<f4")
    if sharded and os.path.exists(d1):
        a1 = np.fromfile(d1, dtype="<f4")
        return np.concatenate([a0, a1])
    return a0


def mse(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def cos(a, b):
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--skein", required=True)
    args = ap.parse_args()

    # (HF .npy name, Skein tap name, is_sharded_across_TP)
    stages = [
        ("embed_out",             "dbg_l0_embed_in",       False),
        ("post_input_layernorm",  "dbg_l0_post_input_ln",  False),
        ("q_proj_out",            "dbg_l0_q_proj",         True),
        ("k_proj_out",            "dbg_l0_k_proj",         True),
        ("attn_out",              "dbg_l0_attn_out",       False),
        ("post_attn_layernorm",   "dbg_l0_post_attn_ln",   False),
        ("router_logits",         "dbg_l0_router_logits",  False),
        ("moe_out",               "dbg_l0_moe_out",        False),
        ("layer0_out",            "hidden_after_block_0",  False),
    ]

    print(f"{'stage':<22} {'hf_shape':>10} {'sk_shape':>10} {'MSE':>12} {'cos':>9} {'hf_norm':>10} {'sk_norm':>10}")
    print("-" * 92)
    first_bad = None
    for hfname, skname, sharded in stages:
        hf_path = os.path.join(args.hf, f"{hfname}.npy")
        if not os.path.exists(hf_path):
            print(f"{hfname:<22} {'(no HF)':>10}")
            continue
        hf = np.load(hf_path).astype(np.float64).reshape(-1)
        sk = load_skein(args.skein, skname, sharded)
        if sk is None:
            print(f"{hfname:<22} {str(hf.shape):>10} {'(no SK)':>10}")
            continue
        m = mse(hf, sk)
        c = cos(hf, sk)
        flag = ""
        if m > 1e-3 and first_bad is None:
            first_bad = (hfname, skname, m)
            flag = "  <== FIRST DIVERGENCE"
        print(f"{hfname:<22} {str(hf.shape):>10} {str(sk.shape):>10} {m:>12.4e} {c:>9.5f} "
              f"{np.linalg.norm(hf):>10.3f} {np.linalg.norm(sk):>10.3f}{flag}")

    print("-" * 92)
    if first_bad:
        print(f"FIRST DIVERGING OP: {first_bad[0]} (skein tap {first_bad[1]}) MSE={first_bad[2]:.4e}")
    else:
        print("No op exceeded MSE 1e-3 — layer 0 matches HF.")


if __name__ == "__main__":
    main()
