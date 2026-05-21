#!/usr/bin/env python3
"""Dump HF Mixtral's layer-0 intermediates for one prompt, to bisect a Skein
parity divergence op-by-op (run on the GPU host that has transformers + weights).

For the LAST prompt token it captures, in forward order:
  embed_out, post_input_layernorm, q_after_rope, k_after_rope, attn_out,
  post_attn_residual, post_attn_layernorm, router_logits, top2_idx, top2_weights,
  moe_out, layer0_out.

Each is saved to <out>/<name>.npy (float32, last-token row). Compare against the
matching Skein tensor (see dump_layer0_intermediates.md): the FIRST tensor whose
MSE jumps far above the ~1e-3 bf16 noise floor is the diverging op.

Usage:
  python dump_layer0_intermediates.py --model /path/to/mixtral-8x7b \
      --prompt "The capital of France is" --out /tmp/hf_layer0
"""
import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", default="/tmp/hf_layer0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids.to("cuda")
    last = ids.shape[1] - 1
    captured = {}

    def save(name, t):
        # last-token row, float32 on host
        arr = t[0, last].detach().to(torch.float32).cpu().numpy()
        captured[name] = arr
        np.save(os.path.join(args.out, f"{name}.npy"), arr)
        print(f"  {name:24s} shape={tuple(arr.shape)}  norm={np.linalg.norm(arr):.4f}")

    layer = model.model.layers[0]
    attn = layer.self_attn
    moe = layer.block_sparse_moe

    hooks = []
    # Embedding output == decoder layer input.
    hooks.append(layer.register_forward_pre_hook(
        lambda m, a, kw=None: save("embed_out", a[0]) or None, with_kwargs=False))
    hooks.append(layer.input_layernorm.register_forward_hook(
        lambda m, i, o: save("post_input_layernorm", o)))
    hooks.append(attn.q_proj.register_forward_hook(
        lambda m, i, o: save("q_proj_out", o)))
    hooks.append(attn.k_proj.register_forward_hook(
        lambda m, i, o: save("k_proj_out", o)))
    hooks.append(attn.o_proj.register_forward_hook(
        lambda m, i, o: save("attn_out", o)))
    hooks.append(layer.post_attention_layernorm.register_forward_hook(
        lambda m, i, o: save("post_attn_layernorm", o)))
    hooks.append(moe.gate.register_forward_hook(
        lambda m, i, o: save("router_logits", o)))
    hooks.append(moe.register_forward_hook(
        lambda m, i, o: save("moe_out", o[0] if isinstance(o, tuple) else o)))
    hooks.append(layer.register_forward_hook(
        lambda m, i, o: save("layer0_out", o[0] if isinstance(o, tuple) else o)))

    with torch.no_grad():
        model(ids, output_hidden_states=False, use_cache=True)

    for h in hooks:
        h.remove()

    # Router top-2 from the captured logits (Mixtral does softmax over all 8,
    # top-2, renormalize).
    rl = torch.tensor(captured["router_logits"])
    probs = torch.softmax(rl, dim=-1)
    w, idx = torch.topk(probs, 2, dim=-1)
    w = w / w.sum()
    np.save(os.path.join(args.out, "top2_idx.npy"), idx.numpy())
    np.save(os.path.join(args.out, "top2_weights.npy"), w.numpy())
    print(f"  top2 experts={idx.tolist()}  weights={w.tolist()}")
    print(f"\nWrote HF layer-0 intermediates to {args.out}")


if __name__ == "__main__":
    main()
