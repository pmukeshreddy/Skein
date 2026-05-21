#!/usr/bin/env python3
"""HuggingFace reference subprocess for skein_parity.

The Rust side sends one JSON object on stdin and receives JSON-lines on
stdout. Diagnostics go to stderr and any failure exits non-zero.
"""

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List


def check_env() -> int:
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised from Rust
        print(f"import failed: {exc}", file=sys.stderr)
        return 1
    return 0


def torch_dtype(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported reference_dtype {name!r}") from exc


def decoder_layers(model: Any) -> Iterable[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("could not locate decoder blocks on HF model")


def tensor_to_list(tensor: Any) -> List[float]:
    return tensor.detach().to("cpu").float().reshape(-1).tolist()


def load_request() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("stdin was empty")
    return json.loads(raw)


def tokenize_only(request: Dict[str, Any]) -> None:
    from transformers import AutoTokenizer

    model_path = request["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    for idx, prompt in enumerate(request.get("prompts", [])):
        text = prompt.get("text")
        if text is None:
            raise ValueError("tokenize-only prompts must contain text")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        print(json.dumps({"prompt_idx": idx, "tokens": tokens}), flush=True)


def forward(request: Dict[str, Any]) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    model_path = request["model_path"]
    dtype = torch_dtype(request.get("reference_dtype", "bfloat16"))
    # Load the reference directly onto GPU 1 (the candidate uses GPU 0). CPU
    # forward of Mixtral is minutes/token and a CPU load spikes ~90GB host RAM;
    # pinning to cuda:1 (idle) makes the reference fast and avoids RAM pressure.
    import os
    ref_device = os.environ.get("SKEIN_REF_DEVICE", "cuda:1")
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": {"": ref_device},
        "low_cpu_mem_usage": True,
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()

    layers = list(decoder_layers(model))
    captures: List[Any] = [None for _ in layers]
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captures[layer_idx] = hidden.detach()

        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            for idx, prompt in enumerate(request.get("prompts", [])):
                tokens = prompt.get("tokens")
                if tokens is None:
                    raise ValueError("forward prompts must contain tokens")
                input_ids = torch.tensor([tokens], dtype=torch.long).to(model.device)
                for i in range(len(captures)):
                    captures[i] = None
                out = model(input_ids=input_ids)
                missing = [i for i, value in enumerate(captures) if value is None]
                if missing:
                    raise RuntimeError(f"missing layer captures: {missing}")
                line = {
                    "prompt_idx": idx,
                    "per_layer_activations": [tensor_to_list(t) for t in captures],
                    "final_logits": tensor_to_list(out.logits),
                }
                print(json.dumps(line, separators=(",", ":")), flush=True)
    finally:
        for handle in handles:
            handle.remove()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--tokenize-only", action="store_true")
    args = parser.parse_args()

    if args.check_env:
        return check_env()

    try:
        request = load_request()
        if args.tokenize_only:
            tokenize_only(request)
        else:
            forward(request)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
