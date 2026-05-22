#!/usr/bin/env python3
"""Compare Skein vs TensorRT-LLM on the same model + prompt, same metrics.

Metrics (single in-flight request, greedy):
  - TTFT  : time-to-first-token (ms)         — prompt seen -> first token out
  - TPOT  : time-per-output-token (ms, p50)  — steady-state inter-token latency
  - throughput (decode tokens/s)             — 1000 / TPOT_p50

Both engines run the SAME HF model dir, prompt, tensor-parallel size, and
generate the same number of tokens, so the numbers are comparable.

Usage:
  python3 bench/compare_trtllm.py \
      --model /home/ubuntu/mixtral-8x7b \
      --skein-artifact /tmp/art_big/<hash> \
      --prompt "The capital of France is" --max-new-tokens 16 --tp 2 \
      [--skein-batched-prefill 5]   # set = prompt token count to enable Skein batched prefill

Run just one side with --only skein|trtllm.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# TensorRT-LLM side (high-level LLM API: builds/loads engine, streams tokens).
# ---------------------------------------------------------------------------
def run_trtllm(model: str, prompt: str, max_new_tokens: int, tp: int) -> dict:
    """Measure TTFT / TPOT / throughput on TensorRT-LLM via the LLM API with
    token streaming. TTFT = wall time to the first streamed token; per-token
    deltas after that give TPOT (p50) and throughput."""
    from tensorrt_llm import LLM, SamplingParams  # noqa: import inside fn

    llm = LLM(model=model, tensor_parallel_size=tp)
    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)  # greedy

    # Warm up (engine warmup + caches) so we measure steady state, like Skein's
    # warm pass.
    for _ in range(2):
        for _o in llm.generate_async(prompt, sampling, streaming=True):
            pass

    t0 = time.perf_counter()
    token_times = []
    text = ""
    for out in llm.generate_async(prompt, sampling, streaming=True):
        token_times.append(time.perf_counter())
        text = out.outputs[0].text
    if not token_times:
        raise RuntimeError("TRT-LLM produced no tokens")

    ttft_ms = (token_times[0] - t0) * 1e3
    deltas_ms = [(b - a) * 1e3 for a, b in zip(token_times, token_times[1:])]
    deltas_ms.sort()
    p50 = deltas_ms[len(deltas_ms) // 2] if deltas_ms else float("nan")
    p95 = deltas_ms[int(len(deltas_ms) * 0.95)] if deltas_ms else float("nan")
    return {
        "engine": "TensorRT-LLM",
        "ttft_ms": round(ttft_ms, 1),
        "tpot_p50_ms": round(p50, 2),
        "tpot_p95_ms": round(p95, 2),
        "throughput_tok_s": round(1000.0 / p50, 2) if p50 == p50 and p50 > 0 else None,
        "decode_steps": len(deltas_ms),
        "text": text[:80],
    }


# ---------------------------------------------------------------------------
# Skein side (skein serve, parse the SKEIN_PERF line it emits).
# ---------------------------------------------------------------------------
SKEIN_PERF_RE = re.compile(
    r"prefill_steps=(?P<prefill_steps>\d+).*?ttft_ms=(?P<ttft>[\d.]+).*?"
    r"tpot_p50_ms=(?P<p50>[\d.]+).*?tpot_p95_ms=(?P<p95>[\d.]+).*?"
    r"decode_tokens_per_s=(?P<tput>[\d.]+)"
)


def run_skein(artifact: str, prompt: str, max_new_tokens: int, tp: int,
              batched_prefill: int | None, warmup: bool = True) -> dict:
    """Run `skein serve` once and parse its SKEIN_PERF line. Optionally a warmup
    pass first so the measured run is steady-state (matches the TRT-LLM warmup)."""
    skein_dir = Path(__file__).resolve().parent.parent
    gpus = ",".join(str(i) for i in range(tp))
    base_env = "RUST_LOG=warn,skein_runtime=info"
    if batched_prefill:
        base_env += f" SKEIN_BATCHED_PREFILL={batched_prefill}"
    cmd = (
        f"{base_env} ./target/release/skein serve --artifact {artifact} "
        f"--workload cluster/sample_trace.jsonl --cost cluster/cost_constants.toml "
        f"--prompt '{prompt}' --max-new-tokens {max_new_tokens} --gpus {gpus}"
    )

    def once():
        # Redirect to a FILE (not a pipe): the multi-process serve's rank
        # children write SKEIN_PERF to the inherited fd, which a pipe capture
        # misses but a file inherits correctly.
        logf = "/tmp/skein_serve_run.log"
        subprocess.run(["bash", "-lc", "rm -f /tmp/skein_rendezvous"], cwd=skein_dir)
        subprocess.run(["bash", "-lc", f"{cmd} > {logf} 2>&1"], cwd=skein_dir)
        return Path(logf).read_text(errors="replace")

    if warmup:
        once()  # populate cubin cache / capture CUDA graphs
    log = once()
    m = None
    for line in log.splitlines():
        mm = SKEIN_PERF_RE.search(line)
        if mm:
            m = mm
    if not m:
        raise RuntimeError("no SKEIN_PERF line found; serve failed:\n" + log[-2000:])
    p50 = float(m["p50"])
    return {
        "engine": "Skein" + (" (batched prefill)" if batched_prefill else " (seq prefill)"),
        "ttft_ms": round(float(m["ttft"]), 1),
        "tpot_p50_ms": round(p50, 2),
        "tpot_p95_ms": round(float(m["p95"]), 2),
        "throughput_tok_s": round(float(m["tput"]), 2),
        "prefill_steps": int(m["prefill_steps"]),
        "decode_steps": max_new_tokens - 1,
    }


def print_table(rows: list[dict]):
    cols = ["engine", "ttft_ms", "tpot_p50_ms", "tpot_p95_ms", "throughput_tok_s",
            "prefill_steps", "decode_steps"]
    widths = {c: max(len(c), *(len(str(r.get(c, "-"))) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r.get(c, "-")).ljust(widths[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/ubuntu/mixtral-8x7b")
    ap.add_argument("--skein-artifact", required=False)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--skein-batched-prefill", type=int, default=None)
    ap.add_argument("--only", choices=["skein", "trtllm"], default=None)
    args = ap.parse_args()

    rows = []
    if args.only in (None, "skein"):
        if not args.skein_artifact:
            sys.exit("--skein-artifact required for the Skein run")
        print(">>> Skein ...", file=sys.stderr)
        rows.append(run_skein(args.skein_artifact, args.prompt, args.max_new_tokens,
                              args.tp, args.skein_batched_prefill))
    if args.only in (None, "trtllm"):
        print(">>> TensorRT-LLM ...", file=sys.stderr)
        rows.append(run_trtllm(args.model, args.prompt, args.max_new_tokens, args.tp))

    print("\n=== Skein vs TensorRT-LLM — Mixtral 8x7B, "
          f"tp={args.tp}, {args.max_new_tokens} tokens, greedy ===")
    print_table(rows)
    print("\n" + json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
