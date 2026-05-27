# Skein build/test/lint targets.
#
# The default build targets CUDA. On a host without `nvcc` the targets fall
# back to `--no-default-features` (the CPU `NativeComputeRuntime` path) so the
# planning + runtime logic still builds and tests. GPU-only targets abort with
# a clear error on a host without CUDA — they never silently degrade.

CARGO        ?= cargo
HAS_CUDA     := $(shell command -v nvcc >/dev/null 2>&1 && echo yes || echo no)

ifeq ($(HAS_CUDA),yes)
FEATURES :=
else
FEATURES := --no-default-features
endif

.PHONY: build test lint fmt fmt-check clean \
        extract-mixtral compile-mixtral verify-mixtral bench \
        require-cuda

build:
	$(CARGO) build --workspace $(FEATURES)

test:
	$(CARGO) test --workspace $(FEATURES)

lint:
	$(CARGO) clippy --workspace --all-targets $(FEATURES) -- -D warnings

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clean:
	$(CARGO) clean

# Plan search only — no GPU required (runs with whatever FEATURES resolves to).
# `extract` requires --drift and --cost (no defaults); they must be passed
# explicitly or the command aborts with a clap "required arguments" error.
extract-mixtral:
	$(CARGO) run -p skein_cli $(FEATURES) -- extract \
	    --model   configs/mixtral_8x7b_config.json \
	    --cluster cluster/rtx6000_2x.toml \
	    --trace   cluster/sample_trace.jsonl \
	    --drift   models/mixtral_8x7b_drift.toml \
	    --cost    cluster/cost_constants.toml \
	    --out     artifacts/plan.json

# GPU pipeline targets.
require-cuda:
	@if [ "$(HAS_CUDA)" != "yes" ]; then \
	    echo "error: this target requires CUDA (nvcc not found on PATH)"; \
	    echo "  hint: run on an NVIDIA GPU host with the CUDA toolkit installed"; \
	    exit 1; \
	fi

# `compile` requires --drift, --cost, and --weights (a real safetensors
# checkpoint dir) in addition to --model/--cluster/--trace. Point
# HF_MODEL_PATH at the downloaded checkpoint, e.g.
#   make compile-mixtral HF_MODEL_PATH=/data/mixtral-8x7b
compile-mixtral: require-cuda
	$(CARGO) run -p skein_cli -- compile \
	    --model   configs/mixtral_8x7b_config.json \
	    --cluster cluster/rtx6000_2x.toml \
	    --trace   cluster/sample_trace.jsonl \
	    --drift   models/mixtral_8x7b_drift.toml \
	    --cost    cluster/cost_constants.toml \
	    --weights $(HF_MODEL_PATH) \
	    --out     artifacts/

# `verify` compares the artifact against a bf16 *Skein* reference artifact.
# With no --reference it uses <artifact>/reference_bf16, which `compile`
# writes during advisory parity — so omit --reference here rather than
# passing an HF checkpoint path (which verify does not accept).
verify-mixtral: require-cuda
	$(CARGO) run -p skein_cli -- verify \
	    --artifact artifacts/LATEST \
	    --sample-from cluster/sample_trace.jsonl

bench: require-cuda
	$(CARGO) run -p skein_cli -- bench \
	    --artifact artifacts/LATEST \
	    --workload cluster/sample_trace.jsonl \
	    --baseline $(VLLM_ENDPOINT) \
	    --metrics throughput,goodput,drift_compliance
