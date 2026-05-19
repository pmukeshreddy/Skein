# Skein build/test/lint targets.
#
# Auto-detect CUDA availability by checking for `nvcc` on PATH. Mac builds run
# without `--features cuda`; H100 builds enable it. Phase B targets abort with
# a clear error when run on a machine without CUDA — never silently degrade.

CARGO        ?= cargo
HAS_CUDA     := $(shell command -v nvcc >/dev/null 2>&1 && echo yes || echo no)

ifeq ($(HAS_CUDA),yes)
FEATURES := --features cuda
else
FEATURES :=
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

# Phase A (works on Mac): emit the Plan only, no Luminal compile.
extract-mixtral:
	$(CARGO) run -p skein_cli -- extract \
	    --model configs/mixtral_8x7b_config.json \
	    --cluster cluster/h100_2x.toml \
	    --trace  cluster/sample_trace.jsonl \
	    --out    artifacts/plan.json

# Phase B (CUDA-only).
require-cuda:
	@if [ "$(HAS_CUDA)" != "yes" ]; then \
	    echo "error: this target requires CUDA (nvcc not found on PATH)"; \
	    echo "  hint: run on an NVIDIA GPU host with the CUDA toolkit installed"; \
	    exit 1; \
	fi

compile-mixtral: require-cuda
	$(CARGO) run -p skein_cli --features cuda -- compile \
	    --model configs/mixtral_8x7b_config.json \
	    --cluster cluster/h100_2x.toml \
	    --trace  cluster/sample_trace.jsonl \
	    --out    artifacts/

verify-mixtral: require-cuda
	$(CARGO) run -p skein_cli --features cuda -- verify \
	    --artifact artifacts/LATEST \
	    --reference $(HF_MODEL_PATH) \
	    --sample-from cluster/sample_trace.jsonl

bench: require-cuda
	$(CARGO) run -p skein_cli --features cuda -- bench \
	    --artifact artifacts/LATEST \
	    --workload cluster/sample_trace.jsonl \
	    --baseline $(VLLM_ENDPOINT) \
	    --metrics throughput,goodput,drift_compliance
