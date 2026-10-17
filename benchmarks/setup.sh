#!/usr/bin/env bash
# Portable CUDA 13 environment. Driver compatibility is an explicit caller choice.
set -euo pipefail
BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$BENCH_DIR/.." && pwd)
UV=${UV:-uv}
BENCH_VENV=${BENCH_VENV:-"$REPO_DIR/.venv"}
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --check ) ]]; then
    echo 'Usage: bash benchmarks/setup.sh [--check]' >&2; exit 2
fi
if [[ ${1:-} != --check ]]; then
    if [[ ! -x "$BENCH_VENV/bin/python" ]]; then
        "$UV" venv --python "${PYTHON_VERSION:-3.12}" "$BENCH_VENV"
    fi
    "$UV" pip sync --python "$BENCH_VENV/bin/python" "$BENCH_DIR/requirements.lock"
    "$UV" pip install --python "$BENCH_VENV/bin/python" "setuptools-rust==1.12.0" "setuptools-scm==9.2.2" "wheel==0.45.1"
    SGLANG_BUILD_RUST_EXTS=none "$UV" pip install --python "$BENCH_VENV/bin/python" --no-deps --no-build-isolation -e "$REPO_DIR/python" -e "$BENCH_DIR"
fi
if [[ -n ${BENCH_CUDA_COMPAT:-} ]]; then
    export LD_LIBRARY_PATH="$BENCH_CUDA_COMPAT:${LD_LIBRARY_PATH:-}"
fi
"$UV" pip check --python "$BENCH_VENV/bin/python"
PYTHONPATH="$REPO_DIR/python:$BENCH_DIR" "$BENCH_VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
import sglang
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
print('Checkout source:', sglang.__file__)
print('CUDA runtime:', torch.version.cuda)
print({name: metadata.version(name) for name in ['torch', 'torchvision', 'torchaudio', 'transformers', 'diffusers', 'triton', 'av', 'sglang-kernel']})
PY
mkdir -p "$REPO_DIR/.cache/benchmark"
"$UV" pip freeze --python "$BENCH_VENV/bin/python" > "$REPO_DIR/.cache/benchmark/environment.freeze.txt"
