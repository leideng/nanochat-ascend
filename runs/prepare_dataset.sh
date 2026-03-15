#!/bin/bash

# This script is used to prepare the dataset for the nanochat-ascend model.
# It will download the dataset from the HuggingFace or URL and prepare it for the nanochat-ascend model.


# exit on error
set -e

source runs/set_env.sh

# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv

# detect NPU vs CPU: use npu-smi (Ascend driver) when available
if command -v npu-smi &>/dev/null && npu-smi info &>/dev/null; then
  UV_EXTRA="npu"
else
  UV_EXTRA="cpu"
fi
echo "Detected device: $UV_EXTRA (using uv sync --extra $UV_EXTRA)"

# install the repo dependencies
uv sync --extra "$UV_EXTRA"

source .venv/bin/activate


python -m nanochat.dataset
