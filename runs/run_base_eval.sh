#!/bin/bash

set -e

source runs/set_env.sh

[ -d ".venv" ] || uv venv
source .venv/bin/activate

# Evaluate the model
python -m scripts.base_eval \
    --model-tag="d4-test" \
    --step=20 \
    --device-batch-size=8 \
    --split-tokens=1024 \
    --max-per-task=16 \
    --eval=bpb,sample
