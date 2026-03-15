#!/bin/bash

set -e

source runs/set_env.sh

[ -d ".venv" ] || uv venv
source .venv/bin/activate

WANDB_RUN=dummy

python -m scripts.chat_sft \
    --model-tag="d4-test" \
    --model-step=20 \
    --max-seq-len=128 \
    --num-iterations=20 \
    --device-batch-size=8 \
    --total-batch-size=1024 \
    --eval-every=10 \
    --eval-tokens=1024 \
    --run=$WANDB_RUN
