#!/bin/bash

set -e


[ -d ".venv" ] || uv venv
source .venv/bin/activate

source runs/set_env.sh

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
LOCAL_ADDR="${LOCAL_ADDR:-127.0.0.1}"

# Evaluate the model
torchrun --nproc_per_node=16 --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --local-addr="$LOCAL_ADDR" -m scripts.chat_eval -- \
    --source="rl" \
    --model-tag="d32" \
    --batch-size=48
