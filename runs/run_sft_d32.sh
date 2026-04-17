#!/bin/bash

set -e

source runs/set_env.sh

[ -d ".venv" ] || uv venv
source .venv/bin/activate

DATE=$(date +%Y%m%d%H%M%S)
WANDB_RUN="nanochat-ascend-sft-d32-${DATE}"
#WANDB_RUN=dummy

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
LOCAL_ADDR="${LOCAL_ADDR:-127.0.0.1}"

DEVICE_TYPE="cpu"
if command -v npu-smi >/dev/null 2>&1 && npu-smi info >/dev/null 2>&1; then
    DEVICE_TYPE="npu"
fi
echo "Detected device type: $DEVICE_TYPE"

if [ "$DEVICE_TYPE" == "npu" ]; then
    echo "Training on NPU"
    torchrun --nproc_per_node=16 --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --local-addr="$LOCAL_ADDR" -m scripts.chat_sft -- \
        --model-tag="d32" \
        --max-seq-len=2048 \
        --num-iterations=-1 \
        --device-batch-size=8 \
        --eval-every=150 \
        --run=$WANDB_RUN
else
    echo "Training on CPU"
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
fi
