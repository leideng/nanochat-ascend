#!/bin/bash

set -e

source runs/set_env.sh

[ -d ".venv" ] || uv venv
source .venv/bin/activate

DATE=$(date +%Y%m%d%H%M%S)
WANDB_RUN="nanochat-ascend-rl-d20-${DATE}"
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
    torchrun --nproc_per_node=16 --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --local-addr="$LOCAL_ADDR" -m scripts.chat_rl -- \
        --model-tag="d20" \
        --num-epochs=1 \
        --device-batch-size=8 \
        --examples-per-step=16 \
        --num-samples=16 \
        --max-new-tokens=256 \
        --temperature=1.0 \
        --top-k=50 \
        --eval-every=60 \
        --save-every=60 \
        --run=$WANDB_RUN
else
    echo "Training on CPU"
    python -m scripts.chat_rl \
        --model-tag="d4-test" \
        --num-epochs=1 \
        --device-batch-size=4 \
        --eval-every=60 \
        --save-every=60 \
        --run=$WANDB_RUN
fi
