#!/bin/bash

# exit on error
set -e

source runs/set_env.sh

# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
source .venv/bin/activate

PARALLEL_TRAIN="${PARALLEL_TRAIN:-0}"

WANDB_RUN=dummy
DEVICE_TYPE="cpu"
if command -v npu-smi >/dev/null 2>&1 && npu-smi info >/dev/null 2>&1; then
    DEVICE_TYPE="npu"
fi
echo "Detected device type: $DEVICE_TYPE"

if [ $DEVICE_TYPE == "npu" ]; then
    echo "Training on NPU"
    torchrun --standalone --nproc_per_node=16 --master-addr=127.0.0.1 -m scripts.base_train -- \
        --depth=4 \
        --head-dim=16 \
        --window-pattern=L \
        --max-seq-len=128 \
        --device-batch-size=8 \
        --total-batch-size=16384 \
        --eval-every=10 \
        --eval-tokens=1024 \
        --core-metric-every=-1 \
        --sample-every=10 \
        --save-every=10 \
        --num-iterations=20 \
        --run=$WANDB_RUN \
        --model-tag="d4-test"
else
    if [ $PARALLEL_TRAIN -eq 1 ]; then # (TODO) parallel training still not working. comming later
        NUM_CPU=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
        echo "$DEVICE_TYPE: NUM_CPU in this machine is: $NUM_CPU"
        #echo "Using half of the CPUs for training: $((NUM_CPU / 2))"
        torchrun --standalone --nproc_per_node=$NUM_CPU --master-addr=127.0.0.1 -m scripts.base_train -- \
            --depth=4 \
            --head-dim=16 \
            --window-pattern=L \
            --max-seq-len=128 \
            --device-batch-size=8 \
            --total-batch-size=16384 \
            --eval-every=10 \
            --eval-tokens=1024 \
            --core-metric-every=-1 \
            --sample-every=10 \
            --save-every=10 \
            --num-iterations=20 \
            --run=$WANDB_RUN \
            --model-tag="d4-test"
    else # single CPU training
        echo "Running single-process training smoke test on $DEVICE_TYPE. Set PARALLEL_TRAIN=1 to use torchrun."
        python -m scripts.base_train \
            --depth=4 \
            --head-dim=16 \
            --window-pattern=L \
            --max-seq-len=128 \
            --device-batch-size=8 \
            --total-batch-size=1024 \
            --eval-every=10 \
            --eval-tokens=1024 \
            --core-metric-every=-1 \
            --sample-every=10 \
            --save-every=10 \
            --num-iterations=20 \
            --run=$WANDB_RUN \
            --model-tag="d4-test"
    fi
fi
