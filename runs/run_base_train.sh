#!/bin/bash

NUM_CPU=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
echo "NUM_CPU in this machine is: $NUM_CPU"

WANDB_RUN=dummy

# train a small 4 layer model
# I tuned this run to complete in about 2 minutes on my PC.
# To get better results, try increasing num_iterations, or get other ideas from your favorite LLM.
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

