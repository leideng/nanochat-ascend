#!/bin/bash

WANDB_RUN=dummy

python -m scripts.chat_sft \
    --model-tag="d4-test" \
    --model-step=20 \
    --max-seq-len=512 \
    --num-iterations=10 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=$WANDB_RUN
