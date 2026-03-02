#!/bin/bash

# Evaluate the model
python -m scripts.base_eval \
    --model-tag="d4-test" \
    --step=20 \
    --device-batch-size=8 \
    --split-tokens=1024 \
    --max-per-task=16 \
    --eval=bpb,sample
