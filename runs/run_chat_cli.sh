#!/bin/bash

# exit on error
set -e

source runs/set_env.sh

# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
source .venv/bin/activate


DEVICE_TYPE="cpu"
if command -v npu-smi >/dev/null 2>&1 && npu-smi info >/dev/null 2>&1; then
    DEVICE_TYPE="npu"
fi
echo "Detected device type: $DEVICE_TYPE"


python -m scripts.chat_cli \
    --model-tag="d20" \
    --source="sft" \
    --device-type=$DEVICE_TYPE
