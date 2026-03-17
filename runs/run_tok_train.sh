#!/bin/bash

set -e

source runs/set_env.sh

[ -d ".venv" ] || uv venv
source .venv/bin/activate

# train the tokenizer
echo "Training the tokenizer..."
python -m scripts.tok_train --max-chars=5000000

# evaluate the tokenizer (report compression ratio etc.)
echo "Evaluating the tokenizer..."
python -m scripts.tok_eval
