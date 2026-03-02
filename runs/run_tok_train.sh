#!/bin/bash

# train the tokenizer
echo "Training the tokenizer..."
python -m scripts.tok_train --max-chars=2000000

# evaluate the tokenizer (report compression ratio etc.)
echo "Evaluating the tokenizer..."
python -m scripts.tok_eval