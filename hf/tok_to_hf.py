# see https://huggingface.co/karpathy/nanochat-d32/discussions/1

from nanochat.tokenizer import RustBPETokenizer

tok = RustBPETokenizer.from_directory(".")

from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from pathlib import Path

output_dir = Path("hf-tokenizer")
output_dir.mkdir(exist_ok=True)
convert_tiktoken_to_fast(tok.enc, output_dir)