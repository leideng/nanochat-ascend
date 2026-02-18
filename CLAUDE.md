# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat-ascend trains a small GPT chat model, adapted from Karpathy's nanochat for Ascend NPU (910C). Supports NPU and CPU backends. Uses UV for package management with Python 3.11.

## Common Commands

```bash
# Environment setup
uv sync                  # install dependencies
uv sync --extra cpu      # install with CPU-only PyTorch

# All scripts run as modules from the repo root
# Tokenizer
python -m scripts.tok_train --max-chars=2000000
python -m scripts.tok_eval

# Pre-training (single device)
python -m scripts.base_train --depth=6 --max-seq-len=512 --device-batch-size=1 --total-batch-size=512 --num-iterations=20

# Pre-training (distributed, 8 NPUs)
torchrun --nproc_per_node=8 -m scripts.base_train

# Evaluation
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384
python -m scripts.chat_eval -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -a ARC-Easy

# SFT and RL
python -m scripts.chat_sft --num-iterations=1500
python -m scripts.chat_rl

# Inference
python -m scripts.chat_cli -p "What is the capital of France?"
python -m scripts.chat_web

# Dataset inspection
python -m nanochat.dataset

# Full CPU demo pipeline
bash runs/runcpu.sh
```

No test suite, linter, or build step is configured.

## Architecture

The codebase has two layers: **nanochat/** (core library) and **scripts/** (entry points).

### Core Library (nanochat/)

- **gpt.py** — GPT model with RoPE, RMSNorm (no learnable params), ReLU² MLP, Group-Query Attention, untied embeddings, sliding window attention (configurable L/S pattern), and Flash Attention 3 support.
- **optim.py** — MuonAdamW optimizer: AdamW for embeddings/scalars, Muon (orthogonal Newton) for matrices, with Polar Express orthogonalization and NorMuon variance reduction.
- **engine.py** — Inference engine with KV cache, top-k/temperature sampling, batch generation, and calculator tool integration.
- **tokenizer.py** — BPE tokenizer with dual backend: HuggingFace Tokenizer or RustBPE+Tiktoken. GPT-4 split pattern. Special tokens: `<|bos|>`, `<|user_start|>`, `<|assistant_start|>`, `<|python_start|>`, etc.
- **dataloader.py** — Distributed data loader using BOS-aligned best-fit cropping (every row starts with BOS, no padding, ~35% tokens cropped at T=2048). Multi-threaded with DDP sharding.
- **flash_attention.py** — PyTorch SDPA attention (FA3 not available on NPU). Handles both training and inference (with KV cache).
- **common.py** — Device autodetection (NPU→CPU), DDP init (HCCL for NPU), logging, `print0()` for rank-0 output, peak FLOPS tables.
- **checkpoint_manager.py** — Save/load model+optimizer+metadata, bfloat16→float32 conversion for CPU, torch.compile key patching.
- **core_eval.py** — CORE metric evaluation (from DCLM paper), few-shot prompting, multiple-choice scoring.
- **loss_eval.py** — Bits-per-byte metric, tokenization-independent, distributed reduction.

### Scripts (scripts/)

- **base_train.py** — Pre-training with configurable depth, attention pattern, compute-budget or Chinchilla-ratio targets.
- **chat_sft.py** — Multi-task SFT mixing GSM8K, MMLU, SmolTalk, SpellingBee, custom JSON.
- **chat_rl.py** — Policy gradient RL training.
- **chat_eval.py** — Generative and categorical evaluation (HumanEval, MMLU, ARC, GSM8K).
- **chat_cli.py** / **chat_web.py** — CLI and FastAPI web chat interfaces.
- **tok_train.py** / **tok_eval.py** — Tokenizer training and evaluation.
- **base_eval.py** — Base model eval (CORE, BPB, sample generation).

## Key Environment Variables

- `NANOCHAT_BASE_DIR` — Cache/checkpoint directory (default: `~/.cache/nanochat`)
- `NANOCHAT_BASE_DATA_DIR` — Path to parquet dataset files (e.g. fineweb-edu-100b-shuffle)
- `WANDB_RUN` — W&B run name

## Key Design Patterns

- **Device-agnostic**: NPU/CPU autodetected; conditional imports and graceful fallbacks throughout.
- **DDP-aware**: `print0()` for rank-0 logging, per-rank optimizer checkpoints, all-reduce for metrics.
- **Checkpoints**: `model_{step:06d}.pt`, `meta_{step:06d}.json`, `optim_{step:06d}_rank{rank}.pt` in base dir.
- **Module execution**: All scripts run via `python -m scripts.<name>` or `python -m nanochat.<name>`, not as standalone files.
