# Project Overview

## Purpose

`nanochat-ascend` trains and evaluates a compact GPT-style chat model stack for Ascend hardware while keeping CPU execution available for small-scale local iteration. The repository is designed to cover the full lifecycle:

- dataset acquisition
- tokenizer training
- base language-model pretraining
- chat-oriented supervised fine-tuning
- reinforcement learning
- offline evaluation
- interactive inference

## Supported Runtime Targets

The codebase is designed for:

- Ascend NPU as the real training and main evaluation target
- CPU as the local development and smoke-test target

The codebase is not designed for CUDA or generic GPU execution.

## Repository Structure

### Core library: `nanochat/`

Important modules include:

- `nanochat/gpt.py`: the transformer implementation
- `nanochat/optim.py`: the MuonAdamW optimizer stack
- `nanochat/tokenizer.py`: tokenizer loading and rendering helpers
- `nanochat/dataloader.py`: pretraining data loading and BOS-aligned packing
- `nanochat/engine.py`: batched generation and inference helpers
- `nanochat/checkpoint_manager.py`: checkpoint save/load utilities
- `nanochat/core_eval.py`: CORE benchmark evaluation logic
- `nanochat/loss_eval.py`: bits-per-byte evaluation
- `nanochat/dataset.py`: dataset download helpers and parquet iteration utilities

### Entry points: `scripts/`

The main operational scripts are:

- `scripts/tok_train.py`: tokenizer training
- `scripts/base_train.py`: base-model pretraining
- `scripts/base_eval.py`: base-model evaluation
- `scripts/chat_sft.py`: supervised fine-tuning
- `scripts/chat_rl.py`: reinforcement-learning fine-tuning
- `scripts/chat_eval.py`: chat-model evaluation
- `scripts/chat_cli.py`: terminal chat interface
- `scripts/chat_web.py`: web chat interface

### Workflow wrappers: `runs/`

The `runs/` shell wrappers encode common repo workflows. They:

- load `configs/global.yaml` through `runs/set_env.sh`
- create or reuse `.venv`
- choose CPU or NPU paths based on whether `npu-smi` is available
- provide practical defaults for smoke tests and full runs

## End-to-End Pipeline

The repo’s intended pipeline is:

1. Prepare datasets with `bash runs/prepare_dataset.sh`.
2. Train and evaluate the tokenizer with `bash runs/run_tok_train.sh`.
3. Pretrain a base model with `bash runs/run_base_train.sh`.
4. Evaluate the base model with `bash runs/run_base_eval.sh`.
5. Run supervised fine-tuning with `bash runs/run_sft.sh`.
6. Run reinforcement learning with `bash runs/run_rl.sh`.
7. Evaluate chat models with `python -m scripts.chat_eval ...` or wrapper scripts such as `runs/run_sft_eval.sh` and `runs/run_rl_eval.sh`.
8. Interact with the resulting model through CLI or web entry points.

## Configuration Model

Runtime paths are centralized in `configs/global.yaml`. The most important groups are:

- dataset locations under `.cache/dataset`
- checkpoint locations under `.cache/checkpoint`
- output and report locations under `.cache/output`
- eager-vs-compile behavior via `enforce_eager`

By default the repository expects:

- pretraining text under `.cache/dataset/pretrain/...`
- CORE/base-eval assets under `.cache/dataset/eval`
- task datasets under `.cache/dataset/task/...`
- base checkpoints under `.cache/checkpoint/base_checkpoints`
- SFT checkpoints under `.cache/checkpoint/chatsft_checkpoints`
- RL checkpoints under `.cache/checkpoint/chatrl_checkpoints`

## Training Stages At A Glance

### Base pretraining

The base model learns causal language modeling over a large parquet text corpus. This stage produces the foundation checkpoint used by later chat stages.

### SFT

SFT starts from a base checkpoint and trains on a mixed conversational/task dataset. The goal is to teach instruction following, identity behavior, multiple-choice formatting, spelling behavior, and early math/tool-style behavior.

### RL

RL starts from an SFT checkpoint and optimizes policy behavior on GSM8K rollouts using a simplified REINFORCE/GRPO-style objective. This stage is focused on reward-driven improvement rather than broad distribution matching.

## Evaluation Modes

The repo uses two evaluation families:

- base-model evaluation:
  - CORE benchmark accuracy
  - bits per byte on pretraining data
  - qualitative sampling
- chat-model evaluation:
  - ARC
  - MMLU
  - GSM8K
  - HumanEval
  - SpellingBee
  - aggregate ChatCORE-style centered metric

## What Makes This Repo Distinct

Compared with a minimal GPT training repo, `nanochat-ascend` adds:

- Ascend-first execution support
- CPU-compatible development workflows
- RoPE, RMSNorm, QK normalization, and Group-Query Attention
- optional sliding-window attention patterns across layers
- MuonAdamW-based optimization
- chat SFT and RL stages in the same repository
- explicit dataset preparation and reporting helpers

## Recommended Reading Order

For a new user, the most effective reading order is:

1. [Datasets Overview](datasets/index.md)
2. [Model](model.md)
3. [Dataset Preparation](processes/dataset-preparation.md)
4. [Pretrain Process](processes/pretrain.md)
5. [SFT Process](processes/sft.md)
6. [RL Process](processes/rl.md)
7. [Results](results.md)
8. [Demo](demo.md)
