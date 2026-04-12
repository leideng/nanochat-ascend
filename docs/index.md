# nanochat-ascend Documentation

`nanochat-ascend` is a small-LLM training project derived from Karpathy's nanochat and adapted for Huawei Ascend NPUs, with CPU support for local development, smoke tests, and documentation-friendly experimentation.

This documentation is organized around how the repository actually works:

- what the project trains and evaluates
- which datasets feed each stage
- how data is prepared locally
- how base pretraining, SFT, and RL are connected
- where to look for checkpoints, reports, and demos

## What This Project Contains

The repository has two main layers:

- `nanochat/`: the reusable library code for the model, optimizer, tokenizer, dataloading, evaluation, checkpointing, and generation
- `scripts/`: entry points for tokenizer training, base model pretraining, supervised fine-tuning, reinforcement learning, evaluation, and interactive chat

At a high level, the training flow is:

1. Prepare datasets into the paths configured by `configs/global.yaml`.
2. Train a tokenizer and/or reuse an existing tokenizer.
3. Pretrain a base causal language model on parquet text.
4. Supervised fine-tune the base model on a mixture of chat and task datasets.
5. Run reinforcement learning on GSM8K rollouts starting from the SFT checkpoint.
6. Evaluate both the base model and the chat models with task-specific evaluation scripts.

## Read This First

- Start with [Project Overview](overview.md) for architecture, runtime scope, and repository layout.
- Read [Datasets](datasets/index.md) to understand the data sources behind pretraining, SFT, RL, and evaluation.
- Read [Model](model.md) for the transformer design choices implemented here.
- Read [Dataset Preparation](processes/dataset-preparation.md), [Pretrain Process](processes/pretrain.md), [SFT Process](processes/sft.md), and [RL Process](processes/rl.md) for the end-to-end workflow.
- Read [Results](results.md) and [Demo](demo.md) for practical expectations.

## Runtime Scope

This repository is intentionally scoped to:

- Ascend NPU for real training and mainline evaluation
- CPU for local development, smoke tests, small demonstrations, and documentation workflows

GPU/CUDA is intentionally unsupported in this codebase.

## Environment Conventions

Most manual commands in this repository assume:

```bash
source runs/set_env.sh
```

That exports:

```bash
NANOCHAT_CONFIG=configs/global.yaml
```

The wrapper scripts under `runs/` source this configuration automatically when needed.

## Documentation Map

- [Project Overview](overview.md)
- [Datasets](datasets/index.md)
- [Model](model.md)
- [Dataset Preparation](processes/dataset-preparation.md)
- [Pretrain Process](processes/pretrain.md)
- [SFT Process](processes/sft.md)
- [RL Process](processes/rl.md)
- [Results](results.md)
- [Demo](demo.md)
