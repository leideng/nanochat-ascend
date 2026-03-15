# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Project Overview

`nanochat-ascend` trains a small GPT chat model, adapted from Karpathy's nanochat, for Huawei Ascend NPU (910C). It supports Ascend NPU and CPU only. GPU/CUDA is intentionally unsupported and all GPU/CUDA-specific code has been removed. The project uses UV for package management with Python 3.11.

## Execution Constraints For This Machine

The current development machine is CPU-only.

- Agents may generate or modify NPU code, but must not execute NPU code on this machine.
- Agents must not run commands that initialize or depend on Ascend runtime availability.
- Agents must not attempt to validate changes with CUDA or GPU commands. GPU is not supported anywhere in this repo.
- Agents may execute only CPU tests and meta-device tests on this machine.
- Meta-device tests are for shape and dtype validation only. They must not be treated as runtime correctness or performance validation.
- For NPU debugging or verification, the user runs commands on an Ascend machine and pastes logs or outputs back into the conversation. Agents should reason from those logs instead of trying to reproduce NPU execution locally.

Practical implications:

- Prefer `--device-type=cpu` when executing repo code locally.
- Keep CPU validation very small: tiny batch sizes, tiny sequence lengths, tiny iteration counts.
- Prefer `NANOCHAT_ENFORCE_EAGER=1` for quick CPU smoke tests unless the task specifically concerns compile behavior.
- Prefer meta-device construction for model-shape checks when a forward or training step is unnecessary.
- Do not run `torchrun` NPU launch commands locally.

## Environment Setup

Use `uv` for dependency and environment management.

Before running repo commands manually that depend on config, agents and users must first run:

```bash
source runs/set_env.sh
```

This exports `NANOCHAT_CONFIG=configs/global.yaml`. Runtime config should be read from `configs/global.yaml` through that environment variable; do not rely on ad hoc config env vars.
If a task uses the wrapper scripts under `runs/`, those scripts may source `runs/set_env.sh` internally.

CPU machine:

```bash
uv sync
```

Ascend NPU machine:

```bash
uv sync --extra npu
```

Notes:

- Run `source runs/set_env.sh` from the repo root before `python -m ...`, `uv run ...`, `torchrun ...`, or pytest commands that need repo config.
- Prefer the `runs/*.sh` wrappers for common repo workflows unless the task specifically requires invoking modules directly.
- In this repo, `uv sync` is the CPU/default setup path and is equivalent in practice to `uv sync --extra cpu`.
- On this CPU-only machine, agents should normally use `uv sync`.
- Agents must not assume that an NPU environment is runnable locally just because `--extra npu` can be resolved or installed elsewhere.
- If the lockfile or dependency graph changes, prefer updating it with `uv lock` and then syncing with the appropriate `uv sync --extra ...` command for the target machine.

## Common Commands

```bash
# Environment setup for manual commands
source runs/set_env.sh
uv sync
uv sync --extra npu

# Common wrapper scripts
bash runs/prepare_dataset.sh
bash runs/run_tok_train.sh
bash runs/run_base_train.sh
bash runs/run_base_eval.sh
bash runs/run_sft.sh
bash runs/run_cpu.sh

# Full NPU pipeline, only on Ascend hardware
bash runs/run_npu.sh

# Direct module examples when wrappers are not appropriate
python -m nanochat.dataset
python -m scripts.tok_train --max-chars=2000000
python -m scripts.base_train --device-type=cpu --depth=6 --max-seq-len=64 --device-batch-size=1 --total-batch-size=8 --num-iterations=2
python -m scripts.base_eval --device-type=cpu --device-batch-size=1 --split-tokens=1024
python -m scripts.chat_cli --device-type=cpu -p "What is the capital of France?"
```

Pytest is configured for the repo under `tests/`. Use `uv run pytest` to run local tests after syncing the environment.

## Validation Policy

When changing code, agents should validate in this order:

1. Static inspection and targeted reasoning.
2. Meta-device checks for model construction, tensor shapes, and checkpoint shape plumbing.
3. Tiny CPU smoke tests for code paths that require real execution.
4. User-provided NPU logs for Ascend-specific behavior, regressions, or runtime failures.

If a change is NPU-specific and cannot be meaningfully exercised on CPU or meta, the agent should still implement the change, explain the validation gap, and tell the user exactly what to run on NPU and which outputs to paste back.

## Architecture

The codebase has two layers: **nanochat/** (core library) and **scripts/** (entry points).

### Core Library (nanochat/)

- **gpt.py** - GPT model with RoPE, RMSNorm (no learnable params), ReLU^2 MLP, Group-Query Attention, untied embeddings, sliding window attention (configurable L/S pattern), and Flash Attention support adapted for Ascend and CPU.
- **optim.py** - MuonAdamW optimizer: AdamW for embeddings/scalars, Muon (orthogonal Newton) for matrices, with Polar Express orthogonalization and NorMuon variance reduction.
- **engine.py** - Inference engine with KV cache, top-k/temperature sampling, batch generation, and calculator tool integration.
- **tokenizer.py** - BPE tokenizer with dual backend: HuggingFace Tokenizer or RustBPE+Tiktoken. GPT-4 split pattern. Special tokens: `<|bos|>`, `<|user_start|>`, `<|assistant_start|>`, `<|python_start|>`, etc.
- **dataloader.py** - Distributed data loader using BOS-aligned best-fit cropping (every row starts with BOS, no padding, ~35% tokens cropped at T=2048). Multi-threaded with DDP sharding.
- **flash_attention.py** - PyTorch SDPA attention. Handles both training and inference, including KV cache paths.
- **common.py** - Device autodetection (NPU -> CPU), DDP init (HCCL for NPU), logging, `print0()` for rank-0 output, peak FLOPS tables.
- **checkpoint_manager.py** - Save/load model+optimizer+metadata, bfloat16->float32 conversion for CPU, torch.compile key patching.
- **core_eval.py** - CORE metric evaluation (from DCLM paper), few-shot prompting, multiple-choice scoring.
- **loss_eval.py** - Bits-per-byte metric, tokenization-independent, distributed reduction.

### Scripts (scripts/)

- **base_train.py** - Pre-training with configurable depth, attention pattern, compute-budget or Chinchilla-ratio targets.
- **chat_sft.py** - Multi-task SFT mixing GSM8K, MMLU, SmolTalk, SpellingBee, custom JSON.
- **chat_rl.py** - Policy gradient RL training.
- **chat_eval.py** - Generative and categorical evaluation (HumanEval, MMLU, ARC, GSM8K).
- **chat_cli.py** / **chat_web.py** - CLI and FastAPI web chat interfaces.
- **tok_train.py** / **tok_eval.py** - Tokenizer training and evaluation.
- **base_eval.py** - Base model eval (CORE, BPB, sample generation).

## Key Environment Variables

- `NANOCHAT_BASE_DIR` - Cache/checkpoint directory (default: `~/.cache/nanochat`)
- `NANOCHAT_BASE_DATA_DIR` - Path to parquet dataset files (e.g. fineweb-edu-100b-shuffle)
- `NANOCHAT_ENFORCE_EAGER` - Set to `"1"` to disable torch.compile (run in eager mode)
- `WANDB_RUN` - W&B run name

## Key Design Patterns

- **Device-agnostic within repo scope**: Ascend NPU and CPU only. No GPU/CUDA support.
- **DDP-aware**: `print0()` for rank-0 logging, per-rank optimizer checkpoints, all-reduce for metrics.
- **Meta-first construction**: training code already uses `with torch.device("meta")` and `to_empty()` for shape-first model initialization.
- **Checkpoints**: `model_{step:06d}.pt`, `meta_{step:06d}.json`, `optim_{step:06d}_rank{rank}.pt` in base dir.
- **Module execution**: All scripts run via `python -m scripts.<name>` or `python -m nanochat.<name>`, not as standalone files.

## Skills

A skill is a local instruction bundle stored in a `SKILL.md` file under `skills/`.

### Available skills

- **meta-test** - Use for shape-only validation, meta-device model construction, tensor shape checks, and checkpoint/model wiring that should not allocate real tensors. File: `skills/meta-test/SKILL.md`
- **cpu-test** - Use for tiny real-execution smoke tests on CPU, especially when validating forward passes, training steps, loaders, checkpoints, or CLI entry points without NPU access. File: `skills/cpu-test/SKILL.md`
- **data-view** - Use when explaining how a dataset is structured, where it is loaded, and how it is transformed for pretraining, CORE eval, chat SFT, chat eval, or chat RL. File: `skills/data-view/SKILL.md`

### How to use skills

- Use `meta-test` when the task is about shapes, initialization, meta tensors, or structure validation without real execution.
- Use `cpu-test` when the task needs a real run but can be reduced to a tiny CPU-only smoke test.
- Use `data-view` when the task is to understand dataset format, sample records, config dataset paths, or code paths consuming the datasets.
- Skills are repo-specific guardrails. They do not override the machine constraint that NPU code must not be executed here.
- If an NPU-only issue cannot be resolved locally, ask the user to run the relevant command on Ascend hardware and paste the output.
