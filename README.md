# nanochat-ascend

Training Karpathy's nanochat on Huawei Ascend NPU, with CPU support for local development and very small-scale tests.

## Environment Setup

This project uses `uv` for package management.

Before running repo commands manually, load the repo config into your current shell:

```bash
source runs/set_env.sh
```

This sets `NANOCHAT_CONFIG=configs/global.yaml`. All runtime configuration is read from `configs/global.yaml`.

If you use the wrapper scripts under `runs/`, they will source `runs/set_env.sh` for you when needed.

CPU environment:

```bash
uv sync --extra cpu
```

Ascend NPU environment:

```bash
uv sync --extra npu
```

## Notes

- Run `source runs/set_env.sh` from the repo root before `python -m ...`, `uv run ...`, `torchrun ...`, or test commands that depend on repo config.
- Prefer the `runs/*.sh` wrappers for common workflows such as dataset preparation, tokenizer training, CPU demo runs, and NPU runs.
- GPU/CUDA is not supported in this repository.
- On CPU machines, only CPU and meta-device tests should be executed.
- NPU tests should be run only on real Ascend hardware.

## Common Workflows

Common entrypoints from the repo root:

```bash
# Load config for manual commands
source runs/set_env.sh

# Prepare datasets and install the right dependency extra for this machine
bash runs/prepare_dataset.sh

# Train and evaluate the tokenizer
bash runs/run_tok_train.sh

# Tiny base-train smoke run
bash runs/run_base_train.sh

# Evaluate the smoke-run checkpoint
bash runs/run_base_eval.sh

# Tiny SFT smoke run
bash runs/run_sft.sh

# Full CPU demo pipeline
bash runs/run_cpu.sh

# Full NPU pipeline, only on Ascend hardware
bash runs/run_npu.sh
```

## Rsync Helpers For A3

These scripts help sync code, datasets, checkpoints, and outputs between local and A3.
Run them from the repo root:

```bash
bash rsync-code-to-a3.sh
bash rsync-dataset-to-a3.sh
bash rsync-ckpt-from-a3.sh
bash rsync-output-from-a3.sh
```

### What Each Script Does

- `rsync-code-to-a3.sh`
  - Pushes code from local repo to `root@a3:/data/ldeng/code/nanochat-ascend`.
  - Syncs: `pyproject.toml`, `.python-version`, `configs/`, `nanochat/`, `runs/`, `scripts/`, `tasks/`, `tests/`.
  - Excludes `__pycache__/` and `*.pyc`.

- `rsync-dataset-to-a3.sh`
  - Pushes selected local datasets to `root@a3:/data/ldeng/code/nanochat-ascend/.cache/dataset/`.
  - Syncs only `.cache/dataset/eval` and `.cache/dataset/task`.
  - Does **not** sync the pretrain dataset; pretrain data should be set manually on A3.

- `rsync-ckpt-from-a3.sh`
  - Pulls checkpoints from `root@a3:/data/ldeng/code/nanochat-ascend/.cache/checkpoint`.
  - Syncs `base`, `chatsft`, and `chatrl` into local `.cache/checkpoint/`.
  - Creates local `.cache/checkpoint/` if missing.

- `rsync-output-from-a3.sh`
  - Pulls the full output directory from `root@a3:/data/ldeng/code/nanochat-ascend/.cache/output/`.
  - Writes into local `.cache/output/`.
  - Creates local `.cache/output/` if missing.



## Demo to Show d20 Performance

### Test 1

[Watch d20 Performance Test 1](assets/d20-chat-cli-videos/d20-test1.webm)

[![d20 Test 1](assets/d20-chat-cli-videos/d20-test1-thumbnail.png)](assets/d20-chat-cli-videos/d20-test1.webm)


### Test 2

[![d20 Test 2](assets/d20-chat-cli-videos/d20-test2-thumbnail.png)](assets/d20-chat-cli-videos/d20-test2.webm)
