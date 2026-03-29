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
