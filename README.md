# nanochat-ascend

Training Karpathy's nanochat on Huawei Ascend NPU, with CPU support for local development and very small-scale tests.

## Environment Setup

This project uses `uv` for package management.

Before running any project command, load the repo config into your current shell:

```bash
source runs/set_env.sh
```

This sets `NANOCHAT_CONFIG=configs/global.yaml`. All runtime configuration is read from `configs/global.yaml`.

CPU environment:

```bash
uv sync
```

Ascend NPU environment:

```bash
uv sync --extra npu
```

In this repo, `uv sync` is the default CPU setup path and is equivalent in practice to `uv sync --extra cpu`.

## Notes

- Run `source runs/set_env.sh` from the repo root before `python -m ...`, `uv run ...`, or test commands that depend on repo config.
- GPU/CUDA is not supported in this repository.
- On CPU machines, only CPU and meta-device tests should be executed.
- NPU tests should be run only on real Ascend hardware.
