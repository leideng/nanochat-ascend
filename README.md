# nanochat-ascend

Training Karpathy's nanochat on Huawei Ascend NPU, with CPU support for local development and very small-scale tests.

## Environment Setup

This project uses `uv` for package management.

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

- GPU/CUDA is not supported in this repository.
- On CPU machines, only CPU and meta-device tests should be executed.
- NPU tests should be run only on real Ascend hardware.
