# nanochat-ascend

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-nanochat--ascend-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/leideng/nanochat-ascend-d32-rl-pt)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-nanochat--ascend-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/datasets/leideng/nanochat-ascend-dataset)

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

Common entrypoints from the repo root.

>[!NOTE]
>
>It is recommended to run workflows phase by phase. The generated output of the current phase will be saved in the corresponding folder specified in [configs/global.yaml] such that the next phase can read from the saved folder.

```bash
# Load config for manual commands
source runs/set_env.sh

# Download and Prepare datasets
bash runs/prepare_dataset.sh

# Train and evaluate the tokenizer
bash runs/run_tok_train.sh

# Pretraining
bash runs/run_base_train.sh

# Evaluate base model
bash runs/run_base_eval.sh

# Run SFT
bash runs/run_sft.sh

# Run RL
bash runs/run_rl.sh

# Evaluate chat model after SFT/RL
bash runs/run_sft_eval.sh
```


## d20 Demo

[![d20 Test 1](assets/d20-chat-cli-videos/d20-test1-thumbnail.png)](assets/d20-chat-cli-videos/d20-test1.webm)
