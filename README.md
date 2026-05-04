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
>It is recommended to run workflows phase by phase. The generated output of the current phase will be saved in the corresponding folder specified in [the config file](configs/global.yaml) such that the next phase can read from the saved folder.

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

## Performance

### Data sources

| Reference | Source |
| --- | --- |
| nanochat-ascend d20 — pretraining | [base-model-training.md](dev/d20_eval_results/base-model-training.md) |
| nanochat-ascend d20 — base evaluation | [base-model-evaluation.md](dev/d20_eval_results/base-model-evaluation.md) |
| nanochat-ascend d32 — pretraining | [base-model-training (iter 16k–17k).md](dev/d32_eval_results/base-model-training-iter-from-16000-to-17000.md) |
| nanochat-ascend d32 — base evaluation | [base-model-evaluation.md](dev/d32_eval_results/base-model-evaluation.md) |
| Karpathy d20 (upstream speedrun) | [nanochat gitHub discussion #1](https://github.com/karpathy/nanochat/discussions/1) |
| Karpathy d32 (upstream $1000 run) | [nanochat gitHub discussion #8](https://github.com/karpathy/nanochat/discussions/8) |

Comparison of base pretraining runs versus upstream nanochat. **Depth labels are not the same architecture:** nanochat-ascend uses a wider configuration at a given depth, so parameter counts and compute differ from Karpathy’s runs.

nanochat-ascend defaults to **eager** execution (`enforce_eager: true` in [`configs/global.yaml`](configs/global.yaml)). Upstream trains with **`torch.compile`** on CUDA (graph mode). **Vocab size** here is 32,768 (2^15) in this fork’s model config ([`nanochat/gpt.py`](nanochat/gpt.py)), versus 65,536 (2^16) in upstream nanochat ([Discussion #1](https://github.com/karpathy/nanochat/discussions/1)).

| Metric | nanochat-ascend d20 | Karpathy d20 | nanochat-ascend d32 | Karpathy d32 |
| --- | --- | --- | --- | --- |
| **Parameters** | 896,535,720 | 560,988,160 | 2,818,580,544 | 1,879,048,192 |
| **Vocab size** | 32,768 (2^15) | 65,536 (2^16) | 32,768 (2^15) | 65,536 (2^16) |
| **Training Model** | Eager | `torch.compile` (CUDA) | Eager | `torch.compile` (CUDA) |
| **Training tokens** | 8,703,180,800 | 11,219,763,200 | 35,651,584,000 | 37,580,963,840 |
| **Tokens∶params** | 9.7 | 20.0 | 12.6 | 20.0 |
| **Iterations** | 8,300 | 21,400 | 17,000 | 71,680 |
| **Total training FLOPs** | 2.82×10¹⁹ | 3.92×10¹⁹ | 4.16×10²⁰ | 4.54×10²⁰ |
| **Final val BPB** | 0.7811 | 0.81 | 0.7026 | 0.7236 |
| **CORE** (`base_eval`) | 0.2167 | 0.2219 | 0.2881 | 0.3168 |


## d20 Demo

[![d20 Test 1](assets/d20-chat-cli-videos/d20-test1-thumbnail.png)](assets/d20-chat-cli-videos/d20-test1.webm)
