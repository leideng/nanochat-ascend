# Pretrain Process

## Objective

The pretrain stage builds the base causal language model from raw text. This stage is implemented by `scripts/base_train.py` and usually started through:

```bash
bash runs/run_base_train.sh
```

## Inputs

Base pretraining depends on:

- prepared parquet text under the configured pretrain dataset directory
- a tokenizer available through the repo tokenizer setup
- runtime configuration from `configs/global.yaml`

## Wrapper Behavior

`runs/run_base_train.sh` selects different defaults based on machine type.

### On Ascend NPU

The wrapper launches distributed training with `torchrun` and a large d20-style configuration, including:

- `depth=20`
- `aspect-ratio=64`
- `head-dim=128`
- `window-pattern=L`
- `max-seq-len=2048`
- `device-batch-size=8`
- `target-param-data-ratio=20`

### On CPU

The wrapper runs a much smaller smoke configuration, typically:

- `depth=4`
- `head-dim=16`
- `max-seq-len=128`
- `device-batch-size=8`
- `total-batch-size=1024`
- `num-iterations=20`

This is for validation and learning, not serious model quality.

## What The Training Script Does

At a high level, `scripts/base_train.py`:

1. parses runtime, model, optimization, and evaluation arguments
2. initializes distributed state and device selection
3. loads the tokenizer and derives vocab size
4. constructs the model on the `meta` device
5. materializes parameters on the actual device with `to_empty(...)`
6. initializes weights
7. builds the optimizer
8. creates the BOS-aligned distributed train/val data loaders
9. computes total iterations from explicit steps, FLOP target, or param/data ratio
10. trains with periodic BPB evaluation, CORE evaluation, sampling, and checkpointing

## Training Horizon Options

The script supports three ways to define how long training should run:

- explicit `--num-iterations`
- derived from `--target-flops`
- derived from `--target-param-data-ratio`

The repo’s main d20 training run uses the parameter-to-data ratio path.

## Batch Size Logic

The script can auto-compute `total_batch_size` when it is set to `-1`. It then scales learning rates and weight decay relative to a reference batch regime.

This means the pretraining script is not only a fixed hyperparameter launcher; it also contains scaling logic for batch-dependent optimization settings.

## Evaluations During Training

The base pretraining loop can perform:

- validation BPB every `eval_every`
- CORE metric evaluation every `core_metric_every`
- qualitative sampling every `sample_every`
- checkpoint saves every `save_every`

These are useful because pretraining quality is not fully captured by a single scalar.

## Outputs

Base pretraining writes to:

- checkpoint directory under `.cache/checkpoint/base_checkpoints/<model_tag>`
- output/report locations configured under `.cache/output`

Checkpoints include:

- model weights
- optimizer state
- metadata such as model config, step, and loader resume state

## Recommended Local Flow

For a local CPU-only machine:

```bash
source runs/set_env.sh
uv sync --extra cpu
bash runs/prepare_dataset.sh
bash runs/run_base_train.sh
```

For real training, users should run the NPU configuration on Ascend hardware.

## What Comes Next

The output of pretraining is the input to supervised fine-tuning. In practice, SFT loads the base checkpoint family using the chosen `model_tag`.
