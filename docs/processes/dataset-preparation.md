# Dataset Preparation

## Goal

The dataset preparation stage downloads all datasets required by the repository into the paths configured by `configs/global.yaml`.

The standard wrapper is:

```bash
bash runs/prepare_dataset.sh
```

## What The Wrapper Does

`runs/prepare_dataset.sh` performs the following steps:

1. `source runs/set_env.sh`
2. create `.venv` if needed
3. detect whether the machine has an Ascend runtime available
4. run `uv sync --extra cpu` or `uv sync --extra npu`
5. activate the virtual environment
6. execute `python -m nanochat.dataset`

## What `python -m nanochat.dataset` Downloads

The dataset module downloads two groups of assets.

### URL-downloaded assets

- CORE eval bundle
- identity conversations JSONL
- simple spelling word list

### Hugging Face dataset snapshots

- `karpathy/fineweb-edu-100b-shuffle`
- `allenai/ai2_arc`
- `openai/gsm8k`
- `openai/openai_humaneval`
- `cais/mmlu`
- `HuggingFaceTB/smol-smoltalk`

The code uses `snapshot_download(...)` for the Hugging Face datasets and downloads entire dataset repos into local directories.

## Resulting Directory Shape

After preparation, users should expect a dataset tree shaped roughly like:

```text
.cache/dataset/
  pretrain/
    fineweb-edu-100b-shuffle/
      *.parquet
  eval/
    core.yaml
    eval_meta_data.csv
    eval_data/
  task/
    identity_conversations.jsonl
    words_alpha.txt
    ai2_arc/
    gsm8k/
    humaneval/
    mmlu/
    smol-smoltalk/
```

## Why Dataset Preparation Is Separate

The project intentionally keeps dataset acquisition out of the training scripts. This separation provides:

- reproducible path conventions
- one shared preparation entry point for CPU and NPU users
- clearer failure modes when a dataset is missing
- easier syncing to a separate Ascend training machine

## Validation Checklist

After preparation, it is reasonable to verify:

- the pretrain dataset directory contains parquet files
- the eval directory contains `core.yaml`
- the task directory contains the expected subdirectories and files

The dataset module itself prints parquet file metadata and sample iteration information after download, which is useful as a first sanity check.

## Environment Prerequisite

Manual commands that depend on config should be run after:

```bash
source runs/set_env.sh
```

This ensures the repository uses `configs/global.yaml` consistently.

## Related But Separate

Dataset preparation is not the same as tokenizer training.

Typical order:

1. prepare datasets
2. train tokenizer if needed
3. pretrain base model
4. fine-tune and evaluate
