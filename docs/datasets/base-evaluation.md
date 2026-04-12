# Base Evaluation Dataset

## Purpose

Base-model evaluation uses two data sources:

- the CORE benchmark bundle for task accuracy
- the pretraining corpus for bits-per-byte evaluation

This page focuses on the CORE benchmark bundle, because it is the dedicated base-evaluation dataset in the repository.

## Source And Location

The eval bundle is downloaded from:

- `https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip`

Configured local path:

```text
.cache/dataset/eval
```

## Bundle Structure

The repository expects the eval directory to contain at least:

- `core.yaml`
- `eval_meta_data.csv`
- `eval_data/**/*.jsonl`

`scripts/base_eval.py` reads:

- task registry and metadata from `core.yaml`
- random baselines from `eval_meta_data.csv`
- task examples from `eval_data`

## Task Types

The CORE bundle includes multiple ICL-style task types, including:

- `multiple_choice`
- `language_modeling`
- `schema`

Each task entry defines:

- a label
- a dataset URI
- few-shot configuration
- task type
- an optional continuation delimiter

## What The Script Computes

`evaluate_core(...)` in `scripts/base_eval.py` computes:

- raw task accuracy
- centered task accuracy using the random baseline
- the mean centered score across tasks, reported as the CORE metric

This makes the CORE metric more informative than raw average accuracy alone, because tasks with high random baselines are normalized.

## Relationship To BPB

The base evaluation pipeline also computes bits per byte on the pretraining dataset. That metric is not part of the CORE bundle, but it is part of the same `scripts/base_eval.py` workflow.

So in practice, base evaluation combines:

- benchmark-style task accuracy from the CORE bundle
- compression/language-model quality from train/val BPB
- qualitative samples from the generation engine

## Why This Dataset Is Base-Only

The CORE bundle evaluates the base model as a language model. It is not the same as the chat evaluation suite and is not used for:

- SFT data generation
- RL reward computation
- chat assistant benchmarking
