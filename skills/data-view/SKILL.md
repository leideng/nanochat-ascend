---
name: data-view
description: Explain how datasets in nanochat-ascend are structured and where they are used. Use this skill when asked to understand dataset formats, inspect sample records, map config dataset paths to training/evaluation code, or explain the role of pretrain, CORE eval, SmolTalk, identity conversations, simple spelling, ARC, GSM8K, HumanEval, MMLU, or other task datasets in the project.
---

# Data View

## Overview

Use this skill to answer dataset questions in `nanochat-ascend` precisely. Focus on three things:

- identify the on-disk format of a dataset
- identify which task/script consumes it
- explain how the raw record is transformed into model inputs or evaluation logic

## Workflow

1. Start from the configured dataset path in `configs/local.yaml` or from the user-provided sample.
2. Distinguish between:
   - base pretraining data
   - CORE/base-eval benchmark data
   - chat task datasets used by `tasks/`
3. Map the dataset to the code path that consumes it:
   - `nanochat/dataset.py` and `nanochat/dataloader.py` for pretrain parquet text
   - `scripts/base_eval.py` and `nanochat/core_eval.py` for CORE eval bundle data
   - `tasks/*.py`, `scripts/chat_sft.py`, `scripts/chat_eval.py`, and `scripts/chat_rl.py` for chat-task datasets
4. Explain whether the raw records are:
   - consumed directly
   - wrapped into conversation objects
   - transformed into synthetic tasks
   - scored by continuation loss, categorical choice, or executable tests
5. Use the bundled reference file for concrete dataset samples and code-path summaries.

## Repo Rules

- Treat `pretrain_dataset` separately from task datasets. It is parquet text used for base LM training, not chat data.
- Treat `eval_dataset` separately from `tasks/`. It is the CORE benchmark bundle for `base_eval`, not chat SFT or chat eval.
- Distinguish `simple_spelling_dataset` from chat JSON datasets. It is a word list used to synthesize spelling tasks.
- When the user shows a sample, explain the exact transformation into the repo's internal representation.

## Reference File

Read [references/datasets.md](references/datasets.md) when you need:
- concrete samples for each configured dataset
- a quick summary of how each dataset is used
- the file/module path that consumes each dataset
