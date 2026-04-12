# Chat Evaluation Datasets

## Purpose

Chat evaluation measures the capabilities of SFT and RL checkpoints on assistant-style tasks. These datasets are consumed by `scripts/chat_eval.py`.

## Supported Evaluation Tasks

The script currently supports:

- ARC-Easy
- ARC-Challenge
- MMLU
- GSM8K
- HumanEval
- SpellingBee

These are built from the task modules under `tasks/`.

## Task Breakdown

### ARC

Source dataset:

- `allenai/ai2_arc`

Variants used:

- `ARC-Easy`
- `ARC-Challenge`

Evaluation type:

- categorical

How it works:

- the conversation is rendered as a prompt
- the model scores only the answer-letter tokens for the available choices
- the highest-scoring letter becomes the prediction

### MMLU

Source dataset:

- `cais/mmlu`

Evaluation type:

- categorical

How it works:

- a multiple-choice prompt is rendered
- the model is evaluated on the logits of answer letters rather than free-form generation

### GSM8K

Source dataset:

- `openai/gsm8k`

Evaluation type:

- generative

How it works:

- the model samples one or more completions
- correctness is determined by task-specific evaluation logic

### HumanEval

Source dataset:

- `openai/openai_humaneval`

Evaluation type:

- generative

How it works:

- the model generates code completions
- correctness is determined by the HumanEval task implementation

### SpellingBee

Derived from:

- the `words_alpha.txt` word list

Evaluation type:

- generative

How it works:

- the model must answer spelling/counting style prompts correctly

## Categorical Vs Generative Evaluation

The chat evaluation pipeline contains two distinct evaluation modes:

- categorical:
  - used for multiple-choice tasks such as ARC and MMLU
  - runs efficiently in batches by looking only at choice-token logits
- generative:
  - used for tasks where the answer must be produced, such as GSM8K and HumanEval
  - uses the generation engine and task-specific correctness checks

## Aggregate Metric

When all chat-eval tasks are run, `scripts/chat_eval.py` also computes a centered aggregate metric referred to as ChatCORE.

This metric:

- normalizes each task against a task-specific baseline
- averages the centered task scores
- makes it easier to compare chat checkpoints at a glance

## Recommended Interpretation

These datasets are best read together rather than separately:

- ARC and MMLU emphasize multiple-choice reasoning
- GSM8K emphasizes arithmetic generation
- HumanEval emphasizes code generation
- SpellingBee emphasizes low-level spelling/counting reliability

A model that improves on one may regress on another, especially after RL.
