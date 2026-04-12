# SFT Process

## Objective

The supervised fine-tuning stage turns the base model into a more usable chat assistant by training on a task mixture of conversations and synthetic/evaluated tasks.

The main entry point is:

```bash
bash runs/run_sft.sh
```

## Starting Point

SFT does not train from scratch. It loads a base-model checkpoint through:

- `load_model("base", ...)`

That means SFT assumes base pretraining has already completed and that the relevant checkpoint is available under the configured base checkpoint directory.

## Wrapper Behavior

`runs/run_sft.sh` chooses different defaults for NPU and CPU:

### On Ascend NPU

- distributed `torchrun`
- `model-tag="d20"`
- `max-seq-len=2048`
- full-epoch SFT with `num-iterations=-1`

### On CPU

- small test configuration
- `model-tag="d4-test"`
- `model-step=20`
- `max-seq-len=128`
- `num-iterations=20`

## Data Construction

The SFT script builds its own conversation-oriented generator instead of reusing the raw-text pretraining loader.

Key properties:

- every example is a rendered conversation
- conversations are packed BOS-aligned
- no conversation tokens are intentionally cropped when a row overflows
- leftover space is padded
- padded target positions are masked with `-1`

This is a better fit for multi-turn data than the pretraining packing path.

## Training Mixture Intent

The training mixture balances multiple goals:

- general assistant behavior from SmolTalk
- repository/project identity behavior from custom JSON conversations
- multiple-choice reasoning from MMLU
- mathematical/chat reasoning from GSM8K
- spelling and character-level reliability from synthetic spelling tasks

The result is a broad assistant finetune rather than a narrowly specialized task head.

## Optimization And Scheduling

The script:

- computes gradient accumulation from `total_batch_size`, `device_batch_size`, `max_seq_len`, and world size
- initializes the same optimizer family used elsewhere
- optionally compiles the model when eager mode is not enforced
- uses a simple late-stage learning-rate rampdown
- adjusts Muon momentum over early training

## Validation

During SFT, the script periodically evaluates validation bits per byte on the SFT validation mixture. That is not the same as final chat evaluation, but it provides a training-time signal for overfitting and improvement.

Final chat capability should be measured with:

- `scripts/chat_eval.py`

## Outputs

SFT checkpoints are written under:

```text
.cache/checkpoint/chatsft_checkpoints/<model_tag>
```

These checkpoints become the starting point for:

- direct chat evaluation
- RL fine-tuning
- interactive CLI or web demos

## Recommended Sequence

The intended stage order is:

1. base pretrain
2. SFT
3. chat evaluation on the SFT checkpoint
4. optional RL
