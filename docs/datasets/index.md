# Datasets Overview

The repository uses different datasets for different stages. They should not be treated as a single generic corpus.

## Dataset Families

There are five major data families in this project:

1. Pretraining text for the base model
2. Chat/task datasets for supervised fine-tuning
3. GSM8K rollouts for reinforcement learning
4. CORE benchmark data for base-model evaluation
5. Chat evaluation task datasets for SFT and RL checkpoints

## Configured Locations

Dataset roots are defined in `configs/global.yaml`:

- `dataset.root`: `.cache/dataset`
- `dataset.pretrain`: `pretrain/fineweb-edu-100b-shuffle`
- `dataset.eval`: `eval`
- `dataset.task.root`: `task`

Under the task root, the repo expects:

- `identity_conversations.jsonl`
- `words_alpha.txt`
- `ai2_arc`
- `gsm8k`
- `humaneval`
- `mmlu`
- `smol-smoltalk`

## How The Repo Uses Data

The main rule is:

- base pretraining consumes raw text
- SFT consumes rendered conversations or synthetic task conversations
- RL consumes prompt/completion rollouts over GSM8K
- base evaluation consumes the CORE bundle and pretraining text
- chat evaluation consumes task-specific chat/eval datasets

## Dataset Pages

- [Pretrain Dataset](pretrain.md)
- [SFT Datasets](sft.md)
- [RL Dataset](rl.md)
- [Base Evaluation Dataset](base-evaluation.md)
- [Chat Evaluation Datasets](chat-evaluation.md)

## Important Distinctions

Do not conflate these categories:

- The pretraining corpus is not a chat dataset.
- The CORE eval bundle is not used for SFT or RL.
- The simple spelling word list is not conversation data by itself; it is used to synthesize spelling tasks.
- GSM8K appears in both SFT and RL, but it is used differently in each stage.
