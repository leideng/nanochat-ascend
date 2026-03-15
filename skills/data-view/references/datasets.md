# Dataset Guide

Use this reference when explaining how datasets in `nanochat-ascend` are structured and consumed.

## Pretrain Dataset

Configured path:
- `pretrain_dataset`

Primary consumers:
- `nanochat/dataset.py`
- `nanochat/dataloader.py`
- `scripts/base_train.py`
- BPB path in `scripts/base_eval.py`

Format:
- Parquet rows with a `text` column

Sample:

```json
{
  "text": "Shipment & Transport-Sea, Air, Rail, Road, Pipeline\nThe mode of transportation is an important consideration when planning the shipment process. ..."
}
```

Usage:
- raw text corpus for base language-model pretraining
- loaded from parquet files
- packed into BOS-aligned token batches by the distributed dataloader
- not a chat dataset

## CORE Eval Bundle

Configured path:
- `eval_dataset`

Primary consumers:
- `scripts/base_eval.py`
- `nanochat/core_eval.py`

Format:
- directory bundle containing `core.yaml`, `eval_meta_data.csv`, and `eval_data/**/*.jsonl`

Task registry sample:

```yaml
icl_tasks:
- label: hellaswag_zeroshot
  dataset_uri: language_understanding/hellaswag.jsonl
  num_fewshot: [0]
  icl_task_type: multiple_choice
```

Row sample:

```json
{
  "query": "Concept: turbulent peace. Question: Which of the following sentences best characterizes turbulent peaces?\nA. Turbulent peace is dangerous for planes.\nB. Turbulent peace is windy.\nC. Turbulent peace is short-lived.\nD. Turbulent peace is full of harmony.\nAnswer: ",
  "choices": [
    "Turbulent peace is dangerous for planes.",
    "Turbulent peace is windy.",
    "Turbulent peace is short-lived.",
    "Turbulent peace is full of harmony."
  ],
  "gold": 2
}
```

Usage:
- benchmark bundle for base-model CORE evaluation
- supports `multiple_choice`, `language_modeling`, and `schema` ICL task types
- not used for chat SFT, chat RL, or chat eval

## Identity Conversations

Configured path:
- `sft_dataset`

Primary consumers:
- `tasks/customjson.py`
- `scripts/chat_sft.py`

Format:
- JSONL, one conversation array per line

Sample:

```json
[
  {
    "role": "user",
    "content": "Hi there! I'm a computer science instructor and I'm looking at using nanochat to teach my students about attention mechanisms. ..."
  },
  {
    "role": "assistant",
    "content": "Hello! It is great to meet an educator. I actually depart from the vanilla Transformer in a few key ways ..."
  },
  {
    "role": "user",
    "content": "Oh, so you just apply LayerNorm before the attention block? ..."
  },
  {
    "role": "assistant",
    "content": "Not quite. While I do use RMSNorm before each attention and MLP block, QK Normalization is a specific extra step. ..."
  }
]
```

Usage:
- synthetic identity and project-explanation conversations
- loaded as chat examples for SFT
- must alternate `user` and `assistant`

## SmolTalk

Configured path:
- `huggingface_tb_smol_smoltalk_dataset`

Primary consumers:
- `tasks/smoltalk.py`
- `scripts/chat_sft.py`

Format:
- Hugging Face conversational dataset with `messages`

Sample:

```json
{
  "messages": [
    {
      "content": "Provide a concise, objective summary of the input text in up to three sentences, focusing on key actions and intentions without using second or third person pronouns.",
      "role": "system"
    },
    {
      "content": "Uruguay hero Fernando Muslera was delighted after his team stunned hosts Argentina in a penalty shootout ...",
      "role": "user"
    },
    {
      "content": "Uruguay's goalkeeper Fernando Muslera saved a crucial penalty from Carlos Tevez, helping his team defeat Argentina 5-4 on penalties ...",
      "role": "assistant"
    }
  ],
  "source": "smol-summarize-20k"
}
```

Usage:
- broad general chat/instruction-following corpus for SFT
- optional `system` message allowed at the start

## Simple Spelling Word List

Configured path:
- `simple_spelling_dataset`

Primary consumers:
- `tasks/spellingbee.py`
- `scripts/chat_sft.py`

Format:
- plain text, one word per line

Sample:

```text
a
aa
aaa
aah
aahed
aahing
aardvark
aardwolves
```

Usage:
- source vocabulary for synthetic spelling tasks
- not used directly as conversation data
- drives `SimpleSpelling` and `SpellingBee`

## MMLU

Configured path:
- `cais_mmlu_dataset`

Primary consumers:
- `tasks/mmlu.py`
- `scripts/chat_sft.py`
- `scripts/chat_eval.py`

Format:
- multiple-choice QA rows

Sample:

```json
{
  "question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
  "subject": "abstract_algebra",
  "choices": ["0", "4", "2", "6"],
  "answer": 1
}
```

Usage:
- converted into a multiple-choice chat prompt
- assistant target is the correct letter `A/B/C/D`
- used for SFT and chat evaluation

## ARC

Configured path:
- `allenai_arc_dataset`

Primary consumers:
- `tasks/arc.py`
- `scripts/chat_eval.py`

Format:
- multiple-choice science QA rows

Sample:

```json
{
  "id": "Mercury_SC_415702",
  "question": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
  "choices": {
    "text": [
      "dry palms",
      "wet palms",
      "palms covered with oil",
      "palms covered with lotion"
    ],
    "label": ["A", "B", "C", "D"]
  },
  "answerKey": "A"
}
```

Usage:
- converted into a multiple-choice chat prompt
- used for chat evaluation only in current code

## GSM8K

Configured path:
- `openai_gsm8k_dataset`

Primary consumers:
- `tasks/gsm8k.py`
- `scripts/chat_sft.py`
- `scripts/chat_eval.py`
- `scripts/chat_rl.py`

Format:
- JSON rows with `question` and worked `answer`

Sample:

```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

Usage:
- converted into a generative reasoning conversation
- `<<expr=result>>` spans are split into `python` and `python_output` message parts
- used for SFT, chat eval, and RL reward training

## HumanEval

Configured path:
- `openai_humaneval_dataset`

Primary consumers:
- `tasks/humaneval.py`
- `scripts/chat_eval.py`

Format:
- coding benchmark rows

Sample:

```json
{
  "task_id": "HumanEval/0",
  "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    ...",
  "canonical_solution": "    for idx, elem in enumerate(numbers):\n        ...",
  "test": "\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    ...",
  "entry_point": "has_close_elements"
}
```

Usage:
- prompt becomes the user coding task
- generated code is executed against the provided tests
- used for chat evaluation only
