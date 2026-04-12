# SFT Datasets

## Purpose

Supervised fine-tuning uses a mixture of datasets rather than a single source. The goal is to turn the base model into a chat-capable assistant with better response formatting, instruction following, identity behavior, spelling behavior, and task-oriented reasoning.

## Training Mixture

`scripts/chat_sft.py` constructs a `TaskMixture` made from:

- `SmolTalk(split="train")`
- `MMLU(subset="auxiliary_train", split="train")`
- `GSM8K(subset="main", split="train")`
- a second copy of `GSM8K(subset="main", split="train")`
- `CustomJSON(filepath=identity_conversations_filepath)`
- a second copy of the identity conversations file
- `SimpleSpelling(size=200000, split="train")`
- `SpellingBee(size=80000, split="train")`

The validation mixture includes:

- `SmolTalk(split="test")`
- `MMLU(subset="all", split="test", stop=5200)`
- `GSM8K(subset="main", split="test", stop=420)`

## Dataset Components

### SmolTalk

Source dataset:

- `HuggingFaceTB/smol-smoltalk`

Format:

- conversational records with a `messages` field

Role in SFT:

- broad instruction-following and conversational behavior
- the largest general chat component in the mixture

### Identity Conversations

Source file:

- `identity_conversations.jsonl`

Format:

- one conversation array per line
- alternating `user` and `assistant` messages

Role in SFT:

- teaches the model how to describe itself and the project
- provides synthetic identity and explanation-oriented chat examples

### MMLU

Source dataset:

- `cais/mmlu`

Role in SFT:

- teaches multiple-choice reasoning and answer formatting
- in SFT, examples are rendered as chat-style conversations with the assistant expected to answer with the correct choice

### GSM8K

Source dataset:

- `openai/gsm8k`

Role in SFT:

- teaches basic mathematical reasoning and answer production
- appears twice in the training mixture to upweight the task

### SimpleSpelling

Source file:

- `words_alpha.txt`

Role in SFT:

- generates synthetic spelling tasks such as spelling a target word

### SpellingBee

Derived from:

- the same word list used by `SimpleSpelling`

Role in SFT:

- generates spelling/counting-style tasks such as counting letters in a word

## How SFT Packing Works

The SFT data path does not use the same raw-text loader as pretraining. Instead:

1. each task yields a conversation object
2. the tokenizer renders the conversation into token ids
3. a custom BOS-aligned best-fit packer fills rows up to `max_seq_len + 1`
4. if no conversation fits in the remaining space, the row is padded instead of cropped
5. padding targets are masked with `-1`

That means the SFT loader preserves full conversations rather than discarding tail tokens to force a fit.

## Why The SFT Mixture Matters

Each component teaches a different capability:

- SmolTalk: general assistant behavior
- identity JSON: repository/project identity
- MMLU: multiple-choice reasoning
- GSM8K: math and structured answers
- spelling tasks: synthetic reliability on character-level tasks

The final SFT model is therefore not just a chat wrapper over the base model. It is a deliberately blended task mixture.
