# Demo

## Purpose

The repository includes both CLI and web chat entry points so users can inspect qualitative behavior after SFT or RL.

## CLI Demo

The simplest interactive path is the chat CLI. The wrapper script is:

```bash
bash runs/run_chat_cli.sh
```

The wrapper:

- loads repo config
- activates `.venv`
- auto-detects CPU vs NPU
- launches `scripts.chat_cli`

By default it points to:

- `--model-tag="d20"`
- `--source="rl"`

So the default demo path is the RL checkpoint when available.

## Direct CLI Example

You can also launch the CLI manually, for example on CPU:

```bash
source runs/set_env.sh
python -m scripts.chat_cli --device-type=cpu -p "What is the capital of France?"
```

## Web Demo

The repository also includes:

- `scripts/chat_web.py`

This provides a browser-based chat interface for qualitative inspection.

## CPU Demo Workflow

For local experimentation on a CPU-only machine, the repository already provides a practical demo script:

```bash
bash runs/run_cpu.sh
```

This script is explicitly educational rather than performance-oriented. It demonstrates a reduced local path that can include:

- optional tokenizer training
- a small base-train run
- SFT on CPU
- CLI or web chat afterward

## Included Media

The repository also contains example d20 demo media under:

```text
assets/d20-chat-cli-videos/
```

These assets are useful for documentation, presentations, and quick visual inspection of the project’s output style.

## What To Expect From Each Stage

### Base model

The base model is useful for qualitative completion tests, but it is not the intended end-user chat experience.

### SFT model

The SFT model is the first checkpoint family that should feel like an assistant.

### RL model

The RL model may be better on reward-targeted tasks such as GSM8K, but that does not always mean it is the best all-around conversational checkpoint.

## Suggested Demo Prompts

Good first prompts for this project are:

- factual: "What is the capital of France?"
- identity: "What is nanochat-ascend?"
- reasoning: "If 5*x + 3 = 13, what is x?"
- spelling: "How many r letters are in strawberry?"

These align well with the capabilities the repository explicitly trains and evaluates.
