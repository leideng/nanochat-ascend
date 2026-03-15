---
name: cpu-test
description: Use this skill when nanochat-ascend changes need real execution on this machine. It focuses on tiny CPU-only smoke tests for forward passes, train/eval entry points, checkpoint flows, and other runtime behavior that cannot be proven with meta-device checks alone.
---

# CPU Test

Use this skill for the smallest possible real execution on CPU.

## When to use

- Forward-pass or loss-path changes that need real tensor values
- Tiny train or eval smoke tests
- CLI or script entry-point validation
- Checkpoint save/load validation on CPU
- Bugs involving control flow, data movement, or ops unsupported on meta

Do not use this skill for performance claims or large runs. The goal is quick behavioral validation only.

## Workflow

1. Reduce the target path to the smallest runnable case.
2. Force CPU explicitly with `--device-type=cpu` when the script supports it.
3. Prefer eager mode for smoke tests with `NANOCHAT_ENFORCE_EAGER=1`.
4. Use tiny settings:
   - `--device-batch-size=1`
   - very small sequence lengths such as `32` or `64`
   - `--num-iterations=1` or `2`
5. If the full script needs unavailable datasets or checkpoints, validate the lowest-level module or a narrower script path instead.
6. If the remaining risk is Ascend-specific, stop after the CPU smoke test and ask the user for NPU output.

## Preferred command patterns

Training-style smoke test:

```bash
NANOCHAT_ENFORCE_EAGER=1 python -m scripts.base_train --device-type=cpu --depth=2 --max-seq-len=64 --device-batch-size=1 --total-batch-size=4 --num-iterations=1
```

Evaluation or inference-style smoke test:

```bash
python -m scripts.chat_cli --device-type=cpu -p "hello"
```

Module-level targeted smoke test:

```bash
python - <<'PY'
import torch

from nanochat.gpt import GPT, GPTConfig

config = GPTConfig(
    sequence_len=32,
    vocab_size=256,
    n_layer=2,
    n_head=2,
    n_kv_head=1,
    n_embd=64,
    window_pattern="SL",
)
model = GPT(config).to("cpu")
model.init_weights()
idx = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)
loss = model(idx, idx)
print(loss.item())
PY
```

Adapt the command to the code you changed. Keep runtime under control.

## Repo-specific guidance

- This repo supports Ascend NPU and CPU only. GPU/CUDA is out of scope.
- Never try to emulate Ascend execution locally. CPU smoke tests only establish local behavioral sanity.
- When a script defaults to autodetect, pass `--device-type=cpu` explicitly to avoid accidental NPU assumptions in future environments.
- If you modify a path that already has a meta-first setup, consider running `meta-test` first and CPU second.

## Expected deliverable

Report:

- the exact CPU command that was run
- whether it passed or failed
- the concrete failure point if it failed
- what still requires user-run validation on a real Ascend machine
