---
name: meta-test
description: Use this skill when changing model construction, checkpoint/model wiring, tensor shapes, attention masks, or other code in nanochat-ascend that should be validated on PyTorch's meta device without allocating real tensors or executing NPU code.
---

# Meta Test

Use this skill for shape-first validation in `nanochat-ascend`.

## When to use

- Model `__init__` changes in [nanochat/gpt.py](../../nanochat/gpt.py)
- Meta-device initialization or `to_empty()` flows
- Tensor shape or dtype plumbing
- Attention mask construction
- Checkpoint loading code that can be validated up to structure and parameter layout
- Bugs where CPU execution is unnecessary or too expensive for the first pass

Do not use this skill as proof that runtime execution works. Meta tests do not validate kernels, real values, data movement, optimizer numerics, or Ascend integration.

## Workflow

1. Inspect the affected code path and identify the smallest shape-only invariant that should hold.
2. Reuse existing meta-device patterns in the repo when possible. `scripts/base_train.py` and `nanochat/checkpoint_manager.py` already build models under `with torch.device("meta")`.
3. Prefer checking:
   - module construction succeeds
   - parameter and buffer shapes are correct
   - forward input and output shapes are correct when the ops support meta tensors
   - dtype propagation is consistent
4. Keep synthetic inputs minimal and explicit.
5. If an operation is unsupported on meta, stop escalating there and switch to a tiny CPU test instead of forcing the issue.

## Preferred command pattern

Use a short one-off Python command from the repo root. Keep it focused on the changed path.

Example pattern:

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
with torch.device("meta"):
    model = GPT(config)

print(next(model.parameters()).device)
print(model.transformer.wte.weight.shape)
PY
```

Adjust the config fields to match the code you touched. Keep dimensions tiny.

## Repo-specific guidance

- Prefer validating model construction before trying a meta forward pass.
- If you touch checkpoint loading, first verify the model can still be instantiated on meta with the same config path used by [nanochat/checkpoint_manager.py](../../nanochat/checkpoint_manager.py).
- If you touch training setup, inspect [scripts/base_train.py](../../scripts/base_train.py) first because it already depends on meta-device initialization.
- Never switch to NPU execution as part of this skill. If the bug is NPU-only, ask the user for Ascend logs after the meta validation is done.

## Expected deliverable

Report exactly what was validated:

- whether construction succeeded
- which shapes or dtypes were checked
- which parts remain unvalidated because they require CPU or NPU execution
