# Model

## Model Family

`nanochat-ascend` uses a GPT-style causal decoder transformer implemented in `nanochat/gpt.py`.

The model is intentionally compact and pragmatic rather than a reproduction of a single published architecture. It combines several modern transformer choices that fit the repo’s goals and hardware targets.

## Core Architectural Features

The implementation explicitly includes:

- rotary positional embeddings instead of learned positional embeddings
- RMSNorm without learnable parameters
- QK normalization
- untied token embedding and output projection weights
- bias-free linear layers
- ReLU squared MLP activations
- Group-Query Attention support
- optional sliding-window attention patterns across layers
- Flash Attention integration with SDPA fallback

## Configuration Fields

The main model config class is `GPTConfig`, with fields such as:

- `sequence_len`
- `vocab_size`
- `n_layer`
- `n_head`
- `n_kv_head`
- `n_embd`
- `window_pattern`

In `scripts/base_train.py`, model width is derived from:

- `depth`
- `aspect_ratio`
- `head_dim`

The script computes:

- `model_dim = round_up(depth * aspect_ratio, head_dim)`
- `num_heads = model_dim // head_dim`

## Attention Design

### Group-Query Attention

The attention layer separates:

- query heads: `n_head`
- key/value heads: `n_kv_head`

That allows grouped key/value sharing for more efficient inference while keeping richer query structure.

### Rotary Embeddings And QK Norm

Queries and keys are:

1. projected from the residual stream
2. rotated with precomputed rotary embeddings
3. normalized with RMS-based QK normalization

This gives the model relative-position handling without learned absolute embeddings.

### Sliding Window Pattern

The config field `window_pattern` determines whether each layer uses:

- `L`: long/full-context attention
- `S`: shorter sliding-window attention

The pattern is tiled across layers, with the final layer forced to full context by the implementation logic.

This lets users trade off global context coverage and efficiency.

## Residual And Value-Embedding Additions

The implementation also contains a few repo-specific features:

- per-layer `resid_lambdas`
- per-layer `x0_lambdas`
- optional value embeddings in alternating layers
- learned gates controlling how value embeddings mix into attention values

These choices push the model beyond a plain reference GPT implementation.

## Initialization Strategy

The model is designed for meta-first construction:

- it can be instantiated on the `meta` device to validate shapes without allocating real parameter tensors
- actual parameter materialization happens later with `to_empty(...)` and `init_weights()`

This is important in the repo because the training scripts use shape-first construction before moving to the real device.

## Optimizer

The project uses MuonAdamW-style optimization from `nanochat/optim.py`.

At a high level:

- embeddings and scalar-like parameters use Adam-style updates
- matrix parameters use Muon-style orthogonalized updates

This optimizer split is built into `model.setup_optimizer(...)` and is used by pretraining, SFT, and RL.

## Checkpoint Interface

Checkpoints are managed through `nanochat/checkpoint_manager.py`. The repository stores model state, optimizer state, and metadata needed to resume or evaluate runs.

Configured checkpoint roots:

- base: `.cache/checkpoint/base_checkpoints`
- SFT: `.cache/checkpoint/chatsft_checkpoints`
- RL: `.cache/checkpoint/chatrl_checkpoints`

## Inference

Generation is handled by `nanochat/engine.py`, which provides:

- batched generation
- top-k and temperature sampling
- KV-cache based inference
- interfaces used by both chat scripts and RL rollouts

## Practical Reading

If you want to understand the model in code, the most useful reading order is:

1. `nanochat/gpt.py`
2. `nanochat/flash_attention.py`
3. `nanochat/optim.py`
4. `nanochat/engine.py`
5. `scripts/base_train.py`
