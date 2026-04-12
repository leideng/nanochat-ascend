# RL Process

## Objective

The reinforcement-learning stage refines the SFT model using sampled rollouts and rewards on GSM8K. It is implemented by `scripts/chat_rl.py` and commonly launched via:

```bash
bash runs/run_rl.sh
```

## Starting Point

RL loads an SFT checkpoint, not a base checkpoint:

- `load_model("sft", device, phase="eval", ...)`

So RL is a post-SFT refinement stage.

## High-Level Algorithm

The script describes itself as a simplified GRPO-like setup, but the implementation is closer to a practical REINFORCE-style loop:

- no trust-region or KL-to-reference term
- no PPO ratio clipping
- token-level objective construction
- centered rewards using `reward - mean(reward)`

In other words, the stage is policy-gradient fine-tuning over sampled completions.

## Data Flow

For each training example:

1. select a GSM8K training conversation
2. render it for completion
3. sample multiple candidate answers from the current model
4. compute rewards for the sampled answers
5. convert the sampled sequences into autoregressive `inputs` and `targets`
6. mask non-trainable positions
7. compute a policy-gradient objective using advantages

This is qualitatively different from SFT, where the model learns directly from target tokens.

## Key RL Hyperparameters

The RL stage is driven by:

- `examples-per-step`
- `num-samples`
- `max-new-tokens`
- `temperature`
- `top-k`
- `device-batch-size`

These control both rollout diversity and optimization throughput.

## Wrapper Defaults

`runs/run_rl.sh` provides:

### On Ascend NPU

- `model-tag="d20"`
- `num-epochs=1`
- `device-batch-size=48`
- `examples-per-step=48`
- `num-samples=48`
- `max-new-tokens=256`

### On CPU

- smaller smoke-test parameters
- `model-tag="d4-test"`
- `device-batch-size=4`

## Evaluation During RL

The script periodically evaluates GSM8K pass@k. A problem is counted as solved at `k` if any of the first `k` samples is correct.

This is useful because RL is explicitly optimizing sampled behavior, so pass@k is often a more faithful online indicator than greedy-only accuracy.

## Outputs

RL checkpoints are written under:

```text
.cache/checkpoint/chatrl_checkpoints/<model_tag>
```

These checkpoints can then be:

- evaluated with `scripts/chat_eval.py -i rl`
- used in CLI or web demos

## Tradeoffs To Keep In Mind

Because RL is specialized, it can improve some capabilities while degrading others. In this repository, the reported results show improvement on GSM8K after RL, with some regressions on other tasks such as HumanEval and SpellingBee.

That tradeoff is normal and should be part of how users interpret RL outcomes.
