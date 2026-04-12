# RL Dataset

## Role In The Pipeline

The RL stage in `scripts/chat_rl.py` uses GSM8K as its training and validation task. Unlike SFT, where GSM8K is one dataset among many, RL focuses narrowly on reward-driven improvement over GSM8K completions.

## Source

The dataset comes from:

- `openai/gsm8k`

Configured local path:

```text
.cache/dataset/task/gsm8k
```

## How It Is Used

`scripts/chat_rl.py` creates:

- `train_task = GSM8K(subset="main", split="train")`
- `val_task = GSM8K(subset="main", split="test")`

The RL loop does not simply replay gold answers as labels. Instead it:

1. loads a conversation/problem from GSM8K
2. renders the prompt for completion
3. samples multiple candidate completions from the current policy
4. scores each completion with `train_task.reward(...)`
5. computes centered advantages by subtracting the mean reward
6. applies a policy-gradient-style update

## Why RL Uses Only GSM8K Here

The RL stage is specialized. It is not trying to preserve the entire SFT data distribution. It is trying to push behavior on a concrete task where sampled outputs can be rewarded automatically.

That makes GSM8K attractive because:

- it has clear correctness criteria
- the task supports automated reward/evaluation logic
- it aligns with arithmetic reasoning improvements

## Batch Structure

The RL script works with three related batch concepts:

- `examples-per-step`: how many prompts are optimized per training step
- `num-samples`: how many rollouts are generated per prompt
- `device-batch-size`: how many sequences fit into a single forward pass

This matters because RL batches are driven by rollout generation, not just fixed supervised targets.

## Validation During RL

The same script periodically evaluates on GSM8K test data using pass@k metrics, where a problem is counted as solved if any of the top `k` sampled completions is correct.

That makes RL evaluation meaningfully different from:

- pretraining evaluation, which uses BPB and CORE
- categorical chat evaluation, which uses answer-choice logits
