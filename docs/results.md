# Results

This page summarizes the example results already captured in the repository under `dev/d20_eval_results/`. These should be read as project results for the documented d20 pipeline, not as universal guarantees for every run.

## Base Model

From the recorded d20 base training and evaluation notes:

- depth: 20
- parameters: 896,535,720
- training steps: 8,300
- training tokens: 8,703,180,800
- minimum validation BPB: 0.7811
- final CORE metric estimate during training: 0.2210
- final recorded CORE metric in evaluation: 0.2167

Selected qualitative samples from the base model show that it already learns:

- basic factual completion such as "The capital of France is Paris"
- short arithmetic pattern completion
- list completion for common world knowledge

The unconditioned samples also show the expected weakness of a partially trained base model:

- fluent but noisy long-form continuation
- local coherence without strong reliability
- better next-token modeling than assistant behavior

## SFT Model

From the recorded SFT notes:

- training iterations: 847
- minimum validation BPB: 0.3673

Recorded chat evaluation on the SFT checkpoint:

- ARC-Easy: 0.4306
- ARC-Challenge: 0.3387
- MMLU: 0.3398
- GSM8K: 0.0235
- HumanEval: 0.0732
- SpellingBee: 0.9844
- ChatCORE metric: 0.2600

The main effect of SFT appears to be:

- much stronger assistant/task formatting
- strong spelling-task performance
- modest gains on multiple-choice reasoning
- limited math accuracy before RL

## RL Model

Recorded chat evaluation on the RL checkpoint:

- ARC-Easy: 0.4272
- ARC-Challenge: 0.3464
- MMLU: 0.3374
- GSM8K: 0.1585
- HumanEval: 0.0122
- SpellingBee: 0.9180
- ChatCORE metric: 0.2617

The most obvious change after RL is:

- GSM8K improves substantially over SFT

At the same time, the recorded metrics suggest some regression on:

- HumanEval
- SpellingBee

This is a typical specialization tradeoff for reward-driven tuning.

## How To Interpret The Numbers

A useful mental model is:

- base model results tell you how strong the language-model foundation is
- SFT results tell you how usable the model is as an assistant
- RL results tell you what happened after specializing the assistant with reward optimization

Users should compare stages, not only absolute scores.

## Recommended Evaluation Practice

When you run your own experiments, evaluate all three layers where relevant:

1. base metrics: CORE, BPB, and samples
2. chat metrics on the SFT checkpoint
3. chat metrics on the RL checkpoint

That is the easiest way to identify whether a later stage truly helped or only shifted capability from one benchmark to another.
