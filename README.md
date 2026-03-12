# tiny-grpo

A minimal, PyTorch-native implementation of **GRPO** (Group Relative Policy Optimization) for reinforcement learning fine-tuning of large language models (LLMs).

---

## What is GRPO?

GRPO is a reinforcement learning algorithm for fine-tuning language models, introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300). Like PPO, it uses a clipped surrogate objective and a KL-divergence penalty against a reference model to keep the policy from drifting too far. The key difference is how advantages are computed: instead of a learned value function, GRPO generates a **group** of completions for each prompt and derives advantages by standardizing the rewards within that group. This makes it simpler and well-suited for settings where a rule-based reward signal is available.

---

## Project Overview

This project trains a causal language model on math word problems from the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset using GRPO. The model learns to produce answers in a structured XML format (`<reasoning>…</reasoning><answer>…</answer>`) and is rewarded for both correctness and adherence to the format.

### Key features

- **Pure PyTorch** — no RL framework dependency; the full GRPO loop is implemented from scratch.
- **Multi-GPU training** — uses PyTorch's FSDP2 (`fully_shard`) for sharding model parameters across GPUs via `torchrun`.
- **Configurable activation checkpointing** — supports full, selective-op, and selective-layer checkpointing to trade memory for compute.
- **Optional `torch.compile`** — compile individual transformer blocks for additional throughput.
- **Weights & Biases integration** — logs loss, average reward, and training progress.

---

## Repository Structure

| File | Description |
|---|---|
| `train.py` | Entry point. Initialises distributed training, loads the model and data, runs evaluation, and calls the GRPO training loop. |
| `loss.py` | Core GRPO logic: completion generation, log-probability computation, advantage estimation, PPO-style surrogate loss, and KL penalty. |
| `model.py` | Model loading helpers. Prepares the trainable **actor/rollout** model and the frozen **reference** model, applies FSDP2 sharding. |
| `data.py` | Dataset utilities. Loads GSM8K from Hugging Face, formats prompts, and creates distributed `DataLoader`s. |
| `reward.py` | Reward functions. `correctness_reward` scores answers; `format_reward` scores XML structure compliance; `combined_reward` sums both. |
| `eval.py` | Evaluation loop. Generates responses and checks correctness with exact-match, single-number, and last-number heuristics. |
| `parallelize.py` | FSDP2 application, `torch.compile` per-block compilation, and activation checkpointing wrappers. |
| `utils.py` | Distributed initialisation (`init_distributed`), random seed setting, and rank helpers. |
| `vlogging.py` | Logging setup. |
| `run.sh` | Convenience script that creates a virtualenv, installs dependencies, and launches training via `torchrun`. |

---

## How It Works

### Training loop

1. **Rollout** — For each training batch, the current policy generates `num_generations` completions per prompt (sampling with temperature 1.0).
2. **Log-probability collection** — Log-probs for each completion token are computed under both the policy model and the frozen reference model.
3. **Reward scoring** — Each completion receives a combined reward: up to **2.0** for a correct answer and up to **0.8** for proper XML formatting (total up to **2.8**).
4. **Advantage estimation** — Rewards are standardised within each prompt group (mean-subtracted, std-normalised) to produce per-token advantages.
5. **GRPO loss** — A PPO-style clipped surrogate objective is combined with a per-token KL penalty between the reference and policy log-probs:

   ```
   loss = -mean[ clip(ratio, 1±ε) * A  -  β * KL(ref ∥ policy) ]
   ```

6. **Parameter update** — AdamW with gradient clipping (max norm 1.0).
7. The outer loop (`num_iterations`) repeats with the same reference model; the inner loop (`num_steps`) iterates over training batches, and `mu` gradient updates are applied per rollout.

### Reward functions

| Reward | Condition | Score |
|---|---|---|
| Correctness | Exact string match | 2.0 |
| Correctness | Numeric equivalence | 1.5 |
| Correctness | No match | 0.0 |
| Format | Each correct XML tag (`<reasoning>`, `</reasoning>`, `<answer>`, `</answer>`) | +0.2 each (max 0.8) |

---

## Requirements

- Python 3.10+
- 8 × NVIDIA A100 80 GB GPUs (default config). Reduce `num_generations` and `max_completion_length` in `train.py` for GPUs with less VRAM.
- PyTorch 2.6+

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
bash run.sh <model> <dataset> <do_eval>
```

| Argument | Description | Example |
|---|---|---|
| `model` | Hugging Face model name or local path | `Qwen/Qwen2.5-1.5B-Instruct` |
| `dataset` | Hugging Face dataset name | `openai/gsm8k` |
| `do_eval` | Run evaluation before and after training (`yes`/`no`) | `yes` |

**Example:**

```bash
bash run.sh Qwen/Qwen2.5-1.5B-Instruct openai/gsm8k yes
```

This will:
1. Create a Python virtual environment and install dependencies.
2. Launch `train.py` across 8 GPUs using `torchrun`.
3. (Optional) Evaluate accuracy before and after GRPO fine-tuning.

Set the `WANDB_PROJECT` environment variable before running to enable Weights & Biases logging:

```bash
export WANDB_PROJECT=my-grpo-project
bash run.sh Qwen/Qwen2.5-1.5B-Instruct openai/gsm8k yes
```

---

## Training Configuration

The default configuration (in `train.py`) is tuned for an 8 × A100 80 GB node:

| Parameter | Default | Description |
|---|---|---|
| `num_iterations` | 1 | Outer GRPO iterations (reference model is fixed per iteration) |
| `num_steps` | 500 | Training steps per iteration |
| `num_generations` | 12 | Completions generated per prompt during rollout |
| `max_completion_length` | 400 | Maximum tokens per completion |
| `beta` | 0.04 | KL penalty coefficient |
| `learning_rate` | 1e-6 | AdamW learning rate |
| `mu` | 1 | Gradient updates per rollout batch |
| `epsilon` | 0.1 | PPO clipping range |

---

## References

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — original GRPO paper.
- [aburkov's theLMbook — GRPO From Scratch (Multi-GPU)](https://github.com/aburkov/theLMbook/blob/main/GRPO_From_Scratch_Multi_GPU_DataParallel_Qwen_2_5_1_5B_Instruct.ipynb)
- [torchtitan](https://github.com/pytorch/torchtitan) — FSDP2 and activation checkpointing patterns.
