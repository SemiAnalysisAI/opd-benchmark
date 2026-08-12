# Mixed GRPO + OPD 4B 100-step multi-eval run

- Status: running
- Started: 2026-08-05 21:46 UTC
- Student: `Qwen/Qwen3-4B-Instruct-2507`
- OPD teacher: `PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL`
- Config: `configs/debug/algo/mixed_grpo_opd.toml`
- W&B: https://wandb.ai/joey00072/algorithms-debug/runs/d2f490a5b4aa4e058e2c47e112c64f8a
- Steps: 0/100
- Training tasks: reverse-text GRPO and reverse-text OPD
- Evaluation tasks: reverse-text, GSM8K, and alphabet-sort every 5 steps
- GPUs: policy inference 0, frozen teacher 1, trainer 2
