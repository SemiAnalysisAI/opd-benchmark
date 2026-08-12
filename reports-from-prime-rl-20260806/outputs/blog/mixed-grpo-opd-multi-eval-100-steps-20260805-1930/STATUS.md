# Mixed GRPO + OPD 100-step multi-eval run

- Status: completed successfully
- Started: 2026-08-05 19:32 UTC
- Config: `configs/debug/algo/mixed_grpo_opd.toml`
- W&B: https://wandb.ai/joey00072/algorithms-debug/runs/00c9c1293b0444289d0e175a9c531d62
- Finished: 2026-08-05 19:40 UTC
- Steps: 100/100
- Training tasks: reverse-text GRPO and reverse-text OPD
- Evaluation tasks: reverse-text, GSM8K, and alphabet-sort every 5 steps
- GPUs: policy inference 0, frozen teacher 1, trainer 2
- Initial eval rewards: reverse-text 0.7401, GSM8K 0.0625, alphabet-sort 0.0099
- Final eval rewards: reverse-text 0.8431, GSM8K 0.0391, alphabet-sort 0.0136
- Final train reward: 0.8116 (reverse-text GRPO 0.8299, reverse-text OPD 0.8090)
- Final rollout health: 0.0% errors; train truncation 1.6%
- Interpretation: reverse-text improved and then plateaued; the reverse-text-only training did not transfer meaningfully to GSM8K or alphabet sorting.
- Caveat: asynchronous evaluation sometimes mixed adjacent policy versions; the final checkpoint used policy v98 consistently across all three evals.
