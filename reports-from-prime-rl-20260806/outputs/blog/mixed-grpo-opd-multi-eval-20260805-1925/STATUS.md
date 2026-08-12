# Mixed GRPO + OPD multi-eval run

- Status: completed successfully
- Started: 2026-08-05 19:26 UTC
- Config: `configs/debug/algo/mixed_grpo_opd.toml`
- W&B: https://wandb.ai/joey00072/algorithms-debug/runs/6c540f2e8b564879b8124fdbcdb52530
- Training tasks: reverse-text GRPO and reverse-text OPD
- Evaluation tasks: reverse-text, GSM8K, and alphabet-sort
- GPUs: policy inference 0, frozen teacher 1, trainer 2
- Finished: 2026-08-05 19:29 UTC
- Steps: 20/20
- Initial eval rewards: reverse-text 0.7250, GSM8K 0.0469, alphabet-sort 0.0065
- Final eval rewards: reverse-text 0.8369, GSM8K 0.0312, alphabet-sort 0.0125
- Final train reward: 0.8020 (reverse-text GRPO 0.7988, reverse-text OPD 0.8244)
- Final rollout health: 0.0% errors; train truncation 0.8%
- Interpretation: reverse-text improved substantially; this short reverse-text-only training run did not transfer to GSM8K or alphabet sorting.
