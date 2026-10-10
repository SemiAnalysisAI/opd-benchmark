# Fireworks: hosted teacher RL, then MOPD

`campaign.py` trains one RL teacher per domain, then trains a fresh student on both domains by multi-teacher on-policy distillation (MOPD). It uses the shared recipe in `shared/recipe.py` (reasoning_gym `caesar_cipher` and `simple_geometry`, thinking on). Unlike the self-hosted frameworks, which distill the frozen `recipe.TEACHERS`, this campaign trains its own teachers. Fireworks settings are at the top of `campaign.py`; budget settings are at the top of `lifecycle.py`.

## Setup

- **Billing.** Uses the dedicated Training API, billed per GPU time: one four-B200 trainer plus a one-replica rollout deployment. It does not use the per-token Serverless API. All weights and checkpoints stay on Fireworks.
- **Base model.** Qwen3.5-35B-A3B, not the Qwen3.6 base used elsewhere, so comparisons with other runs are not same-base. Rollouts use FP8 inference without router replay; the trainer-versus-sampler logprob gap is logged.
- **Prompts.** Qwen3.5 chat template with `enable_thinking=True`, up to 30,720 response tokens, `max_context_length` 32,768, stop at EOS. A response scores 1 only if the text after its last `</think>` contains an `<answer>` that reasoning_gym accepts.
- **Teachers.** Rank-64 LoRAs trained with PPO on group-normalized rewards: 32 prompts × 8 responses per update, LR 1e-5 (10× the recipe's full-finetuning 1e-6), no weight decay, at most 200 updates. Every 25 updates they are evaluated greedily on the full dev split (126 caesar_cipher, 128 simple_geometry problems). The best evaluated checkpoint is kept, even if it misses the target.
- **Teacher targets.** 70% on caesar_cipher and 95% on simple_geometry. These are operational goals set just below the held-out scores of the GRPO teachers in `recipe.TEACHERS` (73.3% and 99.8%, sampled at temperature 1). Reaching them does not mean the teachers match `recipe.TEACHERS`.
- **Student.** 20 updates × 128 responses (64 per domain), LR 1e-5, with teacher-minus-student token advantages. Training is synchronous (policy lag 0, within `recipe.MAX_POLICY_LAG`).
- **Single session.** The backend allows one resident LoRA session. At each MOPD update the controller saves the student, loads each frozen teacher checkpoint to score that teacher's domain, then restores the student's weights and optimizer state before the training step. This checkpoint I/O is counted as teacher time.

## Prerequisites

- `fireworks-ai[training]==1.2.14`
- `reasoning-gym==0.1.25`, installed or with its site-packages directory in `$CAMPAIGN_PYDEPS`
- `FIREWORKS_API_KEY` exported, or an `api_key_profile_label` in `config/fireworks.local.json` so the key is read from `~/.zprofile`
- The account ID, from `--account`, `config/fireworks.local.json`, or `FIREWORKS_ACCOUNT_ID`
- A tokenizer info JSON (`model`, `revision`, `path`, `eos_token_id`) for a locally cached `Qwen/Qwen3.5-35B-A3B` tokenizer

## Run

```
python hosted/fireworks/campaign.py --root NEW_RUN_ROOT --tokenizer-info TOKENIZER_JSON [--account ACCOUNT]
python hosted/fireworks/report.py RUN_ROOT      # Also runs automatically after verified cleanup.
cd hosted/fireworks && python test_campaign.py  # Offline regression tests.
```

The launcher verifies the account, copies `hosted/fireworks`, `shared` and `data` into `RUN_ROOT/source-repo/`, and runs that frozen copy.

- `--teacher-config JSON` overrides the teacher `prompts`, `learning_rate`, `eval_every`, `max_steps` and `target_tolerance`. Any override changes the protocol and must be reported as such.
- `--resume-plan JSON` maps each domain to `{source_root, step, restore_reference, best, [prompt_offset], [complete]}`. Use `cross_job://` references, not sampler snapshots. The source run must match on account, tokenizer revision and data hashes. If the teacher batch size changed, carry `prompt_offset` over.

`report.py` writes `audit.json` and `REPORT.md`, including a check of the student protocol (20 updates, 2,560 responses, 64 per domain per update) and an audit of the checkpoint swaps. A partial run does not count as a completed protocol.

## Cost

Thinking at 32k tokens costs far more than the earlier 256-token protocol. GRPO teacher responses averaged about 11k tokens on caesar_cipher and 4k on simple_geometry, and one teacher can run 200 updates of 256 responses. The default $200 budget will very likely stop the campaign during the first teacher, so set `--budget-usd` deliberately.

## Budget guard

An independent watchdog starts before provisioning. It estimates spend from the start of provisioning at $117/hour (six B200s at the public $13/hour, plus 50% headroom), plus `--prior-spend-usd`. It stops the run when the estimate reaches `--budget-usd` minus a $25 cleanup reserve (175 with the default budget of 200), when training ends, or when the controller dies. It then keeps deleting this run's named trainer and deployment (the deployment with `ignoreChecks`) until their release is observed.

This is a conservative client-side estimate, not an invoice or a provider-enforced cap. Network failures can delay cleanup, and the timings do not measure GPU utilization.
