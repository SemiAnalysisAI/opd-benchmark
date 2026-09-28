# rg-tasks

A verifiers v1 taskset for the OPD reasoning_gym tasks. It loads the campaign's
`data/<domain>-<split>.jsonl` rows (system and user prompt, JSON label) and rewards
the final reply with the shared verifier (`rg_tasks/scoring.py` links to
`shared/scoring.py`). In OPD the reward is diagnostic only.

Install it into the prepared campaign's Prime-RL environment:

```bash
uv pip install --python <campaign>/source/.venv/bin/python --no-deps -e <campaign>/environments/rg_tasks
```

`reasoning-gym==0.1.25` comes from `<campaign>/pydeps` (see docs/SETUP.md).
