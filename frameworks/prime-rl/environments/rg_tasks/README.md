# rg-tasks

A verifiers v1 taskset for the OPD reasoning_gym tasks. It loads the campaign's
`data/<domain>-<split>.jsonl` rows (system and user prompt, JSON label) and scores
the final reply with the shared verifier (`rg_tasks/scoring.py` is a symlink to
`shared/scoring.py`). In OPD the reward is only a diagnostic.

Install it into the prepared campaign's Prime-RL environment:

```bash
uv pip install --python <campaign>/source/.venv/bin/python --no-deps -e <campaign>/environments/rg_tasks
```

`reasoning-gym==0.1.25` is provided by `<campaign>/pydeps` (see docs/SETUP.md).
