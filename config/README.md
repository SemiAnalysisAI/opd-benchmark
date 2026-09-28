# Local configuration

Copy an example to `*.local.json` and edit it. Local files are gitignored, so
cluster addresses, account IDs and account-guard values never enter the repository.

| File | Used by | Needed? |
|---|---|---|
| `site.local.json` | `tools/prepare.py` (Slurm frameworks) | Yes, for Miles, Prime-RL, Slime and verl |
| `site-nemo-rl.local.json` | `tools/prepare.py nemo-rl` | Yes, for NeMo-RL: the site file plus `teacher_node`, `teacher_ip` and `nemo_rl_container` (a third node) |
| `tinker.local.json` | `hosted/tinker/launch.py` | No: without it, `TINKER_API_KEY` is used with no account guard |
| `fireworks.local.json` | `hosted/fireworks/campaign.py` | No: `--account` or `FIREWORKS_ACCOUNT_ID` also work |

API keys never go in these files. By default the hosted launchers read
`TINKER_API_KEY` / `FIREWORKS_API_KEY` from the environment. Set
`api_key_profile_label` to read a literal `LABEL=value` line from `~/.zprofile`
instead, for example when one machine holds keys for several accounts.

Set `expected_email` and `expected_org` to make the Tinker launcher refuse any
other account. Teams running on a shared organization account should keep
these values in their local file.
