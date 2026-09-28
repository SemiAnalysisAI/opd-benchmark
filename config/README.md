# Local configuration

Copy an example to `*.local.json` and edit it. Local files are gitignored, so cluster addresses, account IDs and account-guard values stay out of the repository.

| File | Used by | Required |
|---|---|---|
| `site.local.json` | `tools/prepare.py` | Yes, for Miles, Prime-RL, Slime and verl |
| `site-nemo-rl.local.json` | `tools/prepare.py nemo-rl` | Yes, for NeMo-RL. It is the site file plus `teacher_node`, `teacher_ip` and `nemo_rl_container` for the third node. |
| `tinker.local.json` | `hosted/tinker/launch.py` | No. Without it, `TINKER_API_KEY` is used with no account guard. |
| `fireworks.local.json` | `hosted/fireworks/campaign.py` | No. `--account` or `FIREWORKS_ACCOUNT_ID` also work. |

## API keys

API keys never go in these files. The hosted launchers read `TINKER_API_KEY` or `FIREWORKS_API_KEY` from the environment. To read a `LABEL=value` line from `~/.zprofile` instead, set `api_key_profile_label`. This helps when one machine holds keys for several accounts.

## Tinker account guard

Set `expected_email` and `expected_org` to make the Tinker launcher refuse any other account. On a shared organization account, keep these values in your local file.
