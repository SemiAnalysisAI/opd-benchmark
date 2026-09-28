"""API keys and local account settings for the hosted providers.

Anyone can reproduce with an exported `<PROVIDER>_API_KEY`. Teams can instead
keep account settings in `config/<provider>.local.json` (gitignored; see
`config/README.md`), including a `~/.zprofile` label so that a key exported for
another account is never used by accident. Keys are never written or printed.
"""
import json
import os
from pathlib import Path
import re
import shlex

CONFIG = Path(__file__).resolve().parents[1] / 'config'
PROFILE = Path.home() / '.zprofile'


def local_config(provider, path=None):
    """Settings from `config/<provider>.local.json`, or {} when the file is absent."""
    path = Path(path) if path else CONFIG / f'{provider}.local.json'
    return json.loads(path.read_text()) if path.exists() else {}


def api_key(env_name, profile_label=None):
    """The key from the profile line `profile_label=...` if given, otherwise from `$env_name`."""
    if profile_label:
        return profile_secret(profile_label)
    if not os.environ.get(env_name):
        raise RuntimeError(f'Set {env_name}, or api_key_profile_label in config/, to provide the API key')
    return os.environ[env_name]


def profile_secret(label, profile=PROFILE):
    """Return the value of the single `[export] LABEL=value` line in `profile`, without executing it."""
    pattern = re.compile(r'^\s*(?:export\s+)?' + re.escape(label) + r'\s*=\s*(.*)$')
    values = [shlex.split(match.group(1), comments=True)
              for line in Path(profile).read_text().splitlines()
              if (match := pattern.match(line))]
    if len(values) != 1 or len(values[0]) != 1:
        raise RuntimeError(f'Expected exactly one literal {label} assignment in {profile}')
    return values[0][0]
