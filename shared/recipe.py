"""The experiment every framework runs: multi-teacher OPD (MOPD) on two reasoning_gym tasks.

Each framework translates these values into its own configuration format;
anything framework-specific lives in that framework's directory.

`tools/prepare.py --domains` can narrow a campaign to one task (for example
single-teacher OPD on `caesar_cipher`) by writing the campaign's `experiment.json`.
"""
import json
from pathlib import Path

_EXPERIMENT = Path(__file__).resolve().parents[1] / 'experiment.json'
EXPERIMENT = json.loads(_EXPERIMENT.read_text()) if _EXPERIMENT.is_file() else {}

# Tasks, each with its own frozen GRPO teacher. The rows in `data/` are reasoning_gym 0.1.25
# problems from the default task config (train seed 20000, dev seed 10000, so the splits are
# disjoint) with reasoning_gym's default system prompt, which the teachers were trained with.
ALL_DOMAINS = ('caesar_cipher', 'simple_geometry')
DOMAINS = tuple(EXPERIMENT.get('domains', ALL_DOMAINS))
MIXED_TRAIN_FILE = 'mixed-train.jsonl'  # Both train files interleaved.
TRAIN_EXAMPLES_PER_DOMAIN = 7788
EVAL_EXAMPLES = {'caesar_cipher': 126, 'simple_geometry': 128}  # The whole dev split.


def data_file(domain, split):
    """Name of one domain's split in `data/`, for example `caesar_cipher-dev.jsonl`."""
    return f'{domain}-{split}.jsonl'


# Training prompts: both tasks interleaved, or the single task's own file.
TRAIN_FILE = MIXED_TRAIN_FILE if DOMAINS == ALL_DOMAINS else data_file(DOMAINS[0], 'train')
assert set(DOMAINS) <= set(ALL_DOMAINS) and (len(DOMAINS) == 1 or DOMAINS == ALL_DOMAINS), DOMAINS


# Student training. Thinking is on, so responses are long.
UPDATES = 20
PROMPTS_PER_UPDATE = 128  # Split equally between the domains.
SAMPLES_PER_PROMPT = 1
LEARNING_RATE = 1e-6
ENABLE_THINKING = True  # The scorer only reads answers after a finished `</think>`.
CONTEXT_LENGTH = 32_768
MAX_RESPONSE_TOKENS = 30_720
MAX_POLICY_LAG = 1  # Accepted optimizer updates between sampling and training.
SAVE_INTERVAL = 10  # Checkpoints after updates 10 and 20, so the benchmark also gets a mid-run point.

# Evaluation: greedy decoding on the full dev split of each domain.
EVAL_INTERVAL = 10

# Trainer memory at 32k. No context parallelism: one micro-batch holds up to one full-length
# sequence per GPU, and log probabilities are computed in chunks of this many tokens.
MAX_TOKENS_PER_GPU = CONTEXT_LENGTH
LOG_PROBS_CHUNK_SIZE = 4096

# Pinned Hugging Face snapshots (repository, revision). Teachers are frozen.
BASE_MODEL = ('Qwen/Qwen3.6-35B-A3B', '995ad96eacd98c81ed38be0c5b274b04031597b0')
TEACHERS = {  # GRPO teachers; the revisions are the tags step-125 and step-50.
    'caesar_cipher': ('semianalysisai/Qwen3.6-35B-A3B-caesar-cipher-GRPO-20260924',
                      'dfedf677a5bf08a451733ec48c845c2661a6ab14'),
    'simple_geometry': ('semianalysisai/Qwen3.6-35B-A3B-simple-geometry-GRPO-20260925',
                        '679d4b66018c03249a3bfd0ecca3195ed9cd0e0d'),
}
# Model directory names inside a prepared campaign.
BASE_MODEL_DIR = 'Qwen3.6-35B-A3B'
MEGATRON_MODEL_DIR = 'Qwen3.6-35B-A3B_torch_dist'

# Hardware: two nodes with eight GPUs each. The trainer node trains on all eight.
# The generation node serves one frozen teacher per GPU on GPUs 0-1 (GPU 1 is idle
# in single-teacher runs) and the student policy on GPUs 2-7, two GPUs per engine.
GPUS_PER_NODE = 8
TRAINER_GPUS = 8
TEACHER_GPU = {domain: gpu for gpu, domain in enumerate(DOMAINS)}
POLICY_GPUS = (2, 3, 4, 5, 6, 7)
POLICY_GPUS_PER_ENGINE = 2
CPUS_PER_NODE = 80
RAY_CPUS_PER_NODE = 64
