#!/usr/bin/env python3
"""Run the reverse-text OPD workload on Thinking Machines' hosted Tinker service."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import chz
import tinker
from datasets import load_dataset
from reverse_text import (
    DATASET_ID,
    SYSTEM_PROMPT,
    expected_answer,
    score_response,
    split_prompts,
)
from tinker_cookbook import checkpoint_utils, cli_utils, renderers
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyEnv,
    TeacherConfig,
)
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

DEFAULT_STUDENT = "Qwen/Qwen3.5-4B"
DEFAULT_TEACHER = "Qwen/Qwen3.5-9B"
DEFAULT_RENDERER = "qwen3_5_disable_thinking"


class ReverseTextEnv(PromptOnlyEnv):
    """Zero-reward OPD environment with post-hoc task-quality metrics."""

    def __init__(self, prompt: str, renderer: renderers.Renderer) -> None:
        super().__init__(
            prompt=prompt,
            renderer=renderer,
            convo_prefix=[{"role": "system", "content": SYSTEM_PROMPT}],
        )
        self.answer = expected_answer(prompt)

    async def step(
        self, action: Action, *, extra: ActionExtra | None = None
    ) -> StepResult:
        message, _termination = self.renderer.parse_response(action)
        response = renderers.get_text_content(message)
        task_score, format_valid = score_response(response, self.answer)
        stop_reason = (extra or {}).get("stop_reason")
        stop_value = getattr(stop_reason, "value", stop_reason)

        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "task_score": task_score,
                "format_valid": float(format_valid),
                "truncated": float(stop_value == "length"),
            },
        )


class ReverseTextDataset(RLDataset):
    def __init__(
        self,
        prompts: list[str],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        tokenizer: Tokenizer,
        max_prompt_tokens: int,
        dataset_name: str,
    ) -> None:
        self.prompts = prompts
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.dataset_name = dataset_name

    def _truncate(self, prompt: str) -> str:
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) <= self.max_prompt_tokens:
            return prompt
        return self.tokenizer.decode(tokens[: self.max_prompt_tokens])

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.prompts))
        if start >= end:
            raise IndexError(
                f"batch {index} is outside a dataset of {len(self.prompts)} prompts"
            )
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    ReverseTextEnv, self._truncate(prompt), self.renderer
                ),
                num_envs=self.group_size,
                dataset_name=self.dataset_name,
            )
            for prompt in self.prompts[start:end]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.prompts) / self.batch_size)


@chz.chz
class ReverseTextDatasetBuilder(RLDatasetBuilder):
    groups_per_batch: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    eval_size: int = 128
    max_prompt_tokens: int = 512

    async def __call__(self) -> tuple[ReverseTextDataset, ReverseTextDataset | None]:
        raw = load_dataset(DATASET_ID, split="train")
        prompts = [str(row["prompt"]) for row in raw]
        train_prompts, eval_prompts = split_prompts(prompts, self.eval_size)

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(
            self.renderer_name,
            tokenizer=tokenizer,
            model_name=self.model_name_for_tokenizer,
        )
        train_dataset = ReverseTextDataset(
            prompts=train_prompts,
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            tokenizer=tokenizer,
            max_prompt_tokens=self.max_prompt_tokens,
            dataset_name="reverse-text",
        )
        eval_dataset = (
            ReverseTextDataset(
                prompts=eval_prompts,
                batch_size=self.groups_per_batch,
                group_size=1,
                renderer=renderer,
                tokenizer=tokenizer,
                max_prompt_tokens=self.max_prompt_tokens,
                dataset_name="reverse-text-eval",
            )
            if eval_prompts
            else None
        )
        return train_dataset, eval_dataset


def ensure_tokenizer_compatibility(student_model: str, teacher_model: str) -> None:
    """Fail before paid training if sampled token IDs differ between models."""
    student = get_tokenizer(student_model)
    teacher = get_tokenizer(teacher_model)
    probes = [
        SYSTEM_PROMPT,
        "Reverse this text: abc XYZ 123!",
        "<reversed_text>!321 ZYX cba</reversed_text>",
    ]
    mismatches = [
        probe for probe in probes if student.encode(probe) != teacher.encode(probe)
    ]
    if mismatches:
        raise ValueError(
            "student and teacher tokenizers are incompatible for sampled-token OPD; "
            f"first mismatch: {mismatches[0]!r}"
        )


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent.parent
    default_run = f"tinker-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", default=DEFAULT_STUDENT)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--renderer-name", default=DEFAULT_RENDERER)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--rollouts-per-prompt", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=15)
    parser.add_argument("--compute-post-kl", action="store_true")
    parser.add_argument("--enable-trace", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=script_root / "results" / default_run,
    )
    parser.add_argument(
        "--logdir-behavior",
        choices=("raise", "resume"),
        default="raise",
        help="Refuse an existing result directory by default; resume only when requested.",
    )
    args = parser.parse_args()
    for name in (
        "steps",
        "prompts_per_step",
        "rollouts_per_prompt",
        "max_tokens",
        "lora_rank",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


async def run(args: argparse.Namespace) -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is required; create one in the Tinker console")

    ensure_tokenizer_compatibility(args.student_model, args.teacher_model)
    renderer_name = (
        await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
            model_name=args.student_model,
            explicit_renderer_name=args.renderer_name,
            load_checkpoint_path=args.student_checkpoint,
            base_url=args.base_url,
        )
    )
    dataset_builder = ReverseTextDatasetBuilder(
        groups_per_batch=args.prompts_per_step,
        group_size=args.rollouts_per_prompt,
        model_name_for_tokenizer=args.student_model,
        renderer_name=renderer_name,
        eval_size=args.eval_size,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=TeacherConfig(
            base_model=args.teacher_model,
            load_checkpoint_path=args.teacher_checkpoint,
        ),
        groups_per_batch=args.prompts_per_step,
    )
    config = train_on_policy.Config(
        recipe_name="opd_reverse_text_hosted_comparison",
        learning_rate=args.learning_rate,
        dataset_configs=[dataset_config],
        model_name=args.student_model,
        renderer_name=renderer_name,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        kl_penalty_coef=args.kl_penalty_coef,
        kl_discount_factor=0.0,
        num_substeps=1,
        loss_fn="importance_sampling",
        loss_fn_config=None,
        log_path=str(args.log_path.resolve()),
        base_url=args.base_url,
        load_checkpoint_path=args.student_checkpoint,
        compute_post_kl=args.compute_post_kl,
        eval_every=args.eval_every,
        save_every=args.save_every,
        max_steps=args.steps,
        enable_trace=args.enable_trace,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=args.logdir_behavior)
    await train_on_policy.main(config)


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
