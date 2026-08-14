#!/usr/bin/env python3
"""Validate the hosted OPD inputs without contacting the paid Tinker API."""

from __future__ import annotations

import asyncio
import json

from analyze import PRICING_URL, estimate_cost, load_pricing
from run_opd import (
    DEFAULT_RENDERER,
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    ReverseTextDatasetBuilder,
    ensure_tokenizer_compatibility,
)


async def main() -> None:
    ensure_tokenizer_compatibility(DEFAULT_STUDENT, DEFAULT_TEACHER)
    builder = ReverseTextDatasetBuilder(
        groups_per_batch=8,
        group_size=16,
        model_name_for_tokenizer=DEFAULT_STUDENT,
        renderer_name=DEFAULT_RENDERER,
        eval_size=128,
        max_prompt_tokens=512,
    )
    train, evaluation = await builder()
    pricing = load_pricing(PRICING_URL)
    pricing_available = (
        estimate_cost(DEFAULT_STUDENT, DEFAULT_TEACHER, 1, 1, pricing) is not None
    )
    if not pricing_available:
        raise RuntimeError("default student/teacher pricing is unavailable")

    first_batch = train.get_batch(0)
    if len(first_batch) != 8 or any(group.num_envs != 16 for group in first_batch):
        raise RuntimeError("dataset batch does not match 8 prompts x 16 rollouts")

    print(
        json.dumps(
            {
                "student_model": DEFAULT_STUDENT,
                "teacher_model": DEFAULT_TEACHER,
                "renderer": DEFAULT_RENDERER,
                "tokenizers_compatible": True,
                "train_prompts": len(train.prompts),
                "eval_prompts": len(evaluation.prompts) if evaluation else 0,
                "pricing_available": pricing_available,
                "paid_tinker_api_contacted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
