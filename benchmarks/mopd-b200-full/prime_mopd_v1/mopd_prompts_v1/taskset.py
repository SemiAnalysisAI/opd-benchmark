"""Static single-turn prompts routed by Prime-RL to specialist teachers."""

from __future__ import annotations

import json

import verifiers.v1 as vf


class MOPDPromptData(vf.TaskData):
    domain: str
    answer: str


class MOPDPromptTask(vf.Task[MOPDPromptData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def diagnostic_zero(self, trace: vf.Trace) -> float:
        # OPD is the only training signal in the benchmark.
        return 0.0


class MOPDPromptsConfig(vf.TasksetConfig):
    data_path: str


class MOPDPromptsTaskset(vf.Taskset[MOPDPromptTask, MOPDPromptsConfig]):
    def load(self) -> list[MOPDPromptTask]:
        with open(self.config.data_path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return [
            MOPDPromptTask(
                MOPDPromptData(
                    idx=i,
                    prompt=row["prompt"],
                    system_prompt=row["system"],
                    domain=row["domain"],
                    answer=row["answer"],
                ),
                self.config.task,
            )
            for i, row in enumerate(rows)
        ]
