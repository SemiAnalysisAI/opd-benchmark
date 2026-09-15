import json
from pathlib import Path

import verifiers.v1 as vf

from mopd_puzzles.scoring import score


class MopdPuzzlesData(vf.TaskData):
    label: str
    metadata: dict


class MopdPuzzlesTask(vf.Task[MopdPuzzlesData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def reward(self, trace: vf.Trace) -> float:
        return score(trace.last_reply or '', self.data.label)


class MopdPuzzlesConfig(vf.TasksetConfig):
    data_path: str
    domain: str


class MopdPuzzlesTaskset(vf.Taskset[MopdPuzzlesTask, MopdPuzzlesConfig]):
    def load(self) -> list[MopdPuzzlesTask]:
        tasks = []
        with Path(self.config.data_path).open() as stream:
            for i, line in enumerate(stream):
                row = json.loads(line)
                assert row['metadata']['domain'] == self.config.domain
                assert [message['role'] for message in row['prompt']] == ['system', 'user']
                data = MopdPuzzlesData(idx=i, system_prompt=row['prompt'][0]['content'],
                                       prompt=row['prompt'][1]['content'], label=row['label'], metadata=row['metadata'])
                tasks.append(MopdPuzzlesTask(data, self.config.task))
        return tasks
