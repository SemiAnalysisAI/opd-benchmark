from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze import summarize_tinker


class AnalyzeTest(unittest.TestCase):
    def test_summarizes_timing_quality_and_cost(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary)
            config = {
                "model_name": "student",
                "lora_rank": 8,
                "dataset_configs": [{"teacher_config": {"base_model": "teacher"}}],
            }
            (result_dir / "config.json").write_text(json.dumps(config))
            (result_dir / "wall_time.env").write_text("wall_seconds=10\n")
            rows = []
            for step, task_score, reverse_kl, seconds in (
                (0, 0.1, 0.4, 2.0),
                (1, 0.2, 0.3, 3.0),
            ):
                rows.append(
                    {
                        "step": step,
                        "env/all/task_score": task_score,
                        "env/all/truncated": 0.0,
                        "env/all/ac_tokens_per_turn": 12.5,
                        "env/all/total_episodes": 4,
                        "env/all/total_ob_tokens": 100,
                        "env/all/total_ac_tokens": 50,
                        "teacher_kl": reverse_kl,
                        "time/total": seconds,
                    }
                )
            (result_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (result_dir / "checkpoints.jsonl").write_text(
                json.dumps({"name": "step", "state_path": "tinker://step"})
                + "\n"
                + json.dumps({"name": "final", "state_path": "tinker://final"})
                + "\n"
            )
            pricing = [
                {
                    "tinker_id": "student",
                    "prefill": "$1.00",
                    "cached_prefill": "$0.20",
                    "sample": "$2.00",
                    "train": "$3.00",
                },
                {
                    "tinker_id": "teacher",
                    "prefill": "$4.00",
                    "cached_prefill": "$0.80",
                    "sample": "$0.00",
                    "train": "$0.00",
                },
            ]

            summary = summarize_tinker(result_dir, pricing)

        self.assertEqual(summary["total_trajectories"], 8)
        self.assertEqual(summary["end_to_end_trajectories_per_second"], 0.8)
        self.assertEqual(summary["task_score"]["final"], 0.2)
        self.assertEqual(summary["reverse_kl"]["final"], 0.3)
        self.assertEqual(summary["checkpoint_records"], 2)
        self.assertLess(
            summary["estimated_cost"]["lower_bound_all_prefill_cached"],
            summary["estimated_cost"]["upper_bound_no_prefill_cached"],
        )


if __name__ == "__main__":
    unittest.main()
