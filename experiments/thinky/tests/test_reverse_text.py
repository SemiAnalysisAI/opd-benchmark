from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from reverse_text import expected_answer, score_response, split_prompts


class ReverseTextTest(unittest.TestCase):
    def test_exact_tagged_answer_scores_one(self) -> None:
        answer = expected_answer("abc 123")
        score, valid = score_response("<reversed_text>321 cba</reversed_text>", answer)
        self.assertEqual(score, 1.0)
        self.assertTrue(valid)

    def test_missing_tag_scores_zero(self) -> None:
        score, valid = score_response("cba", "cba")
        self.assertEqual(score, 0.0)
        self.assertFalse(valid)

    def test_split_reserves_source_order_tail(self) -> None:
        train, evaluation = split_prompts(["a", "b", "c", "d"], 2)
        self.assertEqual(train, ["a", "b"])
        self.assertEqual(evaluation, ["c", "d"])


if __name__ == "__main__":
    unittest.main()
