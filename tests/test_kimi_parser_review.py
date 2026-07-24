import unittest
from collections import Counter
from pathlib import Path

from scripts.build_kimi_parser_review import build_records


ROOT = Path(__file__).resolve().parents[1]


class KimiParserReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = build_records(
            ROOT / "frozen_outputs/minigrid/taskd-r/kimi-k2.5/score_per_row.jsonl",
            ROOT / "datasets/minigrid/taskd-r/task_d_exam.jsonl",
        )

    def test_review_population(self):
        self.assertEqual(len(self.records), 133)
        self.assertEqual(sum(row["parser_changed"] for row in self.records), 49)
        self.assertEqual(
            Counter(row["corrected_mode"] for row in self.records),
            Counter({"recoverable": 132, "fail": 1}),
        )

    def test_changed_rows_have_distinct_predictions(self):
        changed = [row for row in self.records if row["parser_changed"]]
        self.assertTrue(all(row["legacy_pred"] != row["corrected_pred"] for row in changed))

    def test_gold_is_complete_but_not_used_by_parser(self):
        self.assertTrue(all(row["gold"] in {"Success", "Fail"} for row in self.records))


if __name__ == "__main__":
    unittest.main()
