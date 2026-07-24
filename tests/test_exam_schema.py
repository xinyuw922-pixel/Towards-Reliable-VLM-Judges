import unittest

from scripts.exam_schema import parse_exam_id, parse_verdict


class ExamSchemaTest(unittest.TestCase):
    def test_taskdr_uid(self):
        parsed = parse_exam_id(
            "DR.doorkey.doorkey_s000000.doorkey_s000001.cf.orig.clean.neu"
        )
        self.assertEqual(parsed["task"], "D")
        self.assertEqual(parsed["group_id"], "doorkey_s000001")
        self.assertEqual(parsed["framing"], "neu")

    def test_strict_verdict(self):
        self.assertEqual(parse_verdict("Fail"), ("Fail", "strict"))

    def test_concluding_verdict_wins(self):
        response = (
            "The reference succeeds. The query never reaches the goal and "
            "does not successfully complete the task.\n\nFail"
        )
        self.assertEqual(parse_verdict(response), ("Fail", "recoverable"))

    def test_concluding_success_wins(self):
        response = "Failure is possible, but the query reaches the goal.\n\nSuccess"
        self.assertEqual(parse_verdict(response), ("Success", "recoverable"))


if __name__ == "__main__":
    unittest.main()
