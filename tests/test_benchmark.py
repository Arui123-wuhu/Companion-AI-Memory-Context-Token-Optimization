"""基准自身的一致性检查。"""

import unittest

from caim.benchmark import build_transcript, format_report, run_benchmark


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_benchmark(total_turns=60, budget=800, window=8)

    def test_transcript_shape(self):
        turns, probes = build_transcript(60)
        self.assertEqual(len(turns), 60)
        self.assertEqual(len(probes), 12)

    def test_caim_cheaper_than_full(self):
        st = self.result["strategies"]
        self.assertLess(st["caim"]["avg_tokens"], st["full_history"]["avg_tokens"])

    def test_caim_keeps_more_than_window(self):
        st = self.result["strategies"]
        self.assertGreater(st["caim"]["retention"], st["sliding_window"]["retention"])

    def test_report_renders(self):
        text = format_report(self.result)
        self.assertIn("CAIM", text)
        self.assertIn("全量历史", text)


if __name__ == "__main__":
    unittest.main()
