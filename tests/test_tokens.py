"""token 估算与截断。"""

import unittest

from caim.tokens import estimate_tokens, join_with_budget, truncate_to_tokens


class TestTokens(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_chinese_heavier_than_english(self):
        zh = estimate_tokens("今天天气不错")
        en = estimate_tokens("nice weather")
        self.assertGreater(zh, en)

    def test_monotonic(self):
        a = estimate_tokens("你好")
        b = estimate_tokens("你好，今天过得怎么样？")
        self.assertLess(a, b)

    def test_truncate_respects_budget(self):
        text = "这是一段比较长的中文文本，用来测试按 token 截断是否生效。" * 5
        clipped = truncate_to_tokens(text, 30)
        self.assertLessEqual(estimate_tokens(clipped), 30)
        self.assertTrue(text.startswith(clipped[:4]))

    def test_truncate_noop_when_short(self):
        text = "短句"
        self.assertEqual(truncate_to_tokens(text, 100), text)

    def test_join_with_budget(self):
        parts = ["一" * 10, "二" * 10, "三" * 10]
        joined = join_with_budget(parts, 12)
        self.assertLessEqual(estimate_tokens(joined), 12)


if __name__ == "__main__":
    unittest.main()
