"""规则抽取。"""

import unittest

from caim.extract import RuleExtractor
from caim.models import MemoryKind


class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.ex = RuleExtractor()

    def texts(self, sentence: str) -> list[str]:
        return [m.text for m in self.ex.extract(sentence)]

    def test_name(self):
        self.assertIn("用户的名字是阿瑞", self.texts("你好，我叫阿瑞"))

    def test_allergy(self):
        self.assertIn("用户对花生过敏", self.texts("我对花生过敏，很严重"))

    def test_pet(self):
        out = self.texts("我养了一只猫叫豆豆")
        self.assertTrue(any("养了猫" in t for t in out))
        self.assertTrue(any("豆豆" in t for t in out))

    def test_family(self):
        self.assertIn("用户的妹妹叫小雨", self.texts("我妹妹叫小雨，在读大学"))

    def test_not_eat(self):
        self.assertIn("用户不吃香菜", self.texts("我不吃香菜"))

    def test_pin(self):
        memories = self.ex.extract("记住我不喝冰的，胃不好")
        self.assertTrue(any(m.pinned for m in memories))
        self.assertTrue(all(m.importance >= 0.95 for m in memories))

    def test_plain_event(self):
        memories = self.ex.extract("今天下班路上堵了半小时")
        self.assertEqual(len(memories), 1)
        self.assertIs(memories[0].kind, MemoryKind.EVENT)
        self.assertLess(memories[0].importance, 0.5)

    def test_empty(self):
        self.assertEqual(self.ex.extract("   "), [])

    def test_no_duplicate_text(self):
        out = self.ex.extract("我叫阿瑞，叫我阿瑞就好")
        self.assertEqual(len(out), len({m.text for m in out}))

    def test_turn_and_time_recorded(self):
        memories = self.ex.extract("我叫阿瑞", turn=7, now=1000.0)
        self.assertEqual(memories[0].turn, 7)
        self.assertEqual(memories[0].created_at, 1000.0)


if __name__ == "__main__":
    unittest.main()
