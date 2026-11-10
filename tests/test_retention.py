"""端到端：长对话后关键事实不丢（抗失忆）。"""

import time
import unittest

from caim.pipeline import MemoryEngine

FACTS = [
    ("我叫阿瑞", "阿瑞"),
    ("我对花生过敏", "花生"),
    ("我养了一只猫叫豆豆", "豆豆"),
    ("我住在杭州", "杭州"),
    ("我妹妹叫小雨", "小雨"),
    ("我不吃香菜", "香菜"),
]


class TestRetention(unittest.TestCase):
    def test_facts_survive_long_history(self):
        start = time.time() - 200 * 86400
        engine = MemoryEngine(":memory:", budget_tokens=700, consolidate_every=25)

        for i in range(1, 201):
            if i % 30 == 0:
                user = FACTS[(i // 30 - 1) % len(FACTS)][0]
            else:
                user = f"今天聊点别的，第{i}天"
            engine.add_turn(user, "嗯嗯", now=start + i * 86400)

        result = engine.build_context("你还记得我有什么忌口吗？", now=start + 201 * 86400)
        self.assertIn("花生", result.text)
        self.assertIn("香菜", result.text)
        self.assertLessEqual(result.tokens, 700)
        engine.close()

    def test_consolidation_reduces_storage(self):
        start = time.time() - 100 * 86400
        engine = MemoryEngine(":memory:", budget_tokens=700, consolidate_every=0)
        for i in range(1, 81):
            engine.add_turn(f"今天发生了第{i}件小事", "记下了", now=start + i * 86400)

        before = engine.stats()["stored_tokens"]
        report = engine.consolidate(now=start + 100 * 86400, older_than_turns=10)
        after = engine.stats()["stored_tokens"]
        self.assertGreater(report.summarized, 0)
        self.assertLess(after, before)
        engine.close()

    def test_pinned_survives_everything(self):
        start = time.time() - 300 * 86400
        engine = MemoryEngine(":memory:", budget_tokens=400, consolidate_every=10)
        engine.add_turn("记住我对芒果也过敏", "好，记牢了", now=start)
        for i in range(300):
            engine.add_turn(f"闲聊第{i}句", "嗯", now=start + i * 86400)
        engine.consolidate(now=start + 300 * 86400)

        result = engine.build_context("我还有什么过敏？", now=start + 300 * 86400)
        self.assertIn("芒果", result.text)
        engine.close()

    def test_stats_shape(self):
        engine = MemoryEngine(":memory:")
        engine.add_turn("我叫阿瑞", "你好阿瑞")
        stats = engine.stats()
        self.assertEqual(stats["turns"], 1)
        self.assertGreaterEqual(stats["memories"], 1)
        self.assertIn("by_kind", stats)
        engine.close()


if __name__ == "__main__":
    unittest.main()
