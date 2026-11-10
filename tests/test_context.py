"""上下文装配：预算、优先级、可观测字段。"""

import time
import unittest

from caim.context import ContextBuilder, DEFAULT_RULES
from caim.models import Memory, MemoryKind
from caim.store import MemoryStore


def build_store(now: float) -> MemoryStore:
    store = MemoryStore(":memory:")
    store.add(Memory.from_kind("用户的名字是阿瑞", MemoryKind.PROFILE, created_at=now))
    store.add(Memory.from_kind("用户对花生过敏", MemoryKind.FACT, created_at=now))
    store.add(Memory.from_kind("用户住在杭州", MemoryKind.FACT, created_at=now))
    store.add(Memory.from_kind("用户要求记住：不喝冰的", MemoryKind.FACT,
                               pinned=True, created_at=now))
    for i in range(1, 6):
        store.add_turn(i, f"第{i}轮用户消息", f"第{i}轮回复", now=now)
        store.add(Memory.from_kind(f"用户说：第{i}轮闲聊内容", MemoryKind.EVENT,
                                   turn=i, created_at=now))
    return store


class TestContextBuilder(unittest.TestCase):
    def setUp(self):
        self.now = time.time()
        self.store = build_store(self.now)

    def tearDown(self):
        self.store.close()

    def test_budget_respected(self):
        builder = ContextBuilder(self.store, budget_tokens=300)
        result = builder.build("我有什么忌口", now=self.now)
        self.assertLessEqual(result.tokens, 300)
        self.assertGreater(result.tokens, 0)

    def test_pinned_always_included(self):
        builder = ContextBuilder(self.store, budget_tokens=200)
        result = builder.build("随便聊聊", now=self.now)
        self.assertIn("不喝冰的", result.text)

    def test_relevant_memory_included(self):
        builder = ContextBuilder(self.store, budget_tokens=600)
        result = builder.build("我花生过敏吗", now=self.now)
        self.assertIn("花生", result.text)

    def test_recent_turns_included(self):
        builder = ContextBuilder(self.store, budget_tokens=600)
        result = builder.build("", now=self.now, recent_turns=2)
        self.assertIn("第5轮用户消息", result.text)

    def test_sections_reported(self):
        builder = ContextBuilder(self.store, budget_tokens=600)
        result = builder.build("花生", now=self.now)
        self.assertIn("rules", result.sections)
        self.assertTrue(any(k in result.sections for k in ("profile", "longterm", "recent")))
        self.assertTrue(result.memory_ids)

    def test_small_budget_does_not_crash(self):
        builder = ContextBuilder(self.store, budget_tokens=128)
        result = builder.build("花生", now=self.now)
        self.assertLessEqual(result.tokens, 128)
        self.assertTrue(result.text.startswith(DEFAULT_RULES[:10]))

    def test_no_duplicate_between_profile_and_longterm(self):
        store = MemoryStore(":memory:")
        store.add(Memory.from_kind("用户的名字是阿瑞", MemoryKind.PROFILE))
        builder = ContextBuilder(store, budget_tokens=400)
        result = builder.build("我叫什么", now=self.now)
        self.assertEqual(result.text.count("用户的名字是阿瑞"), 1)
        store.close()

    def test_utilization_property(self):
        builder = ContextBuilder(self.store, budget_tokens=600)
        result = builder.build("花生", now=self.now)
        self.assertGreater(result.utilization, 0)
        self.assertLessEqual(result.utilization, 1)
        self.assertIn("tokens", result.summary())


if __name__ == "__main__":
    unittest.main()
