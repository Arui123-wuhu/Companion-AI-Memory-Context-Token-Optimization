"""遗忘曲线与间隔重复。"""

import time
import unittest

from caim import decay
from caim.models import Memory, MemoryKind


class TestDecay(unittest.TestCase):
    def test_importance_slows_forgetting(self):
        self.assertGreater(decay.half_life_days(0.95), decay.half_life_days(0.2))

    def test_retention_drops_over_time(self):
        now = time.time()
        memory = Memory.from_kind("用户对花生过敏", MemoryKind.FACT, created_at=now)
        memory.last_access = now
        self.assertAlmostEqual(decay.retention(memory, now), memory.strength, places=5)
        week_later = now + 7 * 86400
        self.assertLess(decay.retention(memory, week_later), decay.retention(memory, now))

    def test_important_memory_retains_more(self):
        now = time.time()
        strong = Memory.from_kind("用户对花生过敏", MemoryKind.FACT, importance=1.0, created_at=now)
        weak = Memory.from_kind("用户昨天吃了面", MemoryKind.EVENT, importance=0.2, created_at=now)
        for m in (strong, weak):
            m.last_access = now
        later = now + 10 * 86400
        self.assertGreater(decay.retention(strong, later), decay.retention(weak, later))

    def test_reinforce_increases_strength_and_count(self):
        memory = Memory.from_kind("事实", MemoryKind.FACT, strength=0.5)
        before = memory.strength
        decay.reinforce(memory, boost=0.3)
        self.assertGreater(memory.strength, before)
        self.assertEqual(memory.access_count, 1)
        self.assertLessEqual(memory.strength, 1.0)

    def test_reinforce_saturates(self):
        memory = Memory.from_kind("事实", MemoryKind.FACT, strength=1.0)
        for _ in range(5):
            decay.reinforce(memory)
        self.assertLessEqual(memory.strength, 1.0)

    def test_pinned_never_forgotten(self):
        now = time.time()
        memory = Memory.from_kind("用户要求记住：不喝冰的", MemoryKind.FACT, pinned=True)
        memory.last_access = now - 3650 * 86400
        self.assertFalse(decay.should_forget(memory, now))

    def test_old_event_forgotten(self):
        now = time.time()
        memory = Memory.from_kind("用户说：今天下雨", MemoryKind.EVENT, importance=0.2)
        memory.last_access = now - 60 * 86400
        memory.strength = 0.5
        self.assertTrue(decay.should_forget(memory, now))

    def test_review_schedule_grows(self):
        memory = Memory.from_kind("事实", MemoryKind.FACT)
        memory.access_count = 0
        first = decay.schedule_review(memory)
        memory.access_count = 3
        later = decay.schedule_review(memory)
        self.assertGreater(later, first)


if __name__ == "__main__":
    unittest.main()
