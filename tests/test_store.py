"""SQLite 持久化。"""

import os
import shutil
import unittest

from caim.models import Memory, MemoryKind
from caim.store import MemoryStore


class TestStore(unittest.TestCase):
    def test_add_get_roundtrip(self):
        store = MemoryStore(":memory:")
        memory = Memory.from_kind("用户对花生过敏", MemoryKind.FACT, entity="花生")
        store.add(memory)
        loaded = store.get(memory.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.text, memory.text)
        self.assertEqual(loaded.entity, "花生")
        store.close()

    def test_pinned_query(self):
        store = MemoryStore(":memory:")
        store.add(Memory.from_kind("普通事实", MemoryKind.FACT))
        store.add(Memory.from_kind("重要事实", MemoryKind.FACT, pinned=True))
        self.assertEqual([m.text for m in store.pinned()], ["重要事实"])
        store.close()

    def test_turns_order(self):
        store = MemoryStore(":memory:")
        for i in range(1, 6):
            store.add_turn(i, f"u{i}", f"r{i}")
        recent = store.recent_turns(3)
        self.assertEqual([t["turn"] for t in recent], [3, 4, 5])
        store.close()

    def test_meta(self):
        store = MemoryStore(":memory:")
        store.set_meta("turn", 42)
        self.assertEqual(store.get_meta("turn"), 42)
        self.assertEqual(store.get_meta("missing", "x"), "x")
        store.close()

    def test_persistence_to_file(self):
        base = os.path.join(os.path.dirname(__file__), "_tmp_store")
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, "m.db")
        try:
            store = MemoryStore(path)
            store.add(Memory.from_kind("持久化测试", MemoryKind.FACT))
            store.close()

            reopened = MemoryStore(path)
            self.assertEqual(reopened.count(), 1)
            self.assertEqual(reopened.all()[0].text, "持久化测试")
            reopened.close()
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_delete(self):
        store = MemoryStore(":memory:")
        memory = store.add(Memory.from_kind("临时", MemoryKind.EVENT))
        store.delete(memory.id)
        self.assertIsNone(store.get(memory.id))
        store.close()


if __name__ == "__main__":
    unittest.main()
