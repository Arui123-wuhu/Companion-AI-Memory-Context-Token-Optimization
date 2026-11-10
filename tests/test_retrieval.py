"""检索打分、MMR 去冗余、召回强化。"""

import time
import unittest

from caim import retrieval
from caim.models import Memory, MemoryKind
from caim.store import MemoryStore


def make_store(items: list[Memory]) -> MemoryStore:
    store = MemoryStore(":memory:")
    for m in items:
        store.add(m)
    return store


class TestRetrieval(unittest.TestCase):
    def test_relevance_prefers_matching(self):
        allergy = Memory.from_kind("用户对花生过敏", MemoryKind.FACT, importance=0.95)
        weather = Memory.from_kind("用户说：今天下雨了", MemoryKind.EVENT)
        self.assertGreater(
            retrieval.relevance("我有什么过敏？", allergy),
            retrieval.relevance("我有什么过敏？", weather),
        )

    def test_synonym_expansion(self):
        dislike = Memory.from_kind("用户不吃香菜", MemoryKind.PREFERENCE)
        self.assertGreater(retrieval.relevance("我讨厌吃什么？", dislike), 0.0)

    def test_retrieve_returns_relevant_first(self):
        now = time.time()
        store = make_store([
            Memory.from_kind("用户对花生过敏", MemoryKind.FACT, created_at=now),
            Memory.from_kind("用户说：今天加班到十点", MemoryKind.EVENT, created_at=now),
            Memory.from_kind("用户住在杭州", MemoryKind.FACT, created_at=now),
        ])
        hits = retrieval.retrieve(store, "我花生过敏怎么办", top_k=2, now=now)
        self.assertIn("花生", hits[0].memory.text)
        store.close()

    def test_pinned_gets_boost(self):
        now = time.time()
        pinned = Memory.from_kind("用户要求记住：不喝冰的", MemoryKind.FACT,
                                  importance=0.2, pinned=True, created_at=now)
        normal = Memory.from_kind("用户说：今天喝了冰可乐", MemoryKind.EVENT, created_at=now)
        s_pinned = retrieval.score_memory(pinned, "", now)
        s_normal = retrieval.score_memory(normal, "", now)
        self.assertGreater(s_pinned.score, s_normal.score)

    def test_mmr_diversifies(self):
        now = time.time()
        store = make_store([
            Memory.from_kind("用户喜欢喝美式咖啡", MemoryKind.PREFERENCE, created_at=now),
            Memory.from_kind("用户喜欢喝美式咖啡不加糖", MemoryKind.PREFERENCE, created_at=now),
            Memory.from_kind("用户住在杭州", MemoryKind.FACT, created_at=now),
        ])
        hits = retrieval.retrieve(store, "咖啡", top_k=3, now=now, diversity=0.9)
        texts = [h.memory.text for h in hits]
        self.assertEqual(len(texts), 3)
        store.close()

    def test_retrieve_reinforces(self):
        now = time.time()
        memory = Memory.from_kind("用户对花生过敏", MemoryKind.FACT, strength=0.5, created_at=now)
        store = make_store([memory])
        hits = retrieval.retrieve(store, "花生", top_k=1, now=now)
        self.assertEqual(hits[0].memory.access_count, 1)
        self.assertGreater(store.get(memory.id).strength, 0.5)
        store.close()

    def test_no_reinforce_flag(self):
        now = time.time()
        memory = Memory.from_kind("用户对花生过敏", MemoryKind.FACT, strength=0.5, created_at=now)
        store = make_store([memory])
        retrieval.retrieve(store, "花生", top_k=1, now=now, reinforce_hits=False)
        self.assertEqual(store.get(memory.id).access_count, 0)
        store.close()

    def test_empty_query_still_returns(self):
        now = time.time()
        store = make_store([Memory.from_kind("用户住在杭州", MemoryKind.FACT, created_at=now)])
        hits = retrieval.retrieve(store, "", top_k=3, now=now)
        self.assertEqual(len(hits), 1)
        store.close()


if __name__ == "__main__":
    unittest.main()
