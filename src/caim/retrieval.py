"""检索：线索依赖的回忆 + 多因子打分 + 去冗余。

对应人脑：

- **线索依赖回忆**：检索效果取决于当前话题与记忆的重合度（relevance）。
- **首因/近因效应**：越新的记忆越容易被想起（recency）。
- **情绪唤醒**：重要的事更容易被提取（importance）。
- **突触强度**：常被回忆的内容更牢固（strength）。
- **侧抑制 / 去冗余**：一次只让最相关的一批记忆进入工作记忆，
  避免相似条目挤占上下文（MMR 多样性重排）。
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from . import decay
from .models import Memory, MemoryKind
from .store import MemoryStore

_PUNCT = re.compile(r"[\s，。！？；：、,.!?;:~…\"'“”‘’()（）\[\]【】]+")

#: 各因子权重（总和为 1）。近因权重刻意压低：最近几轮对话本来就会原文进上下文，
#: 长时记忆的检索更应该看重"相关"和"重要"，否则闲聊会把真正的事实挤出预算。
W_RELEVANCE = 0.55
W_RECENCY = 0.12
W_IMPORTANCE = 0.23
W_STRENGTH = 0.10

#: 记忆的时间尺度（小时）：一周前的记忆近因权重衰减到 1/e
RECENCY_TAU_HOURS = 72.0

#: 同义词扩展表：左词出现时，把右词的字符组也算作查询线索
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "讨厌": ("不喜欢", "不吃", "反感"),
    "不喜欢": ("讨厌", "不吃"),
    "忌口": ("不吃", "过敏", "不能吃"),
    "过敏": ("忌口", "不能吃"),
    "害怕": ("怕", "恐惧"),
    "怕": ("害怕",),
    "喜欢": ("爱", "爱吃"),
    "名字": ("叫", "称呼"),
    "生日": ("出生",),
    "工作": ("职业", "上班"),
    "住": ("城市", "家"),
    "猫": ("宠物", "喵"),
    "狗": ("宠物", "汪"),
}


def _normalize(text: str) -> str:
    return _PUNCT.sub("", text.lower())


def bigrams(text: str) -> set[str]:
    """字符二元组集合；对中文无需分词，对英文也能兜底。"""
    s = _normalize(text)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def relevance(query: str, memory: Memory) -> float:
    """查询覆盖率：query 的二元组有多少出现在记忆里（0~1）。

    额外做一层同义词扩展（"讨厌/不喜欢/不吃"），缓解"问法和记法不一致"导致的漏检；
    扩展词只增加命中，不扩大分母，避免稀释分数。
    """
    base = bigrams(query)
    if not base:
        return 0.0
    expanded = set(base)
    for word, alternatives in _SYNONYMS.items():
        if word in query:
            for alt in alternatives:
                expanded |= bigrams(alt)
    target = bigrams(memory.text)
    if memory.entity:
        target |= bigrams(memory.entity)
    hits = len(expanded & target)
    return min(1.0, hits / len(base))


def recency(memory: Memory, now: float) -> float:
    age_hours = max(0.0, (now - memory.created_at) / 3600.0)
    return math.exp(-age_hours / RECENCY_TAU_HOURS)


@dataclass
class ScoredMemory:
    memory: Memory
    score: float
    relevance: float
    recency: float
    importance: float
    strength: float

    def explain(self) -> str:
        return (
            f"{self.memory.text}  (score={self.score:.2f} "
            f"rel={self.relevance:.2f} rec={self.recency:.2f} "
            f"imp={self.importance:.2f} str={self.strength:.2f})"
        )


def score_memory(memory: Memory, query: str, now: float | None = None) -> ScoredMemory:
    now = time.time() if now is None else now
    rel = relevance(query, memory)
    rec = recency(memory, now)
    imp = memory.importance
    strength = decay.effective_strength(memory, now)
    score = W_RELEVANCE * rel + W_RECENCY * rec + W_IMPORTANCE * imp + W_STRENGTH * strength
    # 置顶/身份类记忆获得固定加成，保证永不被挤出
    if memory.pinned or memory.kind is MemoryKind.PROFILE:
        score += 0.25
    return ScoredMemory(memory, score, rel, rec, imp, strength)


def _similarity(a: Memory, b: Memory) -> float:
    sa, sb = bigrams(a.text), bigrams(b.text)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def retrieve(
    store: MemoryStore,
    query: str,
    *,
    top_k: int = 8,
    now: float | None = None,
    min_relevance: float = 0.0,
    diversity: float = 0.25,
    reinforce_hits: bool = True,
) -> list[ScoredMemory]:
    """检索 top_k 条记忆。

    Args:
        store: 记忆仓库。
        query: 当前话题/问题。
        top_k: 返回条数。
        now: 时间戳（测试可注入）。
        min_relevance: 相关度下限，过滤"风马牛不相及"的记忆。
        diversity: MMR 的多样性惩罚系数，0 表示纯按分数取。
        reinforce_hits: 是否对被召回的记忆做一次"间隔重复"增强。
    """
    now = time.time() if now is None else now
    candidates = [score_memory(m, query, now) for m in store.all()]
    candidates = [c for c in candidates if c.relevance >= min_relevance]
    candidates.sort(key=lambda c: c.score, reverse=True)
    if not candidates:
        return []

    if diversity <= 0:
        chosen = candidates[:top_k]
    else:
        chosen: list[ScoredMemory] = []
        pool = candidates[: max(top_k * 4, top_k)]
        while pool and len(chosen) < top_k:
            best_i, best_val = 0, -1e9
            for i, cand in enumerate(pool):
                if not chosen:
                    val = cand.score
                else:
                    sim = max(_similarity(cand.memory, c.memory) for c in chosen)
                    val = cand.score - diversity * sim
                if val > best_val:
                    best_i, best_val = i, val
            chosen.append(pool.pop(best_i))

    if reinforce_hits:
        for item in chosen:
            decay.reinforce(item.memory, now)
            store.update(item.memory)
    return chosen
