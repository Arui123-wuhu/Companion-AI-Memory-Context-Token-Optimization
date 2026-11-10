"""记忆巩固：去重、摘要、修剪。

对应人脑的"睡眠巩固"：

- **重复抑制**：同一件事反复出现，只保留一条更强的表征。
- **情景 -> 语义**：把零散的情景记忆压缩成阶段性事实，降低存储与检索成本。
- **突触修剪**：长期不用且不重要的连接被剪掉，给重要记忆腾地方。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import decay
from .models import Memory, MemoryKind
from .retrieval import _similarity
from .store import MemoryStore
from .tokens import estimate_tokens


@dataclass
class ConsolidationReport:
    merged: int = 0
    summarized: int = 0
    pruned: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    details: list[str] = field(default_factory=list)

    @property
    def saved_tokens(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


def _store_tokens(store: MemoryStore) -> int:
    return sum(estimate_tokens(m.text) for m in store.all())


def merge_duplicates(store: MemoryStore, threshold: float = 0.85) -> tuple[int, list[str]]:
    """合并高度相似的记忆（保留信息更全、更强的那条）。"""
    memories = store.all()
    removed: set[str] = set()
    details: list[str] = []
    merged = 0

    for i, a in enumerate(memories):
        if a.id in removed:
            continue
        for b in memories[i + 1 :]:
            if b.id in removed or a.kind is not b.kind:
                continue
            if a.kind is MemoryKind.PROFILE:
                continue
            if _similarity(a, b) < threshold:
                continue
            keep, drop = (a, b) if len(a.text) >= len(b.text) else (b, a)
            keep.strength = max(a.strength, b.strength)
            keep.importance = max(a.importance, b.importance)
            keep.pinned = a.pinned or b.pinned
            keep.access_count = max(a.access_count, b.access_count)
            keep.last_access = max(a.last_access, b.last_access)
            store.update(keep)
            store.delete(drop.id)
            removed.add(drop.id)
            merged += 1
            details.append(f"合并重复：{drop.text} -> {keep.text}")
            break
    return merged, details


def summarize_episodes(
    store: MemoryStore,
    *,
    older_than_turns: int = 24,
    group_size: int = 4,
    max_chars: int = 140,
) -> tuple[int, list[str]]:
    """把较早的情景记忆压成"阶段纪要"。

    只处理未被抽取成事实的纯事件（EVENT），避免丢失结构化信息。
    """
    current_turn = int(store.get_meta("turn", 0))
    cutoff = current_turn - older_than_turns
    events = [m for m in store.all() if m.kind is MemoryKind.EVENT and m.turn < cutoff]
    if len(events) < group_size:
        return 0, []

    events.sort(key=lambda m: m.turn)
    details: list[str] = []
    summarized = 0
    for start in range(0, len(events) - group_size + 1, group_size):
        chunk = events[start : start + group_size]
        gists = "；".join(m.text.replace("用户说：", "") for m in chunk)
        summary_text = f"阶段纪要（第{chunk[0].turn}-{chunk[-1].turn}轮）：{gists}"[:max_chars]
        summary = Memory.from_kind(
            summary_text, MemoryKind.EVENT,
            importance=max(m.importance for m in chunk) * 0.9,
            turn=chunk[-1].turn,
            created_at=max(m.created_at for m in chunk),
        )
        store.add(summary)
        for m in chunk:
            store.delete(m.id)
        summarized += len(chunk)
        details.append(f"压缩 {len(chunk)} 条情景 -> 1 条纪要（省 {estimate_tokens(gists) - estimate_tokens(summary_text)} tokens）")
    return summarized, details


def prune(store: MemoryStore, now: float | None = None, floor: float = decay.FORGET_THRESHOLD) -> tuple[int, list[str]]:
    """修剪已经"想不起来"的低价值记忆。"""
    now = time.time() if now is None else now
    pruned = 0
    details: list[str] = []
    for memory in store.all():
        if decay.should_forget(memory, now, floor) and memory.kind is MemoryKind.EVENT:
            store.delete(memory.id)
            pruned += 1
            if len(details) < 10:
                details.append(f"遗忘：{memory.text}")
    return pruned, details


def consolidate(
    store: MemoryStore,
    *,
    now: float | None = None,
    merge_threshold: float = 0.85,
    older_than_turns: int = 24,
) -> ConsolidationReport:
    """执行一轮完整巩固（相当于"睡一觉"）。"""
    now = time.time() if now is None else now
    report = ConsolidationReport(tokens_before=_store_tokens(store))

    merged, d1 = merge_duplicates(store, merge_threshold)
    summarized, d2 = summarize_episodes(store, older_than_turns=older_than_turns)
    pruned, d3 = prune(store, now)

    report.merged = merged
    report.summarized = summarized
    report.pruned = pruned
    report.details = d1 + d2 + d3
    report.tokens_after = _store_tokens(store)
    return report
