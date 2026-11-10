"""遗忘曲线与间隔重复。

对应人脑：

- **艾宾浩斯遗忘曲线**：记忆保持率随时间指数下降。
- **重要/情绪唤醒的记忆衰减更慢**：半衰期随 importance 增大。
- **间隔重复效应**：每次成功回忆都会让记忆更牢固（strength 上升），
  且下一次衰减更慢（last_access 重置）。
"""

from __future__ import annotations

import math
import time

from .models import Memory

#: 基础半衰期（天）：完全不重要的记忆两天掉一半
BASE_HALF_LIFE_DAYS = 2.0
#: importance=1 时额外的半衰期（天）
MAX_EXTRA_HALF_LIFE_DAYS = 40.0
#: 低于该有效强度视为"想不起来了"
FORGET_THRESHOLD = 0.06


def half_life_days(importance: float) -> float:
    """重要度越高，遗忘越慢。"""
    importance = max(0.0, min(1.0, importance))
    return BASE_HALF_LIFE_DAYS + MAX_EXTRA_HALF_LIFE_DAYS * (importance ** 1.5)


def retention(memory: Memory, now: float | None = None) -> float:
    """当前保持率 = strength * 0.5^(距今/半衰期)，取值 0~1。"""
    now = time.time() if now is None else now
    age_days = max(0.0, (now - memory.last_access) / 86400.0)
    return memory.strength * math.pow(0.5, age_days / half_life_days(memory.importance))


def effective_strength(memory: Memory, now: float | None = None) -> float:
    """``retention`` 的别名，语义更贴近"可回忆强度"。"""
    return retention(memory, now)


def reinforce(memory: Memory, now: float | None = None, boost: float = 0.3) -> Memory:
    """回忆一次 → 突触增强（间隔重复）。

    增量与"剩余空间"成正比，避免无限增长：
    ``s' = s + boost * (1 - s)``，然后重置 last_access。
    """
    now = time.time() if now is None else now
    memory.strength = min(1.0, memory.strength + boost * (1.0 - memory.strength))
    memory.last_access = now
    memory.access_count += 1
    return memory


def should_forget(memory: Memory, now: float | None = None, floor: float = FORGET_THRESHOLD) -> bool:
    """是否可以被修剪（置顶记忆永不遗忘）。"""
    if memory.pinned:
        return False
    if memory.kind.value == "profile":
        return False
    return effective_strength(memory, now) < floor


def schedule_review(memory: Memory, now: float | None = None) -> float:
    """给出下一次"该复习"的时间戳（间隔重复调度）。

    间隔随回忆次数指数增长：``interval = 1.6^count`` 天，上限 60 天。
    """
    now = time.time() if now is None else now
    interval_days = min(60.0, 1.6 ** max(0, memory.access_count))
    return now + interval_days * 86400.0
