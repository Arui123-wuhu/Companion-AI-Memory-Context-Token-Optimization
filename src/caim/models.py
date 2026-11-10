"""记忆条目模型。

设计对应关系（人脑 -> 工程）：

===============  ==========================================
人脑             工程
===============  ==========================================
工作记忆         当前上下文窗口（由 ContextBuilder 装配）
情景记忆         MemoryKind.EVENT，带时间戳的事件片段
语义记忆         MemoryKind.FACT / PREFERENCE，实体化事实
身份/自我图式     MemoryKind.PROFILE，置顶、永不遗忘
意图记忆         MemoryKind.PROMISE，承诺与待办
突触强度         strength，随召回增强、随时间衰减
情绪唤醒          importance，决定衰减半衰期与检索权重
===============  ==========================================
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class MemoryKind(str, Enum):
    """记忆类型。"""

    PROFILE = "profile"      # 身份、称呼、长期特征（自我图式）
    FACT = "fact"            # 客观事实：住哪、做什么工作
    PREFERENCE = "preference"  # 偏好与喜恶
    PROMISE = "promise"      # 承诺、约定、待办
    EVENT = "event"          # 情景记忆：某轮对话里发生的事


#: 越"重要"的类型，初始强度与重要度越高
KIND_PRIOR: dict[MemoryKind, tuple[float, float]] = {
    MemoryKind.PROFILE: (0.95, 1.0),
    MemoryKind.PROMISE: (0.85, 1.0),
    MemoryKind.FACT: (0.7, 0.9),
    MemoryKind.PREFERENCE: (0.65, 0.85),
    MemoryKind.EVENT: (0.3, 0.7),
}


@dataclass
class Memory:
    """一条记忆。

    只有 ``strength`` 是"当前强度"，实际可回忆程度还要乘上时间衰减，
    见 :func:`caim.decay.effective_strength`。
    """

    text: str
    kind: MemoryKind = MemoryKind.FACT
    entity: str = ""
    importance: float = 0.5
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    pinned: bool = False
    turn: int = 0
    tags: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = MemoryKind(self.kind)
        self.importance = _clamp(self.importance)
        self.strength = _clamp(self.strength)

    # -- 便捷构造 -------------------------------------------------
    @classmethod
    def from_kind(cls, text: str, kind: MemoryKind, **kwargs) -> "Memory":
        importance, strength = KIND_PRIOR[kind]
        kwargs.setdefault("importance", importance)
        kwargs.setdefault("strength", strength)
        return cls(text=text, kind=kind, **kwargs)

    # -- 展示 -----------------------------------------------------
    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - self.created_at) / 86400.0)

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "entity": self.entity,
            "text": self.text,
            "importance": self.importance,
            "strength": self.strength,
            "created_at": self.created_at,
            "last_access": self.last_access,
            "access_count": self.access_count,
            "pinned": int(self.pinned),
            "turn": self.turn,
            "tags": ",".join(self.tags),
        }

    @classmethod
    def from_row(cls, row: dict) -> "Memory":
        return cls(
            id=row["id"],
            kind=MemoryKind(row["kind"]),
            entity=row["entity"] or "",
            text=row["text"],
            importance=float(row["importance"]),
            strength=float(row["strength"]),
            created_at=float(row["created_at"]),
            last_access=float(row["last_access"]),
            access_count=int(row["access_count"]),
            pinned=bool(row["pinned"]),
            turn=int(row["turn"]),
            tags=tuple(t for t in (row["tags"] or "").split(",") if t),
        )

    def __str__(self) -> str:  # pragma: no cover - 仅调试用
        flag = "*" if self.pinned else " "
        return f"{flag}[{self.kind.value}] {self.text}"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))
