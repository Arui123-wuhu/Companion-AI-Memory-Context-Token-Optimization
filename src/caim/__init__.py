"""Companion AI Memory —— 借鉴人脑记忆机制的上下文压缩与抗遗忘引擎。"""

from .models import Memory, MemoryKind
from .pipeline import MemoryEngine, BuildResult
from .tokens import estimate_tokens

__all__ = [
    "Memory",
    "MemoryKind",
    "MemoryEngine",
    "BuildResult",
    "estimate_tokens",
]

__version__ = "0.1.0"
