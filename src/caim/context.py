"""上下文装配：把"记忆"压进有限的 token 预算里。

对应人脑的**工作记忆**：容量极其有限（7±2 组块），所以必须
"只把当前任务需要的那一小部分记忆调入前额叶"，其余留在长时记忆里。

装配优先级（也是预算分配）：

1. 系统约束（固定，很小）
2. 身份/核心档案（置顶记忆，永不被挤掉）
3. 与当前话题最相关的长期记忆（检索 + 去冗余）
4. 最近几轮原文（保证对话连贯，近因效应）

每一段都有独立的预算上限，任何一段都不会把别的段饿死。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .retrieval import retrieve
from .store import MemoryStore
from .tokens import estimate_tokens, truncate_to_tokens

DEFAULT_RULES = (
    "以下是你与用户相处中积累的长期记忆。回答时优先使用这些记忆；"
    "记忆里没有的信息就直说不知道，不要编造。"
)

#: 预算切分
PINNED_RATIO = 0.28
LONGTERM_RATIO = 0.50


@dataclass
class BuildResult:
    """一次上下文装配的结果，带可观测指标。"""

    text: str
    tokens: int
    budget: int
    sections: dict[str, int] = field(default_factory=dict)
    memory_ids: list[str] = field(default_factory=list)
    recent_turns_used: int = 0
    dropped: int = 0

    @property
    def utilization(self) -> float:
        return self.tokens / self.budget if self.budget else 0.0

    def summary(self) -> str:
        sec = "，".join(f"{k} {v}t" for k, v in self.sections.items())
        return f"{self.tokens}/{self.budget} tokens（{sec}），记忆 {len(self.memory_ids)} 条"


class ContextBuilder:
    """把记忆装配成可直接塞进 LLM 的上下文字符串。"""

    def __init__(
        self,
        store: MemoryStore,
        *,
        budget_tokens: int = 1200,
        rules: str = DEFAULT_RULES,
        use_tiktoken: bool = False,
    ) -> None:
        self.store = store
        self.budget = max(128, int(budget_tokens))
        self.rules = rules
        self.use_tiktoken = use_tiktoken

    # -- 内部工具 -------------------------------------------------
    def _tokens(self, text: str) -> int:
        return estimate_tokens(text, use_tiktoken=self.use_tiktoken)

    def _clip(self, text: str, limit: int) -> str:
        return truncate_to_tokens(text, limit, use_tiktoken=self.use_tiktoken)

    def _fit(self, lines: list[str], max_tokens: int) -> tuple[list[str], int, int]:
        """在预算内尽量多地装入行，返回 (选中行, 消耗 tokens, 丢弃数)。"""
        out: list[str] = []
        used = 0
        for line in lines:
            cost = self._tokens(line) + 1
            if used + cost > max_tokens:
                continue
            out.append(line)
            used += cost
        return out, used, len(lines) - len(out)

    # -- 主流程 ---------------------------------------------------
    def build(
        self,
        query: str = "",
        *,
        now: float | None = None,
        recent_turns: int = 4,
        top_k: int = 14,
        reinforce: bool = True,
    ) -> BuildResult:
        now = time.time() if now is None else now
        rules_tokens = self._tokens(self.rules)
        remaining = max(0, self.budget - rules_tokens)

        sections: dict[str, int] = {"rules": rules_tokens}
        memory_ids: list[str] = []
        dropped = 0
        blocks: list[str] = []

        # 1) 身份 / 置顶记忆
        pinned = [m for m in self.store.all() if m.pinned or m.kind.value == "profile"]
        pinned.sort(key=lambda m: (not m.pinned, -m.importance, m.created_at))
        pinned_lines = [f"- {m.text}" for m in pinned]
        pinned_budget = int(remaining * PINNED_RATIO)
        chosen_pinned, used_pinned, drop_pinned = self._fit(pinned_lines, pinned_budget)
        if not chosen_pinned and pinned_lines and remaining > 0:
            # 预算再紧也至少留一条身份信息
            chosen_pinned = [self._clip(pinned_lines[0], max(8, remaining // 2))]
            used_pinned = self._tokens(chosen_pinned[0]) + 1
            drop_pinned = len(pinned_lines) - 1
        dropped += drop_pinned
        memory_ids.extend(
            m.id for m, line in zip(pinned, pinned_lines) if line in chosen_pinned
        )
        if chosen_pinned:
            blocks.append("【核心档案】\n" + "\n".join(chosen_pinned))
            sections["profile"] = used_pinned

        # 2) 长期记忆（检索 + 去冗余）
        hits = retrieve(self.store, query, top_k=top_k, now=now, reinforce_hits=reinforce)
        used_ids = set(memory_ids)
        long_lines: list[str] = []
        long_ids: list[str] = []
        for hit in hits:
            if hit.memory.id in used_ids:
                continue
            long_lines.append(f"- {hit.memory.text}")
            long_ids.append(hit.memory.id)
        long_budget = int(remaining * LONGTERM_RATIO)
        chosen_long, used_long, drop_long = self._fit(long_lines, long_budget)
        dropped += drop_long
        if chosen_long:
            blocks.append("【长期记忆】\n" + "\n".join(chosen_long))
            sections["longterm"] = used_long
            for line, mid in zip(long_lines, long_ids):
                if line in chosen_long:
                    memory_ids.append(mid)

        # 3) 最近对话（原文，保证连贯）
        spent = sum(sections.values())
        left = max(0, remaining - (spent - rules_tokens))
        recent_block, used_recent, turns_used = self._recent_block(left, recent_turns)
        if recent_block:
            blocks.append("【最近对话】\n" + recent_block)
            sections["recent"] = used_recent

        text = self.rules + "\n\n" + "\n\n".join(blocks) if blocks else self.rules
        tokens = self._tokens(text)
        # 极小概率估算偏差导致超预算：硬截断
        if tokens > self.budget:
            text = self._clip(text, self.budget)
            tokens = self._tokens(text)

        return BuildResult(
            text=text,
            tokens=tokens,
            budget=self.budget,
            sections=sections,
            memory_ids=memory_ids,
            recent_turns_used=turns_used,
            dropped=dropped,
        )

    def _recent_block(self, max_tokens: int, recent_turns: int) -> tuple[str, int, int]:
        if max_tokens <= 0:
            return "", 0, 0
        turns = self.store.recent_turns(recent_turns)
        lines: list[str] = []
        for t in turns:
            lines.append(f"用户：{t['user_text']}")
            lines.append(f"你：{t['reply_text']}")
        # 从最新往回装，保证最近的对话一定在
        chosen: list[str] = []
        used = 0
        for line in reversed(lines):
            cost = self._tokens(line) + 1
            if used + cost > max_tokens:
                if not chosen:
                    clipped = self._clip(line, max(6, max_tokens - 1))
                    chosen.append(clipped)
                    used = self._tokens(clipped) + 1
                break
            chosen.append(line)
            used += cost
        chosen.reverse()
        return "\n".join(chosen), used, len(turns)
