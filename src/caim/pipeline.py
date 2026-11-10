"""门面：把抽取、存储、检索、装配、巩固串成一个引擎。"""

from __future__ import annotations

import time

from .consolidate import ConsolidationReport, consolidate as run_consolidate
from .context import DEFAULT_RULES, BuildResult, ContextBuilder
from .extract import Extractor, RuleExtractor
from .models import Memory, MemoryKind
from .store import MemoryStore
from .tokens import estimate_tokens


class MemoryEngine:
    """陪伴型 AI 的记忆引擎。

    典型用法::

        engine = MemoryEngine("companion.db", budget_tokens=1200)
        engine.add_turn("我对花生过敏，别给我点带花生的", "记住了，以后避开花生")
        ctx = engine.build_context("晚上吃什么好？")
        llm.chat(system=ctx.text, user="晚上吃什么好？")
    """

    def __init__(
        self,
        store_path: str = ":memory:",
        *,
        budget_tokens: int = 1200,
        extractor: Extractor | None = None,
        rules: str = DEFAULT_RULES,
        use_tiktoken: bool = False,
        consolidate_every: int = 20,
    ) -> None:
        self.store = MemoryStore(store_path)
        self.extractor: Extractor = extractor or RuleExtractor()
        self.budget_tokens = budget_tokens
        self.builder = ContextBuilder(
            self.store, budget_tokens=budget_tokens, rules=rules, use_tiktoken=use_tiktoken
        )
        self.use_tiktoken = use_tiktoken
        self.consolidate_every = consolidate_every
        self._last_build: BuildResult | None = None

    # -- 写入 -----------------------------------------------------
    @property
    def turn(self) -> int:
        return int(self.store.get_meta("turn", 0))

    def add_turn(
        self,
        user_text: str,
        reply_text: str = "",
        *,
        now: float | None = None,
        auto_consolidate: bool = True,
    ) -> list[Memory]:
        """记录一轮对话，并从中抽取记忆。返回本轮新增的记忆。"""
        now = time.time() if now is None else now
        turn = self.turn + 1
        self.store.set_meta("turn", turn)
        self.store.add_turn(turn, user_text, reply_text, now=now)

        memories = self.extractor.extract(user_text, turn=turn, now=now)
        if reply_text:
            memories += self._extract_reply_commitments(reply_text, turn, now)
        for memory in memories:
            self.store.add(memory)

        if auto_consolidate and self.consolidate_every and turn % self.consolidate_every == 0:
            self.consolidate(now=now)
        return memories

    def remember(self, text: str, *, kind: MemoryKind = MemoryKind.FACT,
                 importance: float = 0.9, pinned: bool = False,
                 entity: str = "", now: float | None = None) -> Memory:
        """手动写入一条记忆（比如从外部知识库导入）。"""
        memory = Memory.from_kind(
            text, kind, importance=importance, pinned=pinned,
            entity=entity, created_at=time.time() if now is None else now,
        )
        return self.store.add(memory)

    def _extract_reply_commitments(self, reply: str, turn: int, now: float) -> list[Memory]:
        """只从 AI 回复里抓"承诺"，避免把客套话也存成事实。"""
        out: list[Memory] = []
        for marker in ("我答应", "我会提醒", "我记住了", "我保证"):
            idx = reply.find(marker)
            if idx >= 0:
                snippet = reply[idx : idx + 40].strip()
                out.append(Memory.from_kind(
                    f"AI 承诺：{snippet}", MemoryKind.PROMISE,
                    importance=0.85, turn=turn, created_at=now,
                ))
        return out

    # -- 读取 -----------------------------------------------------
    def build_context(
        self,
        query: str = "",
        *,
        budget_tokens: int | None = None,
        now: float | None = None,
        recent_turns: int = 4,
        reinforce: bool = True,
    ) -> BuildResult:
        """按预算装配上下文。"""
        builder = self.builder
        if budget_tokens and budget_tokens != builder.budget:
            builder = ContextBuilder(
                self.store,
                budget_tokens=budget_tokens,
                rules=builder.rules,
                use_tiktoken=self.use_tiktoken,
            )
        # ContextBuilder 内部调用 retrieve，这里用开关控制是否强化
        result = builder.build(
            query, now=now, recent_turns=recent_turns, reinforce=reinforce
        )
        self._last_build = result
        return result

    def retrieve(self, query: str, *, top_k: int = 8, now: float | None = None):
        from .retrieval import retrieve

        return retrieve(self.store, query, top_k=top_k, now=now)

    def stats(self) -> dict:
        memories = self.store.all()
        by_kind: dict[str, int] = {}
        for memory in memories:
            by_kind[memory.kind.value] = by_kind.get(memory.kind.value, 0) + 1
        stored_tokens = sum(estimate_tokens(m.text) for m in memories)
        return {
            "turns": self.turn,
            "memories": len(memories),
            "by_kind": by_kind,
            "pinned": sum(1 for m in memories if m.pinned),
            "stored_tokens": stored_tokens,
            "last_build": self._last_build.summary() if self._last_build else None,
        }

    def consolidate(self, *, now: float | None = None, older_than_turns: int = 24) -> ConsolidationReport:
        return run_consolidate(self.store, now=now, older_than_turns=older_than_turns)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "MemoryEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
