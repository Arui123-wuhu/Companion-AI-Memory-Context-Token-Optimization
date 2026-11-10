"""SQLite 持久化层。

刻意只用标准库：不引入向量数据库、不依赖网络。
记忆数量在陪伴型场景下通常是 10^2 ~ 10^4 量级，线性扫描完全够用，
换来的是可解释、可调试、可离线。
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable

from .models import Memory, MemoryKind

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    entity      TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL,
    importance  REAL NOT NULL,
    strength    REAL NOT NULL,
    created_at  REAL NOT NULL,
    last_access REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    pinned      INTEGER NOT NULL DEFAULT 0,
    turn        INTEGER NOT NULL DEFAULT 0,
    tags        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity);
CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories(pinned);

CREATE TABLE IF NOT EXISTS turns (
    turn        INTEGER PRIMARY KEY,
    user_text   TEXT NOT NULL,
    reply_text  TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class MemoryStore:
    """记忆仓库（SQLite）。"""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- 记忆 -----------------------------------------------------
    def add(self, memory: Memory) -> Memory:
        self.conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, kind, entity, text, importance, strength, created_at,
                last_access, access_count, pinned, turn, tags)
               VALUES (:id, :kind, :entity, :text, :importance, :strength,
                       :created_at, :last_access, :access_count, :pinned, :turn, :tags)""",
            memory.to_row(),
        )
        self.conn.commit()
        return memory

    def add_many(self, memories: Iterable[Memory]) -> list[Memory]:
        items = list(memories)
        for memory in items:
            self.add(memory)
        return items

    def update(self, memory: Memory) -> Memory:
        return self.add(memory)

    def get(self, memory_id: str) -> Memory | None:
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return Memory.from_row(dict(row)) if row else None

    def all(self) -> list[Memory]:
        rows = self.conn.execute("SELECT * FROM memories").fetchall()
        return [Memory.from_row(dict(r)) for r in rows]

    def pinned(self) -> list[Memory]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE pinned = 1 ORDER BY importance DESC, created_at ASC"
        ).fetchall()
        return [Memory.from_row(dict(r)) for r in rows]

    def by_kind(self, kind: MemoryKind) -> list[Memory]:
        rows = self.conn.execute("SELECT * FROM memories WHERE kind = ?", (kind.value,)).fetchall()
        return [Memory.from_row(dict(r)) for r in rows]

    def delete(self, memory_id: str) -> None:
        self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    # -- 对话轮次（工作记忆的持久化备份） --------------------------
    def add_turn(self, turn: int, user_text: str, reply_text: str, now: float | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO turns (turn, user_text, reply_text, created_at) VALUES (?,?,?,?)",
            (turn, user_text, reply_text, time.time() if now is None else now),
        )
        self.conn.commit()

    def recent_turns(self, limit: int = 6) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM turns ORDER BY turn DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # -- meta -----------------------------------------------------
    def set_meta(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_meta(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
