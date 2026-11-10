"""从对话中抽取结构化记忆。

默认实现是**纯规则**的：零依赖、可离线、可解释，中文陪伴场景下召回已知信息够用。
需要更强抽取（隐喻、长句、隐含偏好）时，实现 :class:`Extractor` 协议接入 LLM 即可，
其余模块（衰减/检索/巩固/装配）完全不用改。
"""

from __future__ import annotations

import re
import time
from typing import Protocol, runtime_checkable

from .models import Memory, MemoryKind

_PUNCT = " \t\r\n，。！？；：、,.!?;:~…\"'“”‘’()（）[]【】"


def _clean(text: str, limit: int = 40) -> str:
    text = re.sub(r"\s+", " ", text).strip(_PUNCT)
    return text[:limit]


@runtime_checkable
class Extractor(Protocol):
    """抽取器协议：给一句话，产出若干条记忆。"""

    def extract(self, text: str, turn: int = 0, now: float | None = None) -> list[Memory]:
        ...


class RuleExtractor:
    """基于规则的中文记忆抽取器。"""

    # (正则, 类型, 文本模板, 实体模板, 重要度)
    # 模板里的 {0} {1} 对应正则的捕获组
    _RULES: tuple[tuple[str, MemoryKind, str, str, float], ...] = (
        (r"我(?:的名字)?叫\s*([^\s，。,.!?！？]{1,10})", MemoryKind.PROFILE, "用户的名字是{0}", "", 0.98),
        (r"叫我\s*([^\s，。,.!?！？]{1,10})", MemoryKind.PROFILE, "用户希望被叫作{0}", "", 0.98),
        (r"我的?生日(?:是|在)?\s*([0-9一二三四五六七八九十]{1,4}\s*月\s*[0-9一二三四五六七八九十]{1,3}\s*[日号])",
         MemoryKind.FACT, "用户的生日是{0}", "", 0.9),
        (r"我对\s*([^\s，。,.!?！？]{1,12})\s*过敏", MemoryKind.FACT, "用户对{0}过敏", "{0}", 0.95),
        (r"我(?:住在|家在|定居在)\s*([^\s，。,.!?！？]{1,10})", MemoryKind.FACT, "用户住在{0}", "", 0.75),
        (r"我(?:是|在做|干的?是)\s*(?:一名|一个|个)?\s*([^\s，。,.!?！？]{1,14}"
         r"(?:工程师|程序员|设计师|产品经理|老师|医生|护士|律师|会计|运营|学生|研究生))",
         MemoryKind.FACT, "用户的职业是{0}", "", 0.8),
        (r"我(?:养了|有一只|有个)\s*(?:一只|一个)?\s*(猫|狗|仓鼠|兔子|鸟)",
         MemoryKind.FACT, "用户养了{0}", "{0}", 0.8),
        (r"(?:它|他|她|猫|狗|仓鼠|兔子|鸟)(?:的名字)?叫\s*([^\s，。,.!?！？]{1,8})",
         MemoryKind.FACT, "宠物名叫{0}", "{0}", 0.75),
        (r"我(?:有点|有些|很|挺|特别|超)?(?:怕|害怕|恐惧)\s*([^\s，。,.!?！？]{1,12})",
         MemoryKind.PREFERENCE, "用户害怕{0}", "{0}", 0.8),
        (r"我(?:很)?(?:不吃|不能吃|忌口)\s*([^\s，。,.!?！？]{1,12})",
         MemoryKind.PREFERENCE, "用户不吃{0}", "{0}", 0.85),
        (r"我(?:很)?(?:喜欢|爱|迷上了|在追)\s*([^\s，。,.!?！？]{1,16})",
         MemoryKind.PREFERENCE, "用户喜欢{0}", "{0}", 0.7),
        (r"我(?:不喜欢|讨厌|受不了|反感)\s*([^\s，。,.!?！？]{1,16})",
         MemoryKind.PREFERENCE, "用户不喜欢{0}", "{0}", 0.75),
        (r"我(妹妹|姐姐|哥哥|弟弟|妈妈|爸爸|女朋友|男朋友|老婆|老公)(?:叫|是)\s*([^\s，。,.!?！？]{1,10})",
         MemoryKind.FACT, "用户的{0}叫{1}", "{1}", 0.8),
        (r"(?:答应|约好|说好)(?:了)?\s*([^。！？!?]{1,30})",
         MemoryKind.PROMISE, "用户约定：{0}", "", 0.9),
        (r"(?:记得|提醒我|别忘(?:了)?)\s*[，,：:]?\s*([^。！？!?]{1,40})",
         MemoryKind.PROMISE, "待办：{0}", "", 0.95),
    )

    #: 「记住……」是显式指令，单独处理：转为置顶事实，永不遗忘
    _PIN = re.compile(r"记住[，,：:]?\s*([^。！？!?]{1,40})")

    def __init__(self, keep_events: bool = True) -> None:
        self.keep_events = keep_events
        self._compiled = [(re.compile(p), k, t, e, i) for p, k, t, e, i in self._RULES]

    def extract(self, text: str, turn: int = 0, now: float | None = None) -> list[Memory]:
        now = time.time() if now is None else now
        text = text.strip()
        if not text:
            return []

        out: list[Memory] = []
        seen: set[str] = set()

        pin = self._PIN.search(text)
        if pin:
            value = _clean(pin.group(1), 40)
            if value:
                out.append(Memory(
                    text=f"用户要求记住：{value}",
                    kind=MemoryKind.FACT,
                    importance=1.0,
                    pinned=True,
                    turn=turn,
                    created_at=now,
                ))
                seen.add(out[-1].text)

        for pattern, kind, template, entity_tmpl, importance in self._compiled:
            for match in pattern.finditer(text):
                values = [_clean(g, 40) for g in match.groups() if g]
                if not values:
                    continue
                memory = Memory.from_kind(
                    template.format(*values), kind,
                    entity=_clean(entity_tmpl.format(*values), 20) if entity_tmpl else "",
                    importance=importance, turn=turn, created_at=now,
                )
                if memory.text not in seen:
                    seen.add(memory.text)
                    out.append(memory)

        if self.keep_events and not out:
            gist = _clean(text, 60)
            if len(gist) >= 2:
                out.append(Memory.from_kind(
                    f"用户说：{gist}", MemoryKind.EVENT, turn=turn, created_at=now,
                ))
        return out


def memory_from_llm_json(payload: dict, turn: int = 0, now: float | None = None) -> Memory:
    """把 LLM 抽取结果（JSON）转成 Memory，供自定义 Extractor 复用。"""
    kind = MemoryKind(payload.get("kind", "fact"))
    text = _clean(str(payload.get("text", "")), 120)
    return Memory.from_kind(
        text,
        kind,
        entity=_clean(str(payload.get("entity", "")), 20),
        importance=float(payload.get("importance", 0.7)),
        turn=turn,
        created_at=time.time() if now is None else now,
        tags=tuple(payload.get("tags", ())),
    )
