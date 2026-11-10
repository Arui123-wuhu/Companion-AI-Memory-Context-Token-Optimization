"""可复现的合成基准：全量历史 vs 滑动窗口 vs CAIM。

设计原则：
- 对话内容全部由代码生成，任何人跑出来的数字一致；
- 埋点事实（fact）在对话中间随机分布，用于检验"是否失忆"；
- 每个探针问题单独装配一次上下文，按关键词判断能否答出；
- token 用同一套估算器，三种策略口径一致。
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from .context import DEFAULT_RULES
from .pipeline import MemoryEngine
from .tokens import estimate_tokens

#: 埋点事实：(轮次, 用户台词, 判定关键词, 探针问题)
FACTS: tuple[tuple[int, str, tuple[str, ...], str], ...] = (
    (3, "我叫阿瑞，你以后就这么叫我", ("阿瑞",), "我叫什么名字？"),
    (11, "我的生日是3月14日，别忘了", ("3月14",), "我的生日是哪天？"),
    (19, "我对花生过敏，很严重", ("花生",), "我有什么食物过敏？"),
    (27, "我养了一只猫叫豆豆", ("豆豆",), "我的猫叫什么？"),
    (35, "我住在杭州，已经五年了", ("杭州",), "我住在哪个城市？"),
    (44, "我是后端工程师，主要写 Python", ("工程师",), "我是做什么工作的？"),
    (53, "我很怕打雷，雷雨天会睡不着", ("打雷",), "我害怕什么天气？"),
    (61, "我喜欢万能青年旅店这支乐队", ("万能青年旅店",), "我喜欢哪支乐队？"),
    (70, "我妹妹叫小雨，在读大学", ("小雨",), "我妹妹叫什么？"),
    (79, "我讨厌香菜，闻到就难受", ("香菜",), "我讨厌吃什么？"),
    (88, "记住我不喝冰的，胃不好", ("冰",), "我喝东西有什么忌口？"),
    (97, "我答应周五提醒你交房租", ("房租",), "我答应过你什么？"),
)

#: 填充对话：制造噪声，检验检索能否顶住干扰
FILLERS: tuple[str, ...] = (
    "今天工作好累啊",
    "外面在下雨，不想出门",
    "晚饭吃了牛肉面，还行",
    "周末想去爬山，你有推荐吗",
    "刚看完一部电影，结局有点意外",
    "地铁上人太多了",
    "最近睡眠不太好",
    "帮我看看这个方案怎么样",
    "今天开会开了三个小时",
    "楼下新开了一家咖啡店",
    "有点想吃火锅",
    "电脑又卡了，想换一台",
    "今天走了两万步",
    "天气转凉了，记得加衣服",
    "刚把房间收拾了一遍",
    "同事离职了，有点感慨",
    "想学点新东西，但没想好学什么",
    "今天效率特别低",
    "晚上准备早点睡",
    "路上看到一只橘猫",
)

REPLIES: tuple[str, ...] = (
    "听起来挺不容易的，先歇会儿吧",
    "那确实挺烦的，要不要换个环境试试",
    "嗯嗯，我在听",
    "这个我记下了",
    "慢慢来，不着急",
)


@dataclass
class Probe:
    question: str
    keywords: tuple[str, ...]


def build_transcript(total_turns: int = 120) -> tuple[list[tuple[int, str, str]], list[Probe]]:
    """生成 (turn, user, reply) 列表与探针列表。"""
    fact_map = {t: (u, kw) for t, u, kw, _ in FACTS}
    probes = [Probe(q, kw) for _, _, kw, q in FACTS]

    turns: list[tuple[int, str, str]] = []
    for i in range(1, total_turns + 1):
        if i in fact_map:
            user = fact_map[i][0]
        else:
            user = FILLERS[(i * 7) % len(FILLERS)]
        reply = REPLIES[i % len(REPLIES)]
        turns.append((i, user, reply))
    return turns, probes


def _join_turns(turns: list[tuple[int, str, str]], limit: int | None = None) -> str:
    selected = turns[-limit:] if limit else turns
    return "\n".join(f"用户：{u}\n你：{r}" for _, u, r in selected)


def _retention(text: str, probes: list[Probe]) -> tuple[int, list[str]]:
    hits, missed = 0, []
    for probe in probes:
        if any(kw in text for kw in probe.keywords):
            hits += 1
        else:
            missed.append(probe.question)
    return hits, missed


def run_benchmark(
    *,
    total_turns: int = 120,
    budget: int = 1200,
    window: int = 10,
    hours_per_turn: float = 3.0,
    consolidate: bool = True,
) -> dict:
    """跑三种策略，返回可序列化的结果。"""
    turns, probes = build_transcript(total_turns)
    start = time.time() - total_turns * hours_per_turn * 3600

    rules_tokens = estimate_tokens(DEFAULT_RULES)

    # --- 基线 1：全量历史
    full_text = DEFAULT_RULES + "\n\n" + _join_turns(turns)
    full_hits, full_miss = _retention(full_text, probes)

    # --- 基线 2：滑动窗口
    window_text = DEFAULT_RULES + "\n\n" + _join_turns(turns, window)
    window_hits, window_miss = _retention(window_text, probes)

    # --- CAIM
    engine = MemoryEngine(
        ":memory:",
        budget_tokens=budget,
        consolidate_every=20 if consolidate else 0,
    )
    for i, user, reply in turns:
        engine.add_turn(
            user, reply,
            now=start + i * hours_per_turn * 3600,
            auto_consolidate=consolidate,
        )

    caim_tokens: list[int] = []
    caim_hits, caim_miss = 0, []
    for probe in probes:
        result = engine.build_context(
            probe.question,
            now=start + (total_turns + 1) * hours_per_turn * 3600,
            reinforce=False,
        )
        caim_tokens.append(result.tokens)
        if any(kw in result.text for kw in probe.keywords):
            caim_hits += 1
        else:
            caim_miss.append(probe.question)

    stats = engine.stats()
    engine.close()

    return {
        "config": {
            "turns": total_turns,
            "probes": len(probes),
            "budget_tokens": budget,
            "window_turns": window,
        },
        "strategies": {
            "full_history": {
                "avg_tokens": estimate_tokens(full_text),
                "retention": full_hits / len(probes),
                "hits": full_hits,
                "missed": full_miss,
            },
            "sliding_window": {
                "avg_tokens": estimate_tokens(window_text),
                "retention": window_hits / len(probes),
                "hits": window_hits,
                "missed": window_miss,
            },
            "caim": {
                "avg_tokens": int(statistics.mean(caim_tokens)),
                "max_tokens": max(caim_tokens),
                "retention": caim_hits / len(probes),
                "hits": caim_hits,
                "missed": caim_miss,
            },
        },
        "memory_stats": stats,
    }


def format_report(result: dict) -> str:
    """把结果渲染成 Markdown 表格。"""
    cfg = result["config"]
    st = result["strategies"]
    full_tokens = st["full_history"]["avg_tokens"]
    lines = [
        f"对话 {cfg['turns']} 轮 · 埋点 {cfg['probes']} 条 · 预算 {cfg['budget_tokens']} tokens",
        "",
        "| 策略 | 平均上下文 tokens | 相对全量节省 | 关键事实保留 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, label in (
        ("full_history", "全量历史"),
        ("sliding_window", f"滑动窗口（最近 {cfg['window_turns']} 轮）"),
        ("caim", "CAIM"),
    ):
        item = st[name]
        saving = 1 - item["avg_tokens"] / full_tokens if full_tokens else 0
        lines.append(
            f"| {label} | {item['avg_tokens']:,} | {saving:.0%} | "
            f"{item['hits']}/{cfg['probes']}（{item['retention']:.0%}） |"
        )
    return "\n".join(lines)
