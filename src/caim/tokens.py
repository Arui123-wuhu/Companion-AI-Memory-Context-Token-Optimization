"""Token 估算与预算工具。

默认使用无依赖的启发式估算（中文按字、英文按 3.5 字符 ≈ 1 token），
在中文对话场景下与 tiktoken / DeepSeek 分词器的误差通常在 ±15% 内。
需要精确值时安装 ``caim[tiktoken]`` 并传 ``use_tiktoken=True``。
"""

from __future__ import annotations

import re
from functools import lru_cache

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_WORD = re.compile(r"[A-Za-z0-9_]+")


@lru_cache(maxsize=4)
def _encoder(model: str):
    import tiktoken  # type: ignore

    return tiktoken.encoding_for_model(model)


def estimate_tokens(text: str, *, use_tiktoken: bool = False, model: str = "gpt-4o-mini") -> int:
    """估算一段文本的 token 数。

    Args:
        text: 待估算文本。
        use_tiktoken: 是否使用 tiktoken 精确计数（需安装可选依赖）。
        model: tiktoken 使用的模型名。
    """
    if not text:
        return 0
    if use_tiktoken:
        try:
            return len(_encoder(model).encode(text))
        except Exception:  # noqa: BLE001 - 缺依赖或模型名不支持时回退
            pass
    cjk = len(_CJK.findall(text))
    words = len(_WORD.findall(text))
    others = max(0, len(text) - cjk - sum(len(w) for w in _WORD.findall(text)))
    # 中文 1 字 ≈ 0.95 token；英文单词 ≈ 1.3 token；标点/空白 ≈ 0.5 token
    return int(round(cjk * 0.95 + words * 1.3 + others * 0.5))


def truncate_to_tokens(text: str, max_tokens: int, *, use_tiktoken: bool = False) -> str:
    """按估算 token 截断文本（优先按句/逗号断开，避免硬切）。"""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text, use_tiktoken=use_tiktoken) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid], use_tiktoken=use_tiktoken) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    head = text[:lo]
    for sep in ("。", "！", "？", "；", "，", ",", " "):
        idx = head.rfind(sep)
        if idx >= lo * 0.6:
            return head[: idx + 1].rstrip()
    return head.rstrip()


def join_with_budget(parts: list[str], budget: int, sep: str = "\n") -> str:
    """在 token 预算内按顺序拼接片段。"""
    out: list[str] = []
    used = 0
    for part in parts:
        cost = estimate_tokens(part) + estimate_tokens(sep)
        if used + cost > budget:
            continue
        out.append(part)
        used += cost
    return sep.join(out)
