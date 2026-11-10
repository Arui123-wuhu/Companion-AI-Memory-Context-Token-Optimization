"""最小可跑的"带记忆的聊天"示例。

不调用真实 LLM：把装配好的上下文打印出来，用一句固定回复占位，
方便你观察 token 预算和记忆命中的变化。

    python examples/chat_with_memory.py
    python examples/chat_with_memory.py --db me.db --budget 800
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from caim import MemoryEngine  # noqa: E402


def fake_llm(context: str, user: str) -> str:
    """占位回复：真实项目里换成你的模型调用。"""
    return f"（已收到 {len(context)} 字的上下文）我记住啦。"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="companion.db", help="记忆库路径，默认 companion.db")
    parser.add_argument("--budget", type=int, default=800, help="上下文 token 预算")
    args = parser.parse_args()

    engine = MemoryEngine(args.db, budget_tokens=args.budget)
    print(f"记忆库：{args.db}｜预算：{args.budget} tokens｜输入 /quit 退出")

    while True:
        try:
            user = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/quit", "/exit"}:
            break

        ctx = engine.build_context(user)
        print(f"--- 上下文 {ctx.summary()} ---")
        print(ctx.text)

        reply = fake_llm(ctx.text, user)
        engine.add_turn(user, reply)
        print(f"\nAI > {reply}")

    print("\n=== 记忆统计 ===")
    for key, value in engine.stats().items():
        print(f"{key}: {value}")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
