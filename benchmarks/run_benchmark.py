"""基准脚本：预算扫描 + 巩固消融。

用法::

    python benchmarks/run_benchmark.py            # 默认三档规模
    python benchmarks/run_benchmark.py --sweep    # 预算扫描
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from caim.benchmark import format_report, run_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="CAIM benchmark")
    parser.add_argument("--turns", type=int, default=120)
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--sweep", action="store_true", help="跑预算扫描 + 巩固消融")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.sweep:
        result = run_benchmark(total_turns=args.turns, budget=args.budget, window=args.window)
        print(format_report(result))
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("## 预算扫描（400 轮对话）\n")
    print("| 预算 | CAIM tokens | 事实保留 | 全量 tokens |")
    print("| ---: | ---: | ---: | ---: |")
    for budget in (200, 300, 600, 1200, 2400):
        r = run_benchmark(total_turns=400, budget=budget)
        st = r["strategies"]
        print(f"| {budget} | {st['caim']['avg_tokens']} | "
              f"{st['caim']['hits']}/{r['config']['probes']} | "
              f"{st['full_history']['avg_tokens']:,} |")

    print("\n## 巩固消融（400 轮，budget=800）\n")
    print("| 巩固 | 记忆条数 | 存储 tokens | 上下文 tokens | 事实保留 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for flag in (True, False):
        r = run_benchmark(total_turns=400, budget=800, consolidate=flag)
        st = r["strategies"]["caim"]
        ms = r["memory_stats"]
        print(f"| {'开' if flag else '关'} | {ms['memories']} | {ms['stored_tokens']} | "
              f"{st['avg_tokens']} | {st['hits']}/{r['config']['probes']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
