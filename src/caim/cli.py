"""命令行入口：``caim demo`` / ``caim bench`` / ``caim stats``。"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .benchmark import format_report, run_benchmark
from .pipeline import MemoryEngine


def _print_header(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def cmd_demo(_args: argparse.Namespace) -> int:
    _print_header("CAIM Demo · 一段跨越 20 天的陪伴对话")
    engine = MemoryEngine(":memory:", budget_tokens=480, consolidate_every=8)
    start = time.time() - 20 * 86400

    script = [
        ("你好呀，我叫阿瑞", "阿瑞你好，很高兴认识你"),
        ("我对花生过敏，千万别给我推荐带花生的", "记住了，以后避开花生"),
        ("我养了一只猫叫豆豆，特别黏人", "豆豆听起来很可爱"),
        ("今天上班好累，开了三个小时的会", "辛苦啦，先休息一下"),
        ("我住在杭州，最近一直下雨", "杭州的雨确实多"),
        ("周末想吃火锅，但是我不吃香菜", "那就点个鸳鸯锅，避开香菜"),
        ("我是后端工程师，最近在写 Python", "写 Python 的话注意休息眼睛"),
        ("记住我喝东西不喝冰的，胃不好", "好，以后都给你点常温的"),
        ("最近睡眠很差，老是失眠", "要不要试试睡前不看手机"),
        ("我妹妹叫小雨，她下个月高考", "希望小雨考出好成绩"),
        ("我喜欢万能青年旅店，现场特别棒", "那首《杀死那个石家庄人》很经典"),
        ("今天路过一家新开的咖啡店", "有空可以去坐坐"),
        ("公司楼下的奶茶店排了好长的队", "看来很受欢迎"),
        ("晚上想把之前那部剧看完", "别熬太晚"),
        ("最近在纠结要不要换工作", "可以先把简历更新一下"),
        ("周末去了趟西湖，人特别多", "节假日确实挤"),
        ("我有点怕打雷，昨天半夜被吓醒", "下次雷雨天我陪你聊会儿"),
        ("在学做饭，昨天煎糊了鸡蛋", "新手都这样，多练几次"),
        ("同事推荐了一家川菜馆，说不辣", "川菜说不辣要小心"),
        ("今天难得早睡，感觉精神好多了", "保持住这个节奏"),
        ("下周要出差三天，可能回消息慢", "路上注意安全"),
        ("回来给你带点当地的特产", "那我可记住了"),
    ]
    for i, (user, reply) in enumerate(script):
        now = start + i * 86400
        new = engine.add_turn(user, reply, now=now)
        print(f"\n[{i + 1:2d}] 用户：{user}")
        for m in new:
            star = "★" if m.pinned else " "
            print(f"          {star}抽取 -> [{m.kind.value}] {m.text}")

    _print_header("巩固一轮（去重 / 压缩情景 / 修剪）")
    report = engine.consolidate(older_than_turns=4)
    print(f"合并重复 {report.merged} 条，压缩情景 {report.summarized} 条，"
          f"修剪 {report.pruned} 条；记忆占用 {report.tokens_before} -> {report.tokens_after} tokens")

    _print_header("现在问：今晚想点外卖，你还记得我有什么忌口吗？")
    result = engine.build_context("今晚想点外卖，你还记得我有什么忌口吗？")
    print(result.text)
    print(f"\n-> {result.summary()}")

    naive = "\n".join(f"用户：{u}\n你：{r}" for u, r in script)
    from .tokens import estimate_tokens

    naive_tokens = estimate_tokens(naive)
    saved = 1 - result.tokens / naive_tokens if naive_tokens else 0
    print(f"-> 同样内容全量塞进上下文需要 {naive_tokens} tokens，"
          f"本次只用 {result.tokens} tokens，节省 {saved:.0%}")

    _print_header("引擎状态")
    print(json.dumps(engine.stats(), ensure_ascii=False, indent=2))
    engine.close()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    _print_header(f"CAIM Benchmark · budget={args.budget}")
    result = run_benchmark(budget=args.budget, total_turns=args.turns, window=args.window)
    print(format_report(result))
    if args.json:
        print("\n" + json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    engine = MemoryEngine(args.db, budget_tokens=args.budget)
    print(json.dumps(engine.stats(), ensure_ascii=False, indent=2))
    engine.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caim",
        description="Companion AI Memory —— 人脑式记忆压缩与抗遗忘引擎",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="跑一段演示对话，看看记忆与上下文长什么样")
    demo.set_defaults(func=cmd_demo)

    bench = sub.add_parser("bench", help="跑合成基准：全量 vs 窗口 vs CAIM")
    bench.add_argument("--budget", type=int, default=1200, help="上下文 token 预算")
    bench.add_argument("--turns", type=int, default=120, help="对话轮数")
    bench.add_argument("--window", type=int, default=10, help="滑动窗口轮数")
    bench.add_argument("--json", action="store_true", help="额外输出 JSON")
    bench.set_defaults(func=cmd_bench)

    stats = sub.add_parser("stats", help="查看某个数据库的记忆统计")
    stats.add_argument("--db", default=":memory:", help="SQLite 路径")
    stats.add_argument("--budget", type=int, default=1200)
    stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
