"""Generate the phase-1 synthetic Intent Router dataset.

No ML dependencies -- only the standard library -- so it runs anywhere and is
fast to iterate on. Produces ChatML / conversational JSONL with diverse cases:
normal, colloquial, ambiguous, multi-intent, missing/mixed params, typos,
zh/en mixed, long text and out-of-scope.

Usage:
    python scripts/generate_dataset.py --out data --train 2000 --valid 300 --test 500
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import SYSTEM_PROMPT, make_label  # noqa: E402

DEVICES = [
    "设备A", "设备B", "设备C", "A设备", "B设备", "1号泵", "2号压缩机",
    "P1泵", "风机F3", "reactor R2", "锅炉", "传送带L4", "3号电机", "阀门V7",
    "冷却塔C2", "空压机K1", "搅拌机M5", "输送泵P3", "换热器H8", "离心机X4",
]
TIMES = [
    "最近三个月", "过去一周", "上个月", "今天", "最近", "过去半年", "昨天",
    "这两天", "上周", "近半年",
]


def rec(user, label):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ]
    }


def gen_knowledge_query():
    out = []
    templates = [
        "查询{d}{t}的维修记录",
        "帮我查一下{d}{t}的维护记录",
        "{d}{t}换过哪些零件？",
        "调出{d}的保养历史",
        "{d}上次检修是什么时候？",
        "{d}{t}的报修单都给我看看",
        "{d}的备件更换记录有吗",
        "看看{d}{t}的巡检记录",
    ]
    colloquial = [
        "{d}最近修过几次？",
        "{d}这阵子都保养了啥",
        "{d}以前是不是老出问题",
        "{d}的维修情况咋样",
    ]
    for tpl in templates:
        for d in DEVICES:
            for t in TIMES:
                out.append(rec(tpl.format(d=d, t=t),
                                make_label("knowledge_query", "maintenance", True, True)))
    for tpl in colloquial:
        for d in DEVICES:
            out.append(rec(tpl.format(d=d),
                            make_label("knowledge_query", "maintenance", True, True)))
    # Missing device_id.
    for u in ["查一下维修记录", "维护记录调出来看看", "最近保养情况怎么样",
              "帮我查查维修历史", "看看巡检记录"]:
        out.append(rec(u, make_label("knowledge_query", "maintenance", True, True,
                                     ["device_id"])))
    return out


def gen_diagnosis():
    out = []
    # Templates with a {t} slot -> multiply across TIMES.
    timed = [
        "{d}{t}温度持续升高，可能是什么原因？",
        "{d}{t}振动值超过阈值了",
        "{d}{t}噪音变大，怎么回事",
        "{d}{t}压力异常波动",
        "{d}{t}频繁报警，帮我看看",
        "{d}{t}运行不稳定，是不是有故障",
    ]
    # Templates without time.
    untimed = [
        "{d}好像不太对，运行不稳定",
        "{d}启动后很快就跳闸",
    ]
    colloquial = [
        "{d}最近好像不太对劲",
        "{d}是不是快坏了",
        "感觉{d}越来越吵",
    ]
    for tpl in timed:
        for d in DEVICES:
            for t in TIMES:
                out.append(rec(tpl.format(d=d, t=t),
                                make_label("diagnosis", "equipment", True, True)))
    for tpl in untimed:
        for d in DEVICES:
            out.append(rec(tpl.format(d=d),
                            make_label("diagnosis", "equipment", True, True)))
    for tpl in colloquial:
        for d in DEVICES:
            out.append(rec(tpl.format(d=d),
                            make_label("diagnosis", "equipment", True, True)))
    # Missing device_id.
    for u in ["最近温度一直升高，可能是什么原因？", "设备好像有点问题",
              "振动值超标了怎么办", "总是报警，帮我诊断一下"]:
        out.append(rec(u, make_label("diagnosis", "equipment", True, True,
                                     ["device_id"])))
    return out


def gen_action():
    out = []
    maint_templates = ["把{d}切换到维护模式", "将{d}设为待机", "让{d}进入检修状态"]
    proc_templates = ["重启{d}", "关闭{d}", "启动{d}", "停止{d}运行"]
    for tpl in maint_templates:
        for d in DEVICES:
            out.append(rec(tpl.format(d=d),
                            make_label("action", "maintenance", False, False)))
    for tpl in proc_templates:
        for d in DEVICES:
            out.append(rec(tpl.format(d=d),
                            make_label("action", "process", False, False)))
    # Missing device_id.
    for u in ["切换到维护模式", "重启一下", "关机", "启动设备", "停机"]:
        out.append(rec(u, make_label("action", "process", False, False, ["device_id"])))
    return out


def gen_multi_intent():
    # Primary intent = diagnosis; secondary retrieval implied -> rag/graph true.
    out = []
    templates = [
        "{d}温度升高，顺便把{t}的维修记录也查出来",
        "{d}一直报警，另外看下{t}的保养历史",
    ]
    for tpl in templates:
        for d in DEVICES:
            for t in TIMES:
                out.append(rec(tpl.format(d=d, t=t),
                                make_label("diagnosis", "equipment", True, True)))
    return out


def gen_mixed_lang():
    out = []
    templates = [
        ("check一下{d}的maintenance record", make_label("knowledge_query", "maintenance", True, True)),
        ("{d} temperature 一直升高 what's wrong", make_label("diagnosis", "equipment", True, True)),
        ("restart {d}", make_label("action", "process", False, False)),
    ]
    for tpl, lab in templates:
        for d in DEVICES:
            out.append(rec(tpl.format(d=d), lab))
    return out


def gen_out_of_scope():
    out = []
    users = [
        "今天天气怎么样？", "讲个笑话", "帮我写一首诗", "你是谁？", "1+1等于几",
        "推荐一部电影", "How do I cook pasta?", "明天股票会涨吗", "唱首歌吧",
        "帮我订一张机票", "翻译一下这句英文", "现在几点了", "给我讲讲历史",
        "你会下棋吗", "帮我算个数学题",
    ]
    for u in users:
        out.append(rec(u, make_label("out_of_scope", "unknown", False, False)))
    return out


def add_typos(records, ratio, seed=7):
    """Inject light typos into a fraction of user turns (labels unchanged)."""
    rng = random.Random(seed)
    for r in records:
        if rng.random() < ratio:
            u = r["messages"][1]["content"]
            if len(u) > 3:
                i = rng.randrange(len(u) - 1)
                u = u[:i] + u[i + 1] + u[i] + u[i + 2:]  # swap two adjacent chars
                r["messages"][1]["content"] = u
    return records


def dedupe(records):
    seen, out = set(), []
    for r in records:
        key = r["messages"][1]["content"]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--valid", type=int, default=300)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = []
    records += gen_knowledge_query()
    records += gen_diagnosis()
    records += gen_action()
    records += gen_multi_intent()
    records += gen_mixed_lang()
    records += gen_out_of_scope()

    records = dedupe(records)
    records = add_typos(records, ratio=0.08)

    rng = random.Random(args.seed)
    rng.shuffle(records)

    need = args.train + args.valid + args.test
    if len(records) < need:
        raise SystemExit(f"Only {len(records)} unique samples, need {need}. "
                         "Add more templates in generate_dataset.py.")

    out = Path(args.out)
    train = records[:args.train]
    valid = records[args.train:args.train + args.valid]
    test = records[args.train + args.valid:args.train + args.valid + args.test]

    write_jsonl(out / "raw" / "synthetic.jsonl", records)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "valid.jsonl", valid)
    write_jsonl(out / "test.jsonl", test)

    print(f"total unique: {len(records)}")
    print(f"train={len(train)} valid={len(valid)} test={len(test)}")


if __name__ == "__main__":
    main()
