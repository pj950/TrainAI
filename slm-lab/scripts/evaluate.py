"""Evaluate an Intent Router model on data/test.jsonl.

Builds the Evaluation Matrix: JSON valid rate, intent / domain accuracy,
need_rag / need_graph accuracy, missing_parameters accuracy, and latency.
Writes evaluation/results.json and evaluation/error_cases.jsonl.

Examples:
    # base model (no training)
    python scripts/evaluate.py --model Qwen/Qwen3-0.6B --tag baseline
    # LoRA adapter on top of base
    python scripts/evaluate.py --model Qwen/Qwen3-0.6B --adapter models/lora --tag sft
    # merged / distilled model
    python scripts/evaluate.py --model models/final --tag distill
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema import REQUIRED_KEYS


def load_jsonl(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_json(text):
    """Extract the first JSON object from raw model output."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--test_file", default="data/test.jsonl")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out", default="evaluation")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    samples = load_jsonl(args.test_file)
    n = len(samples)
    stats = {k: 0 for k in ["json_valid", "keys_ok", "intent", "domain",
                            "need_rag", "need_graph", "missing"]}
    latencies = []
    errors = []

    for s in samples:
        msgs = s["messages"]
        gold = json.loads(msgs[2]["content"])
        prompt_msgs = msgs[:2]  # system + user

        inputs = tokenizer.apply_chat_template(
            prompt_msgs, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latencies.append((time.perf_counter() - t0) * 1000)

        gen = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        pred = parse_json(gen)

        ok = pred is not None
        if ok:
            stats["json_valid"] += 1
            if all(k in pred for k in REQUIRED_KEYS):
                stats["keys_ok"] += 1
            if pred.get("intent") == gold["intent"]:
                stats["intent"] += 1
            if pred.get("domain") == gold["domain"]:
                stats["domain"] += 1
            if pred.get("need_rag") == gold["need_rag"]:
                stats["need_rag"] += 1
            if pred.get("need_graph") == gold["need_graph"]:
                stats["need_graph"] += 1
            if sorted(pred.get("missing_parameters", [])) == sorted(gold["missing_parameters"]):
                stats["missing"] += 1

        if not ok or pred.get("intent") != gold["intent"]:
            errors.append({"input": msgs[1]["content"], "gold": gold,
                           "raw_output": gen})

    latencies.sort()
    results = {
        "tag": args.tag,
        "model": args.model,
        "adapter": args.adapter,
        "n": n,
        "json_valid_rate": round(stats["json_valid"] / n, 4),
        "keys_ok_rate": round(stats["keys_ok"] / n, 4),
        "intent_accuracy": round(stats["intent"] / n, 4),
        "domain_accuracy": round(stats["domain"] / n, 4),
        "need_rag_accuracy": round(stats["need_rag"] / n, 4),
        "need_graph_accuracy": round(stats["need_graph"] / n, 4),
        "missing_param_accuracy": round(stats["missing"] / n, 4),
        "latency_ms_mean": round(sum(latencies) / n, 1),
        "latency_ms_p95": round(latencies[int(n * 0.95) - 1], 1),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    all_results = {}
    if results_path.exists():
        all_results = json.loads(results_path.read_text(encoding="utf-8"))
    all_results[args.tag] = results
    results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    with (out_dir / "error_cases.jsonl").open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"errors: {len(errors)} -> {out_dir / 'error_cases.jsonl'}")


if __name__ == "__main__":
    main()
