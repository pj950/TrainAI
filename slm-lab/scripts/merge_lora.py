"""Merge a LoRA adapter into the base model and save FP16 weights.

GGUF conversion needs a full (merged) HF model, not an adapter.

Usage:
    python scripts/merge_lora.py --base Qwen/Qwen3-0.6B \
        --adapter models/lora --out models/final
"""

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="models/final")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype="float16")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
