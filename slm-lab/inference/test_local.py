"""Quick local smoke test of a GGUF Intent Router model via llama.cpp.

Requires: pip install llama-cpp-python

Usage:
    python inference/test_local.py --gguf models/final/model-q4_k_m.gguf
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from schema import SYSTEM_PROMPT  # noqa: E402

SAMPLES = [
    "设备A温度持续升高，同时振动值超过阈值。",
    "查询设备A过去三个月的维修记录。",
    "把设备B切换到维护模式。",
    "查一下维修记录",
    "今天天气怎么样？",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--n_ctx", type=int, default=2048)
    args = ap.parse_args()

    from llama_cpp import Llama

    llm = Llama(model_path=args.gguf, n_ctx=args.n_ctx, verbose=False)

    for user in SAMPLES:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=64,
            temperature=0.0,
        )
        text = out["choices"][0]["message"]["content"]
        try:
            parsed = json.dumps(json.loads(text), ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            parsed = f"[UNPARSEABLE] {text}"
        print(f"IN : {user}")
        print(f"OUT: {parsed}\n")


if __name__ == "__main__":
    main()
