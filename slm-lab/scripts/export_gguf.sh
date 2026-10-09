#!/usr/bin/env bash
# Convert a merged HF model to GGUF and quantize with llama.cpp.
#
# Prerequisite: merge the adapter first, e.g.
#   python scripts/merge_lora.py --base Qwen/Qwen3-0.6B --adapter models/lora --out models/final
#
# Usage:
#   bash scripts/export_gguf.sh models/final models/final
set -euo pipefail

HF_MODEL="${1:-models/final}"
OUT_DIR="${2:-models/final}"
LLAMA_DIR="${LLAMA_CPP_DIR:-third_party/llama.cpp}"

mkdir -p "$OUT_DIR"

# 1. Get llama.cpp (build the quantize binary).
if [ ! -d "$LLAMA_DIR" ]; then
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_NATIVE=ON
  cmake --build "$LLAMA_DIR/build" --config Release -j
fi

# 2. Install conversion deps.
pip install -r "$LLAMA_DIR/requirements.txt"

# 3. HF -> GGUF (F16).
python "$LLAMA_DIR/convert_hf_to_gguf.py" "$HF_MODEL" \
  --outfile "$OUT_DIR/model-f16.gguf" --outtype f16

# 4. Quantize. Binary name differs across builds; try both.
QUANT="$LLAMA_DIR/build/bin/llama-quantize"
[ -x "$QUANT" ] || QUANT="$LLAMA_DIR/build/bin/quantize"

for q in Q8_0 Q5_K_M Q4_K_M; do
  low=$(echo "$q" | cut -d_ -f1 | tr 'A-Z' 'a-z')
  "$QUANT" "$OUT_DIR/model-f16.gguf" "$OUT_DIR/model-${low}.gguf" "$q"
done

echo "GGUF files written to $OUT_DIR"
ls -lh "$OUT_DIR"/*.gguf
