#!/bin/bash
# Reproduce T6 — adversarial-LLM-verifier WITH structural facts in the prompt.
#
# Usage:
#   bash run_T6.sh smoke       # 5-case smoke (~3 min)
#   bash run_T6.sh full        # 194-case slice (~10–15 hours on 1 GPU)
#
# Prereqs:
#   - Ollama running at http://localhost:11444 with qwen3.5:35b loaded
#   - Java retrieval data on disk (paths in --embeddings-path, --meta-path,
#     --java-pool-pkl below). If missing, pass --no-java-examples to disable
#     Stage-3 Java exemplar retrieval (precision will drop a few pp).
#   - Python deps (see requirements.txt)

set -uo pipefail
cd "$(dirname "$0")"

MODE="${1:-smoke}"
case "$MODE" in
    smoke) SLICE=splits/2p5d_smoke_5.json; OUTDIR=results/T6_smoke ;;
    full)  SLICE=splits/2p5d_5pertype_195.json; OUTDIR=results/T6_full ;;
    *)     echo "Usage: bash run_T6.sh [smoke|full]"; exit 1 ;;
esac

mkdir -p "$OUTDIR"

# === T6 environment knobs ===
# LANGCHAIN_PRECISION_MODE=1   — thinking-OFF + 5 precision filters
# LANGCHAIN_FACTS_SRC=...      — structural+intra fact source
# LANGCHAIN_HARD_BLOCK_P=0.20  — auto-drop types whose historical P < 0.20
# LANGCHAIN_ADVERSARIAL=1      — enable per-detection LLM verifier (T5/T6)
# LANGCHAIN_ADVERSARIAL_THRESHOLD=90  — verify only detections with conf < this
# LANGCHAIN_ADVERSARIAL_WITH_FACTS=1  — *** the T6-specific flag ***
#                                       inject gumtree+intra facts into the
#                                       per-detection adversarial prompt
LANGCHAIN_PRECISION_MODE=1 \
LANGCHAIN_FACTS_SRC=gumtree+intra \
LANGCHAIN_HARD_BLOCK_P=0.20 \
LANGCHAIN_ADVERSARIAL=1 \
LANGCHAIN_ADVERSARIAL_THRESHOLD=90 \
LANGCHAIN_ADVERSARIAL_WITH_FACTS=1 \
python3 -u -m langchain_pipeline.concurrent_runner \
    --benchmark "$SLICE" \
    --output-dir "$OUTDIR" \
    --model qwen3.5:35b \
    --base-url http://localhost:11444 \
    --n-workers 2 \
    --sample-per-type 30 \
    --num-ctx 65536 \
    --retriever local \
    --max-java-examples 15 \
    --canon-only \
    --valid-types-file splits/valid_types_39.json \
    --resume \
    "$@"
