# T6 reproduction — adversarial verifier WITH structural facts

This directory is the minimum codebase needed to reproduce the **T6**
experiment from the April 21–23 precision-recovery push: a per-detection
LLM verifier that sees the gumtree+intra-method structural facts in addition
to the cited evidence and the diff window.

T6 differs from the prior `adversarial30` / T5 variant by **one env var**
(`LANGCHAIN_ADVERSARIAL_WITH_FACTS=1`) and ~30 lines of code in
`langchain_pipeline/pipeline.py`. See [T6_DIFF.md](T6_DIFF.md) for the
focused diff.

## Directory layout

```
code/
├── README.md                           ← you are here
├── T6_DIFF.md                          ← the ~30-line diff that makes T6
├── requirements.txt                    ← Python deps
├── run_T6.sh                           ← run smoke / full
├── langchain_pipeline/                 ← the pipeline (qwen3.5:35b harness)
│   ├── pipeline.py                     ← four LLM stages + adversarial + Stage 5
│   ├── prompts.py                      ← Stage prompts + TYPE_DEFINITIONS
│   ├── constants.py                    ← canonical 39 types + routing
│   ├── precision_filters.py            ← 5 precision-mode filters
│   ├── concurrent_runner.py            ← multi-worker harness (entry point)
│   ├── retrieval.py                    ← Java exemplar retriever (Stage 3)
│   ├── data_manager.py                 ← case loader
│   ├── evaluation.py                   ← P/R/F1 scoring
│   ├── gumtree_facts.py                ← AST-level fact extractor (uses GumTree)
│   ├── intra_method_signals.py         ← regex/difflib intra-method patterns
│   ├── structural_facts.py             ← pure-Python AST fallback
│   └── ...
├── splits/
│   ├── 2p5d_smoke_5.json               ← 5-case smoke set (~3 min run)
│   ├── 2p5d_5pertype_195.json          ← 194-case full slice (~10–15 hr per track)
│   └── valid_types_39.json             ← canonical 39-type whitelist
└── scripts/
    └── score_5pt_tracks.py             ← paired-comparison scorer
```

## Prerequisites

### 1. Ollama + the qwen3.5:35b model

Install Ollama (https://ollama.com/) and pull the model:

```bash
ollama pull qwen3.5:35b
```

Run one or more Ollama instances. T6 expects **port 11444** by default
(matching `run_T6.sh`):

```bash
CUDA_VISIBLE_DEVICES=0 \
OLLAMA_HOST=0.0.0.0:11444 \
OLLAMA_NUM_PARALLEL=2 \
OLLAMA_CONTEXT_LENGTH=65536 \
OLLAMA_KEEP_ALIVE=4h \
ollama serve
```

The qwen3.5:35b model loads ~28 GB at full Q4_K_M quantization. Add
KV-cache for `num_ctx=65536` and 2 parallel workers → ~38–42 GB VRAM total.
An A100 80GB is plenty; a 48 GB card is the practical minimum for full
context.

### 2. Python deps

```bash
pip install -r requirements.txt
```

### 3. Java retrieval data + commit pools (download from HuggingFace)

Three big files live in the HuggingFace dataset (not on GitHub due to size):

  **https://huggingface.co/datasets/0726hayate/t6-refactoring-detection-data**

| File | Size | Used for |
|---|---|---|
| `commit_level_train_k15_headline.json` | 2.5 GB | training data (FP-Growth rules + per-type-precision history were mined here) |
| `commit_level_test_k15_headline.json` | 1.1 GB | test pool the slices were drawn from |
| `commit_level_benchmark_k15_headline.json` | 26 MB | historical evaluation set |
| `java_retrieval/unixcoder_java_projected.npy` | ~94 MB | 366k × 64-d UniXcoder embeddings for Java exemplars (Stage 3) |
| `java_retrieval/unixcoder_java_meta.json` | 18 MB | per-record metadata |
| `java_retrieval/java_records_all.pkl` | 2.6 GB | full Java code + commit context per exemplar |

To download:

```bash
pip install huggingface_hub
huggingface-cli download 0726hayate/t6-refactoring-detection-data \
    --repo-type dataset --local-dir .
```

This creates:
- `java_retrieval/` (everything Stage 3 needs — defaults in `run_T6.sh` look here)
- `commit_level_*.json` at top level (the source pools)

If you can't / don't want to download the Java data: pass `--no-java-examples`
to the runner. Stage 3 is disabled, precision drops a few pp, but nothing
else breaks.

### 4. GumTree (optional, for AST facts)

`gumtree_facts.py` shells out to a `gumtree` binary (https://github.com/GumTreeDiff/gumtree).
If GumTree isn't installed, the runner falls back to `structural_facts.py`
(pure-Python `ast` extractor). Install GumTree v3.x and put `gumtree` on PATH
to get richer structural facts.

## Running

### Smoke test (~3 min)

```bash
bash run_T6.sh smoke
```

Processes 5 cases through the full pipeline. Confirms that:

- Ollama responds at port 11444
- All five precision-mode filters fire (check `cases.jsonl` for
  `precision_mode` field)
- The adversarial verifier with facts fires (check for `adversarial_dropped`
  and inspect `adversarial_raw` to see the LLM's per-detection KEEP/DROP
  reasoning *with the structural-facts block* in the prompt)

Output: `results/T6_smoke/cases.jsonl` (one JSON per case).

### Full run (~10–15 hours on 1 GPU)

```bash
bash run_T6.sh full
```

Processes 194 cases (5 per canonical type, stratified). Output:
`results/T6_full/cases.jsonl`.

## Scoring

Compute P/R/F1 against the canonical 39-type ground truth:

```bash
python3 -c "
import json
from pathlib import Path
tp = fp = fn = 0
for line in open('results/T6_full/cases.jsonl'):
    d = json.loads(line)
    tp += d['tp']; fp += d['fp']; fn += d['fn']
P = tp/(tp+fp); R = tp/(tp+fn); F1 = 2*P*R/(P+R)
print(f'TP={tp} FP={fp} FN={fn}  P={P:.3f} R={R:.3f} F1={F1:.3f}')
"
```

## Comparing T6 against T5 (paired)

If you also have a T5 run (same configuration but with
`LANGCHAIN_ADVERSARIAL_WITH_FACTS=0` — the legacy adversarial without
facts), compute paired deltas on shared case_ids:

```bash
python3 scripts/score_5pt_tracks.py
```

The scorer expects results dirs under `results/2p5d_5pt/T*/test/cases.jsonl`.
Adjust paths in the script for your setup.

## Env vars at a glance

| Var | T6 setting | What it does |
|---|:-:|---|
| `LANGCHAIN_PRECISION_MODE` | `1` | Activates 5 precision filters + thinking-OFF |
| `LANGCHAIN_FACTS_SRC` | `gumtree+intra` | Use both AST and intra-method fact sources |
| `LANGCHAIN_HARD_BLOCK_P` | `0.20` | Auto-drop types whose historical P < this |
| `LANGCHAIN_ADVERSARIAL` | `1` | Enable per-detection LLM verifier (T5 + T6) |
| `LANGCHAIN_ADVERSARIAL_THRESHOLD` | `90` | Verify only detections with conf < this |
| **`LANGCHAIN_ADVERSARIAL_WITH_FACTS`** | **`1`** | **T6-specific: inject facts into prompt** |
| `LANGCHAIN_NO_THINK` | (auto via PRECISION_MODE) | Suppress qwen3 reasoning chain |
| `LANGCHAIN_STAGE5` | (T6: not set) | Alternative LLM gating (single batched call) |

## Expected output

Each `cases.jsonl` line is one case with these key fields:

- `case_id`, `url` — case identity
- `ground_truth_known` — list of canonical types in GT
- `final_known` — list of canonical types T6 emitted
- `tp`, `fp`, `fn` — per-case score
- `precision_mode.adversarial_dropped` — list of `{type, verdict}` showing
  what the per-detection LLM verifier dropped (with facts visible to it)
- `adversarial_raw` — raw text of the per-batch adversarial LLM call(s)
  including the **prompts that contained the structural-facts block**

To verify T6 actually wired the facts: grep `adversarial_raw` for
`<gumtree_facts>` or `<intra_method_signals>` — both should appear in every
batched prompt.
