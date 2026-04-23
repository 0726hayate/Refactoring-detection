# T6 — walkthrough (end to end)

This file explains how T6 runs on one commit, from the moment the data is
loaded until a final list of refactorings comes out.

## What T6 is

T6 is the **adversarial-with-facts** pipeline. It takes a Python commit (the
code before, the code after, and the unified diff), runs it through four
LLM stages to get a list of refactoring types, then does one extra LLM call
per uncertain detection to double-check it. The "with-facts" part means the
double-check LLM call now sees the structural AST facts (from GumTree) and
the intra-method patterns (from the regex extractor) in its prompt, so it
can keep a detection if the facts support it — not just if the diff line
happens to match.

## What the adversarial actually uses (just T6)

T6's adversarial verifier is one function plus a small handful of helpers
and inputs. Everything else in the pipeline produces inputs for it but the
verifier itself only touches the items below.

### The function

| File | Function | Role |
|---|---|---|
| `langchain_pipeline/pipeline.py` | **`_stage_adversarial`** | The whole T6 logic: split detections into certain / uncertain by Stage-4 confidence, batch the uncertain ones (≤5 per call), build a prompt that includes the structural facts when `LANGCHAIN_ADVERSARIAL_WITH_FACTS=1`, ask the LLM KEEP / DROP per detection, return the survivors. |

### Direct helpers it calls

| File | Function | What it does for the adversarial |
|---|---|---|
| `langchain_pipeline/pipeline.py` | `_extract_diff_context` | Pulls ±5 lines from the diff around the cited evidence so the LLM has local context |
| `langchain_pipeline/pipeline.py` | `_invoke` | The plain LLM call wrapper — sends `[SystemMessage, HumanMessage]`, returns the raw text |
| `langchain_pipeline/prompts.py` | `TYPE_DEFINITIONS` (dict, not a function) | Lookup: type name → one-line canonical definition that goes into the per-detection prompt |
| `langchain_pipeline/prompts.py` | `_no_think_enabled` | Returns True when precision mode is on, so the system message gets `/no_think` appended |

### Where it is called from (1 call site)

| File | Function | When |
|---|---|---|
| `langchain_pipeline/pipeline.py` | `_stage3_then_stage4` | After the Stage-4 LLM call, if `LANGCHAIN_ADVERSARIAL=1`. Passes the verified types, evidences, confidences, the diff, and the combined `structural_facts_xml`. |

### Inputs it consumes (produced upstream)

| Input | Comes from | File |
|---|---|---|
| `detections` (list of `(type, evidence)`) | Stage 4 LLM output | `pipeline.py` Stage-4 block in `_stage3_then_stage4` |
| `confidences` (dict `type → 0..100`) | Stage 4 LLM output | same |
| `code_diff` | Input case | loaded by `concurrent_runner.main` from the slice JSON |
| `facts_xml` (the T6 special bit) | `_preprocess_case` glues the gumtree + intra blocks | `pipeline.py`, calling `gumtree_facts_for_case` (`gumtree_facts.py`) and `intra_method_signals_for_case` (`intra_method_signals.py`) |
| Per-batch system prompt | Hardcoded one-liner, plus `/no_think` if precision mode | `pipeline.py` inside `_stage_adversarial` |

### Env vars the adversarial reads

| Env var | What it does |
|---|---|
| `LANGCHAIN_ADVERSARIAL` | If unset / `0`, the adversarial never runs. T6 sets it to `1`. |
| `LANGCHAIN_ADVERSARIAL_THRESHOLD` | Detections with Stage-4 confidence ≥ this number are kept without verification. T6 sets `90`. |
| **`LANGCHAIN_ADVERSARIAL_WITH_FACTS`** | The T6-specific switch. When `1`, the structural+intra facts block is prepended to the per-batch prompt and the KEEP rule allows "facts corroborate" as a reason. When `0` (T5 behaviour), the prompt only contains type definition + cited evidence + diff window. |
| `LANGCHAIN_PRECISION_MODE` | Indirectly: when `1`, `_no_think_enabled()` returns True, so the adversarial system message gets `/no_think` appended. |

### What the adversarial does NOT touch

For clarity — these are NOT used by the adversarial verifier:

- The Stage 1 / 2 / 3 / 4 prompts and schemas (`schemas.py`, `build_stage*_messages` in `prompts.py`)
- `precision_filters.py` (those are the rule-based filters that run BEFORE the adversarial)
- `retrieval.py` (no Java exemplar lookup happens during the adversarial call)
- `evaluation.py` (scoring runs after the adversarial returns)
- `data_manager.py` (case loading happens before the pipeline starts)
- `tools.py` (the adversarial does NOT use LangChain tool calling — it's a plain text in / text out call)

## Walkthrough — one commit, start to finish

### Step 1. Load the commit

The runner loads one commit from a slice JSON. Each commit is a dict with
`code_before`, `code_after`, `commit_diff`, `refactoring_types` (the ground
truth — used only for scoring, never shown to the LLM), and
`matched_java_examples` (15 pre-retrieved Java exemplars).

- Entry point: `main` in `langchain_pipeline/concurrent_runner.py`
- Per-commit dispatch to a worker pool, each worker calls `predict` on
  `RefactoringPipeline` in `langchain_pipeline/pipeline.py`

### Step 2. Build the structural facts BEFORE any LLM call

Two fact extractors,  produce XML blocks that
will later be pasted into prompts:

- `gumtree_facts_for_case` in `langchain_pipeline/gumtree_facts.py` — shells
  out to GumTree, produces `<gumtree_facts>` listing added/removed/moved
  classes, methods, signatures, attributes
- `intra_method_signals_for_case` in
  `langchain_pipeline/intra_method_signals.py` — regex + difflib analyzer,
  produces `<intra_method_signals>` with consistent token renames, variable
  extractions, condition inversions, etc.

If GumTree can't parse the commit (some files are broken Python), the
pipeline falls back to `facts_for_case` in
`langchain_pipeline/structural_facts.py` (pure-Python `ast`).

These two blocks are a single `structural_facts_xml`
string that Stages 2/3 and the Stage 5 / adversarial verifier will all see.

### Step 3. Stage 1 — routing (LLM call 1)

The LLM reads the diff and decides whether the change is
**parameter-level**, **method-level**, or **class-level** (it can pick
multiple). This scope controls how many candidate types Stage 3 will probe
later.

- Prompt: `build_stage1_messages` in `langchain_pipeline/prompts.py`
- Output schema: `LevelClassification` in `langchain_pipeline/schemas.py`
- Invocation: `_invoke_structured` in `langchain_pipeline/pipeline.py`

### Step 4. Stage 2 — emit refactoring types the LLM sees (LLM call 2)

The LLM reads: the diff + the `<gumtree_facts>` +
the `<intra_method_signals>` + the canonical 39-type list with definitions.
It returns two lists:
- `defined` — types it recognises from the 39 canonical ones
- `undefined` — novel patterns it couldn't name (prefix `UnknownType:`)

- Prompt: `build_stage2_messages` in `langchain_pipeline/prompts.py`
- Output schema: `DetectedRefactorings` in `langchain_pipeline/schemas.py`
- Types universe: `ALL_KNOWN_TYPES` and `LEVEL_TYPES` in
  `langchain_pipeline/constants.py`

### Step 5. Stage 3 — probe for missing types (LLM call 3)

For each plausible co-occurring type Stage 2 didn't emit (mined from the
training corpus via FP-Growth rules), ask the LLM *"are you sure this
isn't also happening?"*. Java exemplars of each candidate are retrieved and
shown as anchors so the LLM sees what the type looks like in practice.

- Candidate mining: `build_missing_candidates` and `MISSING_HINTS_MINED_V4`
  in `langchain_pipeline/constants.py`
- Java retrieval: `JavaExampleRetriever.retrieve_examples` in
  `langchain_pipeline/retrieval.py`
- Prompt: `build_stage3_messages` in `langchain_pipeline/prompts.py`
- Output schema: `AdditionalDetections` in `langchain_pipeline/schemas.py`

In precision mode, Stage 3's additions go through two filters:
- `filter_stage3_candidates` (drops chronically-bad types)
- `filter_sole_stage3_additions` (needs Stage 2 confirmation OR strong
  association rule OR matching structural evidence)

Both in `langchain_pipeline/precision_filters.py`.

### Step 6. Stage 4 — verify each detection with a confidence score (LLM call 4)

The LLM sees the combined detections and produces, for each one:
- A confidence score 0–100
- An `evidence="..."` attribute quoting the specific diff line that proves
  it

- Prompt: `build_stage4_messages` in `langchain_pipeline/prompts.py`
- Output schema: `VerifiedRefactoringsWithConfidence` in
  `langchain_pipeline/schemas.py`
- Confidence threshold (for filtering): `confidence >= 80` when
  `LANGCHAIN_PRECISION_MODE=1`

### Step 7. THE T6 BIT — adversarial double-check with facts (LLM call 5+)

For every detection whose Stage-4 confidence is below
`LANGCHAIN_ADVERSARIAL_THRESHOLD` (default 90), T6 runs one extra LLM call
per batch of 5 to challenge it.

The per-batch prompt contains:
- The structural+intra facts block (the T6-specific bit — T5 did not have
  this)
- For each detection: the type name, its one-line canonical definition,
  the Stage-4 cited evidence, and ±5 diff lines around that evidence
- Decision rules: KEEP if the diff supports it OR the facts corroborate
  it; DROP otherwise

The LLM replies `KEEP <reason>` or `DROP <reason>` per detection. The
pipeline applies the verdicts and that's the final detection list.

- Function: `_stage_adversarial` in `langchain_pipeline/pipeline.py`
- Env vars that make this run:
  - `LANGCHAIN_ADVERSARIAL=1` (turn it on)
  - `LANGCHAIN_ADVERSARIAL_WITH_FACTS=1` (the T6-specific flag — inject
    facts into the prompt)
  - `LANGCHAIN_ADVERSARIAL_THRESHOLD=90` (only challenge detections below
    this confidence)

One-line canonical definitions used in the prompt: `TYPE_DEFINITIONS` in
`langchain_pipeline/prompts.py`.

### Step 8. Score the case

Once `predict` returns a list of canonical types, the runner scores the
case against the ground truth and writes one JSON line per case to
`results/T6_*/cases.jsonl`.

- Scoring: `compute_multilabel_metrics` and `_process_single_example` in
  `langchain_pipeline/evaluation.py`
- Canonical whitelist: `splits/valid_types_39.json` (passed via
  `--canon-only --valid-types-file splits/valid_types_39.json`)

Each written JSON line has:
- `case_id`, `url` — the commit identity
- `ground_truth_known` — GT types in the canonical 39
- `final_known` — what T6 emitted (after the adversarial filter)
- `tp`, `fp`, `fn` — scored against GT
- `precision_mode.adversarial_dropped` — log of detections the T6
  verifier dropped
- `adversarial_raw` — the raw LLM output for the per-batch verification
  calls (this is where you can grep for `<gumtree_facts>` to confirm T6
  injected the facts)

## Where the data lives

### In this directory

| What | Path | Used for |
|---|---|---|
| 194-case T6 input slice | `splits/2p5d_5pertype_195.json` | full T6 run (`run_T6.sh full`) |
| 5-case smoke slice | `splits/2p5d_smoke_5.json` | quick verification (`run_T6.sh smoke`) |
| Canonical 39-type whitelist | `splits/valid_types_39.json` | scorer + `--canon-only` |

### What's NOT (too big — hosted on HuggingFace)

Dataset: **https://huggingface.co/datasets/0726hayate/t6-refactoring-detection-data**

Download all 6 files into `./` from the dataset repo to reproduce locally.

| What | Path on the server | Why you might need it |
|---|---|---|
| Test commit pool | `/home/25fxvd/summer2025/0807/dspy/langchain_pipeline/commit_level_test_k15_headline.json` (~1.1 GB) | source pool the T6 slice was drawn from |
| Train commit pool | `/home/25fxvd/summer2025/0807/dspy/langchain_pipeline/commit_level_train_k15_headline.json` (~2.5 GB) | where the FP-Growth rules + per-type-precision history were mined from |
| Benchmark pool | `/home/25fxvd/summer2025/0807/dspy/langchain_pipeline/commit_level_benchmark_k15_headline.json` (~26 MB) | historical evaluation set |
| Java embeddings (366k × 64-d) | `/home/25fxvd/summer2025/ELEC825/evaluation_data_v3_ast/contrastive_v5_unixcoder_holdout/projected/unixcoder_java_projected.npy` (~94 MB) | Stage-3 Java exemplar retrieval |
| Java embedding metadata | `/home/25fxvd/summer2025/ELEC825/evaluation_data_v3_ast/embeddings/unixcoder_java_meta.json` | per-record metadata |
| Java record pool | `/home/25fxvd/summer2025/ELEC825/evaluation_data_v3_ast/java/java_records_all.pkl` (~hundreds MB) | full Java code + commit context per exemplar |

### How the data flows into Stages 1/2

**Stage 1 and Stage 2 do NOT read the test/train/benchmark pools directly.**
They only ever see one commit at a time, and that commit comes from the
slice JSON (`splits/2p5d_5pertype_195.json` or
`splits/2p5d_smoke_5.json`). The slice is just a list of commits pulled
from the test pool and saved to disk so we can reproduce the same case
set across experiments.

The Java retrieval data (embeddings + pool) is only used by Stage 3 — to
find Java exemplars of each candidate refactoring type the LLM is
probing. If you pass `--no-java-examples` to the runner, Stage 3 skips
retrieval entirely (precision will drop a few pp but nothing else
breaks).

## How to actually run T6

1. **Python deps**: `pip install -r requirements.txt`
2. **Java retrieval data** on disk at the paths above (optional — pass
   `--no-java-examples` if missing).

Then:

```
bash run_T6.sh smoke   # 5 cases, ~3 min
bash run_T6.sh full    # 194 cases, ~10–15 hr on 1 GPU
```

Output goes to `results/T6_smoke/cases.jsonl` or
`results/T6_full/cases.jsonl`.

## How to check T6 actually injected the facts

After a run, grep the per-case JSONL for `<gumtree_facts>` inside
`adversarial_raw`. If it's there, T6's flag propagated correctly and the
LLM saw the facts when making its per-detection KEEP/DROP call. If it's
not there, the env var didn't make it through — check the `run_T6.sh`
script and the env-var list in the README.
