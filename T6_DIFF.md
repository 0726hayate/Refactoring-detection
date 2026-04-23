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

Two fact extractors run offline-style (no LLM) and produce XML blocks that
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

These two blocks are glued together into a single `structural_facts_xml`
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

The big prompt. The LLM reads: the diff + the `<gumtree_facts>` +
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

### What's bundled in this directory

| What | Path | Used for |
|---|---|---|
| 194-case T6 input slice | `splits/2p5d_5pertype_195.json` | full T6 run (`run_T6.sh full`) |
| 5-case smoke slice | `splits/2p5d_smoke_5.json` | quick verification (`run_T6.sh smoke`) |
| Canonical 39-type whitelist | `splits/valid_types_39.json` | scorer + `--canon-only` |

### What's NOT bundled (too big — hosted on HuggingFace)

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

Three things need to be in place:


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
