#!/usr/bin/env python3
"""
Concurrent 4-worker runner for the LangChain refactoring detection pipeline.

This is the iter 6 entry point. It differs from run.py in three ways:
1. **4 case workers running fully serial through the 4 stages.**
   Each worker holds exactly 1 LLM slot at a time, so total concurrent
   LLM calls = 4 (matching the Ollama instance's hard concurrency ceiling
   for qwen3:32b on the shared workstation). Stage 2's internal level
   parallelism is disabled by passing ``stage2_max_workers=1`` to the
   pipeline so each Stage 2 case runs its 3 level prompts sequentially.

2. **Round-robin type sampling.** The producer cycles through the 39
   refactoring types and yields one un-consumed case per type per round.
   This guarantees per-type coverage even if the run is killed early.

3. **Incremental JSONL persistence with full prompt + response capture.**
   Every LLM call's prompt and raw response is captured via a TracingLLM
   wrapper around the ChatOllama client. After each case completes Stage 4,
   the full per-case trace (including all 5+ LLM calls' prompts and
   responses, the python commit, the java examples used, GT, predictions,
   timestamps) is appended to ``cases.jsonl`` and also written as a
   per-case JSON file at ``cases/{case_id}.json``. Survives kills cleanly.

Usage::
    LANGCHAIN_HINTS_VERSION=v3 \\
    LANGCHAIN_HINTS_HYBRID=1 \\
    LANGCHAIN_DROP_ALWAYS_CHECK=1 \\
        python -m langchain_pipeline.concurrent_runner \\
        --benchmark .../matched_mined_diffs_all_v1_cleaned.json \\
        --model qwen3:32b --sample-per-type 30 --type-avg-filter 0 \\
        --base-url http://localhost:11440 \\
        --output-dir ./results/iteration_6
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .data_manager import DataManager
from .pipeline import MODEL_CONFIGS, RefactoringPipeline
from .retrieval import JavaExampleRetriever

# TracingLLM — wraps the LangChain LLM client to capture prompt + response
class CaseRecorder:
    """Per-case recorder that accumulates LLM call records.

    Each call records the full prompt (formatted from messages), the raw
    response text, the elapsed time, and any error. The runner attaches
    one CaseRecorder per worker per case and reads it after the case
    completes to merge into the case trace dict.
    """
    # init
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    # record
    def record(
        self,
        *,
        prompt: str,
        response: str,
        duration_s: float,
        error: Optional[str] = None,
    ) -> None:
        self.calls.append({
            "prompt": prompt,
            "response_raw": response,
            "duration_s": round(duration_s, 3),
            "error": error,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        })


def _format_messages(messages: List[Any]) -> str:
    """Format LangChain message objects into a single text block.

    Each message becomes ``<role>:\n<content>\n`` so the order and roles
    are preserved in the JSONL trace.
    """
    out = []
    for m in messages:
        role = getattr(m, "type", None) or m.__class__.__name__
        content = getattr(m, "content", str(m))
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", str(p)) if isinstance(p, dict) else str(p)
                for p in content
            )
        out.append(f"--- {role} ---\n{content}")
    return "\n".join(out)


class TracingLLM:
    """Drop-in wrapper around any LangChain LLM client.

    Forwards ``invoke(messages)`` to the inner client and records the
    prompt + response into a CaseRecorder. The recorder reference can be
    swapped between cases via ``set_recorder()`` so the same TracingLLM
    instance is reused across cases on a worker thread.
    """
    # init
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self._recorder: Optional[CaseRecorder] = None

    # set recorder
    def set_recorder(self, recorder: Optional[CaseRecorder]) -> None:
        self._recorder = recorder

    # invoke
    def invoke(self, messages: List[Any]) -> Any:
        prompt_text = _format_messages(messages)
        start = time.monotonic()
        err: Optional[str] = None
        response_text = ""
        try:
            result = self.inner.invoke(messages)
            response_text = (
                result.content if hasattr(result, "content") else str(result)
            )
            return result
        except Exception as exc:
            err = str(exc)[:500]
            raise
        finally:
            duration = time.monotonic() - start
            if self._recorder is not None:
                self._recorder.record(
                    prompt=prompt_text,
                    response=response_text,
                    duration_s=duration,
                    error=err,
                )

    # Forward any other attribute access to the inner client
    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

# Round-robin type producer
def case_key(item: Dict) -> str:
    """Stable per-case identifier used for dedup and resume.

    Mirrors the case_id_safe used by case_worker so we can match a JSONL
    line back to a benchmark item.
    """
    pc = item.get("python_case", {})
    raw = (
        pc.get("python_record_id")
        or pc.get("url")
        or hashlib.md5(pc.get("commit_diff", str(id(item))).encode()).hexdigest()
    )

    return raw.replace("/", "_").replace(":", "_")[-180:]


def round_robin_sample(
    data: List[Dict],
    n_per_type: int = 30,
    seed: int = 42,
    valid_types: Optional[set] = None,
    skip_keys: Optional[set] = None,
    existing_per_type: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, Dict]]:
    """Sample up to ``n_per_type`` cases per refactoring type in round-robin
    order. Cases are deduplicated by python_record_id (or commit url as
    fallback) — each unique commit appears at most once even if it has
    multiple refactoring types.

    Args:
        valid_types: If provided, restricts the type universe to this set.
            Refactoring types in the data outside this set are ignored.
            Use this to enforce the canonical 39-type label space.
        skip_keys: If provided, sampled case_keys in this set are excluded
            (resume support — pass the set of case_ids already in the
            existing cases.jsonl).
        existing_per_type: If provided, ``existing_per_type[T]`` is the
            number of cases already done for type T. The sampler will only
            yield up to ``max(0, n_per_type - existing_per_type[T])`` new
            cases for that type, so the FINAL total per type is at most
            ``n_per_type``. Use this with skip_keys for clean resume.

    Returns a list of ``(scheduled_for_type, case_dict)`` tuples in the
    order the runner should consume them: cycles through types, one new
    case per type per round, until all per-type quotas are met or buckets
    are exhausted.
    """
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for item in data:
        for rt in item.get("python_case", {}).get("refactoring_types", []):
            if valid_types is not None and rt not in valid_types:
                continue
            by_type[rt].append(item)

    rng = random.Random(seed)
    for rt in by_type:
        rng.shuffle(by_type[rt])

    types_in_order = sorted(by_type.keys())
    type_indices: Dict[str, int] = {rt: 0 for rt in types_in_order}

    existing_pt = existing_per_type or {}
    quota: Dict[str, int] = {
        rt: max(0, n_per_type - existing_pt.get(rt, 0))
        for rt in types_in_order
    }
    yielded_pt: Dict[str, int] = {rt: 0 for rt in types_in_order}

    seen_keys: set = set(skip_keys or ())
    sampled: List[Tuple[str, Dict]] = []

    while True:
        progress_this_round = 0
        for rt in types_in_order:
            if yielded_pt[rt] >= quota[rt]:
                continue
            while type_indices[rt] < len(by_type[rt]):
                item = by_type[rt][type_indices[rt]]
                type_indices[rt] += 1
                key = case_key(item)
                if key not in seen_keys:
                    seen_keys.add(key)
                    sampled.append((rt, item))
                    yielded_pt[rt] += 1
                    progress_this_round += 1
                    break
        if progress_this_round == 0:
            break

    return sampled


def load_existing_case_ids(jsonl_path: str) -> set:
    """Read an existing cases.jsonl and return the set of case_ids already
    persisted. Used for resume — the runner skips these cases when sampling.
    """
    if not os.path.exists(jsonl_path):
        return set()
    seen = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("case_id")
                if cid:
                    seen.add(cid)
            except (json.JSONDecodeError, ValueError):
                continue
    return seen


def load_existing_per_type_counts(jsonl_path: str) -> Dict[str, int]:
    """Read an existing cases.jsonl and return per-type counts of how many
    cases were scheduled for each type. Used by round_robin_sample to subtract
    from the new-sample quota when resuming.
    """
    if not os.path.exists(jsonl_path):
        return {}
    counts: Dict[str, int] = defaultdict(int)
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                t = rec.get("scheduled_for_type")
                if t:
                    counts[t] += 1
            except (json.JSONDecodeError, ValueError):
                continue
    return dict(counts)

# Per-case trace builder
def _summarize_python_case(python_case: Dict) -> Dict:
    """Extract a compact representation of the Python case for the JSONL
    trace. Includes the full diff so the supervisor can inspect any case
    without re-loading the original benchmark."""
    return {
        "url": python_case.get("url", ""),
        "repository": python_case.get("repository", ""),
        "sha1": python_case.get("sha1", ""),
        "refactoring_types": python_case.get("refactoring_types", []),
        "commit_diff": python_case.get("commit_diff", ""),

    }


def _summarize_java_examples(java_examples: List[Dict], max_examples: int = 20) -> List[Dict]:
    """Compact view of the Java exemplars Stage 2 saw (anchored from upstream's
    matched_java_examples). Truncates code_text to first ~1500 chars so the
    JSONL doesn't bloat. The full code is still available in the original
    benchmark if we need to re-inspect."""
    out = []
    for ex in java_examples[:max_examples]:
        code = ex.get("code_text", "") or ""
        out.append({
            "id": ex.get("id"),
            "refactoring_type": ex.get("refactoring_type"),
            "similarity": ex.get("similarity"),
            "rerank_score": ex.get("rerank_score"),
            "url": ex.get("url"),
            "code_text_truncated": code[:1500] + ("...[truncated]" if len(code) > 1500 else ""),
        })
    return out


def build_case_trace(
    *,
    case_id: str,
    scheduled_for_type: str,
    round_robin_index: int,
    item: Dict,
    pipeline_trace: Dict,
    llm_calls: List[Dict],
    started_at: float,
    ended_at: float,
    error: Optional[str] = None,
) -> Dict:
    """Assemble the full per-case trace for JSONL persistence."""
    python_case = item.get("python_case", {})
    java_examples = item.get("java_examples", [])
    gt_known = python_case.get("refactoring_types", [])
    final_types = pipeline_trace.get("final_types", [])

    gt_set = set(gt_known)
    final_known = [t for t in final_types if not t.startswith("UnknownType")]
    final_undef = [t for t in final_types if t.startswith("UnknownType")]
    pred_set = set(final_known)
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    return {
        "case_id": case_id,
        "scheduled_for_type": scheduled_for_type,
        "round_robin_index": round_robin_index,
        "python_commit": _summarize_python_case(python_case),
        "java_examples_used": _summarize_java_examples(java_examples),
        "ground_truth_known": gt_known,
        "final_types": final_types,
        "final_known": final_known,
        "final_undefined": final_undef,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "stage1_output": pipeline_trace.get("stage1_output", []),
        "stage2_defined": pipeline_trace.get("stage2_defined", []),
        "stage2_undefined": pipeline_trace.get("stage2_undefined", []),
        "stage3_targets": pipeline_trace.get("stage3_targets", []),
        "stage3_output": pipeline_trace.get("stage3_output", []),
        "stage4_targets": pipeline_trace.get("stage4_targets", []),
        "stage4_output": pipeline_trace.get("stage4_output", []),
        "stage4_confidences": pipeline_trace.get("stage4_confidences", {}),
        "structural_facts_chars": pipeline_trace.get("structural_facts_chars", 0),
        "structural_facts_source": pipeline_trace.get("structural_facts_source", None),
        "trophies_chars": pipeline_trace.get("trophies_chars", 0),
        "precision_mode": pipeline_trace.get("precision_mode", {}),
        "errors": pipeline_trace.get("errors", []),
        "stage_raw": {
            "stage1": pipeline_trace.get("stage1_raw", ""),
            "stage2_traces": pipeline_trace.get("stage2_traces", []),
            "stage3": pipeline_trace.get("stage3_raw", ""),
            "stage4": pipeline_trace.get("stage4_raw", ""),
        },
        "llm_calls": llm_calls,
        "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="milliseconds"),
        "ended_at": datetime.fromtimestamp(ended_at).isoformat(timespec="milliseconds"),
        "wall_clock_total_s": round(ended_at - started_at, 2),
        "n_llm_calls": len(llm_calls),
        "fatal_error": error,
    }

# Async case worker + result writer

# A single case worker
async def case_worker(
    worker_id: int,
    case_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    pipeline: RefactoringPipeline,
    tracing_llm: TracingLLM,
    verbose: bool,
) -> None:
    """A single case worker. Pulls cases from the queue, runs the pipeline
    fully serial (Stage 1 → Stage 2 → Stage 3 → Stage 4) via to_thread,
    pushes the trace to the result queue. Stops when it pulls a None.
    """
    while True:
        msg = await case_queue.get()
        if msg is None:
            case_queue.task_done()
            return

        round_idx, scheduled_type, item = msg
        case_id = (
            item.get("python_case", {}).get("python_record_id")
            or item.get("python_case", {}).get("url", "")
            or f"case_{round_idx}"
        )

        case_id_safe = case_id.replace("/", "_").replace(":", "_")[-180:]

        recorder = CaseRecorder()
        tracing_llm.set_recorder(recorder)

        started_at = time.time()
        fatal_err: Optional[str] = None
        pipeline_trace: Dict[str, Any] = {}

        try:
            python_case = item.get("python_case", {})
            java_examples = item.get("java_examples", [])
            pipeline_trace = await asyncio.to_thread(
                pipeline.predict_with_trace, python_case, java_examples
            )
        except Exception as exc:
            fatal_err = f"{type(exc).__name__}: {str(exc)[:500]}\n{traceback.format_exc()[-1000:]}"
        finally:
            tracing_llm.set_recorder(None)
            ended_at = time.time()

        case_trace = build_case_trace(
            case_id=case_id_safe,
            scheduled_for_type=scheduled_type,
            round_robin_index=round_idx,
            item=item,
            pipeline_trace=pipeline_trace,
            llm_calls=recorder.calls,
            started_at=started_at,
            ended_at=ended_at,
            error=fatal_err,
        )

        if verbose:
            duration = ended_at - started_at
            n_calls = len(recorder.calls)
            tp = case_trace["tp"]
            fp = case_trace["fp"]
            fn = case_trace["fn"]
            print(
                f"  [w{worker_id} #{round_idx:>4} type={scheduled_type[:20]:<20}] "
                f"{duration:>5.1f}s {n_calls} calls TP={tp} FP={fp} FN={fn}",
                flush=True,
            )

        await result_queue.put(case_trace)
        case_queue.task_done()

# Single writer task
async def result_writer(
    result_queue: asyncio.Queue,
    output_dir: str,
    n_total: int,
) -> None:
    """Single writer task. Pulls completed case traces from the result queue
    and appends them to cases.jsonl. Also writes a per-case JSON file for
    easy human inspection. Stops when it pulls a None.
    """
    jsonl_path = os.path.join(output_dir, "cases.jsonl")
    cases_dir = os.path.join(output_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)
    n_written = 0
    started_at = time.time()

    with open(jsonl_path, "a", buffering=1) as fp:
        while True:
            trace = await result_queue.get()
            if trace is None:
                result_queue.task_done()
                break

            fp.write(json.dumps(trace, ensure_ascii=False) + "\n")
            fp.flush()
            try:
                os.fsync(fp.fileno())
            except OSError:
                pass

            try:
                with open(os.path.join(cases_dir, f"{trace['case_id']}.json"), "w") as cf:
                    json.dump(trace, cf, ensure_ascii=False, indent=2)
            except OSError as e:
                print(f"  [writer] WARN: failed to write per-case file: {e}", file=sys.stderr)

            n_written += 1
            if n_written % 5 == 0 or n_written == n_total:
                elapsed = time.time() - started_at
                rate = n_written / max(elapsed, 1) * 3600
                eta_h = (n_total - n_written) / max(rate, 1)
                print(
                    f"  [writer] {n_written}/{n_total} cases written "
                    f"(elapsed={elapsed/60:.1f}min, rate={rate:.0f} cases/hr, "
                    f"eta={eta_h:.1f}h)",
                    flush=True,
                )
            result_queue.task_done()

# Main orchestration

# run async
async def run_async(
    pipeline: RefactoringPipeline,
    tracing_llms: List[TracingLLM],
    cases: List[Tuple[str, Dict]],
    output_dir: str,
    n_workers: int,
    verbose: bool,
) -> None:
    case_queue: asyncio.Queue = asyncio.Queue(maxsize=n_workers * 2)
    result_queue: asyncio.Queue = asyncio.Queue()

    workers = []
    for w_id in range(n_workers):
        t = asyncio.create_task(
            case_worker(
                worker_id=w_id,
                case_queue=case_queue,
                result_queue=result_queue,
                pipeline=pipeline,
                tracing_llm=tracing_llms[w_id],
                verbose=verbose,
            )
        )
        workers.append(t)

    writer_task = asyncio.create_task(
        result_writer(result_queue, output_dir, len(cases))
    )

    for round_idx, (scheduled_type, item) in enumerate(cases):
        await case_queue.put((round_idx, scheduled_type, item))
    for _ in range(n_workers):
        await case_queue.put(None)

    await asyncio.gather(*workers)

    await result_queue.put(None)
    await writer_task

# CLI

# build parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="4-worker concurrent runner for the LangChain refactoring detection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmark", required=True)
    p.add_argument("--definitions", default="./refactoring_defs.json")
    p.add_argument("--model", default="qwen3:32b", choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num-ctx", type=int, default=None)
    p.add_argument("--base-url", default="http://localhost:11440")
    p.add_argument("--max-java-examples", type=int, default=3)
    p.add_argument("--sample-per-type", type=int, default=30)
    p.add_argument("--type-avg-filter", type=float, default=0.0)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument(
        "--embeddings-path",
        default="./java_retrieval/unixcoder_java_projected.npy",
    )
    p.add_argument(
        "--meta-path",
        default="./java_retrieval/unixcoder_java_meta.json",
    )
    p.add_argument(
        "--java-pool-pkl",
        default="./java_retrieval/java_records_all.pkl",
    )
    p.add_argument(
        "--java-pool-dir",
        default="./java_retrieval",
    )
    p.add_argument("--retrieve-k-3a", type=int, default=5)
    p.add_argument("--retrieve-k-3b", type=int, default=3)
    p.add_argument("--no-retrieval", action="store_true")
    p.add_argument("--n-workers", type=int, default=4,
                   help="Number of concurrent case workers (=peak concurrent LLM calls).")
    p.add_argument("--output-dir", default="./results/iteration_6")
    p.add_argument("--verbose", action="store_true")

    p.add_argument(
        "--canon-only",
        action="store_true",
        help="Restrict the round-robin schedule to the canonical 39 refactoring "
             "types (loaded from valid_types_39.json). Without this flag the "
             "sampler iterates over every type that appears in the data, "
             "including out-of-scope types like 'Add Attribute Modifier' that "
             "the LLM cannot detect.",
    )
    p.add_argument(
        "--valid-types-file",
        default="./splits/valid_types_39.json",
        help="Path to the canonical 39-type label space (JSON list).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip case_ids already present in <output-dir>/cases.jsonl. "
             "Use to resume an interrupted run without re-processing cases. "
             "The runner appends new cases to the existing JSONL.",
    )

    p.add_argument(
        "--retriever",
        choices=("headline", "local", "upstream"),
        default="headline",
        help="Retriever implementation. 'headline' (recommended) uses "
             "HeadlineRetriever — the report headline 3-channel cosine "
             "fusion (Same-modal + Desc + GNN with learned 311-entry node "
             "embedding, weights 0.345/0.609/0.046, 47.26 test / 41.02 "
             "bench S-R@10). 'local' uses legacy JavaExampleRetriever "
             "(Java→Java mean-pool anchor similarity, wrong semantics for "
             "Stage 3/4). 'upstream' wraps upstream's RefactoringRetriever "
             "via UpstreamFacadeRetriever — limited to the 2-channel "
             "Same-modal + Desc fusion the upstream class supports.",
    )
    p.add_argument(
        "--upstream-config",
        default="./configs/best_retriever.json",
        help="Config path passed to UpstreamFacadeRetriever. Default is the "
             "cascade-locked headline (UniXcoder + CLIP α=0.2). Only used "
             "when --retriever=upstream.",
    )
    p.add_argument(
        "--rerank",
        action="store_true",
        default=True,
        help="Enable the frozen BAAI/bge-reranker-v2-gemma cross-encoder in "
             "the upstream cascade (matches the locked headline). Default True. "
             "Adds ~300-400ms per retrieval call but gains +16 pp S-R@10 on "
             "benchmark cohort. Only used when --retriever=upstream.",
    )
    p.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Disable the cross-encoder reranker (bi-encoder only). Faster "
             "~30ms per call but loses ~16 pp S-R@10 on benchmark cohort.",
    )
    p.add_argument(
        "--skip-class-level",
        action="store_true",
        help="Sets env var LANGCHAIN_SKIP_CLASS_LEVEL=1, causing the pipeline "
             "to drop class_level from Stage 1's output before Stage 2 runs, "
             "and to filter Stage 3/4 candidate types to parameter+method "
             "types only. Use this for iter 7+ when focusing on the worst "
             "level (method) and the cleanest level (parameter). Also restricts "
             "the round-robin sampler to only schedule cases whose GT contains "
             "≥1 parameter or method type.",
    )
    p.add_argument(
        "--no-java-examples",
        action="store_true",
        help="Disable ALL Java examples: sets max_java_examples=0, removes "
             "retrieve_java_examples_tool from all stages, and skips retriever "
             "loading. Use for no-retrieval baseline ablation.",
    )
    return p

# main
def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("Concurrent Runner — LangChain Refactoring Detection (iter 6)")
    print("=" * 70)
    print(f"Model:           {args.model}")
    print(f"Base URL:        {args.base_url}")
    print(f"Workers:         {args.n_workers}")
    print(f"Sample per type: {args.sample_per_type}")
    print(f"Output dir:      {output_dir}")
    print(f"Hint env vars:")
    for var in [
        "LANGCHAIN_HINTS_VERSION", "LANGCHAIN_HINTS_HYBRID",
        "LANGCHAIN_DROP_ALWAYS_CHECK", "LANGCHAIN_ADAPTIVE_CAP",
        "LANGCHAIN_MISSING_CAP",
    ]:
        v = os.environ.get(var, "<unset>")
        print(f"  {var:<32} = {v}")
    print("=" * 70)
    print()

    print("Loading benchmark...")
    dm = DataManager(
        benchmark_file=os.path.abspath(args.benchmark),
        definitions_file=(
            os.path.abspath(args.definitions)
            if os.path.exists(args.definitions) else None
        ),
    )
    dataset = list(dm.paired_data)
    print(f"  Loaded {len(dataset)} examples")

    model_cfg = MODEL_CONFIGS.get(args.model, {})
    max_tokens = model_cfg.get("num_ctx", 131072)
    dataset = DataManager.filter_long_examples(dataset, max_tokens=max_tokens)
    print(f"  After token filter: {len(dataset)}")

    if args.type_avg_filter and args.type_avg_filter > 0:
        dataset = DataManager.filter_by_type_avg_size(
            dataset, multiplier=args.type_avg_filter,
        )
        print(f"  After type-avg filter: {len(dataset)}")

    if args.skip_class_level:
        os.environ["LANGCHAIN_SKIP_CLASS_LEVEL"] = "1"
        print("  Skip-class-level mode: LANGCHAIN_SKIP_CLASS_LEVEL=1 set. "
              "Stage 2 will not run the class_level prompt; Stage 3/4 "
              "candidate types filtered to parameter+method only.")

    valid_types = None
    if args.canon_only or args.skip_class_level:
        with open(args.valid_types_file) as f:
            canon39 = set(json.load(f))
        if args.skip_class_level:
            from .constants import LEVEL_TYPES
            pm_types = (set(LEVEL_TYPES["parameter_level"]) |
                        set(LEVEL_TYPES["method_level"])) & canon39
            valid_types = pm_types
            print(f"  Valid types: restricted to {len(valid_types)} parameter+method types "
                  f"(from canonical 39)")
        else:
            valid_types = canon39
            print(f"  Canon-only mode: restricted to {len(valid_types)} types from {args.valid_types_file}")

    skip_keys = None
    existing_per_type = None
    n_existing = 0
    if args.resume:
        existing_jsonl = os.path.join(output_dir, "cases.jsonl")
        skip_keys = load_existing_case_ids(existing_jsonl)
        existing_per_type = load_existing_per_type_counts(existing_jsonl)
        n_existing = len(skip_keys)
        print(f"  Resume mode: skipping {n_existing} case_ids already in {existing_jsonl}")
        if existing_per_type:
            min_done = min(existing_per_type.values()) if existing_per_type else 0
            max_done = max(existing_per_type.values()) if existing_per_type else 0
            print(f"  Existing per-type counts: min={min_done} max={max_done} "
                  f"(quota will be {args.sample_per_type} - existing[T])")

    cases = round_robin_sample(
        dataset,
        n_per_type=args.sample_per_type,
        valid_types=valid_types,
        skip_keys=skip_keys,
        existing_per_type=existing_per_type,
    )
    print(f"  Round-robin sampled: {len(cases)} new cases (target {args.sample_per_type}/type total, "
          f"existing kept: {n_existing})")

    if args.max_examples and len(cases) > args.max_examples:
        cases = cases[: args.max_examples]
        print(f"  Capped to {len(cases)} cases (--max-examples)")

    if not cases:
        print("ERROR: no cases after sampling. Exiting.")
        sys.exit(1)

    no_java = getattr(args, 'no_java_examples', False)
    retriever = None
    if no_java:
        print("\n  ⚠ --no-java-examples: skipping retriever, disabling java tool + inline examples")

        from . import tools as _tools
        _tools.set_retriever(None)
    elif not args.no_retrieval:
        if args.retriever == "headline":
            print("\nLoading HeadlineRetriever (3-channel cosine fusion, "
                  "Same-modal+Desc+GNN learned embedding, weights "
                  "0.345/0.609/0.046)...")
            from .retrieval import HeadlineRetriever
            retriever = HeadlineRetriever(
                java_pool_pkl=args.java_pool_pkl,
                java_pool_dir=args.java_pool_dir,
            )
            retriever._ensure_pool()
            print(f"  ✓ Headline retriever loaded "
                  f"(test 47.26 / bench 41.02 S-R@10, cosine-only, no rerank)")
        elif args.retriever == "upstream":
            print(f"\nLoading UpstreamFacadeRetriever from {args.upstream_config}...")
            from .retrieval import UpstreamFacadeRetriever
            retriever = UpstreamFacadeRetriever(
                config_path=args.upstream_config,
                rerank=args.rerank,
            )
            rerank_suffix = "with frozen bge-gemma rerank" if args.rerank else "rerank=False (bi-encoder only)"
            print(f"  ✓ Upstream cascade retriever loaded (Python→Java per-type, "
                  f"2-channel Same-modal+Desc fusion, {rerank_suffix})")
        else:
            print("\nLoading local JavaExampleRetriever "
                  "(Java→Java mean-pool anchor — legacy semantics)...")
            retriever = JavaExampleRetriever(
                embeddings_path=args.embeddings_path,
                meta_path=args.meta_path,
                java_pool_pkl=args.java_pool_pkl,
                java_pool_dir=args.java_pool_dir,
            )
            retriever._ensure_pool()

        if retriever is not None:
            from . import tools as _tools
            _tools.set_retriever(retriever)

    print("\nConstructing pipeline...")
    pipeline = RefactoringPipeline(
        model=args.model,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        base_url=args.base_url,
        max_java_examples=args.max_java_examples,
        retriever=retriever,
        retrieve_k_3a=args.retrieve_k_3a,
        retrieve_k_3b=args.retrieve_k_3b,
        stage2_max_workers=1,
        no_java_examples=no_java,
    )

    base_llm = pipeline.llm
    tracing_llm = TracingLLM(base_llm)
    pipeline.llm = tracing_llm

    print(f"  Constructing {args.n_workers} per-worker pipeline instances...")
    worker_pipelines = []
    worker_tracers = []
    for w in range(args.n_workers):
        wp = RefactoringPipeline(
            model=args.model,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            base_url=args.base_url,
            max_java_examples=args.max_java_examples,
            retriever=retriever,
            retrieve_k_3a=args.retrieve_k_3a,
            retrieve_k_3b=args.retrieve_k_3b,
            stage2_max_workers=1,
            no_java_examples=no_java,
        )
        wt = TracingLLM(wp.llm)
        wp.llm = wt
        worker_pipelines.append(wp)
        worker_tracers.append(wt)
    print(f"  ✓ {args.n_workers} worker pipelines ready")

    snap_dir = output_dir
    try:
        import shutil
        # Snapshot the constants.py and prompts.py we're using so the run is reproducible
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        for src in [
            os.path.join(pkg_dir, "constants.py"),
            os.path.join(pkg_dir, "prompts.py"),
        ]:
            if os.path.exists(src):
                shutil.copy(src, os.path.join(snap_dir, os.path.basename(src) + ".snapshot"))
        print(f"  ✓ snapshot of constants.py + prompts.py written to {snap_dir}")
    except Exception as e:
        print(f"  WARN: snapshot failed: {e}")

    print(f"\nStarting {args.n_workers}-worker run on {len(cases)} cases...")
    start = time.time()

    asyncio.run(run_async_per_worker(
        worker_pipelines=worker_pipelines,
        worker_tracers=worker_tracers,
        cases=cases,
        output_dir=output_dir,
        verbose=args.verbose,
    ))

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes ({len(cases)} cases, "
          f"{len(cases)/max(elapsed/3600,0.001):.0f} cases/hr).")
    print(f"Results: {output_dir}/cases.jsonl ({len(cases)} lines)")

async def run_async_per_worker(
    *,
    worker_pipelines: List[RefactoringPipeline],
    worker_tracers: List[TracingLLM],
    cases: List[Tuple[str, Dict]],
    output_dir: str,
    verbose: bool,
) -> None:
    """Variant of run_async that uses one (pipeline, tracer) pair per worker."""
    n_workers = len(worker_pipelines)
    case_queue: asyncio.Queue = asyncio.Queue(maxsize=n_workers * 2)
    result_queue: asyncio.Queue = asyncio.Queue()

    workers = []
    for w_id in range(n_workers):
        t = asyncio.create_task(
            case_worker(
                worker_id=w_id,
                case_queue=case_queue,
                result_queue=result_queue,
                pipeline=worker_pipelines[w_id],
                tracing_llm=worker_tracers[w_id],
                verbose=verbose,
            )
        )
        workers.append(t)

    writer_task = asyncio.create_task(
        result_writer(result_queue, output_dir, len(cases))
    )

    for round_idx, (scheduled_type, item) in enumerate(cases):
        await case_queue.put((round_idx, scheduled_type, item))
    for _ in range(n_workers):
        await case_queue.put(None)

    await asyncio.gather(*workers)
    await result_queue.put(None)
    await writer_task

if __name__ == "__main__":
    main()
