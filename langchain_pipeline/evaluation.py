"""
Standalone evaluation metrics for multi-label refactoring detection.

No DSPy dependency.  Imports only from sibling ``constants`` module and
the standard library.
"""
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Set

from .constants import ALL_KNOWN_TYPES

# KNOWN REFACTORING TYPES (39 types from RefactoringMiner)

KNOWN_REFACTORING_TYPES: Set[str] = {
    "Extract Method",
    "Rename Class",
    "Move Attribute",
    "Rename Method",
    "Inline Method",
    "Move Method",
    "Move And Rename Method",
    "Pull Up Method",
    "Move Class",
    "Move And Rename Class",
    "Pull Up Attribute",
    "Push Down Attribute",
    "Push Down Method",
    "Extract Superclass",
    "Extract Subclass",
    "Extract Class",
    "Extract And Move Method",
    "Extract Variable",
    "Inline Variable",
    "Rename Variable",
    "Rename Parameter",
    "Rename Attribute",
    "Split Parameter",
    "Replace Variable With Attribute",
    "Replace Attribute With Variable",
    "Parameterize Variable",
    "Localize Parameter",
    "Parameterize Attribute",
    "Change Variable Type",
    "Add Method Annotation",
    "Remove Method Annotation",
    "Add Class Annotation",
    "Add Parameter",
    "Remove Parameter",
    "Reorder Parameter",
    "Encapsulate Attribute",
    "Split Conditional",
    "Invert Condition",
    "Move Code",
}

# Pre-computed normalized lookup set
_KNOWN_TYPES_NORMALIZED: Set[str] = set()

# NORMALIZATION HELPERS
def normalize_type(type_name: str) -> str:
    """Make two type names comparable by stripping whitespace and lowercasing.

    Stops 'Rename Method' from mismatching 'rename method' or 'Rename  Method'.
    """
    if not type_name:
        return ""

    normalized = type_name.strip().lower()

    normalized = " ".join(normalized.split())
    return normalized

# build normalized set
def _build_normalized_set() -> Set[str]:
    return {normalize_type(t) for t in KNOWN_REFACTORING_TYPES}

_KNOWN_TYPES_NORMALIZED = _build_normalized_set()


def is_known_type(type_name: str) -> bool:
    """True if `type_name` is one of the 39 canonical refactoring types."""
    return normalize_type(type_name) in _KNOWN_TYPES_NORMALIZED


def find_matching_type(detected: str, ground_truth_set: Set[str]) -> str:
    """Match a detected type against the GT set (case-/space-insensitive).

    Returns the GT string in its original form (so we keep the original
    spelling), or "" if no match.
    """
    detected_norm = normalize_type(detected)

    for gt in ground_truth_set:
        if normalize_type(gt) == detected_norm:
            return gt

    return ""

# PER-EXAMPLE METRICS
def compute_multilabel_metrics(
    detected_types: List[str],
    ground_truth_types: List[str],
) -> Dict[str, Any]:
    """Compare what we predicted vs the truth, count TP/FP/FN.

    Multi-label means a single commit can have multiple correct types,
    so we work with sets, not single labels.
    """
    detected_set = set(detected_types)
    gt_set = set(ground_truth_types)

    matched_gt: Set[str] = set()
    matched_detected: Set[str] = set()

    for det in detected_set:
        match = find_matching_type(det, gt_set)
        if match:
            matched_gt.add(match)
            matched_detected.add(det)

    tp = len(matched_detected)

    fp = len(detected_set) - len(matched_detected)

    fn = len(gt_set) - len(matched_gt)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "detected": list(detected_set),
        "ground_truth": list(gt_set),
        "matched": list(matched_detected),
        "missed": list(gt_set - matched_gt),
        "false_alarms": list(detected_set - matched_detected),
    }

# PER-TYPE METRICS
def compute_per_type_metrics(
    results: List[Dict[str, Any]],
    known_only: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Per-type breakdown — for every refactoring type, count TP/FP/FN/TN.

    Used to see which types the pipeline does well on (e.g. Rename Method)
    vs poorly on (e.g. Extract Method).
    """
    all_types: Set[str] = set()
    for r in results:
        if r.get("status") in ("SKIPPED", "ERROR"):
            continue
        if known_only:
            gt = r.get("ground_truth_known", r.get("ground_truth", []))
            det = r.get("detected_known", r.get("detected", []))
        else:
            gt = r.get("ground_truth", [])
            det = r.get("detected", [])
        all_types.update(gt)
        all_types.update(det)

    if known_only:
        all_types = {t for t in all_types if is_known_type(t)}

    per_type: Dict[str, Dict[str, Any]] = {}

    for ref_type in sorted(all_types):
        tp = fp = fn = tn = 0
        type_norm = normalize_type(ref_type)

        for r in results:
            if r.get("status") in ("SKIPPED", "ERROR"):
                continue

            if known_only:
                gt_types = set(r.get("ground_truth_known", r.get("ground_truth", [])))
                det_types = set(r.get("detected_known", r.get("detected", [])))
            else:
                gt_types = set(r.get("ground_truth", []))
                det_types = set(r.get("detected", []))

            gt_norm = {normalize_type(t) for t in gt_types}
            det_norm = {normalize_type(t) for t in det_types}

            is_in_gt = type_norm in gt_norm
            is_detected = type_norm in det_norm

            if is_detected and is_in_gt:
                tp += 1
            elif is_detected and not is_in_gt:
                fp += 1
            elif not is_detected and is_in_gt:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_type[ref_type] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
        }

    return per_type

# PIPELINE EVALUATION
def _process_single_example(
    i: int,
    item: Dict[str, Any],
    predict_fn: Callable,
    verbose: bool,
) -> Dict[str, Any]:
    """Run the pipeline on ONE case, score it, return everything as a dict.

    This is what each worker calls per case.
    """
    python_case = item["python_case"]
    java_examples = item.get("java_examples", [])

    gt_types = python_case.get("refactoring_types", [])
    if isinstance(gt_types, str):
        gt_types = [gt_types]

    trace = None
    try:
        result_or_trace = predict_fn(python_case, java_examples)
        if isinstance(result_or_trace, dict) and "final_types" in result_or_trace:
            trace = result_or_trace
            detected_types: List[str] = result_or_trace["final_types"]
        else:
            detected_types: List[str] = result_or_trace
    except Exception as exc:
        if verbose:
            print(f"  [{i}] ERROR - {exc}")

        gt_known = [t for t in gt_types if is_known_type(t)]
        return {
            "index": i,
            "url": python_case.get("url", ""),
            "status": "ERROR",
            "error": str(exc),
            "ground_truth": gt_types,
            "ground_truth_known": gt_known,
            "detected": [],
            "detected_known": [],
            "detected_undefined": [],
            "tp": 0,
            "fp": 0,
            "fn": len(gt_known),
            "matched": [],
            "missed": gt_known,
            "false_alarms": [],
        }

    detected_known = [t for t in detected_types if is_known_type(t)]
    detected_undefined = [t for t in detected_types if not is_known_type(t)]

    gt_known = [t for t in gt_types if is_known_type(t)]

    metrics = compute_multilabel_metrics(detected_known, gt_known)

    if verbose:
        status = "OK" if metrics["fp"] == 0 and metrics["fn"] == 0 else "XX"
        undef_str = (
            f" [+{len(detected_undefined)} undefined]" if detected_undefined else ""
        )
        print(
            f"  [{i}] {status} Detected: {detected_known}, "
            f"GT: {gt_known}{undef_str}"
        )

    result_dict = {
        "index": i,
        "url": python_case.get("url", ""),
        "detected": detected_types,
        "detected_known": detected_known,
        "detected_undefined": detected_undefined,
        "ground_truth": gt_types,
        "ground_truth_known": gt_known,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "matched": metrics["matched"],
        "missed": metrics["missed"],
        "false_alarms": metrics["false_alarms"],
    }
    if trace is not None:
        result_dict["trace"] = trace
    return result_dict


def evaluate_pipeline(
    predict_fn: Callable,
    dataset: List[Dict[str, Any]],
    verbose: bool = False,
    num_threads: int = 1,
) -> Dict[str, Any]:
    """Run the pipeline on ALL cases (sequentially or in parallel) and aggregate.

    Returns the big dict that print_evaluation_report() / save_evaluation_results()
    consume.
    """
    if num_threads <= 1:
        results: List[Dict[str, Any]] = []
        for i, item in enumerate(dataset):
            result = _process_single_example(i, item, predict_fn, verbose)
            results.append(result)
    else:
        results = [None] * len(dataset)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(
                    _process_single_example, i, item, predict_fn, verbose
                ): i
                for i, item in enumerate(dataset)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = {
                        "index": idx,
                        "status": "ERROR",
                        "error": str(exc),
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "detected_known": [],
                        "detected_undefined": [],
                        "ground_truth_known": [],
                        "matched": [],
                        "missed": [],
                        "false_alarms": [],
                    }

    known_tp = 0
    known_fp = 0
    known_fn = 0
    skipped = 0
    undefined_detections: List[str] = []

    for r in results:
        if r.get("status") == "SKIPPED":
            skipped += 1
            continue
        if r.get("status") == "ERROR":
            known_fn += r.get("fn", 0)
            continue
        known_tp += r["tp"]
        known_fp += r["fp"]
        known_fn += r["fn"]
        undefined_detections.extend(r.get("detected_undefined", []))

    known_precision = (
        known_tp / (known_tp + known_fp) if (known_tp + known_fp) > 0 else 0.0
    )
    known_recall = (
        known_tp / (known_tp + known_fn) if (known_tp + known_fn) > 0 else 0.0
    )
    known_f1 = (
        2 * known_precision * known_recall / (known_precision + known_recall)
        if (known_precision + known_recall) > 0
        else 0.0
    )

    known_per_type = compute_per_type_metrics(results, known_only=True)

    known_accuracy = 0.0
    if known_per_type:
        known_accuracy = sum(m["accuracy"] for m in known_per_type.values()) / len(
            known_per_type
        )

    undefined_counts: Dict[str, int] = {}
    for t in undefined_detections:
        undefined_counts[t] = undefined_counts.get(t, 0) + 1

    return {
        "total_examples": len(dataset),
        "skipped": skipped,
        "known_types": {
            "total_tp": known_tp,
            "total_fp": known_fp,
            "total_fn": known_fn,
            "precision": known_precision,
            "recall": known_recall,
            "f1": known_f1,
            "accuracy": known_accuracy,
            "per_type_metrics": known_per_type,
        },
        "undefined_types": {
            "total_detected": len(undefined_detections),
            "unique_types": len(undefined_counts),
            "per_type_counts": dict(
                sorted(undefined_counts.items(), key=lambda x: -x[1])
            ),
        },
        "detailed_results": results,
    }

# REPORTING
def print_evaluation_report(eval_results: Dict[str, Any]) -> None:
    """Pretty-print the evaluation results to the terminal (for human review)."""
    known = eval_results["known_types"]
    undefined = eval_results["undefined_types"]

    print()
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total examples: {eval_results['total_examples']}")
    print(f"Skipped:        {eval_results['skipped']}")

    print()
    print("--- Known Types (RQ1) ---")
    print(f"True Positives:  {known['total_tp']}")
    print(f"False Positives: {known['total_fp']}")
    print(f"False Negatives: {known['total_fn']}")
    print()
    print(f"Precision: {known['precision']:.1%}")
    print(f"Recall:    {known['recall']:.1%}")
    print(f"F1 Score:  {known['f1']:.1%}")
    print(f"Accuracy:  {known['accuracy']:.1%}")

    per_type = known.get("per_type_metrics", {})
    if per_type:
        print()
        print("--- Per-Type Breakdown ---")
        print(f"{'Type':<35s} {'TP':>4s} {'FP':>4s} {'FN':>4s}  {'P':>6s} {'R':>6s} {'F1':>6s}")
        print("-" * 75)
        for t in sorted(per_type.keys()):
            m = per_type[t]

            if m["tp"] + m["fp"] + m["fn"] == 0:
                continue
            print(
                f"{t:<35s} {m['tp']:4d} {m['fp']:4d} {m['fn']:4d}  "
                f"{m['precision']:6.1%} {m['recall']:6.1%} {m['f1']:6.1%}"
            )

    print()
    print("--- Undefined Types (RQ2) ---")
    print(f"Total undefined detections: {undefined['total_detected']}")
    print(f"Unique undefined types:     {undefined['unique_types']}")
    if undefined["per_type_counts"]:
        print("Top undefined types:")
        for t, count in list(undefined["per_type_counts"].items())[:10]:
            print(f"  {t}: {count}")

    print("=" * 60)

# SAVING RESULTS
def save_evaluation_results(
    eval_results: Dict[str, Any],
    output_dir: str,
    timestamp: str,
) -> None:
    """Save the evaluation results to JSON on disk for later analysis."""
    os.makedirs(output_dir, exist_ok=True)

    summary = {k: v for k, v in eval_results.items() if k != "detailed_results"}
    summary_path = os.path.join(output_dir, f"evaluation_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Evaluation summary saved to {summary_path}")

    detailed = eval_results.get("detailed_results", [])
    detailed_path = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, default=str)
    print(f"Detailed results saved to {detailed_path}")
