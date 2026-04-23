"""
Simplified data loading for the LangChain refactoring-detection pipeline.

Standalone module -- only stdlib + json.  No DSPy dependency.
"""
try:
    import orjson as _json_lib

    # load json file
    def _load_json_file(path):
        with open(path, "rb") as f:
            return _json_lib.loads(f.read())
except ImportError:
    import json as _json_lib

    # load json file
    def _load_json_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return _json_lib.load(f)

import json  # kept for non-file operations
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# STANDALONE HELPER: format Python code for the prompt
def format_python_code(python_case: dict) -> str:
    """Format a Python case dict into a prompt-ready text block.

    Priority:
    1. ``commit_diff`` (full unified diff from git) -- best for detection.
    2. ``code_before`` / ``code_after`` file arrays -- fallback.

    Returns ``"N/A"`` if neither source is available.
    """
    commit_diff = python_case.get("commit_diff", "")
    if commit_diff and commit_diff.strip():
        return f"## Python Commit Diff\n\n```diff\n{commit_diff}\n```"

    code_before = python_case.get("code_before", [])
    code_after = python_case.get("code_after", [])

    if code_before and code_after:
        lines: List[str] = ["## Python Code Changes", ""]
        for before_file, after_file in zip(code_before, code_after):
            file_name = before_file.get("file", "unknown.py")
            before_code = before_file.get("code", "")
            after_code = after_file.get("code", "")

            lines.append(f"### File: {file_name}")
            lines.append("")
            lines.append("**Before:**")
            lines.append("```python")
            lines.append(before_code)
            lines.append("```")
            lines.append("")
            lines.append("**After:**")
            lines.append("```python")
            lines.append(after_code)
            lines.append("```")
            lines.append("")
        return "\n".join(lines)

    return "## Python Code Changes\n\nN/A"

# DATA MANAGER
class DataManager:
    """Load, normalize, filter, sample, and split refactoring benchmark data.

    Supports three benchmark JSON formats:
    * **Old format** -- ``python_refactoring`` wrapper with code_before/code_after
      and ``refactoring_type(s)``; ``java_examples`` at the top level.
    * **New format** -- top-level ``code_before``/``code_after`` with a
      ``refactorings`` array containing ``matched_java_examples`` per entry.
    * **Flat format** -- one record per refactoring with ``refactoring_type``,
      ``code_before``/``code_after`` as strings, and ``matched_java_examples``
      (produced by ``match_examples.py`` without ``--python-source``).

    All formats are normalized to the internal representation::
        {
            "python_case": {
                "refactoring_types": [...],
                "commit_diff": "...",
                "code_before": [...],
                "code_after": [...],
                "commit_message": "",
                "url": "...",
                ...
            },
            "java_examples": [...]
        }

    Args:
        benchmark_file: Path to the benchmark JSON file.
        definitions_file: Optional path to a JSON file mapping refactoring
            type names to human-readable definitions.
    """
    # init
    def __init__(
        self,
        benchmark_file: str,
        definitions_file: Optional[str] = None,
    ) -> None:
        self.benchmark_file = benchmark_file

        self.definitions: Dict[str, str] = {}
        if definitions_file is not None:
            defs_list = _load_json_file(definitions_file)
            self.definitions = {
                d["refactoring_type"]: d["definition"] for d in defs_list
            }

        self.paired_data: List[Dict[str, Any]] = self._load_data(benchmark_file)

    # Format detection + dispatch

    @staticmethod
    def _load_data(benchmark_file: str) -> List[Dict[str, Any]]:
        """Load benchmark JSON and normalize every item to internal format.

        Detects format per-item so mixed-format files are handled correctly.
        """
        data = _load_json_file(benchmark_file)

        processed: List[Dict[str, Any]] = []
        for item in data:
            if "refactorings" in item:
                processed.append(DataManager._load_new_format(item))
            elif "python_refactoring" in item:
                processed.append(DataManager._load_old_format(item))
            elif "refactoring_type" in item and "matched_java_examples" in item:
                processed.append(DataManager._load_flat_format(item))
            else:
                processed.append(DataManager._load_old_format(item))

        return processed

    # Format loaders

    @staticmethod
    def _load_old_format(item: Dict) -> Dict:
        """Load old benchmark format (``python_refactoring`` wrapper).

        Also handles flat detection-input format (commit_level_*_k15.json)
        where all fields are at the top level with no wrapper.
        """
        python_ref = item.get("python_refactoring", {})

        if not python_ref and "refactoring_types" in item and "commit_diff" in item:
            return {
                "python_case": {
                    "refactoring_types": item.get("refactoring_types", []),
                    "commit_diff": item.get("commit_diff", ""),
                    "code_before": item.get("code_before", []),
                    "code_after": item.get("code_after", []),
                    "commit_message": "",
                    "url": item.get("url", ""),
                    "repository": "",
                    "sha1": "",
                },
                "java_examples": [
                    {k: ex.get(k, "") for k in (
                        "refactoring_type", "similarity", "rerank_score",
                        "description", "code_text", "code_before", "code_after", "id",
                    )}
                    for ex in item.get("matched_java_examples", [])
                ],
            }

        code_before_files = python_ref.get("code_before", [])
        code_after_files = python_ref.get("code_after", [])

        diff_parts: List[str] = []
        for before_file, after_file in zip(code_before_files, code_after_files):
            file_name = before_file.get("file", "unknown.py")
            before_code = before_file.get("code", "")
            after_code = after_file.get("code", "")
            diff_parts.append(f"=== {file_name} ===")
            diff_parts.append(f"BEFORE:\n{before_code}")
            diff_parts.append(f"AFTER:\n{after_code}")
            diff_parts.append("---")

        commit_diff = "\n".join(diff_parts)

        if item.get("commit_diff"):
            commit_diff = item["commit_diff"]
        elif python_ref.get("commit_diff"):
            commit_diff = python_ref["commit_diff"]

        ref_types = python_ref.get("refactoring_types", None)
        if ref_types is None:
            ref_type = python_ref.get("refactoring_type", "")
            if isinstance(ref_type, str):
                ref_types = [ref_type] if ref_type else []
            else:
                ref_types = ref_type if ref_type else []

        return {
            "python_case": {
                "refactoring_types": ref_types,
                "commit_diff": commit_diff,
                "code_before": code_before_files,
                "code_after": code_after_files,
                "commit_message": "",
                "url": python_ref.get("url", ""),
            },
            "java_examples": item.get("java_examples", []),
        }

    @staticmethod
    def _load_new_format(item: Dict) -> Dict:
        """Load new benchmark format (top-level code arrays + ``refactorings``)."""
        code_before_files = item.get("code_before", [])
        code_after_files = item.get("code_after", [])

        diff_parts: List[str] = []
        for before_file, after_file in zip(code_before_files, code_after_files):
            file_name = before_file.get("file", "unknown.py")
            before_code = before_file.get("code", "")
            after_code = after_file.get("code", "")
            diff_parts.append(f"=== {file_name} ===")
            diff_parts.append(f"BEFORE:\n{before_code}")
            diff_parts.append(f"AFTER:\n{after_code}")
            diff_parts.append("---")

        commit_diff = "\n".join(diff_parts)

        if item.get("commit_diff"):
            commit_diff = item["commit_diff"]

        refactorings = item.get("refactorings", [])
        ref_types = [r.get("type", "") for r in refactorings if r.get("type")]

        seen_examples: Dict[str, Dict] = {}
        for refactoring in refactorings:
            for java_ex in refactoring.get("matched_java_examples", []):
                ex_id = java_ex.get("id", "")
                if not ex_id:
                    ex_id = (
                        f"{java_ex.get('refactoring_type', '')}"
                        f"_{java_ex.get('description', '')}"
                    )
                if (
                    ex_id not in seen_examples
                    or java_ex.get("similarity", 0)
                    > seen_examples[ex_id].get("similarity", 0)
                ):
                    seen_examples[ex_id] = {
                        "refactoring_type": java_ex.get("refactoring_type", ""),
                        "similarity": java_ex.get("similarity", 0),
                        "rerank_score": java_ex.get("rerank_score", None),
                        "description": java_ex.get("description", ""),
                        "code_before": java_ex.get("code_before", ""),
                        "code_after": java_ex.get("code_after", ""),
                        "code_element_type": java_ex.get("code_element_type", ""),
                        "code_element": java_ex.get("code_element", ""),
                        "repository": java_ex.get("repository", ""),
                        "url": java_ex.get("url", ""),
                        "type_match": java_ex.get("type_match", False),
                        "id": java_ex.get("id", ""),
                    }

        java_examples = list(seen_examples.values())

        return {
            "python_case": {
                "refactoring_types": ref_types,
                "commit_diff": commit_diff,
                "code_before": code_before_files,
                "code_after": code_after_files,
                "commit_message": "",
                "url": item.get("url", ""),
                "repository": item.get("repository", ""),
                "sha1": item.get("sha1", ""),
                "refactorings_detail": refactorings,
            },
            "java_examples": java_examples,
        }

    @staticmethod
    def _load_flat_format(item: Dict) -> Dict:
        """Load flat format from ``match_examples.py`` (no ``--python-source``)."""
        code_before = item.get("code_before", "")
        code_after = item.get("code_after", "")

        if isinstance(code_before, str):
            code_before_files = (
                [{"file": "unknown.py", "code": code_before}] if code_before else []
            )
            code_after_files = (
                [{"file": "unknown.py", "code": code_after}] if code_after else []
            )
            diff_parts: List[str] = []
            if code_before:
                diff_parts.append(f"BEFORE:\n{code_before}")
            if code_after:
                diff_parts.append(f"AFTER:\n{code_after}")
            commit_diff = "\n---\n".join(diff_parts)
        else:
            code_before_files = code_before
            code_after_files = code_after
            diff_parts = []
            for bf, af in zip(code_before_files, code_after_files):
                diff_parts.append(f"=== {bf.get('file', 'unknown.py')} ===")
                diff_parts.append(f"BEFORE:\n{bf.get('code', '')}")
                diff_parts.append(f"AFTER:\n{af.get('code', '')}")
                diff_parts.append("---")
            commit_diff = "\n".join(diff_parts)

        if item.get("commit_diff"):
            commit_diff = item["commit_diff"]

        ref_type = item.get("refactoring_type", "")
        ref_types = [ref_type] if ref_type else []

        java_examples: List[Dict] = []
        for java_ex in item.get("matched_java_examples", []):
            java_examples.append(
                {
                    "refactoring_type": java_ex.get("refactoring_type", ""),
                    "similarity": java_ex.get("similarity", 0),
                    "rerank_score": java_ex.get("rerank_score", None),
                    "description": java_ex.get("description", ""),
                    "code_before": java_ex.get("code_before", ""),
                    "code_after": java_ex.get("code_after", ""),
                    "code_element_type": java_ex.get("code_element_type", ""),
                    "code_element": java_ex.get("code_element", ""),
                    "repository": java_ex.get("repository", ""),
                    "url": java_ex.get("url", ""),
                    "type_match": java_ex.get("type_match", False),
                    "id": java_ex.get("id", ""),
                }
            )

        return {
            "python_case": {
                "refactoring_types": ref_types,
                "commit_diff": commit_diff,
                "code_before": code_before_files,
                "code_after": code_after_files,
                "commit_message": "",
                "url": item.get("url", ""),
                "repository": item.get("repository", ""),
                "sha1": item.get("sha1", ""),
            },
            "java_examples": java_examples,
        }

    # Filtering

    def filter_by_types(self, target_types: List[str]) -> List[Dict]:
        """Return only examples that contain at least one of *target_types*.

        Matching is case-insensitive.  Returns all data when *target_types*
        is empty.
        """
        if not target_types:
            return list(self.paired_data)

        target_normalized = {t.lower().strip() for t in target_types}

        filtered: List[Dict] = []
        for item in self.paired_data:
            gt_types = item["python_case"].get("refactoring_types", [])
            gt_normalized = {t.lower().strip() for t in gt_types}
            if gt_normalized & target_normalized:
                filtered.append(item)
        return filtered

    @staticmethod
    def filter_long_examples(
        data: List[Dict],
        max_tokens: int = 131072,
    ) -> List[Dict]:
        """Filter out examples whose estimated token count exceeds the limit.

        Uses 70 % of *max_tokens* as the threshold to leave room for system
        instructions, few-shot demos, and output formatting.

        Token estimation is character-based (``len(text) // 4``) -- no
        external tokenizer dependency.
        """
        threshold = int(max_tokens * 0.7)

        filtered: List[Dict] = []
        for item in data:
            python_case = item.get("python_case", {})
            python_text = python_case.get("commit_diff", "")

            java_text_parts: List[str] = []
            for ex in item.get("java_examples", []):
                java_text_parts.append(ex.get("code_before", ""))
                java_text_parts.append(ex.get("code_after", ""))
            total_text = python_text + "\n".join(java_text_parts)

            estimated_tokens = len(total_text) // 4
            if estimated_tokens < threshold:
                filtered.append(item)
        return filtered

    @staticmethod
    def filter_by_type_avg_size(
        data: List[Dict],
        multiplier: float = 2.0,
        absolute_max: Optional[int] = None,
    ) -> List[Dict]:
        """Filter examples whose diff size exceeds the per-type average by a multiplier.

        For each refactoring type, computes the mean diff length across all
        examples in *data* that contain that type.  An example is kept only if
        its diff length is at most ``mean * multiplier`` for **every** type it
        contains.  This avoids dropping small-type examples because they share
        a commit with a large-type example.

        Args:
            data: Normalised examples.
            multiplier: Keep examples within ``mean * multiplier`` per type
                (default 2.0 = up to 2× the type average).
            absolute_max: Optional hard cap in characters regardless of type
                average (e.g., 200_000).  Applied after the per-type check.

        Returns:
            Filtered list.  Prints per-type threshold and removal stats.
        """
        import statistics

        type_sizes: Dict[str, List[int]] = defaultdict(list)
        item_sizes: List[int] = []
        for item in data:
            diff = item.get("python_case", {}).get("commit_diff", "")
            sz = len(diff)
            item_sizes.append(sz)
            for t in item.get("python_case", {}).get("refactoring_types", []):
                type_sizes[t].append(sz)

        type_threshold: Dict[str, float] = {}
        for t, sizes in type_sizes.items():
            avg = statistics.mean(sizes)
            type_threshold[t] = avg * multiplier

        kept: List[Dict] = []
        removed = 0
        for item, sz in zip(data, item_sizes):
            if absolute_max is not None and sz > absolute_max:
                removed += 1
                continue

            types = item.get("python_case", {}).get("refactoring_types", [])
            if not types:
                kept.append(item)
                continue

            within = all(
                sz <= type_threshold.get(t, float("inf")) for t in types
            )
            if within:
                kept.append(item)
            else:
                removed += 1

        print(
            f"  Type-avg filter (×{multiplier}"
            + (f", cap {absolute_max:,}" if absolute_max else "")
            + f"): {len(data)} → {len(kept)} ({removed} removed)"
        )
        return kept

    # Sampling

    @staticmethod
    def sample_per_type(
        data: List[Dict],
        n: int,
        seed: int = 42,
    ) -> List[Dict]:
        """Sample at most *n* examples for each known refactoring type.

        Groups examples by their ground-truth refactoring types, samples up to
        *n* per type, then deduplicates (an example with multiple types may
        appear in multiple groups but is only included once in the result).

        Returns a deterministically-shuffled list.
        """
        rng = random.Random(seed)

        by_type: Dict[str, List[Dict]] = defaultdict(list)
        for item in data:
            for rt in item["python_case"].get("refactoring_types", []):
                by_type[rt].append(item)

        seen_ids: set = set()
        sampled: List[Dict] = []

        for rt in sorted(by_type.keys()):
            candidates = by_type[rt]
            rng.shuffle(candidates)
            for item in candidates[:n]:
                item_id = id(item)
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    sampled.append(item)

        rng.shuffle(sampled)
        return sampled

    # Splitting

    @staticmethod
    def split(
        data: List[Dict],
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict]]:
        """URL-grouped train/val split.

        All examples sharing the same commit URL are placed in the same
        partition so the same commit never appears in both train and val.

        Args:
            data: List of normalized examples.
            val_fraction: Fraction of unique URLs to hold out for validation.
            seed: Random seed for reproducibility.

        Returns:
            ``(train_data, val_data)`` tuple.
        """
        rng = random.Random(seed)

        by_url: Dict[str, List[Dict]] = defaultdict(list)
        for item in data:
            url = item["python_case"].get("url", "")
            by_url[url].append(item)

        urls = list(by_url.keys())
        rng.shuffle(urls)

        split_idx = max(1, int(len(urls) * (1.0 - val_fraction)))
        train_urls = set(urls[:split_idx])

        train_data: List[Dict] = []
        val_data: List[Dict] = []

        for url, items in by_url.items():
            if url in train_urls:
                train_data.extend(items)
            else:
                val_data.extend(items)

        return train_data, val_data
