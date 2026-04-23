#!/usr/bin/env python3
"""
Preprocess benchmark JSON: clean code in all fields to reduce tokens.

Cleans:
  - code_before[i].code  (Python) — minify_python
  - code_after[i].code   (Python) — minify_python
  - refactorings[j].matched_java_examples[k].code_before (Java mixed) — split header, minify_java
  - refactorings[j].matched_java_examples[k].code_after  (Java) — minify_java
  - commit_diff is NOT touched (diff markers are structural)

Usage:
    python -m langchain_pipeline.preprocess \
        --input  /path/to/matched_mined_diffs_all_v1.json \
        --output /path/to/matched_mined_diffs_all_v1_cleaned.json \
        --workers 8
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any

# Import cleanup functions — handle both package and direct execution
try:
    from .code_cleanup import minify_python, basic_cleanup, minify_java
except ImportError:
    from code_cleanup import minify_python, basic_cleanup, minify_java

# Java code_before splitting
def _clean_java_code_before(text: str) -> str:
    """Clean a Java example's code_before field.

    Format is typically:
        [Type] Description...\n\nBEFORE:\n// FILE: path\n<java code>

    We preserve the description header and clean only the code portion.
    """
    if not text or not text.strip():
        return text

    parts = text.split("BEFORE:\n", 1)
    if len(parts) == 2:
        header = parts[0] + "BEFORE:\n"
        code = parts[1]
        cleaned_code = minify_java(code)
        return header + cleaned_code

    parts = text.split("// FILE:", 1)
    if len(parts) == 2:
        header = parts[0]
        code = "// FILE:" + parts[1]
        cleaned_code = minify_java(code)
        return header + cleaned_code

    return basic_cleanup(text)


def _clean_java_code_after(text: str) -> str:
    """Clean a Java example's code_after field."""
    if not text or not text.strip():
        return text
    return minify_java(text)

# Per-example processing

# Clean all code fields in a single example
def _process_example(item: Dict[str, Any]) -> Dict[str, Any]:
    """Clean all code fields in a single example. Returns the modified example."""
    chars_before = 0
    chars_after = 0

    for cb in item.get("code_before", []):
        code = cb.get("code", "")
        chars_before += len(code)
        if code.strip():
            cb["code"] = minify_python(code)
        chars_after += len(cb.get("code", ""))

    for ca in item.get("code_after", []):
        code = ca.get("code", "")
        chars_before += len(code)
        if code.strip():
            ca["code"] = minify_python(code)
        chars_after += len(ca.get("code", ""))

    for ref in item.get("refactorings", []):
        for jex in ref.get("matched_java_examples", []):
            jcb = jex.get("code_before", "")
            chars_before += len(jcb)
            if jcb.strip():
                jex["code_before"] = _clean_java_code_before(jcb)
            chars_after += len(jex.get("code_before", ""))

            jca = jex.get("code_after", "")
            chars_before += len(jca)
            if jca.strip():
                jex["code_after"] = _clean_java_code_after(jca)
            chars_after += len(jex.get("code_after", ""))

    return item, chars_before, chars_after

# Main

# main
def main():
    parser = argparse.ArgumentParser(description="Preprocess benchmark JSON: clean code fields.")
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output cleaned JSON")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    workers = args.workers or os.cpu_count() or 4

    print(f"Loading {args.input} ...")
    t0 = time.time()
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} examples in {time.time()-t0:.1f}s")

    print(f"Cleaning code with {workers} workers ...")
    t1 = time.time()

    total_before = 0
    total_after = 0
    cleaned_data = [None] * len(data)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_example, item): i for i, item in enumerate(data)}
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                item, cb, ca = future.result()
                cleaned_data[idx] = item
                total_before += cb
                total_after += ca
            except Exception as e:
                print(f"  ERROR on example {idx}: {e}")
                cleaned_data[idx] = data[idx]
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(data)} done ...")

    elapsed = time.time() - t1
    reduction = (1 - total_after / total_before) * 100 if total_before > 0 else 0

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Code chars: {total_before:,} → {total_after:,} ({reduction:.1f}% reduction)")

    print(f"Writing {args.output} ...")
    t2 = time.time()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False)
    print(f"  Written in {time.time()-t2:.1f}s")
    print(f"  File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
