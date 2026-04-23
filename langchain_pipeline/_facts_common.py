"""Shared cross-file move/rename helpers for fact-extractor wrappers.

Mirrors the algorithm in structural_facts.compute_facts so all four parser
backends (gumtree, treesitter, libcst, parso) agree on cross-file behavior.
"""
import difflib
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

SIM_THRESHOLD_FILE = 0.55
SIM_THRESHOLD_CODE = 0.65

_FAILURE_LOG = Path(__file__).resolve().parent / ".fact_failures.jsonl"

# Append one JSON line to .fact_failures.jsonl
def log_failure(source: str, file: str, err: BaseException, code: str = "") -> None:
    """Append one JSON line to .fact_failures.jsonl. Never raises."""
    try:
        sha = hashlib.sha1((code or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
        rec = {
            "source": source,
            "file": file,
            "error": str(err)[:500],
            "error_type": type(err).__name__,
            "sha1": sha,
            "ts": time.time(),
        }
        with _FAILURE_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass

# ratio
def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def match_file_renames(
    removed: List[str],
    added: List[str],
    src_before: Dict[str, str],
    src_after: Dict[str, str],
    threshold: float = SIM_THRESHOLD_FILE,
) -> List[Tuple[str, str, float]]:
    """Greedy best-match removed->added by source-text similarity."""
    pairs: List[Tuple[str, str, float]] = []
    used: Set[str] = set()
    for r in removed:
        best = (0.0, "")
        for a in added:
            if a in used:
                continue
            s = _ratio(src_before.get(r, ""), src_after.get(a, ""))
            if s > best[0]:
                best = (s, a)
        if best[0] >= threshold and best[1]:
            pairs.append((r, best[1], best[0]))
            used.add(best[1])
    return pairs


def cross_file_class_moves(
    classes_before: Dict[Tuple[str, str], Dict],
    classes_after: Dict[Tuple[str, str], Dict],
    rename_to_from: Dict[str, str],
    rename_from_to: Dict[str, str],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Returns (moved, added, removed) class lists.

    Each class entry is a dict {name, file, body_src}.
    """
    moved: List[Dict] = []
    added: List[Dict] = []
    removed: List[Dict] = []

    cb_by_name: Dict[str, List[Tuple[str, Dict]]] = {}
    for (p, n), c in classes_before.items():
        cb_by_name.setdefault(n, []).append((p, c))
    ca_by_name: Dict[str, List[Tuple[str, Dict]]] = {}
    for (p, n), c in classes_after.items():
        ca_by_name.setdefault(n, []).append((p, c))

    for n, after_pairs in ca_by_name.items():
        before_pairs = cb_by_name.get(n, [])
        used: Set[str] = set()
        for pa, ca in after_pairs:
            same = next((pb for pb, _ in before_pairs if pb == pa or rename_to_from.get(pa) == pb), None)
            if same is not None:
                continue
            best = (0.0, "")
            for pb, cb_ in before_pairs:
                if pb in used:
                    continue
                s = _ratio(cb_.get("body_src", ""), ca.get("body_src", ""))
                if s > best[0]:
                    best = (s, pb)
            if best[0] >= SIM_THRESHOLD_CODE:
                moved.append({"name": n, "from_file": best[1], "to_file": pa,
                              "similarity": round(best[0], 2)})
                used.add(best[1])
            else:
                added.append({"name": n, "file": pa})
        for pb, _ in before_pairs:
            if pb in used:
                continue
            same_after = next((pa for pa, _ in after_pairs if pa == pb or rename_from_to.get(pb) == pa), None)
            if same_after is not None:
                continue
            removed.append({"name": n, "file": pb})

    return moved, added, removed


def cross_file_method_moves(
    methods_before: Dict[Tuple[str, Optional[str], str], Dict],
    methods_after: Dict[Tuple[str, Optional[str], str], Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Returns (moved, added, removed) method lists.

    Each method entry is a dict {name, cls, file, body_src}.
    """
    moved: List[Dict] = []
    added: List[Dict] = []
    removed: List[Dict] = []

    keys_before = set(methods_before.keys())
    keys_after = set(methods_after.keys())

    by_name_after: Dict[str, List[Tuple[str, Optional[str], Dict]]] = {}
    for (p, cn, n), m in methods_after.items():
        by_name_after.setdefault(n, []).append((p, cn, m))
    by_name_before: Dict[str, List[Tuple[str, Optional[str], Dict]]] = {}
    for (p, cn, n), m in methods_before.items():
        by_name_before.setdefault(n, []).append((p, cn, m))

    for n, after_list in by_name_after.items():
        before_list = by_name_before.get(n, [])
        used: Set[Tuple[str, Optional[str]]] = set()
        for pa, cna, ma in after_list:
            if (pa, cna, n) in keys_before:
                continue
            best = (0.0, None, None)
            for pb, cnb, mb in before_list:
                if (pb, cnb) in used or (pb, cnb, n) in keys_after:
                    continue
                s = _ratio(mb.get("body_src", ""), ma.get("body_src", ""))
                if s > best[0]:
                    best = (s, pb, cnb)
            if best[0] >= SIM_THRESHOLD_CODE and best[1] is not None:
                moved.append({
                    "name": n, "from_class": best[2], "to_class": cna,
                    "from_file": best[1], "to_file": pa,
                    "similarity": round(best[0], 2),
                })
                used.add((best[1], best[2]))
            else:
                added.append({"name": n, "class": cna, "file": pa})
        for pb, cnb, mb in before_list:
            if (pb, cnb) in used or (pb, cnb, n) in keys_after:
                continue
            removed.append({"name": n, "class": cnb, "file": pb})

    return moved, added, removed

# empty facts
def empty_facts() -> Dict:
    return {
        "files": {"added": [], "removed": [], "renamed": []},
        "classes": {"added": [], "removed": [], "moved": []},
        "methods": {"added": [], "removed": [], "moved": []},
        "signatures": {"changed": []},
        "inheritance": {"changed": []},
        "attributes": {"added": [], "removed": []},
    }


def assemble_facts(
    src_before: Dict[str, str],
    src_after: Dict[str, str],
    file_classes_before: Dict[str, Dict[str, Dict]],
    file_classes_after: Dict[str, Dict[str, Dict]],
    file_funcs_before: Dict[str, Dict[str, Dict]],
    file_funcs_after: Dict[str, Dict[str, Dict]],
) -> Dict:
    """Take per-file class/func extractions and produce the full facts dict.

    Each Dict entry under file_classes_* is keyed by class name -> dict with:
        {name, bases (list[str]), attrs (set[str]), methods (dict[name->method dict]),
         body_src (str)}.
    Each method dict and each top-level func dict has:
        {name, cls (Optional[str]), file, params (list[str]), body_src (str)}.
    """
    files_before = set(src_before)
    files_after = set(src_after)
    added_files = sorted(files_after - files_before)
    removed_files = sorted(files_before - files_after)
    renames = match_file_renames(removed_files, added_files, src_before, src_after)
    rename_from_to = {r: a for r, a, _ in renames}
    rename_to_from = {a: r for r, a, _ in renames}
    added_files_net = [f for f in added_files if f not in rename_to_from]
    removed_files_net = [f for f in removed_files if f not in rename_from_to]

    facts = empty_facts()
    facts["files"] = {
        "added": added_files_net,
        "removed": removed_files_net,
        "renamed": [{"from": r, "to": a, "similarity": round(s, 2)} for r, a, s in renames],
    }

    common = (files_before & files_after) | set(rename_from_to.values())
    for p_after in sorted(common):
        p_before = rename_to_from.get(p_after, p_after)
        cls_b_map = file_classes_before.get(p_before, {})
        cls_a_map = file_classes_after.get(p_after, {})
        for cname, cls_b in cls_b_map.items():
            if cname not in cls_a_map:
                continue
            cls_a = cls_a_map[cname]
            if cls_b.get("bases", []) != cls_a.get("bases", []):
                facts["inheritance"]["changed"].append({
                    "class": cname, "file": p_after,
                    "before_bases": list(cls_b.get("bases", [])),
                    "after_bases": list(cls_a.get("bases", [])),
                })
            attrs_b = set(cls_b.get("attrs", set()))
            attrs_a = set(cls_a.get("attrs", set()))
            for at in sorted(attrs_a - attrs_b):
                facts["attributes"]["added"].append({"class": cname, "file": p_after, "name": at})
            for at in sorted(attrs_b - attrs_a):
                facts["attributes"]["removed"].append({"class": cname, "file": p_after, "name": at})
            mb_map = cls_b.get("methods", {})
            ma_map = cls_a.get("methods", {})
            for mname, mb in mb_map.items():
                if mname in ma_map:
                    ma = ma_map[mname]
                    if mb.get("params", []) != ma.get("params", []):
                        facts["signatures"]["changed"].append({
                            "method": f"{cname}.{mname}", "file": p_after,
                            "before_params": list(mb.get("params", [])),
                            "after_params": list(ma.get("params", [])),
                        })

    classes_before = {(p, n): c for p, m in file_classes_before.items() for n, c in m.items()}
    classes_after = {(p, n): c for p, m in file_classes_after.items() for n, c in m.items()}
    cmoved, cadded, cremoved = cross_file_class_moves(
        classes_before, classes_after, rename_to_from, rename_from_to,
    )
    facts["classes"]["moved"] = cmoved
    facts["classes"]["added"] = cadded
    facts["classes"]["removed"] = cremoved

    methods_before: Dict[Tuple[str, Optional[str], str], Dict] = {}
    methods_after: Dict[Tuple[str, Optional[str], str], Dict] = {}
    for p, cls_map in file_classes_before.items():
        for cn, cls in cls_map.items():
            for mn, m in cls.get("methods", {}).items():
                methods_before[(p, cn, mn)] = m
    for p, cls_map in file_classes_after.items():
        for cn, cls in cls_map.items():
            for mn, m in cls.get("methods", {}).items():
                methods_after[(p, cn, mn)] = m
    for p, fn_map in file_funcs_before.items():
        for fn, m in fn_map.items():
            methods_before[(p, None, fn)] = m
    for p, fn_map in file_funcs_after.items():
        for fn, m in fn_map.items():
            methods_after[(p, None, fn)] = m

    mmoved, madded, mremoved = cross_file_method_moves(methods_before, methods_after)
    facts["methods"]["moved"] = mmoved
    facts["methods"]["added"] = madded
    facts["methods"]["removed"] = mremoved

    return facts
