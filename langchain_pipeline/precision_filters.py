"""Precision-mode filter helpers (M1.L levers 2, 3, 6, bottom-types rescue).

All gating rules used by the pipeline when `LANGCHAIN_PRECISION_MODE=1`. Each
rule is rule-based (no training); calibration comes from pre-built artifacts
mined from sha1-disjoint training data:
  - `experiments/output/type_precision_history.json`  — per-type historical
      precision (built by A9_per_type_precision_history.py).
  - `experiments/output/type_assoc_v4.json`           — FP-Growth association
      rules keyed by singleton + pair antecedents.

Consumed from within `pipeline.py`. No LLM dependency — pure-Python filters.
"""
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent

# Types that live inside method bodies — Lever 6 does NOT gate these (they're
# always class/method-structure-invisible, so corroboration must come from
# <intra_method_signals>, not <structural_facts>).
_CLASS_LEVEL_GATED = {
    "Extract Class", "Extract Subclass", "Extract Superclass",
    "Move Class", "Rename Class", "Move And Rename Class",
    "Pull Up Method", "Pull Up Attribute",
    "Push Down Method", "Push Down Attribute",
}

# Bottom-types rescue: intra-method refactorings each need a specific signal
# from `<intra_method_signals>` to survive precision mode. Absent signal ⇒ drop.
# Maps type → XML-tag name emitted by intra_method_signals.py.
_INTRA_METHOD_GATE: Dict[str, str] = {
    "Rename Variable": "rename",
    "Rename Parameter": "rename",
    "Rename Attribute": "rename",
    "Replace Variable With Attribute": "swap",
    "Replace Attribute With Variable": "swap",
    "Invert Condition": "invert",
    "Split Conditional": "split",
    "Change Variable Type": "type_change",
    "Extract Variable": "extract",
    "Inline Variable": "inline",
}

# Two-step types require BOTH primitive signals in `<structural_facts>`.
# `("moved", "added")` means the structural block must show both a move and
# an add in the same commit.
_TWO_STEP_GATE: Dict[str, Tuple[str, ...]] = {
    "Extract And Move Method": ("moved", "added"),
    "Move And Rename Class": ("moved", "renamed"),
    "Move And Inline Method": ("moved", "removed"),
    "Move And Rename Method": ("moved", "renamed"),
}

# Historical-P floor below which a type is dropped from Stage-3 candidates
# when precision mode is on. Calibrated to kill noisy types without touching
# medium-precision frequent types like Rename Method (0.46).
PRECISION_FLOOR = float(os.environ.get("LANGCHAIN_PRECISION_FLOOR", "0.30"))

# Rule-strength threshold for sole-Stage-3 additions. Additions with
# `max_rule_strength < TAU` AND not confirmed by Stage 2 are dropped.
RULE_STRENGTH_TAU = float(os.environ.get("LANGCHAIN_RULE_STRENGTH_TAU", "0.05"))

# Load the per-type historical-precision table from disk (only once thanks to lru_cache).
@lru_cache(maxsize=1)
def _load_precision_history() -> Dict[str, Dict]:
    path = ROOT / "experiments" / "output" / "type_precision_history.json"

    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)

    return data.get("per_type", {}) if isinstance(data, dict) else {}

# Load the FP-Growth association rules table from disk (also cached).
@lru_cache(maxsize=1)
def _load_assoc_table() -> Dict:
    path = ROOT / "experiments" / "output" / "type_assoc_v4.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def historical_precision(type_name: str, min_support: int = 3) -> Optional[float]:
    """How accurate the pipeline historically is on this type (or None if too rare)."""
    hist = _load_precision_history()
    row = hist.get(type_name)

    if not row or row.get("support", 0) < min_support:
        return None
    return float(row.get("precision", 0.0))


def max_rule_strength(detected_set: Set[str], target: str) -> float:
    """How strongly the already-detected types predict `target`.

    Looks up FP-Growth association rules. Higher = more confident `target`
    co-occurs with what we already detected.
    """
    table = _load_assoc_table()

    if not table:
        return 0.0
    best = 0.0
    detected_l = sorted(detected_set)

    keys: List[str] = list(detected_l)
    for i in range(len(detected_l)):
        for j in range(i + 1, len(detected_l)):
            keys.append("|".join(sorted([detected_l[i], detected_l[j]])))

    for k in keys:
        for r in table.get(k, []):
            if r.get("consequent") == target and r.get("score", 0.0) > best:
                best = float(r["score"])
    return best


def filter_stage3_candidates(
    candidates: List[str],
    detected_set: Set[str],
) -> Tuple[List[str], List[Dict]]:
    """Drop chronically-bad types from the Stage-3 candidate list.

    A candidate type whose historical precision is below the floor is
    dropped — it's almost always wrong, so don't even ask the LLM about it.
    """
    kept: List[str] = []
    dropped: List[Dict] = []

    for t in candidates:
        p = historical_precision(t)

        if p is not None and p < PRECISION_FLOOR:
            dropped.append({"type": t, "reason": "historical_P<floor",
                            "precision": round(p, 3)})
            continue
        kept.append(t)
    return kept, dropped

# Structural / intra-method corroboration

_STRUCTURAL_TAG_RE = re.compile(
    r"<(added|removed|moved|renamed|changed)\b",
    re.IGNORECASE,
)
_INTRA_TAG_RE = re.compile(
    r"<(rename|swap|invert|split|type_change|extract|inline)\b",
    re.IGNORECASE,
)


def structural_signal_present(structural_xml: str, tags: Tuple[str, ...]) -> bool:
    """Did the structural facts mention any of these tags?

    Quick check used by gates to see if e.g. '<methods>' appears in the
    gumtree XML.
    """
    if not structural_xml or not tags:
        return False
    low = structural_xml.lower()

    return all(f"<{tag.lower()}" in low or f"{tag.lower()}=" in low for tag in tags) \
        if len(tags) > 1 else any(f"<{tag.lower()}" in low for tag in tags)


def intra_signal_present(structural_xml: str, tag: str) -> bool:
    """Did the intra-method analyzer emit this tag (e.g. <rename>)?"""
    if not structural_xml:
        return False
    return f"<{tag.lower()}" in structural_xml.lower()


def filter_sole_stage3_additions(
    stage3_additions: List[Tuple[str, str]],
    stage2_defined_names: Set[str],
    structural_xml: str,
) -> Tuple[List[Tuple[str, str]], List[Dict]]:
    """Apply 4 precision filters to Stage-3 ADDITIONS only (not Stage 2 emissions).

    The four filters in order:
      1. Stage-2 corroboration: if Stage 2 also emitted this type, KEEP it
         outright (it's confirmed by both stages).
      2. Rule strength: if the FP-Growth association is weak, the addition
         is probably hallucinated.
      3. Two-step gate: compound types like 'Move And Rename Class' need
         BOTH primitives in the structural XML.
      4. Class-level / intra-method gates: type needs matching evidence
         in the right block.
    """
    kept: List[Tuple[str, str]] = []
    dropped: List[Dict] = []
    for t, ev in stage3_additions:
        if t in stage2_defined_names:
            kept.append((t, ev))
            continue

        rs = max_rule_strength(stage2_defined_names, t)
        if rs < RULE_STRENGTH_TAU:
            dropped.append({"type": t, "reason": "rule_strength<tau",
                            "rule_strength": round(rs, 3)})
            continue

        if t in _TWO_STEP_GATE:
            required = _TWO_STEP_GATE[t]
            if not all(f"<{tag}" in structural_xml.lower() for tag in required):
                dropped.append({"type": t, "reason": "two_step_missing_primitives",
                                "required": list(required)})
                continue

        if t in _CLASS_LEVEL_GATED:
            if not _STRUCTURAL_TAG_RE.search(structural_xml or ""):
                dropped.append({"type": t, "reason": "class_level_no_struct"})
                continue

        gate_tag = _INTRA_METHOD_GATE.get(t)
        if gate_tag and not intra_signal_present(structural_xml, gate_tag):
            dropped.append({"type": t, "reason": "intra_gate_missing",
                            "expected_tag": gate_tag})
            continue

        kept.append((t, ev))
    return kept, dropped

# Name-anchored structural gate helpers

_NAME_ATTR_RE = re.compile(r'\bname="([^"]+)"')
_METHOD_ATTR_RE = re.compile(r'\bmethod="([^"]+)"')
_BEFORE_PARAMS_RE = re.compile(r'\bbefore_params="([^"]*)"')
_AFTER_PARAMS_RE = re.compile(r'\bafter_params="([^"]*)"')
_IDENT_RE_3 = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")

# Pull all `name="..."` values from a specific structural-XML line (e.g. moved methods).
def _names_in_xml_line(xml: str, element: str, sub: str) -> Set[str]:
    """Get the set of names mentioned by a specific element/sub combo.

    e.g. `_names_in_xml_line(xml, "methods", "moved")` returns the names of
    every method gumtree saw moved.
    """
    names: Set[str] = set()
    pat = f"<{element} "
    for line in xml.splitlines():
        low = line.lower()
        if pat.lower() in low and f"{sub}=[" in low:
            for m in _NAME_ATTR_RE.finditer(line):
                names.add(m.group(1).lower())
    return names


def _method_names_in_sigs(xml: str) -> Set[str]:
    """Extract `method="..."` values from <signatures changed=[...]>.
    Includes both the full qualified name and the unqualified suffix
    (e.g. 'ParseRecord.run' → {'parserecord.run', 'run'})."""
    names: Set[str] = set()
    for line in xml.splitlines():
        if "<signatures " in line.lower() and "changed=[" in line.lower():
            for m in _METHOD_ATTR_RE.finditer(line):
                full = m.group(1).lower()
                names.add(full)

                if "." in full:
                    names.add(full.rsplit(".", 1)[-1])
    return names


def _sig_param_delta(xml: str) -> Tuple[Set[str], Set[str]]:
    """Return (before_param_tokens, after_param_tokens) across all changed sigs."""
    before_all: Set[str] = set()
    after_all: Set[str] = set()
    for line in xml.splitlines():
        if "<signatures " in line.lower() and "changed=[" in line.lower():
            for m in _BEFORE_PARAMS_RE.finditer(line):
                before_all.update(_IDENT_RE_3.findall(m.group(1)))
            for m in _AFTER_PARAMS_RE.finditer(line):
                after_all.update(_IDENT_RE_3.findall(m.group(1)))
    return before_all, after_all


def _ev_names(evidence: str) -> Set[str]:
    """Identifiers (≥3 chars) from the evidence string."""
    return set(_IDENT_RE_3.findall(evidence)) if evidence else set()


def _ev_anchors(evidence: str, xml_names: Set[str]) -> bool:
    """True if any evidence identifier overlaps with xml_names, or evidence empty.
    Comparison is case-insensitive (_names_in_xml_line lowercases; evidence preserves case)."""
    if not evidence:
        return True
    ev = {n.lower() for n in _ev_names(evidence)}
    return bool(ev & xml_names)


def _call_site_present(method_name: str, code_diff: str) -> bool:
    """True if method_name appears as a call (not definition) in any plus-line."""
    if not method_name or not code_diff:
        return False
    call_re = re.compile(rf"\b{re.escape(method_name)}\s*\(")
    def_re  = re.compile(rf"\bdef\s+{re.escape(method_name)}\s*\(")
    for line in code_diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            if call_re.search(line) and not def_re.search(line):
                return True
    return False


def filter_stage4_structural(
    final_types: List[str],
    structural_xml: str,
    evidences: Optional[Dict[str, str]] = None,
    code_diff: str = "",
) -> Tuple[List[str], List[str]]:
    """Stage-4 structural evidence gate (M2 + name-anchored M3 extensions).

    For each type that has a registered gate, drops the detection unless
    physical evidence is present in the gumtree <structural_facts> or
    <intra_method_signals> XML blocks.

    When `evidences` is supplied (type → evidence string from Stage 4), the
    gate additionally verifies that the specific name cited in the evidence
    matches a name in the structural fact (name-anchoring).  When evidences
    are absent, falls back to tag-presence only (backward-compat).

    For Extract Method, when both evidences and code_diff are supplied, also
    requires a call-site for the extracted method in the diff (the new method
    must be called, not just defined).

    Gate fires ONLY when <structural_facts> appears in structural_xml — if
    gumtree returned nothing, bypass all gates entirely so we don't penalise
    cases where parsing simply failed.

    Returns (kept_types, dropped_types).
    """
    has_structural = bool(structural_xml) and "<structural_facts" in structural_xml
    has_intra      = bool(structural_xml) and "<intra_method_signals" in structural_xml
    if not has_structural and not has_intra:
        return list(final_types), []

    if evidences is None:
        evidences = {}

    xml_low = structural_xml.lower() if structural_xml else ""

    def _gt_has(element: str, sub: str) -> bool:
        """True if the <element sub=[...]> line exists.
        Returns True (bypass) if gumtree structural facts are absent — we never
        penalise a type just because gumtree failed to parse the commit."""
        if not has_structural:
            return True
        pat = f"<{element} "
        for line in xml_low.splitlines():
            if pat in line and f"{sub}=[" in line:
                return True
        return False

    def _intra_has(tag: str) -> bool:
        """True if the intra-method signal tag is present.
        Returns True (bypass) when intra signals are absent."""
        if not has_intra:
            return True
        return f"<{tag}" in xml_low

    _methods_added  = _names_in_xml_line(structural_xml, "methods", "added")
    _methods_moved  = _names_in_xml_line(structural_xml, "methods", "moved")
    _classes_added  = _names_in_xml_line(structural_xml, "classes", "added")
    _classes_moved  = _names_in_xml_line(structural_xml, "classes", "moved")
    _sig_methods    = _method_names_in_sigs(structural_xml)
    _before_params, _after_params = _sig_param_delta(structural_xml)

    _DEF_RE = re.compile(r"^\+\s*(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)

    def _check_extract_method(ev: str) -> bool:
        if not _gt_has("methods", "added"):
            return False
        if ev and _methods_added:
            ev_ids = _ev_names(ev)

            name_in_gt = bool(ev_ids & _methods_added)
            if not name_in_gt and code_diff:
                diff_added_defs = set(_DEF_RE.findall(code_diff))
                if not (ev_ids & diff_added_defs):
                    return False
        return True

    def _check_rename_method(ev: str) -> bool:
        if not (_intra_has("rename") or _gt_has("methods", "added")
                or _gt_has("methods", "moved")):
            return False

        if ev and not _intra_has("rename") and has_structural:
            ev_ids = _ev_names(ev)
            if not (ev_ids & _methods_added | ev_ids & _methods_moved):
                return False
        return True

    _SIG_PLUS_RE = re.compile(r"^\+\s*(?:async\s+)?def\s+\w+\s*\(", re.MULTILINE)

    def _check_add_param(ev: str) -> bool:
        if _gt_has("signatures", "changed"):
            if ev:
                ev_ids = _ev_names(ev)
                if _sig_methods and not (ev_ids & _sig_methods):
                    return False
                new_params = _after_params - _before_params
                if new_params and not (ev_ids & new_params):
                    return False
            return True

        if ev and _SIG_PLUS_RE.search(ev):
            return True
        return False

    _SIG_MINUS_RE = re.compile(r"^-\s*(?:async\s+)?def\s+\w+\s*\(", re.MULTILINE)

    def _check_remove_param(ev: str) -> bool:
        if _gt_has("signatures", "changed"):
            if ev:
                ev_ids = _ev_names(ev)
                if _sig_methods and not (ev_ids & _sig_methods):
                    return False
                removed_params = _before_params - _after_params
                if removed_params and not (ev_ids & removed_params):
                    return False
            return True

        if ev and _SIG_MINUS_RE.search(ev):
            return True
        return False

    _methods_removed = _names_in_xml_line(structural_xml, "methods", "removed")

    def _check_inline_method(ev: str) -> bool:
        if not _gt_has("methods", "removed"):
            return False
        if ev:
            ev_ids = _ev_names(ev)
            if _methods_removed and not (ev_ids & _methods_removed):
                return False
        return True

    _GATES: Dict[str, any] = {

        "Extract Variable":              lambda ev: _intra_has("extract"),
        "Inline Variable":               lambda ev: _intra_has("inline"),

        "Rename Variable":               lambda ev: True,
        "Rename Parameter":              lambda ev: True,
        "Rename Attribute":              lambda ev: (_intra_has("rename") or
                                                     _gt_has("attributes", "added") or
                                                     _gt_has("attributes", "removed")),
        "Rename Method":                 _check_rename_method,
        "Split Conditional":             lambda ev: _intra_has("split"),
        "Invert Condition":              lambda ev: _intra_has("invert"),
        "Change Variable Type":          lambda ev: _intra_has("type_change"),
        "Replace Variable With Attribute": lambda ev: _intra_has("swap"),
        "Replace Attribute With Variable": lambda ev: _intra_has("swap"),

        "Extract Method":                _check_extract_method,
        "Inline Method":                 _check_inline_method,
        "Move Method":                   lambda ev: (
                                             _gt_has("methods", "moved") and
                                             (not has_structural or _ev_anchors(ev, _methods_moved))),
        "Add Parameter":                 _check_add_param,
        "Remove Parameter":              _check_remove_param,
        "Add Attribute":                 lambda ev: _gt_has("attributes", "added"),
        "Remove Attribute":              lambda ev: _gt_has("attributes", "removed"),
        "Move Attribute":                lambda ev: (_gt_has("attributes", "moved")
                                                     or _gt_has("attributes", "added")),
        "Extract Class":                 lambda ev: (
                                             _gt_has("classes", "added") and
                                             (not has_structural or _ev_anchors(ev, _classes_added))),
        "Extract Subclass":              lambda ev: (
                                             _gt_has("classes", "added") and
                                             (not has_structural or _ev_anchors(ev, _classes_added))),
        "Extract Superclass":            lambda ev: (
                                             _gt_has("classes", "added") and
                                             (not has_structural or _ev_anchors(ev, _classes_added))),
        "Move Class":                    lambda ev: (
                                             _gt_has("classes", "moved") and
                                             (not has_structural or _ev_anchors(ev, _classes_moved))),
        "Rename Class":                  lambda ev: (

                                             not has_structural or
                                             "<classes " in xml_low or
                                             (bool(re.search(r"^-.*\bclass\b", ev or "", re.M)) and
                                              bool(re.search(r"^\+.*\bclass\b", ev or "", re.M)))),
    }

    kept: List[str] = []
    dropped: List[str] = []
    for t in final_types:
        gate = _GATES.get(t)
        ev = evidences.get(t, "")
        if gate is None or gate(ev):
            kept.append(t)
        else:
            dropped.append(t)

    return kept, dropped

# precision mode on
def precision_mode_on() -> bool:
    return os.environ.get("LANGCHAIN_PRECISION_MODE", "").lower() in ("1", "true", "yes")

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


def filter_by_citation(
    types_with_evidence: List[Tuple[str, str]],
    code_diff: str,
    min_overlap: float = 0.70,
) -> Tuple[List[str], List[dict]]:
    """Drop detections whose Stage-4 evidence tokens don't appear in the diff.

    For each (type, evidence_str) pair, tokenise the evidence string into
    identifiers (≥3 chars) and measure what fraction of those tokens appear
    anywhere in code_diff.  If the fraction is below min_overlap the detection
    is dropped — its cited evidence is not anchored to the actual diff.

    Types with empty evidence strings are kept (benefit of the doubt).
    """
    diff_tokens: Set[str] = set(_IDENT_RE.findall(code_diff))
    kept: List[str] = []
    dropped: List[dict] = []
    for t, ev in types_with_evidence:
        ev_tokens = set(_IDENT_RE.findall(ev)) if ev else set()
        if not ev_tokens:
            kept.append(t)
            continue
        overlap = len(ev_tokens & diff_tokens) / len(ev_tokens)
        if overlap >= min_overlap:
            kept.append(t)
        else:
            dropped.append({
                "type": t,
                "overlap": round(overlap, 3),
                "missing_tokens": sorted(ev_tokens - diff_tokens)[:5],
            })
    return kept, dropped
