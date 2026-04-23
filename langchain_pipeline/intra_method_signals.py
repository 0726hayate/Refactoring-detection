"""Intra-method signal extractor — pattern-based, label-free.

Targets the ~25 % of commits whose GT refactorings live inside method bodies
(Rename Variable, Extract / Inline Variable, Invert Condition, Split
Conditional, Replace Variable With Attribute / inverse, Change Variable Type)
— invisible to AST-level diff because they leave the class/method skeleton
unchanged.

Operates on the unified `commit_diff` text directly. Pure-Python: re +
tokenize + difflib. Emits an `<intra_method_signals>` XML block as a sibling
to `<structural_facts>` — facts only, never refactoring labels (no oracle
leak).

Entry point: `intra_method_signals_for_case(case) -> str`
"""
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from xml.sax.saxutils import escape

# Diff parsing — yields (file, hunks) where each hunk is a list of (sign, line)

_FILE_HEADER = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)")
_FILE_HEADER_ALT = re.compile(r"^--- (?:a/)?(?P<f>\S+)")
_HUNK_HEADER = re.compile(r"^@@ ")
_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_PYTHON_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield", "self", "cls",
}
_BUILTIN_NAMES = {
    "print", "len", "range", "list", "dict", "set", "tuple", "str", "int",
    "float", "bool", "None", "True", "False", "abs", "all", "any", "min",
    "max", "sum", "sorted", "reversed", "enumerate", "zip", "map", "filter",
    "isinstance", "type", "open", "input", "format", "repr", "hash", "id",
    "iter", "next", "vars", "dir", "callable", "getattr", "setattr", "hasattr",
}
_NOISE_NAMES = _PYTHON_KEYWORDS | _BUILTIN_NAMES


def _split_hunks(commit_diff: str) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Yield (file_path, [(sign, line), ...]) per hunk in the diff.

    sign in {' ', '+', '-'}.
    Tolerates both git-format and minimal `--- BEFORE` / `+++ AFTER` styles.
    """
    out: List[Tuple[str, List[Tuple[str, str]]]] = []
    cur_file = ""
    cur_hunk: List[Tuple[str, str]] = []
    in_hunk = False

    for raw in commit_diff.splitlines():
        if not raw:
            continue
        m = _FILE_HEADER.match(raw)
        if m:
            if cur_hunk:
                out.append((cur_file, cur_hunk)); cur_hunk = []
            cur_file = m.group("b") or m.group("a")
            in_hunk = False
            continue
        m2 = _FILE_HEADER_ALT.match(raw)
        if m2:
            if cur_hunk:
                out.append((cur_file, cur_hunk)); cur_hunk = []
            f = m2.group("f")
            if f != "/dev/null":
                cur_file = f
            in_hunk = True
            continue
        if raw.startswith("+++"):
            continue
        if _HUNK_HEADER.match(raw):
            if cur_hunk:
                out.append((cur_file, cur_hunk)); cur_hunk = []
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if raw.startswith("+"):
            cur_hunk.append(("+", raw[1:]))
        elif raw.startswith("-"):
            cur_hunk.append(("-", raw[1:]))
        elif raw.startswith(" "):
            cur_hunk.append((" ", raw[1:]))
        else:
            cur_hunk.append((" ", raw))

    if cur_hunk:
        out.append((cur_file, cur_hunk))
    return out

# Token rename detection
def _pair_lines(hunk: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Greedy pairing of - and + lines that look like edits of each other.

    Walk the hunk; when a '-' is followed (with possibly more -'s, then +'s),
    pair them positionally up to the shorter run length. Crude but matches the
    common case where one - line maps to one + line.
    """
    pairs: List[Tuple[str, str]] = []
    i = 0
    n = len(hunk)
    while i < n:
        if hunk[i][0] == "-":
            j = i
            minus = []
            while j < n and hunk[j][0] == "-":
                minus.append(hunk[j][1]); j += 1
            plus = []
            while j < n and hunk[j][0] == "+":
                plus.append(hunk[j][1]); j += 1
            for k in range(min(len(minus), len(plus))):
                pairs.append((minus[k], plus[k]))
            i = j
        else:
            i += 1
    return pairs


def _line_token_subs(before: str, after: str) -> List[Tuple[str, str]]:
    """Tokens that differ at the same position when before/after have same length.

    Returns (old, new) pairs from positions where exactly one token differs and
    surrounding context matches. Filters builtins and keywords on both sides.
    """
    bt = _TOKEN_RE.findall(before)
    at = _TOKEN_RE.findall(after)
    if len(bt) != len(at) or not bt:
        return []
    diffs = [(b, a) for b, a in zip(bt, at) if b != a]
    if not diffs:
        return []
    out: List[Tuple[str, str]] = []
    for b, a in diffs:
        if b in _NOISE_NAMES or a in _NOISE_NAMES:
            continue
        if not (b.isidentifier() and a.isidentifier()):
            continue
        out.append((b, a))
    return out


def detect_renames(hunks: List[Tuple[str, List]], min_hits: int = 2) -> List[Dict]:
    """Find consistent (old, new) token substitutions.

    Aggregates across all hunks. A pair qualifies if it fires >= min_hits AND
    no contradictory mapping exists (old also maps to a different new with
    >= min_hits/2 hits).
    """
    counts: Counter = Counter()
    by_old: Dict[str, Counter] = defaultdict(Counter)
    by_new: Dict[str, Counter] = defaultdict(Counter)
    for _file, hunk in hunks:
        for before, after in _pair_lines(hunk):
            for old, new in _line_token_subs(before, after):
                counts[(old, new)] += 1
                by_old[old][new] += 1
                by_new[new][old] += 1

    qualified = []
    for (old, new), hits in counts.items():
        if hits < min_hits:
            continue

        alt_news = by_old[old]
        if len(alt_news) > 1:
            second = alt_news.most_common(2)[1][1] if len(alt_news) >= 2 else 0
            if second >= max(2, hits // 2):
                continue

        alt_olds = by_new[new]
        if len(alt_olds) > 1:
            second = alt_olds.most_common(2)[1][1] if len(alt_olds) >= 2 else 0
            if second >= max(2, hits // 2):
                continue
        qualified.append({"old": old, "new": new, "hits": hits})
    qualified.sort(key=lambda d: -d["hits"])
    return qualified

# Self-prefix swap (subset of rename: x ↔ self.x)
def detect_self_swaps(hunks: List[Tuple[str, List]]) -> List[Dict]:
    """`x` ↔ `self.x` substitutions — same-line text replacement detection."""
    var_to_attr: Counter = Counter()
    attr_to_var: Counter = Counter()
    for _f, hunk in hunks:
        for before, after in _pair_lines(hunk):
            for m in re.finditer(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\b", after):
                t = m.group(1)
                if t in before and "self." + t not in before:
                    var_to_attr[t] += 1
            for m in re.finditer(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\b", before):
                t = m.group(1)
                if t in after and "self." + t not in after:
                    attr_to_var[t] += 1
    out = []
    for t, h in var_to_attr.items():
        if h >= 2:
            out.append({"kind": "var_to_attr", "old": t, "new": "self." + t, "hits": h})
    for t, h in attr_to_var.items():
        if h >= 2:
            out.append({"kind": "attr_to_var", "old": "self." + t, "new": t, "hits": h})
    out.sort(key=lambda d: -d["hits"])
    return out

# Variable extraction / inlining

_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")


def detect_extractions(hunks: List[Tuple[str, List]]) -> List[Dict]:
    """`+ name = expr` line followed by `<name>` replacing `<expr>` at use sites."""
    out: List[Dict] = []
    for _f, hunk in hunks:
        for idx, (sign, line) in enumerate(hunk):
            if sign != "+":
                continue
            m = _ASSIGN_RE.match(line)
            if not m:
                continue
            name, expr = m.group(1), m.group(2)
            if not name.isidentifier() or name in _NOISE_NAMES:
                continue
            if len(expr) < 8 or "(" not in expr and "." not in expr:
                continue

            hits = 0
            seen_minus = 0
            for sign2, line2 in hunk[idx + 1:]:
                if sign2 == "-" and expr in line2:
                    seen_minus += 1
                elif sign2 == "+" and name in line2 and expr not in line2:
                    hits += 1
            if hits >= 2 and seen_minus >= 1:
                out.append({"var": name, "expr": expr[:80], "hits": hits})
    out.sort(key=lambda d: -d["hits"])
    return out


def detect_inlinings(hunks: List[Tuple[str, List]]) -> List[Dict]:
    """Mirror of extraction: `- name = expr` then `expr` reappears at use sites."""
    out: List[Dict] = []
    for _f, hunk in hunks:
        for idx, (sign, line) in enumerate(hunk):
            if sign != "-":
                continue
            m = _ASSIGN_RE.match(line)
            if not m:
                continue
            name, expr = m.group(1), m.group(2)
            if not name.isidentifier() or name in _NOISE_NAMES:
                continue
            if len(expr) < 8:
                continue
            hits = 0
            seen_plus = 0
            for sign2, line2 in hunk[idx + 1:]:
                if sign2 == "+" and expr in line2:
                    seen_plus += 1
                elif sign2 == "-" and name in line2 and expr not in line2:
                    hits += 1
            if hits >= 2 and seen_plus >= 1:
                out.append({"var": name, "expr": expr[:80], "hits": hits})
    out.sort(key=lambda d: -d["hits"])
    return out

# Condition inversion + split

_COND_KEYWORD = re.compile(r"^\s*(if|elif|while|assert)\s+")
_COMPARATOR_FLIPS = [
    ("==", "!="), ("!=", "=="),
    ("<", ">="), (">=", "<"),
    (">", "<="), ("<=", ">"),
    ("is", "is not"), ("is not", "is"),
    (" in ", " not in "), (" not in ", " in "),
]

# detect inversions
def detect_inversions(hunks: List[Tuple[str, List]]) -> List[Dict]:
    out: List[Dict] = []
    for _f, hunk in hunks:
        for before, after in _pair_lines(hunk):
            if not (_COND_KEYWORD.match(before) and _COND_KEYWORD.match(after)):
                continue
            b_strip = before.strip()
            a_strip = after.strip()
            inverted = False

            if (" not " in a_strip and " not " not in b_strip) or \
               (" not " in b_strip and " not " not in a_strip):
                inverted = True

            for old_op, new_op in _COMPARATOR_FLIPS:
                if old_op in b_strip and new_op in a_strip and \
                        b_strip.replace(old_op, new_op, 1) == a_strip:
                    inverted = True
                    break
            if inverted:
                out.append({"before": b_strip[:120], "after": a_strip[:120]})
    return out[:20]

_SPLIT_AND_RE = re.compile(r"^\s*if\s+(.+?)\s+(?:and|or)\s+(.+?):\s*$")


def detect_splits(hunks: List[Tuple[str, List]]) -> List[Dict]:
    """`- if A and B:` matched by `+ if A:` followed by `+    if B:` (deeper indent)."""
    out: List[Dict] = []
    for _f, hunk in hunks:
        i = 0
        while i < len(hunk):
            sign, line = hunk[i]
            if sign == "-" and _SPLIT_AND_RE.match(line):
                j = i + 1
                while j < len(hunk) and hunk[j][0] == "-":
                    j += 1
                if j + 1 < len(hunk) and hunk[j][0] == "+" and hunk[j + 1][0] == "+":
                    a_indent = len(hunk[j][1]) - len(hunk[j][1].lstrip())
                    b_indent = len(hunk[j + 1][1]) - len(hunk[j + 1][1].lstrip())
                    if b_indent > a_indent and re.match(r"^\s*if\s+", hunk[j][1]) and \
                            re.match(r"^\s*if\s+", hunk[j + 1][1]):
                        out.append({"before": line.strip()[:120],
                                    "after_outer": hunk[j][1].strip()[:80],
                                    "after_inner": hunk[j + 1][1].strip()[:80]})
            i += 1
    return out[:10]

# Type-annotation change

_VAR_ANNOT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^=]+?)(?:=|$)")

# detect type changes
def detect_type_changes(hunks: List[Tuple[str, List]]) -> List[Dict]:
    out: List[Dict] = []
    seen: set = set()
    for _f, hunk in hunks:
        for before, after in _pair_lines(hunk):
            mb = _VAR_ANNOT_RE.match(before)
            ma = _VAR_ANNOT_RE.match(after)
            if not (mb and ma):
                continue
            if mb.group(1) != ma.group(1):
                continue
            tb, ta = mb.group(2).strip(), ma.group(2).strip()
            if tb == ta or not tb or not ta:
                continue
            key = (mb.group(1), tb, ta)
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": mb.group(1), "before": tb[:60], "after": ta[:60]})
    return out[:20]

# Render

# render attrs
def _render_attrs(d: Dict) -> str:
    return " ".join(f'{k}="{escape(str(v))}"' for k, v in d.items())

# render block
def _render_block(tag: str, items: List[Dict], inner_tag: str) -> str:
    if not items:
        return ""
    lines = [f"  <{tag}>"]
    for it in items:
        lines.append(f"    <{inner_tag} {_render_attrs(it)} />")
    lines.append(f"  </{tag}>")
    return "\n".join(lines)

# compute signals
def compute_signals(commit_diff: str) -> Dict[str, List[Dict]]:
    if not commit_diff:
        return {"renames": [], "self_swaps": [], "extractions": [],
                "inlinings": [], "inversions": [], "splits": [], "type_changes": []}
    hunks = _split_hunks(commit_diff)
    return {
        "renames": detect_renames(hunks),
        "self_swaps": detect_self_swaps(hunks),
        "extractions": detect_extractions(hunks),
        "inlinings": detect_inlinings(hunks),
        "inversions": detect_inversions(hunks),
        "splits": detect_splits(hunks),
        "type_changes": detect_type_changes(hunks),
    }

# render signals xml
def render_signals_xml(signals: Dict[str, List[Dict]]) -> str:
    blocks = [
        _render_block("renames", signals["renames"], "rename"),
        _render_block("self_swaps", signals["self_swaps"], "swap"),
        _render_block("extractions", signals["extractions"], "extract"),
        _render_block("inlinings", signals["inlinings"], "inline"),
        _render_block("inversions", signals["inversions"], "invert"),
        _render_block("splits", signals["splits"], "split"),
        _render_block("type_changes", signals["type_changes"], "type_change"),
    ]
    body = "\n".join(b for b in blocks if b)
    if not body.strip():
        return ""
    return f"<intra_method_signals>\n{body}\n</intra_method_signals>"

# intra method signals for case
def intra_method_signals_for_case(case: Dict) -> str:
    return render_signals_xml(compute_signals(case.get("commit_diff", "") or ""))
