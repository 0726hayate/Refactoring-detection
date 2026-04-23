"""GumTree backend: parse each file with `gumtree parse`, walk the indented
text dump, then defer to the shared cross-file mover for renames/moves.

We don't use `gumtree textdiff` directly because its action stream is
per-token; reconstructing class/method-level deltas is brittle.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .structural_facts import render_facts_xml
from . import _facts_common as common

ROOT = Path(__file__).resolve().parent.parent
GUMTREE_BIN = ROOT / "tools" / "gumtree" / "bin" / "gumtree"
JAVA_HOME = ROOT / "tools" / "jdk-21"
TMP_DIR = ROOT / "langchain_pipeline" / "tmp"

_AVAILABLE = GUMTREE_BIN.exists() and JAVA_HOME.exists()
TIMEOUT_S = 30


def _run_parse(src: str, path_for_log: str) -> Optional[str]:
    """Invoke `gumtree parse`; return stdout text, or None on any failure."""
    if not _AVAILABLE:
        return None
    try:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    env = os.environ.copy()
    env["JAVA_HOME"] = str(JAVA_HOME)
    env["PATH"] = f"{JAVA_HOME / 'bin'}:{env.get('PATH', '')}"
    f = None
    try:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=str(TMP_DIR),
                                        delete=False, encoding="utf-8")
        f.write(src)
        f.close()
        try:
            proc = subprocess.run([str(GUMTREE_BIN), "parse", f.name],
                                  capture_output=True, timeout=TIMEOUT_S,
                                  env=env, text=True)
        except subprocess.TimeoutExpired as e:
            common.log_failure("gumtree", path_for_log, e, src)
            return None
        if proc.returncode != 0:
            common.log_failure("gumtree", path_for_log,
                RuntimeError(f"gumtree rc={proc.returncode}: {proc.stderr[:200]}"), src)
            return None
        return proc.stdout
    except Exception as e:
        common.log_failure("gumtree", path_for_log, e, src)
        return None
    finally:
        try:
            if f is not None:
                os.unlink(f.name)
        except Exception:
            pass

# Each row is (depth, type, value, start_byte, end_byte). Lines look like:
#   "    function_definition [20,59]"  or  "        identifier: m [24,25]"
_LINE_RE = re.compile(
    r"^(?P<indent> *)(?P<type>[a-zA-Z_][\w-]*)(?:: (?P<value>.+?))? \[(?P<s>\d+),(?P<e>\d+)\]$"
)

# parse tree
def _parse_tree(text: str):
    rows = []
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        depth = len(m.group("indent")) // 4
        rows.append((depth, m.group("type"), (m.group("value") or "").rstrip(),
                     int(m.group("s")), int(m.group("e"))))
    return rows


def _slice_subtree(rows, start_idx):
    """Return rows[start..end) with end at the first row of equal-or-shallower depth."""
    base = rows[start_idx][0]
    end = len(rows)
    for j in range(start_idx + 1, len(rows)):
        if rows[j][0] <= base:
            end = j
            break
    return rows[start_idx:end]


def _direct_children(sub_rows, type_name=None):
    """Yield (idx, row) for direct children (depth = root+1) of the subtree root."""
    root = sub_rows[0][0]
    for i in range(1, len(sub_rows)):
        d = sub_rows[i][0]
        if d <= root:
            break
        if d == root + 1 and (type_name is None or sub_rows[i][1] == type_name):
            yield i, sub_rows[i]


def _named_child(sub_rows, type_name) -> Optional[str]:
    """Return value of first direct child of the given type."""
    for _i, rec in _direct_children(sub_rows, type_name):
        return rec[2]
    return None

_PARAM_KINDS = {"typed_parameter": "", "default_parameter": "",
                "typed_default_parameter": "", "list_splat_pattern": "*",
                "dictionary_splat_pattern": "**"}


def _params_from_function(fn_rows) -> List[str]:
    """Walk the parameters subtree and collect param names with */** prefixes."""
    out: List[str] = []
    for i, _rec in _direct_children(fn_rows, "parameters"):
        sub = _slice_subtree(fn_rows, i)
        for j, rec in _direct_children(sub):
            tt = rec[1]
            if tt == "identifier":
                out.append(rec[2])
            elif tt in _PARAM_KINDS:
                nm = _named_child(_slice_subtree(sub, j), "identifier")
                if nm:
                    out.append(_PARAM_KINDS[tt] + nm)
        break
    return out


def _bases_from_class(cls_rows) -> List[str]:
    """Pull base names from the class's argument_list child."""
    out: List[str] = []
    for i, _rec in _direct_children(cls_rows, "argument_list"):
        sub = _slice_subtree(cls_rows, i)
        for j, rec in _direct_children(sub):
            tt, vv = rec[1], rec[2]
            if tt == "identifier":
                out.append(vv)
            elif tt == "attribute":
                inner = _slice_subtree(sub, j)
                parts = [r[2] for _, r in _direct_children(inner) if r[1] == "identifier"]
                if parts:
                    out.append(".".join(parts))
        break
    return out

_RE_SELF_ATTR = re.compile(r"\b(?:self|cls)\.([A-Za-z_]\w*)\s*(?:=|:)")
_RE_CLASS_ATTR_LINE = re.compile(r"^\s{0,8}([A-Za-z_]\w*)\s*:\s*[^=]+(?:=.*)?$")


def _self_attrs_in_function(src: str) -> set:
    """self.X / cls.X assignment targets in a function body (regex on body src)."""
    return {m.group(1) for m in _RE_SELF_ATTR.finditer(src)}


def _class_level_attrs(body_src: str) -> set:
    """Class-body annotated assignments at top-of-class indent (`x: int = 0`)."""
    return {m.group(1) for m in (_RE_CLASS_ATTR_LINE.match(l) for l in body_src.splitlines()) if m}

# slice text
def _slice_text(src_bytes: bytes, rec) -> str:
    return src_bytes[rec[3]:rec[4]].decode("utf-8", errors="replace")


def _build_class(rows, idx, src_bytes, path) -> Tuple[str, Dict]:
    """Build a class entry from rows[idx] (must be class_definition)."""
    rec = rows[idx]
    sub = _slice_subtree(rows, idx)
    cname = _named_child(sub, "identifier") or "?"
    body_src = _slice_text(src_bytes, rec)
    attrs = _class_level_attrs(body_src)
    methods: Dict[str, Dict] = {}
    for j, _ in _direct_children(sub, "block"):
        block_sub = _slice_subtree(sub, j)
        for k, krec in _direct_children(block_sub):
            fn_idx = k
            if krec[1] == "decorated_definition":
                deco = _slice_subtree(block_sub, k)
                inner = next((kk for kk, rr in _direct_children(deco, "function_definition")),
                             None)
                if inner is None:
                    continue
                fn_idx = k + inner
            elif krec[1] != "function_definition":
                continue
            fn_rec = block_sub[fn_idx]
            fn_sub = _slice_subtree(block_sub, fn_idx)
            mname = _named_child(fn_sub, "identifier") or "?"
            fn_src = _slice_text(src_bytes, fn_rec)
            attrs |= _self_attrs_in_function(fn_src)
            methods[mname] = {
                "name": mname, "cls": cname, "file": path,
                "params": _params_from_function(fn_sub), "body_src": fn_src,
            }
        break
    return cname, {
        "name": cname, "bases": _bases_from_class(sub), "attrs": attrs,
        "methods": methods, "body_src": body_src,
    }

# build func
def _build_func(rows, idx, src_bytes, path) -> Tuple[str, Dict]:
    rec = rows[idx]
    sub = _slice_subtree(rows, idx)
    fname = _named_child(sub, "identifier") or "?"
    return fname, {
        "name": fname, "cls": None, "file": path,
        "params": _params_from_function(sub),
        "body_src": _slice_text(src_bytes, rec),
    }

# extract
def _extract(text: str, src: str, path: str) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    classes: Dict[str, Dict] = {}
    funcs: Dict[str, Dict] = {}
    rows = _parse_tree(text)
    if not rows:
        return classes, funcs
    src_bytes = src.encode("utf-8", errors="replace")
    target = rows[0][0] + 1
    for i, rec in enumerate(rows):
        if rec[0] != target:
            continue
        t = rec[1]
        if t == "class_definition":
            cname, entry = _build_class(rows, i, src_bytes, path)
            classes[cname] = entry
        elif t == "function_definition":
            fname, entry = _build_func(rows, i, src_bytes, path)
            funcs[fname] = entry
        elif t == "decorated_definition":
            sub = _slice_subtree(rows, i)
            for j, jrec in _direct_children(sub):
                if jrec[1] == "class_definition":
                    cname, entry = _build_class(rows, i + j, src_bytes, path)
                    classes[cname] = entry
                    break
                if jrec[1] == "function_definition":
                    fname, entry = _build_func(rows, i + j, src_bytes, path)
                    funcs[fname] = entry
                    break
    return classes, funcs

# parse file
def _parse_file(path: str, src: str):
    if not _AVAILABLE or not src.strip():
        return {}, {}
    text = _run_parse(src, path)
    if text is None:
        return {}, {}
    try:
        return _extract(text, src, path)
    except Exception as e:
        common.log_failure("gumtree", path, e, src)
        return {}, {}


def compute_facts(code_before: List[Dict], code_after: List[Dict]) -> Dict:
    """gumtree backend; same signature/schema as structural_facts.compute_facts."""
    if not _AVAILABLE:
        return common.empty_facts()
    src_before = {f.get("file", ""): f.get("code", "") for f in code_before}
    src_after = {f.get("file", ""): f.get("code", "") for f in code_after}

    def split(srcs):
        cls, fns = {}, {}
        for p, s in srcs.items():
            c, f = _parse_file(p, s)
            cls[p], fns[p] = c, f
        return cls, fns
    fcb, ffb = split(src_before)
    fca, ffa = split(src_after)
    return common.assemble_facts(src_before, src_after, fcb, fca, ffb, ffa)

# gumtree facts for case
def gumtree_facts_for_case(case: Dict) -> str:
    code_before = case.get("code_before") or []
    code_after = case.get("code_after") or []
    if not code_before and not code_after:
        return ""
    try:
        facts = compute_facts(code_before, code_after)
    except Exception as e:
        common.log_failure("gumtree", "<case>", e)
        facts = common.empty_facts()
    return render_facts_xml(facts)
