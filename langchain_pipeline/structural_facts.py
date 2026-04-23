"""Type-agnostic AST structural-facts extractor for Stage 2/3 prompts.

Reads the per-file before/after blobs already present on every train/test case
(`code_before` / `code_after` arrays), runs Python ast.parse, and emits a
type-agnostic XML block listing structural changes the LLM cannot reliably
recover from a unified diff:
  files: added / removed / renamed
  classes: added / removed / moved
  methods: added / removed / moved (with from_class / to_class)
  signatures: changed (param list deltas)
  inheritance: changed (base list deltas)
  attributes: added / removed (self.x in __init__ + class-level annotations)

Pure-Python; no external tools. Falls back gracefully on parse errors -- the
fact block is always best-effort.
"""
import ast
import difflib
import warnings

# Train commits include legacy Python with deprecated escape sequences in
# string literals (e.g. "\d" instead of r"\d"). Python 3.12+ flags these as
# SyntaxWarning during ast.parse — harmless but pollutes logs. Silence them
# only inside this module's parse calls.
warnings.filterwarnings("ignore", category=SyntaxWarning)
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from xml.sax.saxutils import escape

SIM_THRESHOLD_FILE = 0.55
SIM_THRESHOLD_CODE = 0.65


@dataclass
class MethodInfo:
    name: str
    cls: Optional[str]
    file: str
    params: List[str]
    body_src: str

    # key
    @property
    def key(self) -> Tuple[Optional[str], str]:
        return (self.cls, self.name)


@dataclass
class ClassInfo:
    name: str
    file: str
    bases: List[str]
    methods: Dict[str, MethodInfo] = field(default_factory=dict)
    attrs: Set[str] = field(default_factory=set)
    body_src: str = ""


@dataclass
class FileSnapshot:
    path: str
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    funcs: Dict[str, MethodInfo] = field(default_factory=dict)
    raw: str = ""
    parsed: bool = True

# ast funcs
def _ast_funcs(node: ast.AST):
    return [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

# params of
def _params_of(fn) -> List[str]:
    a = fn.args
    out = [p.arg for p in a.posonlyargs + a.args]
    if a.vararg:
        out.append("*" + a.vararg.arg)
    out += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        out.append("**" + a.kwarg.arg)
    return out

# base name
def _base_name(b: ast.AST) -> str:
    if isinstance(b, ast.Name):
        return b.id
    if isinstance(b, ast.Attribute):
        return f"{_base_name(b.value)}.{b.attr}"
    try:
        return ast.unparse(b)
    except Exception:
        return "?"

# attrs in
def _attrs_in(node: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for tgt in n.targets:
                if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) \
                        and tgt.value.id in ("self", "cls"):
                    out.add(tgt.attr)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Attribute) \
                and isinstance(n.target.value, ast.Name) and n.target.value.id in ("self", "cls"):
            out.add(n.target.attr)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) \
                and isinstance(node, ast.ClassDef):
            out.add(n.target.id)
    return out

# parse file
def _parse_file(path: str, src: str) -> FileSnapshot:
    snap = FileSnapshot(path=path, raw=src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        snap.parsed = False
        return snap
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cls = ClassInfo(
                name=node.name, file=path,
                bases=[_base_name(b) for b in node.bases],
                attrs=_attrs_in(node),
            )
            try:
                cls.body_src = ast.unparse(node)
            except Exception:
                cls.body_src = ""
            for fn in _ast_funcs(node):
                try:
                    body = ast.unparse(fn)
                except Exception:
                    body = ""
                m = MethodInfo(name=fn.name, cls=node.name, file=path,
                               params=_params_of(fn), body_src=body)
                cls.methods[fn.name] = m
            snap.classes[node.name] = cls
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                body = ast.unparse(node)
            except Exception:
                body = ""
            snap.funcs[node.name] = MethodInfo(
                name=node.name, cls=None, file=path,
                params=_params_of(node), body_src=body,
            )
    return snap

# ratio
def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

# match renames
def _match_renames(removed: List[str], added: List[str], src_before, src_after,
                   threshold: float = SIM_THRESHOLD_FILE) -> List[Tuple[str, str, float]]:
    pairs = []
    used_added: Set[str] = set()
    for r in removed:
        best = (0.0, "")
        for a in added:
            if a in used_added:
                continue
            s = _ratio(src_before.get(r, ""), src_after.get(a, ""))
            if s > best[0]:
                best = (s, a)
        if best[0] >= threshold and best[1]:
            pairs.append((r, best[1], best[0]))
            used_added.add(best[1])
    return pairs

# all classes
def _all_classes(snaps: Dict[str, FileSnapshot]):
    return [(p, c) for p, s in snaps.items() for c in s.classes.values()]

# all methods
def _all_methods(snaps: Dict[str, FileSnapshot]):
    out = []
    for p, s in snaps.items():
        for c in s.classes.values():
            for m in c.methods.values():
                out.append((p, c.name, m))
        for fn in s.funcs.values():
            out.append((p, None, fn))
    return out


def compute_facts(code_before: List[Dict], code_after: List[Dict]) -> Dict:
    """Return the structural-facts dict for a single commit.

    Args:
        code_before / code_after: lists of {"file": ..., "code": ...}.

    Returns:
        Dict with keys files / classes / methods / signatures / inheritance /
        attributes, each holding sub-categories. Empty categories omitted at
        XML render time.
    """
    src_before = {f.get("file", ""): f.get("code", "") for f in code_before}
    src_after = {f.get("file", ""): f.get("code", "") for f in code_after}

    files_before = set(src_before)
    files_after = set(src_after)
    added_files = sorted(files_after - files_before)
    removed_files = sorted(files_before - files_after)
    renames = _match_renames(removed_files, added_files, src_before, src_after)
    rename_from_to = {r: a for r, a, _ in renames}
    rename_to_from = {a: r for r, a, _ in renames}
    added_files_net = [f for f in added_files if f not in rename_to_from]
    removed_files_net = [f for f in removed_files if f not in rename_from_to]

    snaps_before = {p: _parse_file(p, src_before[p]) for p in src_before}
    snaps_after = {p: _parse_file(p, src_after[p]) for p in src_after}

    facts = {
        "files": {
            "added": added_files_net,
            "removed": removed_files_net,
            "renamed": [{"from": r, "to": a, "similarity": round(s, 2)} for r, a, s in renames],
        },
        "classes": {"added": [], "removed": [], "moved": []},
        "methods": {"added": [], "removed": [], "moved": []},
        "signatures": {"changed": []},
        "inheritance": {"changed": []},
        "attributes": {"added": [], "removed": []},
    }

    common_files = (files_before & files_after) | {a for _, a in rename_from_to.items()}
    for p_after in sorted(common_files):
        p_before = rename_to_from.get(p_after, p_after)
        sb = snaps_before.get(p_before)
        sa = snaps_after.get(p_after)
        if not sb or not sa or not sb.parsed or not sa.parsed:
            continue
        for cname, cls_b in sb.classes.items():
            if cname not in sa.classes:
                continue
            cls_a = sa.classes[cname]
            if cls_b.bases != cls_a.bases:
                facts["inheritance"]["changed"].append({
                    "class": cname, "file": p_after,
                    "before_bases": cls_b.bases, "after_bases": cls_a.bases,
                })
            attrs_added = sorted(cls_a.attrs - cls_b.attrs)
            attrs_removed = sorted(cls_b.attrs - cls_a.attrs)
            for at in attrs_added:
                facts["attributes"]["added"].append({"class": cname, "file": p_after, "name": at})
            for at in attrs_removed:
                facts["attributes"]["removed"].append({"class": cname, "file": p_after, "name": at})
            for mname, mb in cls_b.methods.items():
                if mname in cls_a.methods:
                    ma = cls_a.methods[mname]
                    if mb.params != ma.params:
                        facts["signatures"]["changed"].append({
                            "method": f"{cname}.{mname}", "file": p_after,
                            "before_params": mb.params, "after_params": ma.params,
                        })

    classes_before = {(p, c.name): c for p, c in _all_classes(snaps_before)}
    classes_after = {(p, c.name): c for p, c in _all_classes(snaps_after)}
    cb_keyed_by_name: Dict[str, List[Tuple[str, ClassInfo]]] = {}
    for (p, n), c in classes_before.items():
        cb_keyed_by_name.setdefault(n, []).append((p, c))
    ca_keyed_by_name: Dict[str, List[Tuple[str, ClassInfo]]] = {}
    for (p, n), c in classes_after.items():
        ca_keyed_by_name.setdefault(n, []).append((p, c))

    for n, after_pairs in ca_keyed_by_name.items():
        before_pairs = cb_keyed_by_name.get(n, [])
        used: Set[str] = set()
        for pa, ca in after_pairs:
            same_path = next((pb for pb, _ in before_pairs if pb == pa or rename_to_from.get(pa) == pb), None)
            if same_path is not None:
                continue
            best = (0.0, "")
            for pb, cb_ in before_pairs:
                if pb in used:
                    continue
                s = _ratio(cb_.body_src, ca.body_src)
                if s > best[0]:
                    best = (s, pb)
            if best[0] >= SIM_THRESHOLD_CODE:
                facts["classes"]["moved"].append({"name": n, "from_file": best[1], "to_file": pa,
                                                  "similarity": round(best[0], 2)})
                used.add(best[1])
            else:
                facts["classes"]["added"].append({"name": n, "file": pa})
        for pb, _ in before_pairs:
            if pb in used:
                continue
            same_path_after = next((pa for pa, _ in after_pairs if pa == pb or rename_from_to.get(pb) == pa), None)
            if same_path_after is not None:
                continue
            facts["classes"]["removed"].append({"name": n, "file": pb})

    method_keys_before = {(p, cn, m.name) for p, cn, m in _all_methods(snaps_before)}
    methods_before = {(p, cn, m.name): m for p, cn, m in _all_methods(snaps_before)}
    method_keys_after = {(p, cn, m.name) for p, cn, m in _all_methods(snaps_after)}
    methods_after = {(p, cn, m.name): m for p, cn, m in _all_methods(snaps_after)}

    by_name_after: Dict[str, List[Tuple[str, Optional[str], MethodInfo]]] = {}
    for (p, cn, n), m in methods_after.items():
        by_name_after.setdefault(n, []).append((p, cn, m))
    by_name_before: Dict[str, List[Tuple[str, Optional[str], MethodInfo]]] = {}
    for (p, cn, n), m in methods_before.items():
        by_name_before.setdefault(n, []).append((p, cn, m))

    for n, after_list in by_name_after.items():
        before_list = by_name_before.get(n, [])
        used: Set[Tuple[str, Optional[str]]] = set()
        for pa, cna, ma in after_list:
            if (pa, cna, n) in method_keys_before:
                continue
            best = (0.0, None, None)
            for pb, cnb, mb in before_list:
                if (pb, cnb) in used or (pb, cnb, n) in method_keys_after:
                    continue
                s = _ratio(mb.body_src, ma.body_src)
                if s > best[0]:
                    best = (s, pb, cnb)
            if best[0] >= SIM_THRESHOLD_CODE and best[1] is not None:
                facts["methods"]["moved"].append({
                    "name": n, "from_class": best[2], "to_class": cna,
                    "from_file": best[1], "to_file": pa, "similarity": round(best[0], 2),
                })
                used.add((best[1], best[2]))
            else:
                facts["methods"]["added"].append({"name": n, "class": cna, "file": pa})
        for pb, cnb, mb in before_list:
            if (pb, cnb) in used or (pb, cnb, n) in method_keys_after:
                continue
            facts["methods"]["removed"].append({"name": n, "class": cnb, "file": pb})

    return facts

# is empty
def _is_empty(facts: Dict) -> bool:
    for v in facts.values():
        if isinstance(v, dict):
            for sub in v.values():
                if sub:
                    return False
        elif v:
            return False
    return True

# xml attrs
def _xml_attrs(d: Dict) -> str:
    return " ".join(f'{k}="{escape(str(v))}"' for k, v in d.items())

# render list
def _render_list(items: List[Dict]) -> str:
    if not items:
        return ""
    return "[" + ", ".join("{" + _xml_attrs(it) + "}" for it in items) + "]"


def render_facts_xml(facts: Dict, max_per_category: int = 40) -> str:
    """Render the facts dict as the <structural_facts> block.

    `max_per_category` caps each sub-list (cheap defence against pathological
    commits; the embedding-based selector in fact_selection.py is the proper
    overflow path).
    """
    if _is_empty(facts):
        return ""
    out = ["<structural_facts>"]
    for cat in ("files", "classes", "methods", "signatures", "inheritance", "attributes"):
        sub = facts.get(cat, {})
        line_parts = []
        for sub_name, items in sub.items():
            if not items:
                continue
            if len(items) > max_per_category:
                items = items[:max_per_category] + [{"_truncated": len(sub[sub_name]) - max_per_category}]
            if isinstance(items, list) and items and not isinstance(items[0], dict):
                line_parts.append(f'{sub_name}="{escape(", ".join(map(str, items)))}"')
            else:
                line_parts.append(f"{sub_name}={_render_list(items)}")
        if line_parts:
            out.append(f"  <{cat} " + " ".join(line_parts) + " />")
    out.append("</structural_facts>")
    return "\n".join(out)


def facts_for_case(case: Dict) -> str:
    """Convenience: take a normalized python_case dict, return XML or ''."""
    code_before = case.get("code_before") or []
    code_after = case.get("code_after") or []
    if not code_before and not code_after:
        return ""
    facts = compute_facts(code_before, code_after)
    return render_facts_xml(facts)
