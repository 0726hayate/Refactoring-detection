"""libcst backend for structural-fact extraction.

Uses concrete-syntax-tree visitors. Slightly more tolerant of formatting and
whitespace than the stdlib ast, but still requires syntactically valid Python.
"""
from typing import Dict, List, Optional

from .structural_facts import render_facts_xml
from . import _facts_common as common

_AVAILABLE = True
try:
    import libcst as cst  # type: ignore
except Exception as _e:
    _AVAILABLE = False
    _IMPORT_ERR = _e


def _attr_name(node) -> str:
    """Render a CST expression node as a dotted name (best-effort)."""
    if not _AVAILABLE:
        return "?"
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_attr_name(node.value)}.{node.attr.value}"
    try:
        mod = cst.Module(body=[])
        return mod.code_for_node(node).strip()
    except Exception:
        return "?"

# params of
def _params_of(params) -> List[str]:
    out: List[str] = []
    for p in list(params.posonly_params) + list(params.params):
        out.append(p.name.value)
    if params.star_arg is not None and isinstance(params.star_arg, cst.Param):
        out.append("*" + params.star_arg.name.value)
    for p in params.kwonly_params:
        out.append(p.name.value)
    if params.star_kwarg is not None:
        out.append("**" + params.star_kwarg.name.value)
    return out

# bases of
def _bases_of(class_node) -> List[str]:
    return [_attr_name(arg.value) for arg in class_node.bases]


def _self_attrs_in(node, mod) -> set:
    """Find self.X / cls.X assignment targets in the subtree."""
    if not _AVAILABLE:
        return set()
    out = set()

    class V(cst.CSTVisitor):
        def visit_Assign(self, n):
            for tgt in n.targets:
                t = tgt.target
                if isinstance(t, cst.Attribute) and isinstance(t.value, cst.Name) \
                        and t.value.value in ("self", "cls"):
                    out.add(t.attr.value)

        def visit_AnnAssign(self, n):
            t = n.target
            if isinstance(t, cst.Attribute) and isinstance(t.value, cst.Name) \
                    and t.value.value in ("self", "cls"):
                out.add(t.attr.value)
    try:
        node.visit(V())
    except Exception:
        pass
    return out

# class level attrs
def _class_level_attrs(class_node) -> set:
    out = set()
    if not _AVAILABLE:
        return out
    body = class_node.body
    if not isinstance(body, cst.IndentedBlock):
        return out
    for stmt in body.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for sub in stmt.body:
                if isinstance(sub, cst.AnnAssign) and isinstance(sub.target, cst.Name):
                    out.add(sub.target.value)
    return out

# node src
def _node_src(mod, node) -> str:
    try:
        return mod.code_for_node(node)
    except Exception:
        return ""

# parse file
def _parse_file(path: str, src: str):
    classes: Dict[str, Dict] = {}
    funcs: Dict[str, Dict] = {}
    if not _AVAILABLE:
        return classes, funcs
    try:
        mod = cst.parse_module(src)
    except Exception as e:
        common.log_failure("libcst", path, e, src)
        return classes, funcs
    try:
        for stmt in mod.body:
            node = stmt
            if isinstance(stmt, cst.ClassDef):
                _add_class(node, mod, path, classes)
            elif isinstance(stmt, cst.FunctionDef):
                _add_func(node, mod, path, funcs)

    except Exception as e:
        common.log_failure("libcst", path, e, src)
    return classes, funcs

# add class
def _add_class(node, mod, path, out_classes):
    cname = node.name.value
    bases = _bases_of(node)
    attrs = _class_level_attrs(node)
    methods: Dict[str, Dict] = {}
    body = node.body
    if isinstance(body, cst.IndentedBlock):
        for ch in body.body:
            if isinstance(ch, cst.FunctionDef):
                mname = ch.name.value
                params = _params_of(ch.params)
                attrs |= _self_attrs_in(ch, mod)
                methods[mname] = {
                    "name": mname, "cls": cname, "file": path,
                    "params": params,
                    "body_src": _node_src(mod, ch),
                }
    out_classes[cname] = {
        "name": cname, "bases": bases, "attrs": attrs,
        "methods": methods, "body_src": _node_src(mod, node),
    }

# add func
def _add_func(node, mod, path, out_funcs):
    fname = node.name.value
    out_funcs[fname] = {
        "name": fname, "cls": None, "file": path,
        "params": _params_of(node.params),
        "body_src": _node_src(mod, node),
    }


def compute_facts(code_before: List[Dict], code_after: List[Dict]) -> Dict:
    """libcst backend; same signature/schema as structural_facts.compute_facts."""
    if not _AVAILABLE:
        return common.empty_facts()
    src_before = {f.get("file", ""): f.get("code", "") for f in code_before}
    src_after = {f.get("file", ""): f.get("code", "") for f in code_after}
    fcb: Dict[str, Dict[str, Dict]] = {}
    ffb: Dict[str, Dict[str, Dict]] = {}
    fca: Dict[str, Dict[str, Dict]] = {}
    ffa: Dict[str, Dict[str, Dict]] = {}
    for p, s in src_before.items():
        cls, fn = _parse_file(p, s)
        fcb[p] = cls
        ffb[p] = fn
    for p, s in src_after.items():
        cls, fn = _parse_file(p, s)
        fca[p] = cls
        ffa[p] = fn
    return common.assemble_facts(src_before, src_after, fcb, fca, ffb, ffa)

# libcst facts for case
def libcst_facts_for_case(case: Dict) -> str:
    code_before = case.get("code_before") or []
    code_after = case.get("code_after") or []
    if not code_before and not code_after:
        return ""
    try:
        facts = compute_facts(code_before, code_after)
    except Exception as e:
        common.log_failure("libcst", "<case>", e)
        facts = common.empty_facts()
    return render_facts_xml(facts)
