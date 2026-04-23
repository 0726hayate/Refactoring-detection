"""parso backend for structural-fact extraction.

parso supports error_recovery, so it returns a partial tree even for files
with syntax errors -- handy for the noisy commits in the train slice.
"""
from typing import Dict, List, Optional, Tuple

from .structural_facts import render_facts_xml
from . import _facts_common as common

_AVAILABLE = True
try:
    import parso  # type: ignore
except Exception as _e:
    _AVAILABLE = False
    _IMPORT_ERR = _e


def _bases_of(class_node) -> List[str]:
    """Pull base-class names out of the parens after the class name."""
    out: List[str] = []
    children = class_node.children
    lp_idx = None
    for i, c in enumerate(children):
        if getattr(c, "value", None) == "(":
            lp_idx = i
            break
    if lp_idx is None:
        return out
    rp_idx = None
    for j in range(lp_idx + 1, len(children)):
        if getattr(children[j], "value", None) == ")":
            rp_idx = j
            break
    if rp_idx is None:
        return out
    inner = children[lp_idx + 1:rp_idx]
    if not inner:
        return out
    if len(inner) == 1 and inner[0].type == "arglist":
        for sub in inner[0].children:
            if getattr(sub, "value", None) == ",":
                continue
            txt = sub.get_code().strip()
            if txt and "=" not in txt:
                out.append(txt)
    else:
        for sub in inner:
            if getattr(sub, "value", None) == ",":
                continue
            txt = sub.get_code().strip()
            if txt and "=" not in txt:
                out.append(txt)
    return out

# params of
def _params_of(fn_node) -> List[str]:
    out: List[str] = []
    try:
        for p in fn_node.get_params():
            star = getattr(p, "star_count", 0)
            name = p.name.value if hasattr(p, "name") and p.name is not None else p.get_code().strip()
            if star == 1:
                out.append("*" + name)
            elif star == 2:
                out.append("**" + name)
            else:
                out.append(name)
    except Exception:
        pass
    return out


def _walk(node):
    """Iterate every descendant (and self)."""
    yield node
    for ch in getattr(node, "children", []) or []:
        yield from _walk(ch)


def _self_attrs_in(node) -> set:
    """Detect `self.X = ...` assignments anywhere inside the subtree."""
    out = set()
    for n in _walk(node):
        if n.type == "expr_stmt":
            children = n.children

            if len(children) >= 2 and getattr(children[1], "value", None) == "=":
                left = children[0]
                attr = _trailer_self_attr(left)
                if attr:
                    out.add(attr)
            elif len(children) >= 2 and children[1].type == "annassign":
                left = children[0]
                attr = _trailer_self_attr(left)
                if attr:
                    out.add(attr)
    return out


def _trailer_self_attr(node) -> Optional[str]:
    """If node is `self.X` or `cls.X`, return X."""
    if node.type != "atom_expr" and node.type != "power":
        return None
    children = node.children
    if len(children) < 2:
        return None
    base = children[0]
    if not (hasattr(base, "value") and base.value in ("self", "cls")):
        return None
    trailer = children[1]
    if trailer.type != "trailer" or len(trailer.children) < 2:
        return None
    if getattr(trailer.children[0], "value", None) != ".":
        return None
    name_node = trailer.children[1]
    if hasattr(name_node, "value"):
        return name_node.value
    return None


def _class_level_attrs(class_node) -> set:
    """Class-body annotated assignments: `x: int = 0`."""
    out = set()
    suite = class_node.children[-1]
    if suite.type != "suite":
        return out
    for ch in suite.children:
        if ch.type == "simple_stmt":
            for sub in ch.children:
                if sub.type == "expr_stmt":
                    cs = sub.children
                    if len(cs) >= 2 and cs[1].type == "annassign":
                        left = cs[0]
                        if hasattr(left, "value"):
                            out.add(left.value)
    return out

# iter class body funcdefs
def _iter_class_body_funcdefs(class_node):
    suite = class_node.children[-1]
    if suite.type != "suite":
        return
    for ch in suite.children:
        if ch.type == "funcdef":
            yield ch
        elif ch.type == "decorated":
            inner = ch.children[-1]
            if inner.type == "funcdef":
                yield inner

# parse file
def _parse_file(path: str, src: str):
    classes: Dict[str, Dict] = {}
    funcs: Dict[str, Dict] = {}
    if not _AVAILABLE:
        return classes, funcs
    try:
        tree = parso.parse(src, error_recovery=True)
    except Exception as e:
        common.log_failure("parso", path, e, src)
        return classes, funcs
    try:
        for node in tree.children:
            real = node
            if node.type == "decorated":
                real = node.children[-1]
            if real.type == "classdef":
                _add_class(real, path, classes)
            elif real.type == "funcdef":
                _add_func(real, path, funcs)
    except Exception as e:
        common.log_failure("parso", path, e, src)
    return classes, funcs

# add class
def _add_class(node, path, out_classes):
    cname = node.name.value
    bases = _bases_of(node)
    attrs = _class_level_attrs(node)
    methods: Dict[str, Dict] = {}
    for fn in _iter_class_body_funcdefs(node):
        mname = fn.name.value
        params = _params_of(fn)
        attrs |= _self_attrs_in(fn)
        methods[mname] = {
            "name": mname, "cls": cname, "file": path,
            "params": params,
            "body_src": fn.get_code(),
        }
    out_classes[cname] = {
        "name": cname, "bases": bases, "attrs": attrs,
        "methods": methods, "body_src": node.get_code(),
    }

# add func
def _add_func(node, path, out_funcs):
    fname = node.name.value
    out_funcs[fname] = {
        "name": fname, "cls": None, "file": path,
        "params": _params_of(node),
        "body_src": node.get_code(),
    }


def compute_facts(code_before: List[Dict], code_after: List[Dict]) -> Dict:
    """parso backend; same signature/schema as structural_facts.compute_facts."""
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

# parso facts for case
def parso_facts_for_case(case: Dict) -> str:
    code_before = case.get("code_before") or []
    code_after = case.get("code_after") or []
    if not code_before and not code_after:
        return ""
    try:
        facts = compute_facts(code_before, code_after)
    except Exception as e:
        common.log_failure("parso", "<case>", e)
        facts = common.empty_facts()
    return render_facts_xml(facts)
