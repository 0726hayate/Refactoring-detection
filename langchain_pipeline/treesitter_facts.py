"""tree-sitter Python backend for structural-fact extraction.

Same compute_facts/render schema as structural_facts (the ast version);
parses each file with tree_sitter_python and walks the CST.
"""
from typing import Dict, List, Optional, Tuple

from .structural_facts import render_facts_xml
from . import _facts_common as common

_PARSER = None
_AVAILABLE = True
try:
    import tree_sitter  # type: ignore
    import tree_sitter_python  # type: ignore
    _LANG = tree_sitter.Language(tree_sitter_python.language())
    _PARSER = tree_sitter.Parser(_LANG)
except Exception as _e:
    _AVAILABLE = False
    _IMPORT_ERR = _e

# node text
def _node_text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _child_field(node, field: str):
    """Return the first child for a named field, or None."""
    return node.child_by_field_name(field)

# name of
def _name_of(node, src: bytes) -> str:
    n = _child_field(node, "name")
    return _node_text(n, src) if n is not None else "?"


def _params_of_function(node, src: bytes) -> List[str]:
    """Extract parameter names from function_definition's parameters node."""
    params_node = _child_field(node, "parameters")
    out: List[str] = []
    if params_node is None:
        return out
    for ch in params_node.children:
        t = ch.type
        if t == "identifier":
            out.append(_node_text(ch, src))
        elif t == "typed_parameter":
            for sub in ch.children:
                if sub.type == "identifier":
                    out.append(_node_text(sub, src))
                    break
        elif t == "default_parameter" or t == "typed_default_parameter":
            n = _child_field(ch, "name")
            if n is not None:
                out.append(_node_text(n, src))
            else:
                for sub in ch.children:
                    if sub.type == "identifier":
                        out.append(_node_text(sub, src))
                        break
        elif t == "list_splat_pattern":
            for sub in ch.children:
                if sub.type == "identifier":
                    out.append("*" + _node_text(sub, src))
                    break
        elif t == "dictionary_splat_pattern":
            for sub in ch.children:
                if sub.type == "identifier":
                    out.append("**" + _node_text(sub, src))
                    break
    return out

# bases of class
def _bases_of_class(node, src: bytes) -> List[str]:
    arglist = _child_field(node, "superclasses")
    if arglist is None:
        return []
    out: List[str] = []
    for ch in arglist.children:
        if ch.type in ("identifier", "attribute"):
            out.append(_node_text(ch, src))
    return out


def _self_attrs_in(node, src: bytes) -> set:
    """Walk subtree, collect self.X / cls.X assignment targets."""
    out = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "assignment":
            left = _child_field(n, "left")
            if left is not None and left.type == "attribute":
                obj = _child_field(left, "object")
                attr = _child_field(left, "attribute")
                if obj is not None and attr is not None and _node_text(obj, src) in ("self", "cls"):
                    out.add(_node_text(attr, src))
        for ch in n.children:
            stack.append(ch)
    return out


def _class_level_annotated(class_node, src: bytes) -> set:
    """Class-body annotated assignments: `x: int = 0` -> {'x'}."""
    out = set()
    body = _child_field(class_node, "body")
    if body is None:
        return out
    for ch in body.children:
        if ch.type == "expression_statement":
            for sub in ch.children:
                if sub.type == "assignment":
                    left = _child_field(sub, "left")
                    typ = _child_field(sub, "type")
                    if typ is not None and left is not None and left.type == "identifier":
                        out.add(_node_text(left, src))
    return out


def _walk_top(tree_root):
    """Yield top-level statements inside a module node."""
    for ch in tree_root.children:
        yield ch


def _parse_file(path: str, src: str) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Return (classes_map, funcs_map) for a file's classes + module-level funcs."""
    classes: Dict[str, Dict] = {}
    funcs: Dict[str, Dict] = {}
    if not _AVAILABLE or _PARSER is None:
        return classes, funcs
    try:
        src_bytes = src.encode("utf-8", errors="replace")
        tree = _PARSER.parse(src_bytes)
    except Exception as e:
        common.log_failure("treesitter", path, e, src)
        return classes, funcs
    try:
        root = tree.root_node
        for top in _walk_top(root):
            if top.type == "class_definition":
                cname = _name_of(top, src_bytes)
                bases = _bases_of_class(top, src_bytes)
                attrs = _class_level_annotated(top, src_bytes)
                methods: Dict[str, Dict] = {}
                body = _child_field(top, "body")
                if body is not None:
                    for ch in body.children:
                        if ch.type == "function_definition":
                            mname = _name_of(ch, src_bytes)
                            params = _params_of_function(ch, src_bytes)
                            attrs |= _self_attrs_in(ch, src_bytes)
                            methods[mname] = {
                                "name": mname, "cls": cname, "file": path,
                                "params": params,
                                "body_src": _node_text(ch, src_bytes),
                            }
                classes[cname] = {
                    "name": cname, "bases": bases, "attrs": attrs,
                    "methods": methods, "body_src": _node_text(top, src_bytes),
                }
            elif top.type == "function_definition":
                fname = _name_of(top, src_bytes)
                funcs[fname] = {
                    "name": fname, "cls": None, "file": path,
                    "params": _params_of_function(top, src_bytes),
                    "body_src": _node_text(top, src_bytes),
                }
            elif top.type == "decorated_definition":
                inner = top.child_by_field_name("definition")
                if inner is None:
                    continue
                if inner.type == "class_definition":
                    cname = _name_of(inner, src_bytes)
                    bases = _bases_of_class(inner, src_bytes)
                    attrs = _class_level_annotated(inner, src_bytes)
                    methods = {}
                    body = _child_field(inner, "body")
                    if body is not None:
                        for ch in body.children:
                            real = ch
                            if ch.type == "decorated_definition":
                                real = ch.child_by_field_name("definition") or ch
                            if real.type == "function_definition":
                                mname = _name_of(real, src_bytes)
                                params = _params_of_function(real, src_bytes)
                                attrs |= _self_attrs_in(real, src_bytes)
                                methods[mname] = {
                                    "name": mname, "cls": cname, "file": path,
                                    "params": params,
                                    "body_src": _node_text(real, src_bytes),
                                }
                    classes[cname] = {
                        "name": cname, "bases": bases, "attrs": attrs,
                        "methods": methods, "body_src": _node_text(inner, src_bytes),
                    }
                elif inner.type == "function_definition":
                    fname = _name_of(inner, src_bytes)
                    funcs[fname] = {
                        "name": fname, "cls": None, "file": path,
                        "params": _params_of_function(inner, src_bytes),
                        "body_src": _node_text(inner, src_bytes),
                    }
    except Exception as e:
        common.log_failure("treesitter", path, e, src)
    return classes, funcs


def compute_facts(code_before: List[Dict], code_after: List[Dict]) -> Dict:
    """tree-sitter backend; same signature/schema as structural_facts.compute_facts."""
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

# treesitter facts for case
def treesitter_facts_for_case(case: Dict) -> str:
    code_before = case.get("code_before") or []
    code_after = case.get("code_after") or []
    if not code_before and not code_after:
        return ""
    try:
        facts = compute_facts(code_before, code_after)
    except Exception as e:
        common.log_failure("treesitter", "<case>", e)
        facts = common.empty_facts()
    return render_facts_xml(facts)
