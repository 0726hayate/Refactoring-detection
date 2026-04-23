"""
Static prompt templates and message builders for the 4-stage LangChain
refactoring-detection pipeline.

Stage layout:
    1. Level Classifier         (1 LLM call)
    2. Open-ended detection     (1 LLM call per selected level)
       - Step 1: defined types from the level's known set
       - Step 2: undefined patterns ("UnknownType: <description>")
    3. Missing-type detector    (1 LLM call) — recall direction
       Adds types Stage 2 missed using MISSING_HINTS-driven candidates and
       retrieved Java exemplars per candidate.
    4. Confusion verifier       (1 LLM call) — precision direction
       Trims / swaps the combined (Stage 2 ∪ Stage 3) detection list using
       retrieved confusion-partner Java exemplars. Cannot add new types.

All prompts use XML for structured I/O so the LLM output can be parsed
unambiguously.

Standalone module — no DSPy dependency.
"""
import os
import re
from typing import Dict, List

from .constants import (
    CONFUSION_HINTS,
    LEVEL_TYPES,
    PARAMETER_LEVEL_TYPES,
    METHOD_LEVEL_TYPES,
    CLASS_LEVEL_TYPES,
    build_level_definitions_xml,
)


def _no_think_enabled() -> bool:
    """True when reasoning-mode should be suppressed via `/no_think` directive.

    Triggered by LANGCHAIN_NO_THINK=1 or LANGCHAIN_PRECISION_MODE=1. The
    qwen3 family honours this tag in the system prompt.
    """
    for key in ("LANGCHAIN_NO_THINK", "LANGCHAIN_PRECISION_MODE"):
        if os.environ.get(key, "").lower() in ("1", "true", "yes"):
            return True
    return False


def _apply_no_think(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Append `/no_think` to the system message when thinking-mode is disabled.

    Done as a prompt directive (works on langchain_ollama 1.0.1 which doesn't
    expose Ollama's `think` option at the API level). <think>…</think> blocks
    that still leak through are stripped at parse time by
    `_preprocess_malformed_xml` in pipeline.py.
    """
    if not _no_think_enabled() or not messages:
        return messages
    out = list(messages)
    if out and out[0].get("role") == "system":
        sys = out[0]
        if "/no_think" not in sys.get("content", ""):
            out[0] = {**sys, "content": sys["content"].rstrip() + "\n\n/no_think"}
    return out

# STAGE 1 -- Level Classifier

STAGE1_SYSTEM: str = """\
You are a refactoring-level classifier for Python code changes.

Classify which abstraction level(s) the code change operates at.

Levels:
- parameter_level: changes to function/method signatures
  (add, remove, rename, reorder, split parameters; parameterize or localize a variable)
- method_level: changes within or to method/function bodies
  (extract, inline, move, rename methods; variable operations;
   condition restructuring: split conditional, invert condition; move code)
- class_level: changes to class hierarchy or structure
  (move/rename/extract class/superclass/subclass; pull up/push down members;
   attribute operations; decorator/annotation changes)

A change can operate at multiple levels simultaneously. Respond with ALL that apply.

Output ONLY the applicable levels wrapped in XML tags.
Example: <levels><level>parameter_level</level><level>method_level</level></levels>
"""
# STAGE 2 -- Open-ended two-step detection (per level)

_STAGE2_TEMPLATE: str = """\
You are a {level_human} refactoring detector for Python code changes.

"Refactoring is a disciplined technique for restructuring an existing body of \
code, altering its internal structure without changing its external behavior." \
— Martin Fowler

You receive THREE inputs in the user message:
1. <python_diff> — the Python code change to analyze.
2. <java_reference_examples> — Java refactorings that an upstream retrieval
   pipeline matched to this case. Use them as PATTERNS to recognize the same
   operations in the Python diff.
3. <reference> — definitions of the {n_defined} {level_human} known types.

Your task has TWO STEPS. Do BOTH.

STEP 1 — DEFINED TYPES.
List which of the {n_defined} {level_human} known types appear in the Python
diff. For each detection, you MUST quote a specific +/- diff line that proves
it. NO type without an evidence quote.

STEP 2 — UNDEFINED TYPES.
After Step 1, scan the diff once more. If you see ANY refactoring pattern that
does NOT match any of the {n_defined} known types, list it as
"UnknownType: <one-line description>" with the same evidence requirement.
This is how novel refactoring patterns are surfaced — be specific about what
was changed (e.g. "convert method to cached_property descriptor").

CRITICAL RULES
- Evidence is mandatory. A type without an evidence quote is dropped.
- Be specific. Quote the literal +/- line from the diff, not a paraphrase.
- The two steps are independent — undefined patterns can coexist with defined ones.
- Stay within the {level_human}: leave other levels to their dedicated detectors.
- Read the diff direction carefully: '+' lines are 'after', '-' lines are 'before'.

OUTPUT FORMAT (XML, strictly):
<detected>
  <defined>
    <type evidence="+def foo(self, new_param):">Add Parameter</type>
    <type evidence="-class A(B):">Rename Class</type>
  </defined>
  <undefined>
    <type evidence="+@functools.cached_property">UnknownType: replace method with cached_property</type>
  </undefined>
</detected>

If no defined types apply, leave <defined/> empty.
If no undefined patterns apply, leave <undefined/> empty.
Output ONLY the <detected>...</detected> block. No prose, no thinking trace.
"""
# make stage2 system
def _make_stage2_system(level: str) -> str:
    types = LEVEL_TYPES.get(level, [])
    return _STAGE2_TEMPLATE.format(
        level_human=level.replace("_", "-"),
        n_defined=len(types),
    )

STAGE2_PARAMETER_SYSTEM: str = _make_stage2_system("parameter_level")
STAGE2_METHOD_SYSTEM: str = _make_stage2_system("method_level")
STAGE2_CLASS_SYSTEM: str = _make_stage2_system("class_level")

_STAGE2_SYSTEM_PROMPTS: Dict[str, str] = {
    "parameter_level": STAGE2_PARAMETER_SYSTEM,
    "method_level": STAGE2_METHOD_SYSTEM,
    "class_level": STAGE2_CLASS_SYSTEM,
}

# STAGE 4 -- Confusion verifier (precision direction; runs AFTER Stage 3)

STAGE4_VERIFIER_SYSTEM: str = """\
You are a refactoring verification expert. Your single job is to TRIM AND
CORRECT a list of refactoring detections that has been built up by earlier
stages. You may REMOVE wrong detections or SWAP a wrong type for a confusion
partner. You must NOT add wholly new types.

You receive:
1. <python_diff> — the diff to re-examine.
2. <detections_so_far> — the types collected by earlier stages, with the
   evidence quotes used to support them. Some come from the initial detection
   (Stage 2) and some from a missing-type expansion step (Stage 3); treat
   them all the same — verify each one independently against the diff.
3. <retrieved_examples> — for each known confusion partner of those types,
   several Java refactoring examples (from the same type) showing what that
   pattern looks like in real code.
4. <confusion_hints> — terse rules for the most common confusion patterns.

Your task — for each detection, decide one of:
- KEEP: the detection is correct (the python diff matches the type's pattern,
  and the retrieved examples agree).
- REMOVE: there is no real evidence; the detection was a guess.
- SWAP: the python pattern actually matches a confusion partner type better
  than the originally detected one — output the partner type instead.

CRITICAL RULES
- Output evidence is mandatory for KEEP and SWAP. Quote the actual diff line.
- A SWAP must point to a type that was either in <detections_so_far> or in
  one of its confusion partners (do not invent new types).
- Be RUTHLESS about REMOVE: if you cannot quote a clear +/- line that proves
  the type, REMOVE it. The cost of a wrong type is high.
- Compound types (Move And Rename Method, Extract And Move Method,
  Move And Rename Class) are atomic. If the initial detection contains BOTH
  components AND the compound, KEEP only the compound.
- For class-hierarchy direction: members moving INTO an existing parent →
  Pull Up; FROM the parent INTO a child → Push Down; new parent created →
  Extract Superclass; new child created → Extract Subclass.

OUTPUT FORMAT (XML, strictly):
<verified>
  <type confidence="85" evidence="+def foo(self, x, y):">Add Parameter</type>
  <type confidence="40" evidence="-class Old:&#10;+class New:">Rename Class</type>
</verified>

confidence is an integer 0-100:
  90-100 = certain (clear structural evidence in the diff)
  60-89  = probable (evidence present but pattern is ambiguous)
  30-59  = uncertain (weak or indirect evidence)
  0-29   = very unlikely (almost no evidence; consider REMOVE instead)

If nothing survives, output <verified/>.
Output ONLY the <verified>...</verified> block. No prose.
"""
# STAGE 3 -- Missing-type detector (recall direction; runs BEFORE Stage 4)

STAGE3_MISSING_SYSTEM: str = """\
You are a refactoring discovery expert. Stage 2 already detected some types
in this Python diff. Your job is to find ADDITIONAL types it may have missed.
Stage 4 will later verify your additions, so you can be slightly more
permissive than Stage 2 — but every YES still requires concrete evidence.

You receive:
1. <python_diff> — the diff to re-examine.
2. <stage2_detected> — the types Stage 2 ALREADY found. Do NOT re-list these.
3. <candidate_missing_types> — the small set of refactoring types that are
   PLAUSIBLE additions based on co-occurrence with what Stage 2 found.
4. <retrieved_examples> — for each candidate missing type, several Java
   refactoring examples showing what that pattern looks like.

Your task — for EACH candidate missing type, decide present=yes or
present=no. For each YES, quote a specific +/- diff line as evidence.

CRITICAL RULES
- NO is the safe default. Only say YES when you can point to clear +/-
  evidence in the python diff.
- Do NOT re-confirm or repeat any type already in <stage2_detected>.
- Do NOT propose types not in <candidate_missing_types>.
- A YES without an evidence quote is dropped.
- The retrieved Java examples are PATTERNS, not python code — use them to
  recognize the SAME pattern in the python diff, not to copy code.

OUTPUT FORMAT (XML, strictly):
<additional>
  <type evidence="+    @cached">Add Method Annotation</type>
  <type evidence="-class Foo(Bar):  +class Foo(Baz):">Extract Superclass</type>
</additional>

If nothing should be added, output <additional/>.
Output ONLY the <additional>...</additional> block. No prose.
"""
# XML FORMATTING HELPERS
def _xml_escape(s: str) -> str:
    """Minimal XML escaping for attribute / text content."""
    if not s:
        return ""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _clean_java_code(text: str) -> str:
    """Remove pure-comment lines (// only), trailing whitespace, and collapse blank runs."""
    if not text:
        return text
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.rstrip()

        if re.match(r'^\s*//', stripped):
            continue
        cleaned.append(stripped)

    return re.sub(r'\n{3,}', '\n\n', '\n'.join(cleaned)).strip()


def _clean_diff_text(text: str) -> str:
    """Strip trailing whitespace per line and collapse 3+ consecutive blank lines."""
    if not text:
        return text
    lines = [line.rstrip() for line in text.splitlines()]

    return re.sub(r'(^[ \t]*$\n){3,}', '\n\n', '\n'.join(lines), flags=re.MULTILINE)


def _clean_code_text(text: str) -> str:
    """Clean code_text (no truncation — caller controls how many examples to include)."""
    if not text:
        return ""
    return _clean_java_code(text)

# sort java examples
def _sort_java_examples(java_examples: list, max_n: int) -> list:
    return sorted(
        java_examples,
        key=lambda x: (
            x.get("rerank_score")
            if x.get("rerank_score") is not None
            else x.get("similarity", 0)
        ),
        reverse=True,
    )[:max_n]

# format code diff xml
def format_code_diff_xml(code_text: str) -> str:
    if not code_text or not code_text.strip():
        return "<python_diff/>"
    return f"<python_diff>\n{_clean_diff_text(code_text)}\n</python_diff>"


def _example_chars(ex: dict) -> str:
    """Return cleaned code_text for a Java example (no truncation)."""
    code_text = ex.get("code_text") or ""
    if not code_text:
        cb = ex.get("code_before", "") or ""
        ca = ex.get("code_after", "") or ""
        if cb or ca:
            code_text = (f"BEFORE:\n{cb}\n" if cb else "") + (f"AFTER:\n{ca}" if ca else "")
    return _clean_code_text(code_text)


def format_java_reference_examples_xml(
    java_examples: list,
    level: str,
    max_n: int = 15,
    char_budget: int = None,
) -> str:
    """Format upstream-matched Java examples for use in Stage 2 prompts.

    Filters to examples whose ``refactoring_type`` is in the level's known
    type set, sorts by score, then greedily includes full examples until
    ``char_budget`` is exhausted (never truncates individual examples).
    """
    level_types = set(LEVEL_TYPES.get(level, []))
    relevant = [
        ex for ex in java_examples
        if ex.get("refactoring_type", "") in level_types
    ]
    if not relevant:
        return "<java_reference_examples/>"

    candidates = _sort_java_examples(relevant, max_n)

    selected = []
    used_chars = 0
    for ex in candidates:
        ct = _example_chars(ex)
        ex_chars = len(ct) + len(ex.get("description", "") or "")
        if char_budget is not None and used_chars + ex_chars > char_budget:
            continue
        selected.append((ex, ct))
        used_chars += ex_chars

    if not selected:
        return "<java_reference_examples/>"

    parts: List[str] = ["<java_reference_examples>"]
    for ex, code_text in selected:
        ref_type = _xml_escape(ex.get("refactoring_type", "Unknown"))
        description = _xml_escape(ex.get("description", ""))
        parts.append(f'  <example type="{ref_type}">')
        if description:
            parts.append(f"    <description>{description}</description>")
        if code_text:
            parts.append(f"    <code_text>{_xml_escape(code_text)}</code_text>")
        parts.append("  </example>")
    parts.append("</java_reference_examples>")
    return "\n".join(parts)


def format_retrieved_examples_xml(
    results_by_type: Dict[str, list],
    char_budget: int = None,
) -> str:
    """Format the per-type retrieval output from JavaExampleRetriever for a
    Stage 3/4 prompt.

    Args:
        results_by_type: ``{type_name: [java_record, ...]}``.
        char_budget: Max chars for all example code_text combined. Examples
            that would exceed the budget are omitted (never truncated).

    Returns:
        XML block with one ``<for_type>`` group per target type.
    """
    if not results_by_type:
        return "<retrieved_examples/>"

    parts: List[str] = ["<retrieved_examples>"]
    used_chars = 0
    for type_name, records in results_by_type.items():
        if not records:
            continue
        type_parts = [f'  <for_type name="{_xml_escape(type_name)}">']
        for r in records:
            ref_type = _xml_escape(r.get("refactoring_type", "Unknown"))
            description = _xml_escape(r.get("description", ""))
            code_text = _clean_code_text(r.get("code_text", ""))
            ex_chars = len(code_text) + len(description)
            if char_budget is not None and used_chars + ex_chars > char_budget:
                continue
            used_chars += ex_chars
            type_parts.append(f'    <example type="{ref_type}">')
            if description:
                type_parts.append(f"      <description>{description}</description>")
            if code_text:
                type_parts.append(
                    f"      <code_text>{_xml_escape(code_text)}</code_text>"
                )
            type_parts.append("    </example>")
        if len(type_parts) > 1:
            type_parts.append("  </for_type>")
            parts.extend(type_parts)
    parts.append("</retrieved_examples>")
    return "\n".join(parts)


def format_stage2_detected_xml(detected: List[tuple]) -> str:
    """Format Stage 2 output (list of (type, evidence) tuples) for Stage 3."""
    if not detected:
        return "<stage2_detected/>"
    parts = ["<stage2_detected>"]
    for type_name, evidence in detected:
        ev = _xml_escape(evidence) if evidence else ""
        parts.append(f'  <type evidence="{ev}">{_xml_escape(type_name)}</type>')
    parts.append("</stage2_detected>")
    return "\n".join(parts)


def format_combined_detected_xml(detected: List[tuple]) -> str:
    """Format the combined Stage 2 + Stage 3 detection list for Stage 4.

    Same shape as ``format_stage2_detected_xml`` but uses the
    ``<detections_so_far>`` wrapper that Stage 4's prompt expects.
    """
    if not detected:
        return "<detections_so_far/>"
    parts = ["<detections_so_far>"]
    for type_name, evidence in detected:
        ev = _xml_escape(evidence) if evidence else ""
        parts.append(f'  <type evidence="{ev}">{_xml_escape(type_name)}</type>')
    parts.append("</detections_so_far>")
    return "\n".join(parts)


def format_candidate_missing_xml(candidates: List[str]) -> str:
    """Format the list of candidate missing types for Stage 3 Call B."""
    if not candidates:
        return "<candidate_missing_types/>"
    parts = ["<candidate_missing_types>"]
    for c in candidates:
        parts.append(f"  <type>{_xml_escape(c)}</type>")
    parts.append("</candidate_missing_types>")
    return "\n".join(parts)


def format_confusion_hints_xml(detected_types: list) -> str:
    """Build XML confusion hints (used by Stage 3A only)."""
    hints: List[str] = []
    for t in detected_types:
        if t in CONFUSION_HINTS:
            hints.append(
                f'  <hint type="{_xml_escape(t)}">'
                f"{_xml_escape(CONFUSION_HINTS[t])}</hint>"
            )

    compound_trigger = {
        "Move Method", "Rename Method", "Move Class", "Rename Class",
        "Extract Method", "Move Code",
    }
    if any(t in compound_trigger for t in detected_types):
        for compound in [
            "Move And Rename Method",
            "Move And Rename Class",
            "Extract And Move Method",
        ]:
            if compound not in detected_types and compound in CONFUSION_HINTS:
                hints.append(
                    f'  <hint type="{compound}" status="check_compound">'
                    f"{_xml_escape(CONFUSION_HINTS[compound])}</hint>"
                )
    if not hints:
        return "<confusion_hints/>"
    return "\n".join(["<confusion_hints>"] + hints + ["</confusion_hints>"])

# MESSAGE BUILDERS

# build stage1 messages
def build_stage1_messages(code_diff: str) -> List[Dict[str, str]]:
    user_content = (
        "Classify which abstraction level(s) this Python code change "
        "operates at.\n\n"
        + format_code_diff_xml(code_diff)
    )
    return _apply_no_think([
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": user_content},
    ])

_STAGE_OVERHEAD_CHARS = 12000  # conservative: system prompt + XML structure + LLM output


def build_stage2_messages(
    code_diff: str,
    level: str,
    java_examples: list = None,
    max_java: int = 15,
    num_ctx: int = 40960,
    use_tools: bool = False,
    structural_facts_xml: str = "",
    trophies_xml: str = "",
) -> List[Dict[str, str]]:
    """Build messages for Stage 2 (open-ended two-step detection).

    If use_tools=True, strips inline examples/definitions and adds tool-use
    instructions. The LLM should call get_definition_tool, retrieve_java_examples_tool,
    lookup_term_tool, and pull_full_commit_tool as needed.
    """
    system_prompt = _STAGE2_SYSTEM_PROMPTS.get(level)
    if system_prompt is None:
        valid = ", ".join(sorted(_STAGE2_SYSTEM_PROMPTS.keys()))
        raise ValueError(f"Unknown level {level!r}. Valid levels: {valid}")

    diff_xml = format_code_diff_xml(code_diff)
    facts_block = f"{structural_facts_xml}\n\n" if structural_facts_xml else ""
    trophies_block = f"{trophies_xml}\n\n" if trophies_xml else ""

    if use_tools:
        level_types = LEVEL_TYPES.get(level, [])
        type_list = ", ".join(level_types)
        user_content = (
            f"Detect refactoring types at the {level.replace('_', '-')} "
            f"in this Python code change.\n\n"
            f"{diff_xml}\n\n"
            f"{facts_block}"
            f"{trophies_block}"
            f"Candidate types at this level: {type_list}\n\n"
            f"REQUIRED WORKFLOW:\n"
            f"1. First, identify 2-5 candidate types from the diff patterns.\n"
            f"2. For EACH candidate, call get_definition_tool to check its definition.\n"
            f"3. For EACH candidate, call retrieve_java_examples_tool to see real "
            f"code examples of that type — compare the Java pattern with the Python diff.\n"
            f"4. If the diff context is unclear, call pull_full_commit_tool to see "
            f"the full Python source files.\n"
            f"5. After gathering evidence, produce your final XML output.\n\n"
            f"You MUST call retrieve_java_examples_tool at least once to verify "
            f"your candidates against real examples."
        )
    else:
        char_budget = int(num_ctx * 4 * 0.65) - len(code_diff) - _STAGE_OVERHEAD_CHARS
        java_xml = format_java_reference_examples_xml(
            java_examples or [], level, max_n=max_java, char_budget=max(0, char_budget)
        )
        ref_xml = build_level_definitions_xml(level)
        user_content = (
            f"Detect refactoring types at the {level.replace('_', '-')} "
            f"in this Python code change.\n\n"
            f"{diff_xml}\n\n"
            f"{facts_block}"
            f"{trophies_block}"
            f"{java_xml}\n\n"
            f"{ref_xml}"
        )
    return _apply_no_think([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ])

# Build messages for Stage 3 (missing-type detector)
def build_stage3_messages(
    code_diff: str,
    stage2_detected: List[tuple],
    candidate_missing: List[str],
    retrieved_examples: Dict[str, list],
    num_ctx: int = 40960,
    use_tools: bool = False,
    structural_facts_xml: str = "",
    trophies_xml: str = "",
) -> List[Dict[str, str]]:
    """Build messages for Stage 3 (missing-type detector). Runs BEFORE Stage 4."""
    diff_xml = format_code_diff_xml(code_diff)
    detected_xml = format_stage2_detected_xml(stage2_detected)
    facts_block = f"{structural_facts_xml}\n\n" if structural_facts_xml else ""
    trophies_block = f"{trophies_xml}\n\n" if trophies_xml else ""

    if use_tools:
        candidates_str = ", ".join(candidate_missing) if candidate_missing else "none"
        user_content = (
            "Decide which candidate missing types are present in the diff. "
            "ADD types only — do NOT re-confirm Stage 2 detections.\n\n"
            f"{diff_xml}\n\n"
            f"{facts_block}"
            f"{trophies_block}"
            f"{detected_xml}\n\n"
            f"Candidate missing types to check: {candidates_str}\n\n"
            f"REQUIRED WORKFLOW:\n"
            f"1. For each candidate, call get_definition_tool to understand it.\n"
            f"2. For each candidate, call retrieve_java_examples_tool to see what "
            f"the pattern looks like in real code.\n"
            f"3. Compare the Java examples against the Python diff.\n"
            f"4. Only ADD types with clear evidence in the diff.\n\n"
            f"You MUST call retrieve_java_examples_tool for at least the top "
            f"2 candidates before deciding."
        )
    else:
        candidates_xml = format_candidate_missing_xml(candidate_missing)
        char_budget = int(num_ctx * 4 * 0.5) - len(code_diff) - _STAGE_OVERHEAD_CHARS
        retrieved_xml = format_retrieved_examples_xml(retrieved_examples, char_budget=max(0, char_budget))
        user_content = (
            "Decide which candidate missing types are present in the diff. "
            "ADD types only — do NOT re-confirm Stage 2 detections.\n\n"
            f"{diff_xml}\n\n"
            f"{facts_block}"
            f"{trophies_block}"
            f"{detected_xml}\n\n"
            f"{candidates_xml}\n\n"
            f"{retrieved_xml}"
        )
    return _apply_no_think([
        {"role": "system", "content": STAGE3_MISSING_SYSTEM},
        {"role": "user", "content": user_content},
    ])


def build_stage4_messages(
    code_diff: str,
    combined_detected: List[tuple],
    retrieved_examples: Dict[str, list],
    num_ctx: int = 40960,
    use_tools: bool = False,
) -> List[Dict[str, str]]:
    """Build messages for Stage 4 (confusion verifier).

    *combined_detected* is the union of Stage 2 detections and Stage 3
    additions, as ``[(type, evidence), ...]`` tuples.
    """
    diff_xml = format_code_diff_xml(code_diff)
    detected_xml = format_combined_detected_xml(combined_detected)

    if use_tools:
        detected_type_names = [t for t, _ in combined_detected]
        type_list = ", ".join(detected_type_names)
        user_content = (
            "Verify every detection collected so far. Trim, swap, or keep — but "
            "do NOT add new types.\n\n"
            f"{diff_xml}\n\n"
            f"{detected_xml}\n\n"
            f"Detected types to verify: {type_list}\n\n"
            f"REQUIRED WORKFLOW:\n"
            f"1. For EACH detected type, call get_confusion_hints_tool to check "
            f"what it is commonly confused with.\n"
            f"2. For types with confusion partners, call retrieve_java_examples_tool "
            f"for BOTH the detected type and its confusion partner — compare them.\n"
            f"3. TRIM types with no clear evidence. SWAP if a partner fits better.\n"
            f"4. Assign confidence 0-100 per type in the output XML.\n\n"
            f"You MUST call get_confusion_hints_tool for each detected type."
        )
    else:
        char_budget = int(num_ctx * 4 * 0.5) - len(code_diff) - _STAGE_OVERHEAD_CHARS
        retrieved_xml = format_retrieved_examples_xml(retrieved_examples, char_budget=max(0, char_budget))
        detected_type_names = [t for t, _ in combined_detected]
        hints_xml = format_confusion_hints_xml(detected_type_names)
        user_content = (
            "Verify every detection collected so far. Trim, swap, or keep — but "
            "do NOT add new types.\n\n"
            f"{diff_xml}\n\n"
            f"{detected_xml}\n\n"
            f"{retrieved_xml}\n\n"
            f"{hints_xml}"
        )
    return _apply_no_think([
        {"role": "system", "content": STAGE4_VERIFIER_SYSTEM},
        {"role": "user", "content": user_content},
    ])

# One-line canonical definitions for the 39 refactoring types.
# Used by the adversarial per-detection prompt (M3 Lever B) to give the model
# a concise type contract to verify against.
TYPE_DEFINITIONS: dict = {
    "Add Parameter":                  "a new parameter is added to a method signature",
    "Remove Parameter":               "an existing parameter is removed from a method signature",
    "Rename Parameter":               "a parameter is renamed within a method signature",
    "Rename Method":                  "a method is renamed while its body and parameters remain",
    "Move Method":                    "a method is moved from one class to another",
    "Extract Method":                 "a code block is extracted from an existing method into a new method",
    "Inline Method":                  "a method call is replaced by the method's body",
    "Extract And Move Method":        "a code block is extracted into a new method and moved to another class",
    "Move And Rename Method":         "a method is moved to another class and also renamed",
    "Add Method Annotation":          "a new annotation is added to a method declaration",
    "Remove Method Annotation":       "an existing annotation is removed from a method declaration",
    "Rename Class":                   "a class is renamed while its members remain unchanged",
    "Move Class":                     "a class is moved to a different module or package",
    "Move And Rename Class":          "a class is moved to a different module and also renamed",
    "Extract Class":                  "members are moved out of an existing class into a new class",
    "Extract Subclass":               "members are moved into a new subclass of the original class",
    "Extract Superclass":             "common members of two or more classes are moved into a new superclass",
    "Add Class Annotation":           "a new annotation is added to a class declaration",
    "Remove Class Annotation":        "an existing annotation is removed from a class declaration",
    "Rename Attribute":               "a field or attribute is renamed within its class",
    "Move Attribute":                 "an attribute is moved from one class to another",
    "Pull Up Attribute":              "an attribute is moved from a subclass up to its superclass",
    "Push Down Attribute":            "an attribute is moved from a superclass down to a subclass",
    "Pull Up Method":                 "a method is moved from a subclass up to its superclass",
    "Push Down Method":               "a method is moved from a superclass down to a subclass",
    "Extract Variable":               "a sub-expression is extracted into a named local variable",
    "Inline Variable":                "a variable is inlined — usages replaced by its initialiser expression",
    "Rename Variable":                "a local variable is renamed within its scope",
    "Replace Variable With Attribute":"a local variable is replaced by a class attribute (self.x replaces x)",
    "Replace Attribute With Variable":"a class attribute is replaced by a local variable",
    "Encapsulate Attribute":          "direct attribute access is replaced by getter/setter methods",
    "Parameterize Attribute":         "a hard-coded value in a class is turned into a constructor parameter",
    "Change Variable Type":           "the type annotation of a variable, parameter, or attribute is changed",
    "Invert Condition":               "a boolean condition is logically negated (not added or comparator flipped)",
    "Split Parameter":                "a single parameter is replaced by multiple parameters",
    "Localize Parameter":             "a parameter is removed and its value is computed locally instead",
    "Move Code":                      "a code block is moved to a different location without being extracted into a method",
    "Push Down Attribute":            "an attribute is moved from a superclass down to a subclass",
    "Split Conditional":              "a compound conditional is split into nested simpler conditionals",
}

STAGE5_SYSTEM = (
    "You are doing the FINAL evidence-review step for a Python refactoring "
    "detector. You see the full diff, the AST-level structural facts, the "
    "intra-method patterns, and the Stage-4 detections with their cited "
    "evidence and confidence scores. For each detection, you decide KEEP, "
    "DROP, or UNCERTAIN based on whether the evidence physically supports "
    "the type. UNCERTAIN means 'plausible but I can't prove it' — the "
    "system will fall back to the Stage-4 confidence threshold."
)


def build_stage5_prompt(
    code_diff: str,
    facts_xml: str,
    detections: list,
) -> str:
    """Assemble the Stage 5 evidence-review prompt body.

    Args:
        code_diff: full unified diff
        facts_xml: combined structural-facts XML (gumtree + intra blocks
            already concatenated by `_preprocess_case`); may be "".
        detections: list of dicts {type, evidence, confidence, definition}
    """
    det_lines = []
    for d in detections:
        det_lines.append(
            f"- {d['type']} (confidence {d.get('confidence', '?')}): "
            f"definition: {d.get('definition', 'a code-level refactoring')} "
            f"| evidence quoted by Stage 4: {d.get('evidence') or '(none)'}"
        )
    detections_block = "\n".join(det_lines) if det_lines else "(no detections)"

    facts_block = (facts_xml or "").strip() or "(no structural / intra-method facts available — parsers may have failed)"

    return (
        "Review the detections below using ALL the evidence shown.\n\n"
        "=== Full diff ===\n"
        f"{code_diff}\n\n"
        "=== Structural and intra-method facts ===\n"
        f"{facts_block}\n\n"
        "=== Stage 4 detections to verify ===\n"
        f"{detections_block}\n\n"
        "=== Output format ===\n"
        "For EACH detection above, output exactly one line:\n"
        "  <type> | KEEP | <one-line reason citing a fact tag OR diff line>\n"
        "  <type> | DROP | <one-line reason: evidence absent, paraphrased, or contradicted>\n"
        "  <type> | UNCERTAIN | <one-line reason>\n\n"
        "Rules:\n"
        "1. KEEP only if (a) the cited evidence line genuinely appears in the "
        "diff above AND (b) at least one structural fact OR intra-method "
        "pattern corroborates the type (e.g. <methods><added> for Extract "
        "Method, <renames> for Rename Variable, <classes><moved> for Move Class).\n"
        "2. DROP if the evidence is fabricated, paraphrased, or the "
        "structural facts actively contradict the type.\n"
        "3. UNCERTAIN if the evidence is plausible but neither confirmed nor "
        "contradicted — system will fall back to Stage-4 confidence.\n"
        "4. Output one line per detection, in the same order they were listed. "
        "Do not output anything else."
    )
