"""
Standalone tool definitions for the LangChain refactoring-detection pipeline.

Provides:
  - RefactoringDefinitionTool  -- looks up definitions from refactoring_defs.json
  - lookup_term                -- looks up action/target term definitions
  - get_full_code              -- fetches full file content from GitHub
  - LangChain @tool wrappers   -- get_definition_tool, lookup_term_tool
"""
import json
import os
import re
from typing import Dict, Optional

import requests
from langchain_core.tools import tool

from .code_cleanup import minify_python, basic_cleanup
from .constants import ACTION_TERMS, TARGET_TERMS

# REFACTORING DEFINITION TOOL
class RefactoringDefinitionTool:
    """Loads definitions from ``refactoring_defs.json`` and provides fuzzy lookup.

    Fuzzy matching is case-insensitive and normalises ``"and"`` / ``"And"``
    so that, e.g., ``"move and rename method"`` matches
    ``"Move And Rename Method"``.
    """
    # init
    def __init__(self, defs_path: Optional[str] = None):
        if defs_path is None:
            defs_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "refactoring_defs.json",
            )
        with open(defs_path, "r", encoding="utf-8") as f:
            raw: list = json.load(f)

        self._defs: Dict[str, str] = {}

        self._norm_map: Dict[str, str] = {}

        for entry in raw:
            name = entry["refactoring_type"]
            defn = entry["definition"]
            self._defs[name] = defn
            self._norm_map[self._normalise(name)] = name

    @staticmethod
    def _normalise(name: str) -> str:
        """Lower-case, collapse whitespace, normalise 'and'/'And'."""
        n = name.strip().lower()

        n = re.sub(r"\s+", " ", n)

        return n

    def get_definition(self, type_name: str) -> str:
        """Return the definition for *type_name*, with fuzzy matching.

        Returns a formatted string with the type name and definition, or
        an error message listing available types if no match is found.
        """
        if type_name in self._defs:
            return f"{type_name}: {self._defs[type_name]}"

        norm = self._normalise(type_name)
        if norm in self._norm_map:
            canonical = self._norm_map[norm]
            return f"{canonical}: {self._defs[canonical]}"

        available = sorted(self._defs.keys())
        return (
            f"Unknown refactoring type: '{type_name}'. "
            f"Available types ({len(available)}): {', '.join(available)}"
        )

    def get_all_types(self) -> list:
        """Return a sorted list of all known refactoring type names."""
        return sorted(self._defs.keys())

# Singleton instance (loaded once at import time)
_DEFINITION_TOOL = RefactoringDefinitionTool()

# TERM LOOKUP
def lookup_term(term: str) -> str:
    """Look up an action or target term definition.

    Searches both :data:`ACTION_TERMS` and :data:`TARGET_TERMS`
    (case-insensitive).  Returns a formatted definition or lists available
    terms if no match is found.
    """
    for label, term_dict in [("Action", ACTION_TERMS), ("Target", TARGET_TERMS)]:
        if term in term_dict:
            return f"[{label}] {term}: {term_dict[term]}"

        for key, value in term_dict.items():
            if key.lower() == term.lower():
                return f"[{label}] {key}: {value}"

    all_actions = sorted(ACTION_TERMS.keys())
    all_targets = sorted(TARGET_TERMS.keys())
    return (
        f"Unknown term: '{term}'. "
        f"Available action terms: {', '.join(all_actions)}. "
        f"Available target terms: {', '.join(all_targets)}."
    )

# GET FULL CODE (GitHub fetcher)

_CODE_CACHE: Dict[str, str] = {}


def get_full_code(
    url: str,
    file_path: str,
    version: str = "after",
) -> str:
    """Fetch full file content from GitHub for a given commit URL.

    Parameters
    ----------
    url : str
        Commit URL in the format
        ``https://github.com/owner/repo/commit/sha1``.
    file_path : str
        Path to the file within the repository (e.g., ``src/main.py``).
    version : str
        ``"before"`` to fetch the parent commit (``sha1~1``), or
        ``"after"`` (default) to fetch the commit itself.

    Returns
    -------
    str
        Cleaned file content, or an error message on failure.
    """
    match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/commit/([0-9a-fA-F]+)",
        url,
    )
    if not match:
        return f"Error: Could not parse GitHub commit URL: {url}"

    owner, repo, sha = match.group(1), match.group(2), match.group(3)

    ref = sha if version == "after" else f"{sha}~1"

    cache_key = f"{owner}/{repo}/{ref}/{file_path}"
    if cache_key in _CODE_CACHE:
        return _CODE_CACHE[cache_key]

    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        f"?ref={ref}"
    )
    headers = {"Accept": "application/vnd.github.v3.raw"}

    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        resp = requests.get(api_url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return (
                f"Error fetching {file_path} at ref {ref}: "
                f"HTTP {resp.status_code} -- {resp.text[:200]}"
            )
        code = resp.text
    except requests.RequestException as exc:
        return f"Error fetching {file_path}: {exc}"

    if file_path.endswith(".py"):
        cleaned = minify_python(code)
    else:
        cleaned = basic_cleanup(code)

    _CODE_CACHE[cache_key] = cleaned
    return cleaned

# LANGCHAIN @tool WRAPPERS
@tool
def get_definition_tool(type_name: str) -> str:
    """Look up the definition of a refactoring type by name.

    Use this when you need to understand what a refactoring type means
    before deciding whether it applies to the code diff. Accepts exact
    names (e.g. 'Rename Method') or fuzzy variants (case-insensitive).
    """
    return _DEFINITION_TOOL.get_definition(type_name)


@tool
def lookup_term_tool(term: str) -> str:
    """Look up the definition of a refactoring action or target term.

    Use this when you encounter a term you are unsure about, such as
    'Extract', 'Inline', 'Parameter', 'Attribute', etc. Action terms
    describe what is done; target terms describe what it is done to.
    """
    return lookup_term(term)


@tool
def retrieve_java_examples_tool(type_name: str, count: int = 3) -> str:
    """Retrieve Java code examples of a specific refactoring type.

    Use this when you need to see real-world examples of a refactoring
    type to compare against the Python diff. Returns code snippets showing
    what the pattern looks like in Java (the closest available reference).
    """
    retriever = _get_retriever()
    if retriever is None:
        return f"Retriever not available. Cannot fetch examples for {type_name}."
    try:
        results = retriever.retrieve_for_types(
            anchor_ids=[], target_types=[type_name], k=count,
        )
        examples = results.get(type_name, [])
        if not examples:
            return f"No Java examples found for {type_name}."
        parts = []
        for i, ex in enumerate(examples[:count], 1):
            code = ex.get("code_text", "") or ex.get("code_before", "")
            desc = ex.get("description", "")
            parts.append(f"Example {i}: {desc}\n{code[:2000]}")
        return "\n---\n".join(parts)
    except Exception as exc:
        return f"Error retrieving examples for {type_name}: {exc}"


@tool
def get_missing_type_hints_tool(detected_type: str) -> str:
    """Get commonly co-occurring types that are often MISSED when a given
    type is detected.

    Use this in Stage 3 to find additional refactoring types that should
    be checked. For example, if 'Extract Method' was detected, this might
    return 'Add Parameter' and 'Rename Variable' as commonly co-occurring
    types that the LLM should also look for in the diff.

    Source resolution:
      LANGCHAIN_HINTS_VERSION=v4 (default) -> FP-Growth association rules
        mined from the full 14k RefactoringMiner-Python train corpus, ranked
        by lift x confidence. Stays consistent with the v4 candidate scorer
        feeding Stage 3's <candidate_missing_types> block.
      Other values fall back through V3/NPMI -> V2 -> curated, matching
        constants._active_missing_hints().
    """
    import os
    from .constants import MISSING_HINTS, _active_missing_hints

    active = _active_missing_hints()
    source_label = os.environ.get("LANGCHAIN_HINTS_VERSION", "default") or "default"

    curated = MISSING_HINTS.get(detected_type, [])
    mined = active.get(detected_type, []) if active is not MISSING_HINTS else []
    combined = list(dict.fromkeys(curated + mined))

    if not combined:
        return f"No missing-type hints found for '{detected_type}'."
    return (
        f"Types commonly co-occurring with '{detected_type}' "
        f"(source: {source_label}): {', '.join(combined[:7])}"
    )


@tool
def get_confusion_hints_tool(detected_type: str) -> str:
    """Get confusion hints for a detected refactoring type — types that are
    commonly CONFUSED with this one, and rules to distinguish them.

    Use this in Stage 4 verification to check whether a detection might
    actually be a different (similar-looking) refactoring type.
    """
    from .constants import CONFUSION_HINTS

    hint = CONFUSION_HINTS.get(detected_type)
    if hint:
        return f"Confusion hint for '{detected_type}': {hint}"
    return f"No confusion hints available for '{detected_type}'."


@tool
def pull_full_commit_tool(commit_url: str) -> str:
    """Fetch the full Python source files from a GitHub commit.

    Use this when the diff alone is insufficient to determine the
    refactoring type — for example, when you need to see the full
    function/class context around a change. Only Python files are fetched
    and cleaned (comments, extra whitespace, docstrings removed).

    Returns cleaned source code for all .py files changed in the commit.
    """
    match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/commit/([0-9a-fA-F]+)",
        commit_url,
    )
    if not match:
        return f"Error: Could not parse GitHub commit URL: {commit_url}"

    owner, repo, sha = match.group(1), match.group(2), match.group(3)

    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        resp = requests.get(api_url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return f"Error fetching commit: HTTP {resp.status_code}"
        commit_data = resp.json()
    except Exception as exc:
        return f"Error fetching commit: {exc}"

    files = [f for f in commit_data.get("files", []) if f["filename"].endswith(".py")]
    if not files:
        return "No Python files found in this commit."

    parts = []
    for f in files[:10]:
        fname = f["filename"]

        code = get_full_code(commit_url, fname, version="after")
        if code.startswith("Error"):
            parts.append(f"# {fname}\n{code}")
        else:
            parts.append(f"# {fname}\n{code[:5000]}")
    return "\n\n".join(parts)

# INTERNAL HELPERS

_RETRIEVER_INSTANCE = None

# Default retriever paths, matching concurrent_runner.py CLI defaults
_DEFAULT_EMBEDDINGS = "/home/25fxvd/summer2025/ELEC825/evaluation_data_v3_ast/contrastive_v5_unixcoder_holdout/projected/unixcoder_java_projected.npy"
_DEFAULT_META       = "/home/25fxvd/summer2025/ELEC825/evaluation_data_v3_ast/embeddings/unixcoder_java_meta.json"
_DEFAULT_POOL_PKL   = "/home/25fxvd/summer2025/ELEC825/evaluation_data_v3_ast/java/java_records_all.pkl"
_DEFAULT_POOL_DIR   = "/home/25fxvd/summer2025/ELEC825/evaluation_data_v3_ast/java"


def set_retriever(retriever):
    """Allow the concurrent_runner to inject the already-loaded retriever
    so the tool doesn't need to construct a second copy of the embeddings."""
    global _RETRIEVER_INSTANCE
    _RETRIEVER_INSTANCE = retriever


def _get_retriever():
    """Lazily initialize a retriever singleton for tool use."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is not None:
        return _RETRIEVER_INSTANCE
    try:
        from .retrieval import JavaExampleRetriever
        _RETRIEVER_INSTANCE = JavaExampleRetriever(
            embeddings_path=_DEFAULT_EMBEDDINGS,
            meta_path=_DEFAULT_META,
            java_pool_pkl=_DEFAULT_POOL_PKL,
            java_pool_dir=_DEFAULT_POOL_DIR,
        )
        _RETRIEVER_INSTANCE._ensure_pool()
        return _RETRIEVER_INSTANCE
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Failed to load retriever: %s", exc)
        return None

# Convenience: all tools as a list for bind_tools()
ALL_TOOLS = [
    get_definition_tool,
    lookup_term_tool,
    retrieve_java_examples_tool,
    get_missing_type_hints_tool,
    get_confusion_hints_tool,
    pull_full_commit_tool,
]
