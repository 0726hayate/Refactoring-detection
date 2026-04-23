"""
4-stage LangChain refactoring detection pipeline.

Stages:
    1. Level Classification    -- classify code change into parameter/method/class levels.
    2. Open-ended detection    -- per level, detect defined types AND undefined patterns
                                  using Java reference examples + evidence quotes.
    3. Missing-type detector   -- recall direction. Add types Stage 2 missed,
                                  using MISSING_HINTS + retrieved Java examples.
    4. Confusion verifier      -- precision direction. Verify the combined
                                  (Stage 2 ∪ Stage 3) detection list. Trim or
                                  swap using retrieved confusion-partner Java
                                  examples. Cannot add new types.
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_ollama import ChatOllama

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # pip install langchain-openai

from .schemas import (
    LevelClassification,
    DetectedRefactorings,
    AdditionalDetections,
    VerifiedRefactoringsWithConfidence,
)
from .tools import (
    get_definition_tool,
    lookup_term_tool,
    retrieve_java_examples_tool,
    get_missing_type_hints_tool,
    get_confusion_hints_tool,
    pull_full_commit_tool,
)

# Per-stage tool sets
STAGE2_TOOLS = [get_definition_tool, lookup_term_tool, retrieve_java_examples_tool, pull_full_commit_tool]
STAGE3_TOOLS = [get_missing_type_hints_tool, retrieve_java_examples_tool, get_definition_tool]
STAGE4_TOOLS = [get_confusion_hints_tool, retrieve_java_examples_tool, get_definition_tool]
from .prompts import (
    build_stage1_messages,
    build_stage2_messages,
    build_stage3_messages,
    build_stage4_messages,
)
from .constants import (
    CONFUSION_HINTS,
    LEVEL_TYPES,
    build_missing_candidates,
)

logger = logging.getLogger(__name__)

# MODEL CONFIGURATIONS

MODEL_CONFIGS: Dict[str, Any] = {

    # Local Ollama models
    "qwen3.5:35b": {"provider": "ollama", "num_ctx": 262144, "default_ctx": 262144},
    "qwen3:32b":   {"provider": "ollama", "num_ctx": 40960,  "default_ctx": 40960},
    # OpenAI API models (pinned snapshots for reproducibility)
    "gpt-4o-mini": {"provider": "openai", "model_id": "gpt-4o-mini-2024-07-18", "max_tokens": 4096},
}

# CODE FORMATTING
def format_python_code(python_case: dict) -> str:
    """Format a Python case dict into a prompt-ready text block."""
    commit_diff = python_case.get("commit_diff", "")
    if commit_diff and commit_diff.strip():
        return f"## Python Commit Diff\n\n```diff\n{commit_diff}\n```"

    code_before = python_case.get("code_before", [])
    code_after = python_case.get("code_after", [])

    if code_before and code_after:
        lines: List[str] = ["## Python Code Changes", ""]
        for before_file, after_file in zip(code_before, code_after):
            file_name = before_file.get("file", "unknown.py")
            before_code = before_file.get("code", "")
            after_code = after_file.get("code", "")
            lines.append(f"### File: {file_name}")
            lines.append("")
            lines.append("**Before:**")
            lines.append("```python")
            lines.append(before_code)
            lines.append("```")
            lines.append("")
            lines.append("**After:**")
            lines.append("```python")
            lines.append(after_code)
            lines.append("```")
            lines.append("")
        return "\n".join(lines)

    return "## Python Code Changes\n\nN/A"

# PIPELINE
class RefactoringPipeline:
    """4-stage LangChain refactoring detection pipeline.

    Args:
        model: Ollama model name (key into ``MODEL_CONFIGS``).
        temperature: Sampling temperature.
        num_ctx: Override context window size (default: from ``MODEL_CONFIGS``).
        base_url: Ollama server URL.
        max_java_examples: Max Java examples in Stage 2 reference block.
        retriever: Optional ``JavaExampleRetriever``. If provided, the
            pipeline runs Stage 3 (missing-type detector) → Stage 4
            (confusion verifier) sequentially; otherwise both are skipped
            (Stage 2 output is final).
        retrieve_k_3a: Examples per confusion partner for Stage 4.
        retrieve_k_3b: Examples per missing candidate for Stage 3.
    """
    # init
    def __init__(
        self,
        model: str = "qwen3.5:35b",
        temperature: float = 0.1,
        num_ctx: Optional[int] = None,
        base_url: str = "http://localhost:11434",
        max_java_examples: int = 5,
        retriever: Optional[Any] = None,
        retrieve_k_3a: int = 5,
        retrieve_k_3b: int = 3,
        stage2_max_workers: int = 3,
        no_java_examples: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_java_examples = 0 if no_java_examples else max_java_examples
        self.no_java_examples = no_java_examples
        self.retriever = retriever
        self.retrieve_k_3a = retrieve_k_3a
        self.retrieve_k_3b = retrieve_k_3b

        self.stage2_max_workers = max(1, stage2_max_workers)

        config = MODEL_CONFIGS.get(
            model, {"provider": "ollama", "num_ctx": 8192, "default_ctx": 8192}
        )
        provider = config.get("provider", "ollama")

        if provider == "openai":
            if ChatOpenAI is None:
                raise ImportError("pip install langchain-openai")
            self.llm = ChatOpenAI(
                model=config["model_id"],
                temperature=temperature,
                max_tokens=config.get("max_tokens", 4096),
                timeout=300,
                max_retries=8,
            )
            self.num_ctx = 128000
        else:
            self.num_ctx = num_ctx if num_ctx is not None else config["default_ctx"]
            self.llm = ChatOllama(
                model=model,
                temperature=temperature,
                num_ctx=self.num_ctx,
                num_predict=4096,
                base_url=base_url,
                timeout=300,
            )

    def _filter_tools(self, tools: list) -> list:
        """Remove retrieve_java_examples_tool when no_java_examples is set."""
        if not self.no_java_examples:
            return tools
        return [t for t in tools if t.name != "retrieve_java_examples_tool"]

    # LLM invocation helper

    # invoke
    def _invoke(self, messages: list) -> str:
        try:
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.warning("LLM call failed: %s", str(exc)[:200])
            return ""

    def _invoke_structured(self, messages: list, schema):
        """Invoke LLM with LangChain structured output (Pydantic schema).
        Returns a schema instance or None on failure."""
        structured_llm = self.llm.with_structured_output(
            schema, method="json_schema",
        )
        try:
            return structured_llm.invoke(messages)
        except Exception as exc:
            logger.warning("Structured LLM call failed: %s", str(exc)[:200])
            return None

    def _invoke_with_tools(
        self, messages: list, tools: list, schema=None, max_rounds: int = 6,
    ):
        """Invoke LLM with tool calling, then optionally produce structured output.

        Flow:
        1. LLM with bound tools generates response (may include tool_calls)
        2. Execute tool calls, feed results back as tool messages
        3. Repeat up to max_rounds until LLM produces a final text response
        4. If schema is provided, make one final structured-output call
           with the accumulated context (all tool results included)

        Returns:
            If schema: a Pydantic model instance (or None on failure)
            If no schema: the final text content as str
        """
        from langchain_core.messages import ToolMessage

        tool_map = {t.name: t for t in tools}
        llm_with_tools = self.llm.bind_tools(tools)

        conversation = list(messages)
        tool_call_log = []

        for _round in range(max_rounds):
            try:
                response = llm_with_tools.invoke(conversation)
            except Exception as exc:
                logger.warning("Tool-calling LLM failed: %s", str(exc)[:200])
                break

            conversation.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_fn = tool_map.get(tool_name)

                if tool_fn:
                    try:
                        result = tool_fn.invoke(tool_args)
                    except Exception as exc:
                        result = f"Error calling {tool_name}: {exc}"
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_call_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": str(result)[:2000],
                })
                conversation.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

        final_content = ""
        if conversation and hasattr(conversation[-1], "content"):
            final_content = conversation[-1].content or ""

        from langchain_core.messages import ToolMessage, HumanMessage
        last_msg_is_tool = conversation and isinstance(conversation[-1], ToolMessage)
        if last_msg_is_tool or not final_content:
            try:
                conversation.append(HumanMessage(
                    content="Based on the tool results so far, produce your final answer now. Do not call any more tools — respond with the final output in the requested format."
                ))
                forced = self.llm.invoke(conversation)
                if hasattr(forced, "content") and forced.content:
                    final_content = forced.content
            except Exception as exc:
                logger.warning("Forced final response failed: %s", str(exc)[:200])

        if schema is not None:
            tool_summary = ""
            if tool_call_log:
                parts = []
                for tc in tool_call_log:
                    parts.append(f"[{tc['tool']}({tc['args']})]: {tc['result']}")
                tool_summary = "\n".join(parts)

            from langchain_core.messages import SystemMessage, HumanMessage
            structured_msgs = list(messages)
            if tool_summary:
                structured_msgs.append(HumanMessage(
                    content=f"Tool results from your analysis:\n{tool_summary}"
                ))
            if final_content:
                structured_msgs.append(HumanMessage(
                    content=f"Your preliminary analysis:\n{final_content}\n\nNow produce the final structured output."
                ))

            structured_llm = self.llm.with_structured_output(
                schema, method="json_schema",
            )
            try:
                result = structured_llm.invoke(structured_msgs)
                return result, tool_call_log
            except Exception as exc:
                logger.warning("Structured output after tools failed: %s", str(exc)[:200])
                return None, tool_call_log

        return final_content, tool_call_log

    def _retriever_takes_commit_url(self) -> bool:
        """True if self.retriever's retrieve_for_types signature accepts a
        ``commit_url`` kwarg (UpstreamFacadeRetriever) vs only anchor_ids
        (legacy JavaExampleRetriever). Cached on first call.
        """
        cached = getattr(self, "_retriever_commit_url_flag", None)
        if cached is not None:
            return cached
        import inspect
        try:
            sig = inspect.signature(self.retriever.retrieve_for_types)
            flag = "commit_url" in sig.parameters
        except (TypeError, ValueError):
            flag = False
        self._retriever_commit_url_flag = flag
        return flag

    # Main prediction

    # predict
    def predict(
        self,
        python_case: dict,
        java_examples: list,
    ) -> List[str]:
        return self.predict_with_trace(python_case, java_examples)["final_types"]

    # predict with trace
    def predict_with_trace(
        self,
        python_case: dict,
        java_examples: list,
    ) -> Dict[str, Any]:
        import os as _os
        trace: Dict[str, Any] = {
            "final_types": [],
            "code_diff": "",
            "stage1_output": [],
            "stage2_traces": [],
            "stage2_defined": [],
            "stage2_undefined": [],
            "stage3_output": [],
            "stage4_output": [],
            "errors": [],
        }

        code_diff = format_python_code(python_case)
        trace["code_diff"] = code_diff

        if _os.environ.get("LANGCHAIN_DISABLE_STRUCTURAL_FACTS", "").lower() in ("1", "true", "yes"):
            structural_facts_xml = ""
            trace["structural_facts_source"] = "disabled"
        else:
            facts_src = _os.environ.get("LANGCHAIN_FACTS_SRC", "ast").lower()
            sources = [s.strip() for s in facts_src.split("+") if s.strip()]
            blocks = []
            for src in sources:
                try:
                    if src == "ast":
                        from .structural_facts import facts_for_case as _f
                    elif src == "gumtree":
                        from .gumtree_facts import gumtree_facts_for_case as _f
                    elif src == "treesitter":
                        from .treesitter_facts import treesitter_facts_for_case as _f
                    elif src == "libcst":
                        from .libcst_facts import libcst_facts_for_case as _f
                    elif src == "parso":
                        from .parso_facts import parso_facts_for_case as _f
                    elif src == "intra":
                        from .intra_method_signals import intra_method_signals_for_case as _f
                    else:
                        trace["errors"].append({"stage": "structural_facts",
                                                "error": f"unknown FACTS_SRC: {src}"})
                        continue
                    block = _f(python_case)
                    if block:
                        blocks.append(block)
                except Exception as exc:
                    trace["errors"].append({"stage": f"structural_facts:{src}",
                                            "error": str(exc)[:200]})
            structural_facts_xml = "\n\n".join(blocks)
            trace["structural_facts_source"] = facts_src
        trace["structural_facts_chars"] = len(structural_facts_xml)

        trophies_xml = ""
        if _os.environ.get("LANGCHAIN_USE_TROPHIES", "").lower() in ("1", "true", "yes"):
            try:
                from .trophy_retrieval import trophies_for_case
                trophies_xml = trophies_for_case(python_case, k=3)
            except Exception as exc:
                trace["errors"].append({"stage": "trophy_retrieval", "error": str(exc)[:200]})
        trace["trophies_chars"] = len(trophies_xml)

        commit_url = python_case.get("url") or python_case.get("commit_url")

        levels = self._stage1(code_diff, trace)
        if not levels:
            levels = ["parameter_level", "method_level", "class_level"]

        skip_class = _os.environ.get("LANGCHAIN_SKIP_CLASS_LEVEL", "").lower() in ("1", "true", "yes")
        if skip_class:
            levels = [l for l in levels if l != "class_level"]
            if not levels:
                levels = ["parameter_level", "method_level"]
            trace["stage1_output_filtered"] = levels

        _n_s2_calls = int(_os.environ.get("LANGCHAIN_STAGE2_CALLS", "1"))
        if _n_s2_calls > 1:
            stage2_defined, stage2_undefined = self._stage2_consensus(
                code_diff, levels, java_examples, trace,
                structural_facts_xml=structural_facts_xml,
                trophies_xml=trophies_xml,
                n_calls=_n_s2_calls,
            )
        else:
            stage2_defined, stage2_undefined = self._stage2_parallel(
                code_diff, levels, java_examples, trace,
                structural_facts_xml=structural_facts_xml,
                trophies_xml=trophies_xml,
            )
        trace["stage2_defined"] = stage2_defined
        trace["stage2_undefined"] = stage2_undefined

        if not stage2_defined and not stage2_undefined and self.retriever is None:
            trace["final_types"] = []
            return trace

        if self.retriever is not None:
            verified_defined = self._stage3_then_stage4(
                code_diff, stage2_defined, levels, java_examples, trace,
                commit_url=commit_url,
                structural_facts_xml=structural_facts_xml,
                trophies_xml=trophies_xml,
            )
        else:
            verified_defined = [t for t, _ in stage2_defined]

        final = _apply_mutex_collapse(set(verified_defined))

        undefined_pairs = list(stage2_undefined)
        if _os.environ.get("LANGCHAIN_PRECISION_MODE", "").lower() in ("1", "true", "yes"):
            try:
                min_overlap = float(_os.environ.get("LANGCHAIN_UNKNOWN_OVERLAP", "0.80"))
            except ValueError:
                min_overlap = 0.80
            import re as _re
            diff_tokens = set(_re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code_diff))
            kept_pairs, dropped_pairs = [], []
            for desc, ev in undefined_pairs:
                ev_tokens = set(_re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", ev or ""))
                if not ev_tokens:
                    dropped_pairs.append({"desc": desc, "reason": "no_evidence_tokens"})
                    continue
                overlap = len(ev_tokens & diff_tokens) / len(ev_tokens)
                if overlap >= min_overlap:
                    kept_pairs.append((desc, ev))
                else:
                    dropped_pairs.append({"desc": desc, "overlap": round(overlap, 2)})
            trace.setdefault("precision_mode", {})
            trace["precision_mode"]["unknown_kept"] = len(kept_pairs)
            trace["precision_mode"]["unknown_dropped"] = len(dropped_pairs)
            trace["precision_mode"]["unknown_dropped_detail"] = dropped_pairs[:10]
            undefined_pairs = kept_pairs

        undefined_strs = [u for u, _ in undefined_pairs]
        trace["final_types"] = sorted(final) + undefined_strs
        return trace

    # Stage 1

    # stage1
    def _stage1(self, code_diff: str, trace: dict) -> List[str]:
        msgs = build_stage1_messages(code_diff)
        try:
            raw = self._invoke(msgs)
            trace["stage1_raw"] = raw
            levels = _parse_xml_levels(raw)
            trace["stage1_output"] = levels
            return levels
        except Exception as exc:
            err = str(exc)[:200]
            trace["errors"].append({"stage": "stage1", "error": err})
            return ["parameter_level", "method_level", "class_level"]

    # Stage 2

    def _stage2_parallel(
        self,
        code_diff: str,
        levels: List[str],
        java_examples: list,
        trace: dict,
        structural_facts_xml: str = "",
        trophies_xml: str = "",
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Run Stage 2 for each level in parallel.

        Returns ``(defined, undefined)`` where each is a list of
        ``(type_or_description, evidence)`` tuples, deduplicated across levels.
        """
        def _run_one(level: str) -> dict:
            msgs = build_stage2_messages(
                code_diff, level, java_examples=java_examples,
                max_java=self.max_java_examples, num_ctx=self.num_ctx,
                use_tools=True,
                structural_facts_xml=structural_facts_xml,
                trophies_xml=trophies_xml,
            )
            s2 = {"level": level, "defined": [], "undefined": [], "tool_calls": []}
            try:
                raw, tool_log = self._invoke_with_tools(msgs, self._filter_tools(STAGE2_TOOLS))
                s2["tool_calls"] = tool_log
                s2["raw"] = raw
                parsed = _parse_xml_two_step(raw)
                s2["defined"] = parsed["defined"]
                s2["undefined"] = parsed["undefined"]
            except Exception as exc:
                s2["error"] = str(exc)[:200]
            return s2

        all_defined: List[Tuple[str, str]] = []
        all_undefined: List[Tuple[str, str]] = []

        if len(levels) == 1 or self.stage2_max_workers <= 1:
            traces = [_run_one(lvl) for lvl in levels]
        else:
            traces = []
            with ThreadPoolExecutor(max_workers=self.stage2_max_workers) as pool:
                futures = {pool.submit(_run_one, lvl): lvl for lvl in levels}
                for fut in as_completed(futures):
                    traces.append(fut.result())

        for s2 in traces:
            trace["stage2_traces"].append(s2)
            if "error" in s2:
                trace["errors"].append(
                    {"stage": f"stage2_{s2['level']}", "error": s2["error"]}
                )
            all_defined.extend(s2["defined"])
            all_undefined.extend(s2["undefined"])

        seen_def: Set[str] = set()
        deduped_def: List[Tuple[str, str]] = []
        for t, ev in all_defined:
            key = t.strip().lower()
            if key and key not in seen_def:
                seen_def.add(key)
                deduped_def.append((t, ev))

        seen_undef: Set[str] = set()
        deduped_undef: List[Tuple[str, str]] = []
        for desc, ev in all_undefined:
            key = desc.strip().lower()
            if key and key not in seen_undef:
                seen_undef.add(key)
                deduped_undef.append((desc, ev))

        return deduped_def, deduped_undef

    def _stage2_consensus(
        self,
        code_diff: str,
        levels: List[str],
        java_examples: list,
        trace: dict,
        structural_facts_xml: str = "",
        trophies_xml: str = "",
        n_calls: int = 3,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Run Stage 2 n_calls times at varying temperatures; return intersection.

        Each call uses a fresh ChatOllama at temperatures [0.0, 0.2, 0.4, ...].
        A defined type survives only if it appears in ALL n_calls outputs.
        UnknownType (undefined) survives with majority vote (>= ceil(n/2)).
        Evidence strings come from the first call that produced each type.
        """
        base_temps = [0.0, 0.2, 0.4, 0.6]
        temps = base_temps[:n_calls]

        orig_llm = self.llm
        all_results: List[dict] = []
        for i, temp in enumerate(temps):
            try:
                self.llm = ChatOllama(
                    model=orig_llm.model,
                    temperature=temp,
                    num_ctx=self.num_ctx,
                    num_predict=4096,
                    base_url=orig_llm.base_url,
                    timeout=300,
                )
            except Exception:
                self.llm = orig_llm
                break
            sub_trace: dict = {"stage2_traces": [], "errors": []}
            try:
                defined, undefined = self._stage2_parallel(
                    code_diff, levels, java_examples, sub_trace,
                    structural_facts_xml=structural_facts_xml,
                    trophies_xml=trophies_xml,
                )
            finally:
                self.llm = orig_llm
            all_results.append({"temp": temp, "defined": defined, "undefined": undefined,
                                 "traces": sub_trace["stage2_traces"]})
            trace["stage2_traces"].extend(sub_trace["stage2_traces"])

        if not all_results:
            return [], []

        type_sets = [
            {t.strip().lower() for t, _ in r["defined"]} for r in all_results
        ]
        consensus_names: Set[str] = type_sets[0]
        for s in type_sets[1:]:
            consensus_names &= s

        consensus_defined = [
            (t, ev) for t, ev in all_results[0]["defined"]
            if t.strip().lower() in consensus_names
        ]

        min_count = (len(all_results) + 1) // 2
        undef_votes: dict = {}
        for r in all_results:
            for desc, ev in r["undefined"]:
                key = desc.strip().lower()
                if key not in undef_votes:
                    undef_votes[key] = {"count": 0, "desc": desc, "ev": ev}
                undef_votes[key]["count"] += 1
        consensus_undefined = [
            (v["desc"], v["ev"])
            for v in undef_votes.values()
            if v["count"] >= min_count
        ]

        trace.setdefault("precision_mode", {})["stage2_consensus"] = {
            "n_calls": len(all_results),
            "per_call_counts": [len(r["defined"]) for r in all_results],
            "consensus_defined": [t for t, _ in consensus_defined],
        }
        return consensus_defined, consensus_undefined

    # Stage 4.5 — adversarial per-detection verification (M3 Lever B)

    @staticmethod
    def _extract_diff_context(evidence: str, code_diff: str, window: int = 5) -> str:
        """Return ±window diff lines around the line best matching evidence."""
        import re as _re
        ev_toks = set(_re.findall(r"\b\w{3,}\b", evidence))
        if not ev_toks:
            return code_diff[:1500]
        lines = code_diff.split("\n")
        best_idx, best_score = -1, 0.0
        for i, line in enumerate(lines):
            line_toks = set(_re.findall(r"\b\w{3,}\b", line))
            if line_toks:
                score = len(ev_toks & line_toks) / len(ev_toks)
                if score > best_score:
                    best_score, best_idx = score, i
        if best_idx == -1 or best_score < 0.4:
            return code_diff[:1500]
        start = max(0, best_idx - window)
        end = min(len(lines), best_idx + window + 1)
        return "\n".join(lines[start:end])

    def _stage_adversarial(
        self,
        detections: List[Tuple[str, str]],
        confidences: Dict[str, int],
        threshold: int,
        code_diff: str,
        trace: dict,
        facts_xml: str = "",
    ) -> List[str]:
        """Adversarial per-detection verification for uncertain Stage-4 outputs.

        Only challenges types whose Stage-4 confidence < threshold.  Batches
        ≤5 uncertain types per LLM call.  Returns the surviving type list.

        When LANGCHAIN_ADVERSARIAL_WITH_FACTS=1, the structural+intra facts
        block is prepended to the per-batch prompt so the LLM can use
        positive AST corroboration to keep marginal-confidence TPs (closes
        the recall floor that the fact-less variant hits at R~0.29).
        """
        from .prompts import TYPE_DEFINITIONS
        import os as _os

        certain = [t for t, _ in detections if confidences.get(t, threshold) >= threshold]
        uncertain = [(t, ev) for t, ev in detections if confidences.get(t, threshold) < threshold]

        if not uncertain:
            return [t for t, _ in detections]

        adv_dropped: List[dict] = []
        kept_uncertain: List[str] = []

        with_facts = _os.environ.get("LANGCHAIN_ADVERSARIAL_WITH_FACTS", "0") == "1"

        batch_size = 5
        for batch_start in range(0, len(uncertain), batch_size):
            batch = uncertain[batch_start: batch_start + batch_size]
            lines = []

            for t, ev in batch:
                defn = TYPE_DEFINITIONS.get(t, "a code-level refactoring operation")

                ctx = self._extract_diff_context(ev, code_diff)
                lines.append(
                    f"--- Detection ---\n"
                    f"Type: {t}\n"
                    f"Definition: {defn}\n"
                    f"Stage-4 evidence: {ev or '(none)'}\n"
                    f"Diff context (±5 lines around evidence):\n{ctx}\n"
                    f"Reply: KEEP <one-line reason>  OR  DROP <one-line reason>\n"
                )

            facts_block = ""
            if with_facts and facts_xml.strip():
                facts_block = (
                    "Structural and intra-method facts for this commit "
                    "(use as POSITIVE corroboration when keeping a detection):\n"
                    f"{facts_xml.strip()}\n\n"
                )

            decision_rules = (
                "KEEP if EITHER (a) the cited evidence line is present in the diff context AND "
                "matches the type, OR (b) the structural/intra facts above contain a tag that "
                "directly corroborates this type (e.g. <methods><added> for Extract Method, "
                "<renames> for Rename Variable, <classes><moved> for Move Class).\n"
                "DROP if the evidence is absent/paraphrased AND no fact tag corroborates the type.\n"
            ) if with_facts else (
                "KEEP only if the evidence line is present in the diff context AND proves the type.\n"
                "DROP if the evidence is absent, paraphrased, or doesn't match the type.\n"
            )

            prompt = (
                "You are verifying refactoring detections. For EACH detection below, "
                "answer KEEP or DROP on its own line.\n"
                + decision_rules + "\n"
                + facts_block
                + "\n".join(lines)
            )

            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                from .prompts import _no_think_enabled

                _adv_sys = "You are a precise code-review assistant."
                if _no_think_enabled():
                    _adv_sys = _adv_sys + "\n\n/no_think"
                msgs = [
                    SystemMessage(content=_adv_sys),
                    HumanMessage(content=prompt),
                ]

                raw = self._invoke(msgs)

                trace.setdefault("adversarial_raw", []).append(raw)

                verdict_lines = [ln.strip() for ln in raw.split("\n")
                                 if ln.strip().upper().startswith(("KEEP", "DROP"))]

                for (t, _), verdict in zip(batch, verdict_lines):
                    if verdict.upper().startswith("KEEP"):
                        kept_uncertain.append(t)
                    else:
                        adv_dropped.append({"type": t, "verdict": verdict[:120]})

                for t, _ in batch[len(verdict_lines):]:
                    kept_uncertain.append(t)
            except Exception as exc:
                kept_uncertain.extend(t for t, _ in batch)
                trace.setdefault("errors", []).append(
                    {"stage": "adversarial", "error": str(exc)[:200]}
                )

        trace.setdefault("precision_mode", {})["adversarial_dropped"] = adv_dropped
        return certain + kept_uncertain

    def _stage5_evidence_review(
        self,
        detections: List[Tuple[str, str]],
        confidences: Dict[str, int],
        code_diff: str,
        facts_xml: str,
        trace: dict,
        uncertain_keep_floor: int = 70,
    ) -> List[str]:
        """Stage 5: LLM-driven evidence review.

        Single batched LLM call shows the model the diff + structural facts +
        intra patterns + Stage-4 detections. Model decides KEEP/DROP/UNCERTAIN
        per detection. UNCERTAIN falls back to Stage-4 confidence threshold.
        """
        if not detections:
            return []

        from .prompts import TYPE_DEFINITIONS, STAGE5_SYSTEM, build_stage5_prompt

        det_dicts = [
            {
                "type": t,
                "evidence": ev,
                "confidence": confidences.get(t, "?"),
                "definition": TYPE_DEFINITIONS.get(t, "a code-level refactoring"),
            }
            for t, ev in detections
        ]

        prompt = build_stage5_prompt(
            code_diff=code_diff,
            facts_xml=facts_xml or "",
            detections=det_dicts,
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from .prompts import _no_think_enabled

            _s5_sys = STAGE5_SYSTEM
            if _no_think_enabled():
                _s5_sys = _s5_sys + "\n\n/no_think"
            msgs = [
                SystemMessage(content=_s5_sys),
                HumanMessage(content=prompt),
            ]

            raw = self._invoke(msgs)
        except Exception as exc:
            trace.setdefault("errors", []).append(
                {"stage": "stage5", "error": str(exc)[:200]}
            )
            return [t for t, _ in detections]

        trace.setdefault("stage5_raw", raw)

        decisions: Dict[str, str] = {}
        reasons: Dict[str, str] = {}
        for ln in raw.splitlines():
            parts = [p.strip() for p in ln.split("|")]

            if len(parts) < 2:
                continue
            t = parts[0]
            verdict = parts[1].upper()
            reason = parts[2] if len(parts) > 2 else ""

            if verdict in ("KEEP", "DROP", "UNCERTAIN"):
                decisions[t] = verdict
                reasons[t] = reason[:160]

        kept: List[str] = []
        dropped: List[dict] = []
        uncertain: List[dict] = []
        unparsed: List[str] = []

        for t, _ in detections:
            v = decisions.get(t)
            if v is None:
                kept.append(t)
                unparsed.append(t)
            elif v == "KEEP":
                kept.append(t)
            elif v == "DROP":
                dropped.append({"type": t, "reason": reasons.get(t, "")})
            else:
                conf = confidences.get(t, 0)
                if conf >= uncertain_keep_floor:
                    kept.append(t)
                    uncertain.append({"type": t, "kept_via_conf": conf, "reason": reasons.get(t, "")})
                else:
                    dropped.append({"type": t, "reason": "UNCERTAIN, conf<{}".format(uncertain_keep_floor)})
                    uncertain.append({"type": t, "dropped_via_conf": conf, "reason": reasons.get(t, "")})

        trace.setdefault("precision_mode", {})["stage5_decisions"] = {
            "kept": kept,
            "dropped": dropped,
            "uncertain": uncertain,
            "unparsed": unparsed,
        }
        return kept

    # Stage 3 → Stage 4 (sequential: missing detector, then verifier)

    def _stage3_then_stage4(
        self,
        code_diff: str,
        stage2_defined: List[Tuple[str, str]],
        stage1_levels: List[str],
        java_examples: list,
        trace: dict,
        commit_url: Optional[str] = None,
        structural_facts_xml: str = "",
        trophies_xml: str = "",
    ) -> List[str]:
        """Run Stage 3 (missing detector) then Stage 4 (confusion verifier)
        sequentially.

        Args:
            commit_url: Optional commit URL used by UpstreamFacadeRetriever
                to do Python→Java per-type retrieval. JavaExampleRetriever
                (legacy Java→Java anchor approach) ignores this argument.

        Returns the final list of verified type names (no evidence).
        """
        import os as _os

        anchor_ids = []
        if hasattr(self.retriever, "extract_anchor_ids"):
            anchor_ids = self.retriever.extract_anchor_ids(java_examples)
        else:
            anchor_ids = [
                ex.get("id") for ex in java_examples if ex.get("id")
            ]

        stage2_type_names = [t for t, _ in stage2_defined]

        skip_class = _os.environ.get(
            "LANGCHAIN_SKIP_CLASS_LEVEL", ""
        ).lower() in ("1", "true", "yes")
        pm_types_set: Optional[Set[str]] = None
        if skip_class:
            pm_types_set = set(LEVEL_TYPES.get("parameter_level", [])) | \
                           set(LEVEL_TYPES.get("method_level", []))

        skip_s3 = _os.environ.get(
            "LANGCHAIN_SKIP_STAGE3", ""
        ).lower() in ("1", "true", "yes")
        if skip_s3:
            missing_candidates = []
        else:
            missing_candidates = build_missing_candidates(
                stage2_type_names, stage1_levels=stage1_levels, cap=7,
            )

            if pm_types_set is not None:
                missing_candidates = [c for c in missing_candidates if c in pm_types_set]

            from . import precision_filters as _pf
            if _pf.precision_mode_on():
                before_n = len(missing_candidates)
                missing_candidates, pm_drop = _pf.filter_stage3_candidates(
                    missing_candidates, set(stage2_type_names),
                )
                trace.setdefault("precision_mode", {})
                trace["precision_mode"]["stage3_candidate_drops"] = pm_drop
                trace["precision_mode"]["stage3_candidates_before"] = before_n
                trace["precision_mode"]["stage3_candidates_after"] = len(missing_candidates)
        trace["stage3_targets"] = missing_candidates

        stage3_additions: List[Tuple[str, str]] = []
        if missing_candidates:
            retrieved_3 = self.retriever.retrieve_for_types(
                anchor_ids, missing_candidates, k=self.retrieve_k_3b,
                commit_url=commit_url,
            ) if self._retriever_takes_commit_url() else self.retriever.retrieve_for_types(
                anchor_ids, missing_candidates, k=self.retrieve_k_3b,
            )
            msgs_3 = build_stage3_messages(
                code_diff, stage2_defined, missing_candidates, retrieved_3,
                num_ctx=self.num_ctx, use_tools=True,
                structural_facts_xml=structural_facts_xml,
                trophies_xml=trophies_xml,
            )
            try:
                raw_3, tool_log_3 = self._invoke_with_tools(msgs_3, self._filter_tools(STAGE3_TOOLS))
                trace["stage3_tool_calls"] = tool_log_3
                trace["stage3_raw"] = raw_3
                parsed_3 = _parse_xml_block(
                    _preprocess_malformed_xml(raw_3), "additional",
                )

                stage2_set = {t.strip().lower() for t in stage2_type_names}
                stage3_additions = [
                    (t, ev) for t, ev in parsed_3
                    if t.strip().lower() not in stage2_set
                ]

                if pm_types_set is not None:
                    stage3_additions = [
                        (t, ev) for t, ev in stage3_additions if t in pm_types_set
                    ]
            except Exception as exc:
                trace["errors"].append({"stage": "stage3", "error": str(exc)[:200]})
                stage3_additions = []

        from . import precision_filters as _pf
        if _pf.precision_mode_on() and stage3_additions:
            before_n = len(stage3_additions)
            stage3_additions, add_drops = _pf.filter_sole_stage3_additions(
                stage3_additions, set(stage2_type_names),
                structural_facts_xml or "",
            )
            trace.setdefault("precision_mode", {})
            trace["precision_mode"]["stage3_addition_drops"] = add_drops
            trace["precision_mode"]["stage3_additions_before"] = before_n
            trace["precision_mode"]["stage3_additions_after"] = len(stage3_additions)

        if _os.environ.get("LANGCHAIN_STAGE2_ONLY", "0") == "1" and stage3_additions:
            trace.setdefault("precision_mode", {})["stage2_only_dropped"] = [
                t for t, _ in stage3_additions
            ]
            stage3_additions = []

        trace["stage3_output"] = [t for t, _ in stage3_additions]

        combined_detected: List[Tuple[str, str]] = list(stage2_defined)
        seen = {t.strip().lower() for t, _ in stage2_defined}
        for t, ev in stage3_additions:
            key = t.strip().lower()
            if key and key not in seen:
                seen.add(key)
                combined_detected.append((t, ev))

        if not combined_detected:
            trace["stage4_output"] = []
            return []

        combined_type_names = [t for t, _ in combined_detected]

        confusion_targets = self._collect_confusion_targets(combined_type_names)
        if pm_types_set is not None:
            confusion_targets = [t for t in confusion_targets if t in pm_types_set]
        trace["stage4_targets"] = confusion_targets

        retrieved_4 = (
            (self.retriever.retrieve_for_types(
                anchor_ids, confusion_targets, k=self.retrieve_k_3a,
                commit_url=commit_url,
             ) if self._retriever_takes_commit_url() else self.retriever.retrieve_for_types(
                anchor_ids, confusion_targets, k=self.retrieve_k_3a,
             ))
            if confusion_targets else {}
        )
        msgs_4 = build_stage4_messages(code_diff, combined_detected, retrieved_4, num_ctx=self.num_ctx, use_tools=True)

        try:
            raw_4, tool_log_4 = self._invoke_with_tools(msgs_4, self._filter_tools(STAGE4_TOOLS))
            trace["stage4_tool_calls"] = tool_log_4
            trace["stage4_raw"] = raw_4
            parsed_4 = _parse_xml_block(
                _preprocess_malformed_xml(raw_4), "verified",
            )
            verified_types = [t for t, _ in parsed_4]

            stage4_evidences: Dict[str, str] = {t: ev for t, ev in parsed_4}

            stage4_confidences = _parse_stage4_confidences(
                _preprocess_malformed_xml(raw_4)
            )
            if not verified_types:
                verified_types = combined_type_names
        except Exception as exc:
            trace["errors"].append({"stage": "stage4", "error": str(exc)[:200]})
            verified_types = combined_type_names
            stage4_confidences = {}
            stage4_evidences = {}

        if _os.environ.get("LANGCHAIN_PRECISION_MODE", "").lower() in ("1", "true", "yes"):
            from . import precision_filters as _pf
            try:
                threshold = int(_os.environ.get("LANGCHAIN_PRECISION_THRESHOLD", "80"))
            except ValueError:
                threshold = 80
            try:
                hard_floor = float(_os.environ.get("LANGCHAIN_HARD_BLOCK_P", "0.15"))
            except ValueError:
                hard_floor = 0.15
            kept = []
            dropped = []
            for t in verified_types:
                hp = _pf.historical_precision(t)
                if hp is not None and hp < hard_floor:
                    dropped.append({"type": t, "reason": "hard_block_low_P",
                                    "historical_P": round(hp, 3)})
                    continue
                conf = stage4_confidences.get(t)

                if conf is None or conf >= threshold:
                    kept.append(t)
                else:
                    dropped.append({"type": t, "reason": "confidence<threshold",
                                    "confidence": conf})
            trace.setdefault("precision_mode", {})
            trace["precision_mode"]["stage4_threshold"] = threshold
            trace["precision_mode"]["stage4_hard_floor"] = hard_floor
            trace["precision_mode"]["stage4_kept"] = len(kept)
            trace["precision_mode"]["stage4_dropped"] = len(dropped)
            trace["precision_mode"]["stage4_dropped_detail"] = dropped
            verified_types = kept

        if (_os.environ.get("LANGCHAIN_PRECISION_MODE", "").lower() in ("1", "true", "yes") and
                _os.environ.get("LANGCHAIN_STRUCT_GATES", "0") == "1"):
            from . import precision_filters as _pf_sg
            verified_types, struct_dropped = _pf_sg.filter_stage4_structural(
                verified_types, structural_facts_xml or "",
                evidences=stage4_evidences, code_diff=code_diff)
            trace.setdefault("precision_mode", {})["struct_gate_dropped"] = struct_dropped

        if _os.environ.get("LANGCHAIN_CITATION_CHECK", "0") == "1" and verified_types:
            from . import precision_filters as _pf_cit
            try:
                _cit_overlap = float(_os.environ.get("LANGCHAIN_CITATION_OVERLAP", "0.70"))
            except ValueError:
                _cit_overlap = 0.70
            _pairs = [(t, stage4_evidences.get(t, "")) for t in verified_types]
            verified_types, cit_dropped = _pf_cit.filter_by_citation(
                _pairs, code_diff, min_overlap=_cit_overlap)
            trace.setdefault("precision_mode", {})["citation_dropped"] = cit_dropped

        if _os.environ.get("LANGCHAIN_ADVERSARIAL", "0") == "1" and verified_types:
            try:
                _adv_thresh = int(_os.environ.get("LANGCHAIN_ADVERSARIAL_THRESHOLD", "90"))
            except ValueError:
                _adv_thresh = 90
            verified_types = self._stage_adversarial(
                [(t, stage4_evidences.get(t, "")) for t in verified_types],
                stage4_confidences, _adv_thresh, code_diff, trace,
                facts_xml=structural_facts_xml or "",
            )

        if _os.environ.get("LANGCHAIN_STAGE5", "0") == "1" and verified_types:
            try:
                _s5_floor = int(_os.environ.get("LANGCHAIN_STAGE5_UNCERTAIN_FLOOR", "70"))
            except ValueError:
                _s5_floor = 70
            verified_types = self._stage5_evidence_review(
                detections=[(t, stage4_evidences.get(t, "")) for t in verified_types],
                confidences=stage4_confidences,
                code_diff=code_diff,
                facts_xml=structural_facts_xml or "",
                trace=trace,
                uncertain_keep_floor=_s5_floor,
            )

        trace["stage4_output"] = verified_types
        trace["stage4_confidences"] = stage4_confidences
        return verified_types

    @staticmethod
    def _collect_confusion_targets(detected: List[str]) -> List[str]:
        """For Stage 3 Call A: derive a small list of confusion partner types
        to retrieve examples for, based on detected types.

        Returns: union of detected types AND their known confusion partners,
        capped at ~6 to keep retrieval cost bounded.
        """
        targets: Set[str] = set()
        for t in detected:
            if t in CONFUSION_HINTS:
                targets.add(t)

        compound_trigger = {
            "Move Method", "Rename Method", "Move Class", "Rename Class",
            "Extract Method", "Move Code",
        }
        if any(t in compound_trigger for t in detected):
            for compound in [
                "Move And Rename Method",
                "Move And Rename Class",
                "Extract And Move Method",
            ]:
                targets.add(compound)

        return sorted(targets)[:6]

# MUTEX COLLAPSE (compound > components)
def _apply_mutex_collapse(types: Set[str]) -> Set[str]:
    """If a compound type is present, remove its component parts."""
    types = set(types)
    if "Move And Rename Method" in types:
        types -= {"Move Method", "Rename Method"}
    if "Move And Rename Class" in types:
        types -= {"Move Class", "Rename Class"}
    if "Extract And Move Method" in types:
        types -= {"Extract Method", "Move Method"}
    return types

# XML OUTPUT PARSERS

# parse xml levels
def _parse_xml_levels(text: str) -> List[str]:
    text = _preprocess_malformed_xml(text)
    valid = {"parameter_level", "method_level", "class_level"}
    found = re.findall(r"<level>(.*?)</level>", text, re.IGNORECASE | re.DOTALL)
    if found:
        return [l.strip() for l in found if l.strip().lower() in valid]
    result = []
    for level in valid:
        if level in text.lower():
            result.append(level)
    return result

# Match `<type evidence="..."[>?]NAME</type>` — the `>?` after the closing
# quote handles a common model error where it omits the closing `>`.
# Match <type [confidence="N"] evidence="..."[>]NAME</type>
# The evidence value uses `.*?` (non-greedy) and terminates at `">` specifically,
# so inner unescaped double-quotes inside the evidence don't break parsing.
_TYPE_WITH_EVIDENCE_RE = re.compile(
    r'<type\s+(?:confidence="(\d+)"\s+)?evidence="(.*?)"\s*/?>\s*([^<>]*?)\s*</type>',
    re.IGNORECASE | re.DOTALL,
)
_TYPE_NO_EVIDENCE_RE = re.compile(
    r"<type[^/>]*>([^<>]*?)</type>",
    re.IGNORECASE | re.DOTALL,
)


def _clean_type_name(name: str) -> str:
    """Strip any XML/attribute leftovers from a type name."""
    if not name:
        return ""

    name = re.sub(r'^\s*<[^>]*>?\s*', "", name)

    name = re.sub(r'\s*<[^>]*$', "", name)

    name = name.strip().strip('"').strip("'").strip()
    return name

# Parse a single XML block (e.g. <detected>...</detected>)
def _parse_xml_block(text: str, block_name: str) -> List[Tuple[str, str]]:
    """Parse a single XML block (e.g. ``<verified>``, ``<additional>``,
    ``<defined>``, ``<undefined>``) and return a list of
    ``(type_name, evidence)`` tuples.

    - Returns ``[]`` if the block is self-closing (``<defined/>``).
    - Returns ``[]`` if no block is found at all (no fall-back to whole text,
      to avoid leaking types between sibling blocks).
    """
    if re.search(rf"<{block_name}\s*/>", text, re.IGNORECASE):
        return []

    block_re = re.compile(
        rf"<{block_name}\b[^>]*>(.*?)</{block_name}\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    m = block_re.search(text)
    if not m:
        return []
    inner = m.group(1)

    items: List[Tuple[str, str]] = []
    seen_keys: set = set()

    for match in _TYPE_WITH_EVIDENCE_RE.finditer(inner):
        ev = match.group(2).strip()
        name = _clean_type_name(match.group(3))
        if not name or not ev:
            continue
        key = name.lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        items.append((_unescape(name), _unescape(ev)))

    if not items:
        for match in _TYPE_NO_EVIDENCE_RE.finditer(inner):
            name = _clean_type_name(match.group(1))
            if not name or name.lower() in ("none", "n/a", "empty"):
                continue
            key = name.lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            items.append((_unescape(name), ""))

    return items

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)


def _preprocess_malformed_xml(text: str) -> str:
    """Repair common model XML mistakes before parsing.

    - Strips any ``<think>…</think>`` block so qwen3 reasoning traces can't
      contaminate downstream XML parsing even when thinking mode is off but
      the model still emits them.
    - Inserts missing ``>`` in ``<type evidence="X"NAME</type>``.
    """
    if not text:
        return text
    text = _THINK_BLOCK_RE.sub("", text)
    return re.sub(
        r'(<type\s+evidence="[^"]*")(?=[A-Za-z])',
        r"\1>",
        text,
        flags=re.IGNORECASE,
    )


def _parse_stage4_confidences(text: str) -> Dict[str, int]:
    """Extract {type_name: confidence} from the Stage 4 <verified> block.

    Looks for ``<type confidence="N" ...>Name</type>`` inside <verified>.
    Returns an empty dict if no confidences are present.
    """
    m = re.search(
        r'<verified\b[^>]*>(.*?)</verified\s*>',
        text, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return {}
    inner = m.group(1)
    confidences: Dict[str, int] = {}
    for match in _TYPE_WITH_EVIDENCE_RE.finditer(inner):
        conf_str = match.group(1)
        if not conf_str:
            continue
        name = _clean_type_name(match.group(3))
        if not name:
            continue
        try:
            confidences[_unescape(name)] = int(conf_str)
        except ValueError:
            pass
    return confidences


def _parse_xml_two_step(text: str) -> Dict[str, List[Tuple[str, str]]]:
    """Parse Stage 2 output: ``<detected><defined>...</defined>
    <undefined>...</undefined></detected>``.

    Returns ``{"defined": [(type, evidence), ...],
               "undefined": [(description, evidence), ...]}``.
    """
    text = _preprocess_malformed_xml(text)
    defined = _parse_xml_block(text, "defined")
    undefined = _parse_xml_block(text, "undefined")

    if not defined and not undefined:
        legacy = _parse_xml_block(text, "detected_types")
        defined = legacy

    return {"defined": defined, "undefined": undefined}


def _unescape(s: str) -> str:
    """Reverse the minimal XML escaping used in our prompts."""
    return (
        s.replace("&quot;", '"')
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
    )

# SERIALIZATION HELPER

# messages to serializable
def _messages_to_serializable(messages: list) -> list:
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
        elif hasattr(msg, "type") and hasattr(msg, "content"):
            result.append({"role": msg.type, "content": msg.content})
        else:
            result.append({"role": "unknown", "content": str(msg)})
    return result
