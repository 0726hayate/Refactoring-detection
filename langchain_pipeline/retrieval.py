"""
Java refactoring example retrievers for the LangChain refactoring-detection
pipeline.

Three implementations, listed in current preference order for Stage 3/4:
  1. ``HeadlineRetriever`` (new, 2026-04-17) — matches the report headline
     directly. Loads SupCon full-diff projections for Same-modal + Desc + GNN
     (learned 311-entry node-type embedding) and computes the 3-channel
     cosine fusion at query time with the train-tuned weights
     (α=0.345, β=0.609, γ=0.046). Test S-R@10 = 47.26%, bench = 41.02%.
     Self-contained, no upstream call, fast (pure numpy/torch matmul).

  2. ``UpstreamFacadeRetriever`` — wraps upstream's ``RefactoringRetriever``.
     Same query semantics (commit-level, type-filtered) but limited to the
     2-channel Same-modal + Desc fusion exposed by the upstream class.
     Current best 2-channel cosine: ~44.99% test / 37.06% bench. Use only
     if you need rerank or other upstream-only features.

  3. ``JavaExampleRetriever`` (legacy) — Java→Java mean-pool anchor
     similarity. Different semantics from the headline (queries with Java
     anchor IDs, not a Python commit). Kept for backward compatibility with
     callers that still use the anchor-id interface.

Standalone module — depends only on numpy, torch, and stdlib.
"""
import json
import os
import pickle
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

# MMR
def mmr(
    query: np.ndarray,
    cand_embs: np.ndarray,
    k: int = 5,
    lambda_: float = 0.5,
) -> List[int]:
    """Maximum Marginal Relevance over candidate embeddings.

    Args:
        query: 1-D unit-norm query vector (D,).
        cand_embs: 2-D unit-norm candidate matrix (N, D), already pre-ranked
            by similarity to ``query`` is OK but not required.
        k: Number of items to select.
        lambda_: Trade-off between relevance (1.0) and diversity (0.0).

    Returns:
        Indices into ``cand_embs`` for the K selected items, in selection
        order.
    """
    n = len(cand_embs)
    if n == 0:
        return []
    if n <= k:
        return list(range(n))

    sims_q = cand_embs @ query
    sims_c = cand_embs @ cand_embs.T

    selected = [int(np.argmax(sims_q))]
    while len(selected) < k:
        best_i, best_score = -1, -1e18
        sel_set = set(selected)
        for i in range(n):
            if i in sel_set:
                continue
            div_pen = max(sims_c[i, j] for j in selected)
            score = lambda_ * sims_q[i] - (1.0 - lambda_) * div_pen
            if score > best_score:
                best_i, best_score = i, score
        if best_i < 0:
            break
        selected.append(best_i)
    return selected

# JAVA POOL LOADER
def _split_code_text(code_text: str) -> Dict[str, str]:
    """Split a BEFORE:/AFTER:-delimited Java ``code_text`` blob back into
    ``code_before`` and ``code_after`` fields. Mirrors the upstream
    implementation in ``match_examples.split_code_text``."""
    code_before = ""
    code_after = ""
    if not code_text:
        return {"code_before": "", "code_after": ""}
    if "\n\nAFTER:\n" in code_text:
        parts = code_text.split("\n\nAFTER:\n", 1)
        code_before = parts[0]
        code_after = parts[1] if len(parts) > 1 else ""
        if code_before.startswith("BEFORE:\n"):
            code_before = code_before[len("BEFORE:\n"):]
    elif "BEFORE:\n" in code_text:
        code_before = code_text.replace("BEFORE:\n", "", 1)
    elif "AFTER:\n" in code_text:
        code_after = code_text.replace("AFTER:\n", "", 1)
    else:
        code_before = code_text
    return {"code_before": code_before, "code_after": code_after}


def _load_java_pool_from_pkl(pkl_path: str) -> Dict[str, dict]:
    """Load Java pool records from a pickle and index by id."""
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    return {r["id"]: r for r in records if "id" in r}


def _load_java_pool_from_json_dir(java_dir: str) -> Dict[str, dict]:
    """Load Java pool records from JSON files (slow fallback)."""
    import glob
    from concurrent.futures import ThreadPoolExecutor

    files = sorted(
        f for f in glob.glob(os.path.join(java_dir, "java_records_*.json"))
        if not f.endswith(".no_prefix_backup")
        and not f.endswith(".normalized_backup")
    )

    def _load_one(fp: str) -> List[dict]:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f).get("records", [])

    pool: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for recs in ex.map(_load_one, files):
            for r in recs:
                if "id" in r:
                    pool[r["id"]] = r
    return pool

# RETRIEVER
class JavaExampleRetriever:
    """Legacy Java→Java retriever. Use ``HeadlineRetriever`` for Stage 3/4
    in current pipelines; this class is kept for callsites that still rely
    on the anchor-id interface.

    Loads a single set of pre-computed Java embeddings, L2-normalizes them,
    and at query time builds a unit-norm query by mean-pooling the
    embeddings of the supplied ``anchor_ids``. Returns the top-K (or MMR)
    Java records whose label is in the target type set.

    Args:
        embeddings_path: Path to a ``*_java_*projected*.npy`` file. For the
            current best single-channel substrate, point at the SupCon
            full-diff Same-modal projection
            (``evaluation_data_v3_ast/embeddings/java_sub_proj_supcon_fulldiff.npy``).
        meta_path: Path to the matching ``unixcoder_java_meta.json`` (or
            equivalent) with aligned ids/labels.
        java_pool_pkl: Path to a pickle cache of Java records (built once).
        java_pool_dir: Fallback directory of ``java_records_*.json`` files.
    """
    # init
    def __init__(
        self,
        embeddings_path: str,
        meta_path: str,
        java_pool_pkl: Optional[str] = None,
        java_pool_dir: Optional[str] = None,
    ) -> None:
        t0 = time.time()
        embeddings = np.load(embeddings_path)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = (embeddings / np.maximum(norms, 1e-8)).astype(np.float32)
        print(
            f"  Retriever loaded embeddings: {self.embeddings.shape} "
            f"in {time.time()-t0:.1f}s"
        )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.ids: List[str] = meta["ids"]
        self.labels: List[str] = meta["labels"]
        if len(self.ids) != self.embeddings.shape[0]:
            raise ValueError(
                f"meta ids ({len(self.ids)}) != embeddings rows "
                f"({self.embeddings.shape[0]})"
            )

        self.id_to_row: Dict[str, int] = {id_: i for i, id_ in enumerate(self.ids)}
        self.type_to_rows: Dict[str, List[int]] = defaultdict(list)
        for i, lbl in enumerate(self.labels):
            self.type_to_rows[lbl].append(i)

        self._java_pool: Optional[Dict[str, dict]] = None
        self._java_pool_pkl = java_pool_pkl
        self._java_pool_dir = java_pool_dir

    # ensure pool
    def _ensure_pool(self) -> Dict[str, dict]:
        if self._java_pool is not None:
            return self._java_pool
        t0 = time.time()
        if self._java_pool_pkl and os.path.exists(self._java_pool_pkl):
            self._java_pool = _load_java_pool_from_pkl(self._java_pool_pkl)
            src = "pkl"
        elif self._java_pool_dir:
            self._java_pool = _load_java_pool_from_json_dir(self._java_pool_dir)
            src = "json_dir"
            if self._java_pool_pkl:
                with open(self._java_pool_pkl, "wb") as f:
                    pickle.dump(list(self._java_pool.values()), f, protocol=4)
        else:
            raise ValueError(
                "JavaExampleRetriever needs either java_pool_pkl or "
                "java_pool_dir to load records."
            )
        print(
            f"  Retriever loaded Java pool ({src}): "
            f"{len(self._java_pool):,} records in {time.time()-t0:.1f}s"
        )
        return self._java_pool

    def _build_query(self, anchor_ids: Sequence[str]) -> Optional[np.ndarray]:
        """Average the embeddings of the given anchor IDs into a unit-norm
        query vector. Returns ``None`` if no anchor IDs match the index.
        """
        rows = [self.id_to_row[a] for a in anchor_ids if a in self.id_to_row]
        if not rows:
            return None
        q = self.embeddings[rows].mean(axis=0)
        n = float(np.linalg.norm(q))
        if n < 1e-8:
            return None
        return (q / n).astype(np.float32)

    def retrieve_for_types(
        self,
        anchor_ids: Sequence[str],
        target_types: Sequence[str],
        k: int = 5,
        top_n: int = 20,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
    ) -> Dict[str, List[dict]]:
        """For each target type, return up to *k* Java records (with
        ``code_text``) most relevant to the anchors and diverse from each
        other.

        Args:
            anchor_ids: IDs of upstream-matched Java examples for the case.
                Used to build the query embedding (mean of their embeddings).
            target_types: Refactoring types to retrieve examples for. Each
                type's candidates are filtered to only that label.
            k: Number of examples per type.
            top_n: Pre-MMR top-N by similarity (only matters if use_mmr).
            use_mmr: If True, MMR over the top-N candidates. Else top-K.
            mmr_lambda: MMR trade-off (1.0 = pure relevance, 0.0 = pure
                diversity).

        Returns:
            ``{type_name: [java_record, ...]}`` — empty list per type if no
            examples are available.
        """
        results: Dict[str, List[dict]] = {}
        if not target_types:
            return results

        query = self._build_query(anchor_ids)
        pool = self._ensure_pool()

        for t in target_types:
            cand_rows = self.type_to_rows.get(t, [])
            if not cand_rows:
                results[t] = []
                continue

            if query is None:
                chosen_rows = cand_rows[:k]
            else:
                cand_embs = self.embeddings[cand_rows]
                sims = cand_embs @ query
                n = min(top_n, len(cand_rows))

                top_local = np.argpartition(-sims, n - 1)[:n]
                top_local = top_local[np.argsort(-sims[top_local])]
                top_embs = cand_embs[top_local]

                if use_mmr and n > k:
                    chosen_local = mmr(query, top_embs, k=k, lambda_=mmr_lambda)
                else:
                    chosen_local = list(range(min(k, n)))

                chosen_rows = [cand_rows[top_local[i]] for i in chosen_local]

            records: List[dict] = []
            for r in chosen_rows:
                rid = self.ids[r]
                rec = pool.get(rid)
                if rec is not None:
                    records.append(rec)
            results[t] = records

        return results

    @staticmethod
    def extract_anchor_ids(java_examples: Sequence[dict]) -> List[str]:
        """Pull the ``id`` field from each upstream-matched Java example."""
        out: List[str] = []
        for ex in java_examples:
            i = ex.get("id")
            if i:
                out.append(i)
        return out

# Upstream facade adapter (iter 7+, commit-level Apr 9)

#
# The JavaExampleRetriever above does Java→Java mean-pool anchor similarity,
# which is the wrong retrieval semantics for Stage 3/4. Stage 3/4 need:
#     "Given a Python commit and a SET of candidate refactoring types,
#      retrieve top-K Java exemplars matching that commit, restricted to
#      records whose type is in the candidate set (union)."
#
# Upstream's RefactoringRetriever (in /home/25fxvd/summer2025/ELEC825/)
# provides this via its `retrieve_commit_level` method (Apr 9), which:
#   1. Mean-pools the commit's per-refactoring substrate + CLIP py-diff
#      projected vectors into a single query vector
#   2. Filters the 366k Java pool to records whose type ∈ candidate set
#   3. Scores with CLIP α=0.2 fusion (substrate cosine + CLIP cosine)
#   4. Takes top-100 by fused cosine
#   5. Optionally reranks with frozen BAAI/bge-reranker-v2-gemma (2.5B)
#   6. Returns a SINGLE ranked list of top-K Java exemplars
#
# This adapter wraps that method. It exposes two call shapes:
#
#   * retrieve_for_commit_union(commit_url, target_types, k)
#       → List[dict] of length ≤ k, a single pooled ranked list.
#         PREFERRED for Stage 3/4: the candidate set is the UNION of the
#         types the stage is considering, the returned list is a shared
#         pool of in-context examples the LLM can reason over.
#
#   * retrieve_for_types(anchor_ids, target_types, k, commit_url=None)
#       → Dict[str, List[dict]] — backward-compatible with the old
#         JavaExampleRetriever signature. Internally calls
#         retrieve_for_commit_union with budget = k * len(target_types),
#         then re-buckets by refactoring_type so each type gets up to k.
class UpstreamFacadeRetriever:
    """Adapter that wraps upstream's commit-level ``RefactoringRetriever``.

    Note: upstream's class supports only the 2-channel (Same-modal + Desc)
    fusion via ``clip_alpha`` plus an optional cross-encoder reranker. It
    does NOT support the 3-channel Same-modal + Desc + GNN fusion that
    forms the report headline. For the full headline use
    ``HeadlineRetriever`` instead.

    Recommended entry point: ``retrieve_for_commit_union(commit_url,
    target_types, k)`` — returns one pooled ranked list of Java examples
    whose type is in the ``target_types`` set, retrieved by a single
    commit-level query (Same-modal + Desc fusion, optional rerank).

    Backward-compat entry point: ``retrieve_for_types(...)`` — returns a
    per-type dict for call sites that still expect the old interface.

    Note: ``anchor_ids`` is accepted for interface compatibility with the
    legacy JavaExampleRetriever but is NOT used. Only ``commit_url``
    matters. The pipeline must pass ``commit_url`` as a kwarg.
    """
    # init
    def __init__(
        self,
        config_path: str = None,
        rerank: bool = False,
    ) -> None:
        import json as _json
        import sys as _sys
        import os as _os

        # The upstream retriever needs ELEC825 on disk. Set ELEC825_DIR env var
        # to point at it, or pass config_path explicitly. Not used by T6 (which
        # uses --retriever local).
        self._upstream_dir = _os.environ.get("ELEC825_DIR", "./ELEC825")
        if config_path is None:
            config_path = _os.path.join(self._upstream_dir, "configs", "best_retriever.json")
        if self._upstream_dir not in _sys.path:
            _sys.path.insert(0, self._upstream_dir)
        from refactoring_retriever import RefactoringRetriever

        with open(config_path) as f:
            cfg = _json.load(f)

        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        cfg["rerank"] = rerank

        if not rerank:
            for k in ("rerank_model", "rerank_candidates",
                      "rerank_batch_size", "rerank_lora_dir"):
                cfg.pop(k, None)

        if "scale_weights" in cfg and isinstance(cfg["scale_weights"], list):
            cfg["scale_weights"] = tuple(cfg["scale_weights"])

        prev_cwd = _os.getcwd()
        try:
            _os.chdir(self._upstream_dir)
            self._upstream = RefactoringRetriever(**cfg)
        finally:
            _os.chdir(prev_cwd)

        self._rerank = rerank

        self._retrieve_lock = threading.Lock()

    def retrieve_for_commit_union(
        self,
        commit_url: str,
        target_types: Sequence[str],
        k: int = 15,
    ) -> List[dict]:
        """PREFERRED commit-level retrieval: one pooled list per commit.

        Single mean-pooled query over the commit's refactorings, with the
        Java pool restricted to records whose type ∈ ``target_types``
        (UNION of the caller's candidate set). Returns up to ``k`` Java
        exemplars as a single ranked list.

        Use this whenever the downstream consumer shows the LLM one commit
        diff and wants a shared in-context example set across several
        candidate types. The caller supplies ``k`` directly.

        Args:
            commit_url: Python commit URL, must exist in upstream's
                python_records.json. If empty or not found, returns [].
            target_types: sequence of refactoring type names. The Java
                pool is filtered to records whose type is in this set
                BEFORE scoring. Empty/None ⇒ no type filter (full pool).
            k: number of Java examples to return.

        Returns:
            List of up to ``k`` Java exemplar dicts, each with:
                id, refactoring_type, similarity, rerank_score (if
                reranker is enabled), description, code_text, code_before,
                code_after, repository, sha1, url.
        """
        if not commit_url:
            return []
        try:
            with self._retrieve_lock:
                return self._upstream.retrieve_commit_level(
                    commit_url,
                    refactoring_types_filter=list(target_types) if target_types else None,
                    top_k=k,
                )
        except Exception:
            return []

    def retrieve_for_types(
        self,
        anchor_ids: Sequence[str],
        target_types: Sequence[str],
        k: int = 5,
        top_n: int = 20,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
        commit_url: Optional[str] = None,
    ) -> Dict[str, List[dict]]:
        """Backward-compatible per-type dict interface over commit-level retrieval.

        Internally calls ``retrieve_for_commit_union`` with a budget of
        ``k * len(target_types)`` (so every type gets a fair share of the
        candidate pool), then re-buckets the returned pooled list by
        ``refactoring_type`` and caps each bucket at ``k``.

        Use ``retrieve_for_commit_union`` instead if your caller wants a
        single shared pool of examples — it's the natural match for the
        LLM's consumption model.

        Fallback if commit_url is missing or the commit isn't found: empty
        dict (caller's ``retrieved or {}`` handling takes over).
        """
        results: Dict[str, List[dict]] = {t: [] for t in target_types}
        if not target_types or not commit_url:
            return results

        budget = min(k * max(len(target_types), 1), 100)
        pooled = self.retrieve_for_commit_union(
            commit_url=commit_url,
            target_types=target_types,
            k=budget,
        )
        if not pooled:
            return results

        seen_ids: set = set()
        for ex in pooled:
            jid = ex.get("id")
            if jid and jid in seen_ids:
                continue
            if jid:
                seen_ids.add(jid)
            ex_type = ex.get("refactoring_type")
            if ex_type in results and len(results[ex_type]) < k:
                results[ex_type].append(ex)

        return results

    # Interface-compatible stub
    @staticmethod
    def extract_anchor_ids(java_examples: Sequence[dict]) -> List[str]:
        """Interface-compatible stub. Not used by the facade retriever
        (which keys off ``commit_url`` instead of Java anchor IDs)."""
        out: List[str] = []
        for ex in java_examples:
            i = ex.get("id")
            if i:
                out.append(i)
        return out

    # Interface-compatible stub
    def _ensure_pool(self) -> Dict:
        """Interface-compatible stub. Upstream loads its own pool internally."""
        return {}

# Headline retriever (3-channel SupCon full-diff fusion, learned node-type
# embeddings; matches report Table~\ref{tab:retrieval_test/bench} headline)
class HeadlineRetriever:
    """3-channel cosine-only retrieval matching the report headline.

    Loads SupCon full-diff projections for Same-modal, Desc, and GNN
    (learned 311-entry node-type embedding) and the corresponding commit-
    level query vectors for both train+test commits and the benchmark set.
    Scores Java candidates as

        score = α * (q_sub · ja_sub) + β * (q_desc · ja_desc) + γ * (q_gnn · ja_gnn)

    with weights (α, β, γ) = (0.345, 0.609, 0.046) — Nelder-Mead optimum on
    the 13,197-commit train split (no test or benchmark leakage). Test
    S-R@10 = 47.26%, bench = 41.02%, train = 49.01%.

    Filters by ``target_types`` (UNION of candidate types) before topk. Pure
    cosine, no reranker.

    Interface matches ``UpstreamFacadeRetriever``: callers pass
    ``commit_url`` and ``target_types``; ``anchor_ids`` is accepted for
    legacy-callsite compatibility but ignored.
    """
    # Train-tuned headline weights from
    # ablation_results/retuned_weights_fulldiff_v2.json
    DEFAULT_ALPHA = 0.345
    DEFAULT_BETA = 0.609
    DEFAULT_GAMMA = 0.046

    # Set ELEC825_DIR env var to override (e.g. ELEC825_DIR=/path/to/ELEC825).
    # Not used by T6 (T6 uses JavaExampleRetriever with --retriever local).
    import os as _os
    _UPSTREAM_DIR = _os.environ.get("ELEC825_DIR", "./ELEC825")
    del _os

    # init
    def __init__(
        self,
        upstream_dir: str = _UPSTREAM_DIR,
        java_pool_pkl: Optional[str] = None,
        java_pool_dir: Optional[str] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        device: str = "cuda",
    ) -> None:
        import torch as _torch

        self.alpha = alpha if alpha is not None else self.DEFAULT_ALPHA
        self.beta = beta if beta is not None else self.DEFAULT_BETA
        self.gamma = gamma if gamma is not None else self.DEFAULT_GAMMA
        self._device_str = device
        self._device = _torch.device(device if _torch.cuda.is_available() else "cpu")

        emb_dir = os.path.join(upstream_dir, "evaluation_data_v3_ast", "embeddings")
        gnn_dir = os.path.join(upstream_dir, "path_a_stage4")

        t0 = time.time()

        sub_ja = np.load(os.path.join(emb_dir, "java_sub_proj_supcon_fulldiff.npy")).astype(np.float32)
        desc_ja = np.load(os.path.join(emb_dir, "java_desc_proj_supcon_fulldiff.npy")).astype(np.float32)
        gnn_ja = np.load(os.path.join(gnn_dir, "java_gnn_proj_supcon_fulldiff.npy")).astype(np.float32)

        self._sub_ja = _torch.from_numpy(sub_ja).to(self._device)
        self._desc_ja = _torch.from_numpy(desc_ja).to(self._device)
        self._gnn_ja = _torch.from_numpy(gnn_ja).to(self._device)

        self._sub_c = np.load(os.path.join(emb_dir, "commit_sub_query_supcon_fulldiff.npy")).astype(np.float32)
        self._desc_c = np.load(os.path.join(emb_dir, "commit_clip_query_supcon_fulldiff.npy")).astype(np.float32)
        self._gnn_c = np.load(os.path.join(gnn_dir, "gnn_commit_query_supcon_fulldiff.npy")).astype(np.float32)
        self._sub_b = np.load(os.path.join(emb_dir, "bench_commit_sub_query_supcon_fulldiff.npy")).astype(np.float32)
        self._desc_b = np.load(os.path.join(emb_dir, "bench_commit_clip_query_supcon_fulldiff.npy")).astype(np.float32)
        self._gnn_b = np.load(os.path.join(gnn_dir, "bench_gnn_commit_query_supcon_fulldiff.npy")).astype(np.float32)

        with open(os.path.join(emb_dir, "commit_urls.json"), "r", encoding="utf-8") as f:
            urls_all = json.load(f)
        with open(os.path.join(emb_dir, "bench_commit_urls.json"), "r", encoding="utf-8") as f:
            urls_bench = json.load(f)
        self._url_to_idx = {u: i for i, u in enumerate(urls_all)}
        self._url_to_idx_bench = {u: i for i, u in enumerate(urls_bench)}

        with open(os.path.join(emb_dir, "unixcoder_java_meta.json"), "r", encoding="utf-8") as f:
            ja_meta = json.load(f)
        self._java_ids: List[str] = ja_meta["ids"]
        self._java_labels = np.asarray(ja_meta["labels"])

        print(
            f"  HeadlineRetriever loaded in {time.time()-t0:.1f}s — "
            f"java {sub_ja.shape[0]}, commits {len(urls_all)}, bench {len(urls_bench)} "
            f"(α={self.alpha}, β={self.beta}, γ={self.gamma})"
        )

        self._java_pool: Optional[Dict[str, dict]] = None
        self._java_pool_pkl = java_pool_pkl
        self._java_pool_dir = java_pool_dir
        self._retrieve_lock = threading.Lock()

    # ensure pool
    def _ensure_pool(self) -> Dict[str, dict]:
        if self._java_pool is not None:
            return self._java_pool
        t0 = time.time()
        if self._java_pool_pkl and os.path.exists(self._java_pool_pkl):
            self._java_pool = _load_java_pool_from_pkl(self._java_pool_pkl)
            src = "pkl"
        elif self._java_pool_dir:
            self._java_pool = _load_java_pool_from_json_dir(self._java_pool_dir)
            src = "json_dir"
            if self._java_pool_pkl:
                with open(self._java_pool_pkl, "wb") as f:
                    pickle.dump(list(self._java_pool.values()), f, protocol=4)
        else:
            raise ValueError(
                "HeadlineRetriever needs either java_pool_pkl or java_pool_dir "
                "to load Java records (for code_text in returned exemplars)."
            )
        print(
            f"  HeadlineRetriever loaded Java pool ({src}): "
            f"{len(self._java_pool):,} records in {time.time()-t0:.1f}s"
        )
        return self._java_pool

    def _lookup_query(self, commit_url: str):
        """Return (sub_q, desc_q, gnn_q) for a commit URL or None if not found."""
        if not commit_url:
            return None
        i = self._url_to_idx.get(commit_url)
        if i is not None:
            return self._sub_c[i], self._desc_c[i], self._gnn_c[i]
        i = self._url_to_idx_bench.get(commit_url)
        if i is not None:
            return self._sub_b[i], self._desc_b[i], self._gnn_b[i]
        return None

    def retrieve_for_commit_union(
        self,
        commit_url: str,
        target_types: Sequence[str],
        k: int = 15,
    ) -> List[dict]:
        """Top-K Java exemplars for a commit, restricted to ``target_types``.

        Returns a single ranked list (highest-similarity first) drawn from
        the union of all records whose type ∈ target_types. If
        ``target_types`` is empty/None, the full Java pool is scored.
        """
        import torch as _torch

        q = self._lookup_query(commit_url)
        if q is None:
            return []
        sub_q, desc_q, gnn_q = q

        if target_types:
            wanted = set(target_types)
            mask = np.fromiter(
                (lbl in wanted for lbl in self._java_labels),
                dtype=bool, count=len(self._java_labels)
            )
            cand_rows = np.flatnonzero(mask)
            if cand_rows.size == 0:
                return []
        else:
            cand_rows = None

        with self._retrieve_lock:
            sub_q_t = _torch.from_numpy(sub_q).to(self._device).unsqueeze(0)
            desc_q_t = _torch.from_numpy(desc_q).to(self._device).unsqueeze(0)
            gnn_q_t = _torch.from_numpy(gnn_q).to(self._device).unsqueeze(0)

            if cand_rows is not None:
                idx_t = _torch.from_numpy(cand_rows).to(self._device)
                sub_ja = self._sub_ja.index_select(0, idx_t)
                desc_ja = self._desc_ja.index_select(0, idx_t)
                gnn_ja = self._gnn_ja.index_select(0, idx_t)
            else:
                sub_ja, desc_ja, gnn_ja = self._sub_ja, self._desc_ja, self._gnn_ja

            scores = (
                self.alpha * (sub_q_t @ sub_ja.T)
                + self.beta * (desc_q_t @ desc_ja.T)
                + self.gamma * (gnn_q_t @ gnn_ja.T)
            ).squeeze(0)
            n = min(k, scores.shape[0])
            top_vals, top_local = _torch.topk(scores, n)
            top_vals = top_vals.cpu().numpy()
            top_local = top_local.cpu().numpy()

        if cand_rows is not None:
            top_global = cand_rows[top_local]
        else:
            top_global = top_local

        pool = self._ensure_pool()
        out: List[dict] = []
        for rank in range(top_global.shape[0]):
            j = int(top_global[rank])
            jid = self._java_ids[j]
            rec = pool.get(jid, {})
            ex = dict(rec)
            ex["id"] = jid
            ex["refactoring_type"] = str(self._java_labels[j])
            ex["similarity"] = float(top_vals[rank])

            if "code_before" not in ex or "code_after" not in ex:
                ex.update(_split_code_text(ex.get("code_text", "")))
            out.append(ex)
        return out

    def retrieve_for_types(
        self,
        anchor_ids: Sequence[str] = (),
        target_types: Sequence[str] = (),
        k: int = 5,
        top_n: int = 20,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
        commit_url: Optional[str] = None,
    ) -> Dict[str, List[dict]]:
        """Per-type dict view over commit-level retrieval.

        Calls ``retrieve_for_commit_union`` with budget = k * len(target_types)
        (capped at 100), then re-buckets by ``refactoring_type`` and caps
        each bucket at ``k``. ``anchor_ids`` is ignored — kept for legacy
        callsite compatibility.
        """
        results: Dict[str, List[dict]] = {t: [] for t in target_types}
        if not target_types or not commit_url:
            return results
        budget = min(k * max(len(target_types), 1), 100)
        pooled = self.retrieve_for_commit_union(
            commit_url=commit_url, target_types=target_types, k=budget,
        )
        if not pooled:
            return results
        seen: set = set()
        for ex in pooled:
            jid = ex.get("id")
            if jid and jid in seen:
                continue
            if jid:
                seen.add(jid)
            t = ex.get("refactoring_type")
            if t in results and len(results[t]) < k:
                results[t].append(ex)
        return results

    @staticmethod
    def extract_anchor_ids(java_examples: Sequence[dict]) -> List[str]:
        """Interface-compatible stub — HeadlineRetriever keys off commit_url."""
        return [ex.get("id") for ex in java_examples if ex.get("id")]
