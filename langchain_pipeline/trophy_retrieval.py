"""Trophy-case retrieval for in-context exemplars at inference time.

At inference: embed the new commit's diff via UniXcoder, find the K nearest
trophies in the pre-built pool by cosine, render as `<successful_examples>`
XML block to inject into Stage 2/3 prompts.

The trophy pool is built once by `experiments/A8_build_trophy_case.py`.
The UniXcoder embedding for the live commit is looked up from the upstream
embeddings file by URL. Live commits NOT in the pre-computed pool fall back
gracefully to "no trophies retrieved" (returns empty XML).

Toggle: env var `LANGCHAIN_USE_TROPHIES=1`.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TROPHY_JSONL = ROOT / "experiments" / "output" / "trophy_case.jsonl"
TROPHY_NPY = ROOT / "experiments" / "output" / "trophy_embeddings.npy"
EMB_NPY = ROOT.parent.parent / "ELEC825" / "evaluation_data_v3_ast" / "embeddings" / "unixcoder_python_commit_embeddings.npy"
EMB_META = ROOT.parent.parent / "ELEC825" / "evaluation_data_v3_ast" / "embeddings" / "unixcoder_python_commit_meta.json"

_TROPHY_CACHE: Optional[Dict] = None

# load trophy pool
def _load_trophy_pool() -> Dict:
    global _TROPHY_CACHE
    if _TROPHY_CACHE is not None:
        return _TROPHY_CACHE
    if not TROPHY_JSONL.exists() or not TROPHY_NPY.exists():
        _TROPHY_CACHE = {"trophies": [], "emb": None, "url_to_idx": {}}
        return _TROPHY_CACHE
    trophies = [json.loads(l) for l in open(TROPHY_JSONL) if l.strip()]
    emb = np.load(TROPHY_NPY)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_n = emb / norms

    if EMB_NPY.exists() and EMB_META.exists():
        live_emb = np.load(EMB_NPY)
        live_norms = np.linalg.norm(live_emb, axis=1, keepdims=True)
        live_norms[live_norms == 0] = 1.0
        live_emb_n = live_emb / live_norms
        with open(EMB_META) as f:
            meta = json.load(f)
        if isinstance(meta, dict) and "urls" in meta:
            urls = meta["urls"]
        elif isinstance(meta, list):
            urls = meta
        else:
            urls = list(meta.keys()) if isinstance(meta, dict) else []
        url_to_idx = {u: i for i, u in enumerate(urls)}
    else:
        live_emb_n = None
        url_to_idx = {}

    _TROPHY_CACHE = {
        "trophies": trophies,
        "emb": emb_n,
        "live_emb": live_emb_n,
        "url_to_idx": url_to_idx,
    }
    return _TROPHY_CACHE


def retrieve_trophies(case: Dict, k: int = 3, min_sim: float = 0.4) -> List[Dict]:
    """Return up to k trophies most similar to the given case by UniXcoder cosine.

    Returns [] if trophy pool empty, or live commit isn't in the embedding
    pool, or no trophy clears `min_sim`.
    """
    pool = _load_trophy_pool()
    if not pool["trophies"] or pool.get("live_emb") is None:
        return []
    url = case.get("url", "")
    if not url or url not in pool["url_to_idx"]:
        return []
    q = pool["live_emb"][pool["url_to_idx"][url]]
    sims = pool["emb"] @ q

    own_idx = next(
        (i for i, t in enumerate(pool["trophies"]) if t.get("url") == url), -1
    )
    if own_idx >= 0:
        sims[own_idx] = -1.0
    order = np.argsort(-sims)
    out = []
    for i in order[:k]:
        s = float(sims[i])
        if s < min_sim:
            break
        t = dict(pool["trophies"][i])
        t["_sim"] = round(s, 3)
        out.append(t)
    return out


def render_trophies_xml(trophies: List[Dict]) -> str:
    """Render trophies as `<successful_examples>` XML.

    Notes for prompt economy:
    - GT types are deduplicated and order-preserved (commits often have many
      duplicates of the same type).
    - Diff snippet is rendered as-is (already capped at 600 chars at build time).
    """
    if not trophies:
        return ""
    lines = ["<successful_examples>"]
    for t in trophies:
        sim = t.get("_sim", 0.0)
        seen, unique_types = set(), []
        for ty in t.get("ground_truth", []):
            if ty and ty not in seen:
                seen.add(ty); unique_types.append(ty)
        gt = ", ".join(unique_types)
        snippet = t.get("diff_snippet", "").strip()
        lines.append(f'  <example sim="{sim}" types="{escape(gt)}">')
        lines.append("    <diff>")
        for raw_line in snippet.splitlines():
            lines.append(f"      {escape(raw_line)}")
        lines.append("    </diff>")
        lines.append("  </example>")
    lines.append("</successful_examples>")
    return "\n".join(lines)


def trophies_for_case(case: Dict, k: int = 3) -> str:
    """Convenience: returns the rendered XML block (or '') ready to inject."""
    if os.environ.get("LANGCHAIN_USE_TROPHIES", "").lower() not in ("1", "true", "yes"):
        return ""
    return render_trophies_xml(retrieve_trophies(case, k=k))
