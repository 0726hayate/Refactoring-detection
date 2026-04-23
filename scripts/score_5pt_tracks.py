"""Score the 5-track precision experiment and produce a side-by-side
comparison + per-type CSVs.

Outputs:
  /home/25fxvd/summer2025/Review/HEADLINE_5PERTYPE.md
  /home/25fxvd/summer2025/Review/per_type/2p5d_5pt_T1_baseline.csv
  /home/25fxvd/summer2025/Review/per_type/2p5d_5pt_T2_heuristic.csv
  /home/25fxvd/summer2025/Review/per_type/2p5d_5pt_T3_stage5.csv
  /home/25fxvd/summer2025/Review/per_type/2p5d_5pt_T4_hybrid.csv
  /home/25fxvd/summer2025/Review/per_type/2p5d_5pt_T5_adversarial.csv
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path("/home/25fxvd/summer2025/0807/dspy/results/2p5d_5pt")
VALID_TYPES_FILE = Path("/home/25fxvd/summer2025/ELEC825/splits/valid_types_39.json")
REVIEW = Path("/home/25fxvd/summer2025/Review")
PER_TYPE_DIR = REVIEW / "per_type"
PER_TYPE_DIR.mkdir(parents=True, exist_ok=True)

VALID_TYPES = set(json.loads(VALID_TYPES_FILE.read_text()))
TRACKS = [
    ("T1_baseline", "No gates / no Stage 5 (pm_v2 config)"),
    ("T2_heuristic", "Structural gates + citation filter"),
    ("T3_stage5", "Stage 5 LLM only (no heuristic gates)"),
    ("T4_hybrid", "Heuristic gates + citation + Stage 5"),
    ("T5_adversarial", "Adversarial per-detection verification"),
]


def load_cases(track: str):
    path = RESULTS / track / "test" / "cases.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def score_case(final_known, gt_known):
    pred = {t for t in final_known if t in VALID_TYPES}
    gt = {t for t in gt_known if t in VALID_TYPES}
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    return tp, fp, fn, pred, gt


def score_track(cases):
    tp = fp = fn = 0
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for c in cases:
        ctp, cfp, cfn, pred, gt = score_case(
            c.get("final_known", []), c.get("ground_truth_known", [])
        )
        tp += ctp
        fp += cfp
        fn += cfn
        for t in pred & gt:
            per_type[t]["tp"] += 1
        for t in pred - gt:
            per_type[t]["fp"] += 1
        for t in gt - pred:
            per_type[t]["fn"] += 1
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return dict(n=len(cases), tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1, per_type=dict(per_type))


def write_per_type_csv(track, per_type, out_path):
    rows = []
    for t in sorted(VALID_TYPES):
        d = per_type.get(t, {"tp": 0, "fp": 0, "fn": 0})
        pp = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0
        rr = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0
        f1 = 2 * pp * rr / (pp + rr) if pp + rr else 0
        rows.append([t, d["tp"], d["fp"], d["fn"], f"{pp:.4f}", f"{rr:.4f}", f"{f1:.4f}"])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "tp", "fp", "fn", "precision", "recall", "f1"])
        rows.sort(key=lambda r: (-float(r[6]), -r[1], r[0]))
        w.writerows(rows)


def best_track_by_metric(track_results, metric):
    best = max(track_results.items(), key=lambda kv: kv[1].get(metric, 0))
    return best[0], best[1][metric]


def main():
    track_cases_raw = {tid: load_cases(tid) for tid, _ in TRACKS}
    track_results = {tid: (score_track(c) if c else None)
                     for tid, c in track_cases_raw.items()}

    # Tracks killed early (stopped when user changed direction); we still score
    # them but we EXCLUDE them from the paired intersection so the paired n can
    # grow past the stopping point as live tracks continue.
    STOPPED_EARLY = {"T2_heuristic", "T4_hybrid"}

    # Paired comparison: intersection of case_ids across non-empty LIVE tracks only
    populated = {tid: c for tid, c in track_cases_raw.items() if c}
    live_populated = {tid: c for tid, c in populated.items() if tid not in STOPPED_EARLY}
    paired_results = None
    if len(live_populated) >= 2:
        case_id_sets = [set(d["case_id"] for d in c) for c in live_populated.values()]
        common_ids = set.intersection(*case_id_sets)
        if common_ids:
            paired_results = {}
            # Score live tracks on intersection
            for tid, cases in live_populated.items():
                cases_filtered = [d for d in cases if d["case_id"] in common_ids]
                paired_results[tid] = score_track(cases_filtered)
            # Also score stopped-early tracks on the intersection for comparison
            # (only count cases they share with the intersection)
            for tid in STOPPED_EARLY:
                cases = populated.get(tid)
                if not cases:
                    continue
                cases_filtered = [d for d in cases if d["case_id"] in common_ids]
                if cases_filtered:
                    paired_results[tid + "_partial"] = score_track(cases_filtered)

    # Per-type CSVs (using full per-track data, not paired)
    for track_id, _ in TRACKS:
        res = track_results.get(track_id)
        if not res:
            continue
        out = PER_TYPE_DIR / f"2p5d_5pt_{track_id}.csv"
        write_per_type_csv(track_id, res["per_type"], out)
        # Also copy to Sup/per_type/ for the supervisor briefing
        sup_out = Path("/home/25fxvd/summer2025/Review/Sup/per_type") / f"2p5d_5pt_{track_id}.csv"
        sup_out.parent.mkdir(parents=True, exist_ok=True)
        write_per_type_csv(track_id, res["per_type"], sup_out)

    # Headline markdown
    md = ["# 5-Cases-Per-Type Precision Experiment — Results", ""]
    md.append("## Headline (per-track full results — NOT paired, n differs across tracks)")
    md.append("")
    md.append("| Track | Label | n | TP | FP | FN | **P** | **R** | **F1** |")
    md.append("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for track_id, label in TRACKS:
        res = track_results.get(track_id)
        if res is None:
            md.append(f"| {track_id} | {label} | *(no data yet)* | | | | | | |")
        else:
            md.append(
                f"| {track_id} | {label} | {res['n']} | {res['tp']} | {res['fp']} | "
                f"{res['fn']} | **{res['p']:.3f}** | **{res['r']:.3f}** | **{res['f1']:.3f}** |"
            )

    if paired_results:
        n_common = len(common_ids)
        md.append("")
        md.append(f"## Paired comparison (intersection: {n_common} cases shared by all running tracks)")
        md.append("")
        md.append("| Track | Label | TP | FP | FN | **P** | **R** | **F1** |")
        md.append("|---|---|--:|--:|--:|--:|--:|--:|")
        for track_id, label in TRACKS:
            res = paired_results.get(track_id)
            if res is None:
                continue
            md.append(
                f"| {track_id} | {label} | {res['tp']} | {res['fp']} | "
                f"{res['fn']} | **{res['p']:.3f}** | **{res['r']:.3f}** | **{res['f1']:.3f}** |"
            )
        if "T1_baseline" in paired_results:
            base = paired_results["T1_baseline"]
            md.append("")
            md.append(f"### Paired deltas vs T1_baseline (n={n_common})")
            md.append("")
            md.append("| Track | ΔP | ΔR | ΔF1 |")
            md.append("|---|--:|--:|--:|")
            for tid, _ in TRACKS:
                if tid == "T1_baseline" or tid not in paired_results:
                    continue
                r = paired_results[tid]
                md.append(
                    f"| {tid} | {r['p']-base['p']:+.3f} | "
                    f"{r['r']-base['r']:+.3f} | {r['f1']-base['f1']:+.3f} |"
                )

    valid = {k: v for k, v in track_results.items() if v}
    if valid:
        md.append("")
        md.append("## Winners")
        md.append("")
        best_f1, vf1 = best_track_by_metric(valid, "f1")
        best_p, vp = best_track_by_metric(valid, "p")
        best_r, vr = best_track_by_metric(valid, "r")
        md.append(f"- **Best F1**: `{best_f1}` at {vf1:.3f}")
        md.append(f"- **Best P**:  `{best_p}` at {vp:.3f}")
        md.append(f"- **Best R**:  `{best_r}` at {vr:.3f}")

        # Deltas vs T1 baseline
        if "T1_baseline" in valid:
            base = valid["T1_baseline"]
            md.append("")
            md.append("## Deltas vs `T1_baseline`")
            md.append("")
            md.append("| Track | ΔP | ΔR | ΔF1 |")
            md.append("|---|--:|--:|--:|")
            for track_id, _ in TRACKS:
                if track_id == "T1_baseline" or track_id not in valid:
                    continue
                r = valid[track_id]
                md.append(
                    f"| {track_id} | {r['p']-base['p']:+.3f} | "
                    f"{r['r']-base['r']:+.3f} | {r['f1']-base['f1']:+.3f} |"
                )

    # Per-type files
    md.append("")
    md.append("## Per-type breakdowns")
    md.append("")
    for track_id, _ in TRACKS:
        md.append(f"- [`per_type/2p5d_5pt_{track_id}.csv`](per_type/2p5d_5pt_{track_id}.csv)")

    (REVIEW / "HEADLINE_5PERTYPE.md").write_text("\n".join(md) + "\n")
    sup_path = Path("/home/25fxvd/summer2025/Review/Sup/HEADLINE_5PERTYPE.md")
    sup_path.parent.mkdir(parents=True, exist_ok=True)
    sup_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {REVIEW / 'HEADLINE_5PERTYPE.md'}")
    print(f"Wrote {sup_path}")
    for track_id, _ in TRACKS:
        if track_results.get(track_id):
            print(f"  {track_id}: P={track_results[track_id]['p']:.3f} "
                  f"R={track_results[track_id]['r']:.3f} "
                  f"F1={track_results[track_id]['f1']:.3f} "
                  f"(n={track_results[track_id]['n']})")


if __name__ == "__main__":
    main()
