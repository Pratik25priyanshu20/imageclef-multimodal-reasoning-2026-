"""
Post-hoc MCQ error analysis using released gold labels.

Downloads gold labels from HuggingFace, compares against predictions,
and outputs LaTeX-ready tables for the working notes paper.

Usage:
    # Full analysis with predictions
    python src/error_analysis.py \
        --visual_preds outputs/mcq_visual_final/predictions.json \
        --textual_preds outputs/mcq_textual_final/predictions.json

    # Gold-label-only analysis (no predictions needed)
    python src/error_analysis.py --gold-only

    # Generate LaTeX output
    python src/error_analysis.py \
        --visual_preds outputs/mcq_visual_final/predictions.json \
        --textual_preds outputs/mcq_textual_final/predictions.json \
        --latex
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import get_token


def download_gold_labels(cache_dir: Path | None = None, token: str | None = None):
    """Download MCQ test gold labels from HuggingFace.

    For gated datasets, either:
      1. Run `huggingface-cli login` first, or
      2. Pass --hf_token YOUR_TOKEN, or
      3. Use --gold_dir to load from local JSON files
    """
    if token is None:
        token = get_token()
    gold = {}

    for name, hf_id in [
        ("mcq_visual", "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual"),
        ("mcq_textual", "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Textual"),
    ]:
        print(f"Downloading {name} gold labels from {hf_id}...")
        ds = load_dataset(
            hf_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            token=token,
        )

        test_split = ds.get("test", ds.get("default"))
        if test_split is None:
            for split_name in ds.keys():
                test_split = ds[split_name]
                break

        labels = {}
        for item in test_split:
            qid = item["question_id"]
            labels[qid] = {
                "answer_key": item.get("answer_key", ""),
                "language": item.get("language", ""),
                "subject": item.get("subject", ""),
                "type": item.get("type", ""),
            }
        gold[name] = labels
        print(f"  {name}: {len(labels)} samples with gold labels")

    return gold


def load_gold_from_local(gold_dir: Path):
    """Load gold labels from local JSON files.

    Supports two layouts:
      1. export_gold_labels.py format:
           gold_dir/mcq_visual_gold.json
           gold_dir/mcq_textual_gold.json
      2. download_data.py format:
           gold_dir/mcq_visual/test_metadata.json
           gold_dir/mcq_textual/test_metadata.json
    """
    gold = {}
    for name in ["mcq_visual", "mcq_textual"]:
        # Try export format first
        candidates = [
            gold_dir / f"{name}_gold.json",
            gold_dir / name / "test_metadata.json",
        ]
        found = None
        for path in candidates:
            if path.exists():
                found = path
                break

        if found is None:
            print(f"  Gold labels for {name} not found in {gold_dir}, skipping")
            continue

        with open(found) as f:
            data = json.load(f)
        labels = {}
        for item in data:
            qid = item.get("question_id", item.get("id"))
            labels[qid] = {
                "answer_key": item.get("answer_key", ""),
                "language": item.get("language", ""),
                "subject": item.get("subject", ""),
                "type": item.get("type", ""),
            }
        gold[name] = labels
        print(f"  {name}: {len(labels)} samples from {found}")
    return gold


def load_predictions(path: str) -> dict:
    """Load prediction JSON (our submission format)."""
    with open(path) as f:
        preds = json.load(f)

    pred_dict = {}
    for p in preds:
        qid = p.get("question_id", p.get("id"))
        ans = p.get("answer_key", "")
        pred_dict[qid] = {
            "answer_key": ans,
            "language": p.get("language", ""),
            "subject": p.get("subject", ""),
        }
    return pred_dict


def analyse_track(gold: dict, preds: dict, track_name: str):
    """Run full error analysis for one MCQ track."""
    print(f"\n{'='*70}")
    print(f"  {track_name}")
    print(f"{'='*70}")

    # Match predictions to gold
    matched = 0
    correct = 0
    missing = 0
    results = []

    for qid, g in gold.items():
        if qid not in preds:
            missing += 1
            continue
        matched += 1
        p = preds[qid]
        is_correct = p["answer_key"] == g["answer_key"]
        if is_correct:
            correct += 1
        results.append({
            "qid": qid,
            "gold": g["answer_key"],
            "pred": p["answer_key"],
            "correct": is_correct,
            "language": g["language"],
            "subject": g["subject"],
            "type": g.get("type", ""),
        })

    overall_acc = correct / matched if matched else 0
    print(f"\nOverall: {correct}/{matched} = {overall_acc:.4f}")
    if missing:
        print(f"  WARNING: {missing} gold questions have no prediction!")

    # ── Per-Language ──────────────────────────────────────────────
    print(f"\nPer-Language Accuracy:")
    lang_stats = defaultdict(lambda: {"c": 0, "t": 0})
    for r in results:
        lang_stats[r["language"]]["t"] += 1
        if r["correct"]:
            lang_stats[r["language"]]["c"] += 1

    lang_order = ["English", "Bulgarian", "Chinese", "Croatian", "Italian", "Serbian"]
    for lang in lang_order:
        if lang in lang_stats:
            s = lang_stats[lang]
            acc = s["c"] / s["t"] if s["t"] else 0
            print(f"  {lang:<15} {s['c']:>4}/{s['t']:<4} = {acc:.4f}")
    for lang in sorted(lang_stats):
        if lang not in lang_order:
            s = lang_stats[lang]
            acc = s["c"] / s["t"] if s["t"] else 0
            print(f"  {lang:<15} {s['c']:>4}/{s['t']:<4} = {acc:.4f}")

    # ── Per-Subject (top 15) ─────────────────────────────────────
    print(f"\nPer-Subject Accuracy (top 15 by sample count):")
    subj_stats = defaultdict(lambda: {"c": 0, "t": 0})
    for r in results:
        subj_stats[r["subject"]]["t"] += 1
        if r["correct"]:
            subj_stats[r["subject"]]["c"] += 1

    top_subjects = sorted(subj_stats.items(), key=lambda x: -x[1]["t"])[:15]
    for subj, s in top_subjects:
        acc = s["c"] / s["t"] if s["t"] else 0
        print(f"  {subj:<30} {s['c']:>4}/{s['t']:<4} = {acc:.4f}")

    # ── Per-Type ─────────────────────────────────────────────────
    type_stats = defaultdict(lambda: {"c": 0, "t": 0})
    for r in results:
        t = r["type"] if r["type"] else "unknown"
        type_stats[t]["t"] += 1
        if r["correct"]:
            type_stats[t]["c"] += 1

    if len(type_stats) > 1 or "unknown" not in type_stats:
        print(f"\nPer-Type Accuracy:")
        for t in sorted(type_stats):
            s = type_stats[t]
            acc = s["c"] / s["t"] if s["t"] else 0
            print(f"  {t:<20} {s['c']:>4}/{s['t']:<4} = {acc:.4f}")

    # ── Confusion Matrix (A-E) ───────────────────────────────────
    print(f"\nConfusion Matrix (rows=gold, cols=predicted):")
    options = ["A", "B", "C", "D", "E"]
    confusion = defaultdict(Counter)
    for r in results:
        confusion[r["gold"]][r["pred"]] += 1

    header = f"  {'':>5} " + " ".join(f"{o:>5}" for o in options) + "  total"
    print(header)
    for g in options:
        row = confusion[g]
        total = sum(row.values())
        cells = " ".join(f"{row.get(p, 0):>5}" for p in options)
        print(f"  {g:>5} {cells}  {total:>5}")

    # ── Error Patterns ───────────────────────────────────────────
    print(f"\nMost Common Error Patterns (gold -> pred):")
    error_patterns = Counter()
    for r in results:
        if not r["correct"]:
            error_patterns[(r["gold"], r["pred"])] += 1

    for (g, p), count in error_patterns.most_common(10):
        print(f"  {g} -> {p}: {count} errors")

    # ── Hardest Subjects ─────────────────────────────────────────
    print(f"\nHardest Subjects (lowest accuracy, min 5 samples):")
    hard_subjects = [
        (subj, s["c"] / s["t"], s["t"])
        for subj, s in subj_stats.items()
        if s["t"] >= 5
    ]
    hard_subjects.sort(key=lambda x: x[1])
    for subj, acc, n in hard_subjects[:10]:
        print(f"  {subj:<30} {acc:.4f}  (n={n})")

    # ── Easiest Subjects ─────────────────────────────────────────
    print(f"\nEasiest Subjects (highest accuracy, min 5 samples):")
    easy_subjects = sorted(hard_subjects, key=lambda x: -x[1])
    for subj, acc, n in easy_subjects[:5]:
        print(f"  {subj:<30} {acc:.4f}  (n={n})")

    return {
        "overall_acc": overall_acc,
        "matched": matched,
        "correct": correct,
        "lang_stats": dict(lang_stats),
        "subj_stats": dict(subj_stats),
        "type_stats": dict(type_stats),
        "confusion": {g: dict(row) for g, row in confusion.items()},
        "results": results,
    }


def gold_label_analysis(gold: dict, track_name: str):
    """Analyse gold label distribution (no predictions needed)."""
    print(f"\n{'='*70}")
    print(f"  {track_name} — Gold Label Distribution")
    print(f"{'='*70}")
    print(f"  Total: {len(gold)} samples")

    # Answer distribution
    ans_dist = Counter(g["answer_key"] for g in gold.values())
    print(f"\n  Answer distribution:")
    for opt in ["A", "B", "C", "D", "E"]:
        count = ans_dist.get(opt, 0)
        pct = count / len(gold) * 100 if gold else 0
        print(f"    {opt}: {count:>4} ({pct:.1f}%)")

    # Language distribution
    lang_dist = Counter(g["language"] for g in gold.values())
    print(f"\n  Language distribution:")
    for lang, count in sorted(lang_dist.items(), key=lambda x: -x[1]):
        print(f"    {lang:<15} {count:>4}")

    # Subject distribution (top 10)
    subj_dist = Counter(g["subject"] for g in gold.values())
    print(f"\n  Top 10 subjects:")
    for subj, count in subj_dist.most_common(10):
        print(f"    {subj:<30} {count:>4}")


def generate_latex(visual_analysis: dict, textual_analysis: dict):
    """Generate LaTeX tables for the paper."""
    print(f"\n{'='*70}")
    print(f"  LATEX OUTPUT — paste into paper Section 6.5")
    print(f"{'='*70}")

    # ── Table: Per-Subject Accuracy Comparison ────────────────────
    # Find subjects common to both tracks with enough samples
    v_subj = visual_analysis["subj_stats"]
    t_subj = textual_analysis["subj_stats"]
    common = set(v_subj.keys()) & set(t_subj.keys())
    common = [s for s in common if v_subj[s]["t"] >= 5 and t_subj[s]["t"] >= 5]
    common.sort(key=lambda s: -(v_subj[s]["t"] + t_subj[s]["t"]))

    print(r"""
\begin{table}[t]
\centering
\caption{Post-hoc per-subject accuracy on MCQ test sets (subjects with $\geq$5 samples in both tracks).}
\label{tab:per_subject}
\begin{tabular}{lcccc}
\toprule
\textbf{Subject} & \multicolumn{2}{c}{\textbf{Visual MCQ}} & \multicolumn{2}{c}{\textbf{Textual MCQ}} \\
 & Acc. & $n$ & Acc. & $n$ \\
\midrule""")
    for subj in common[:12]:
        v = v_subj[subj]
        t = t_subj[subj]
        v_acc = v["c"] / v["t"]
        t_acc = t["c"] / t["t"]
        subj_clean = subj.replace("&", r"\&")
        print(f"{subj_clean:<30} & {v_acc:.3f} & {v['t']} & {t_acc:.3f} & {t['t']} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # ── Table: Per-Language Comparison ─────────────────────────────
    lang_order = ["English", "Bulgarian", "Chinese", "Croatian", "Italian", "Serbian"]
    lang_abbr = {"English": "EN", "Bulgarian": "BG", "Chinese": "ZH",
                 "Croatian": "HR", "Italian": "IT", "Serbian": "SR"}

    print(r"""
\begin{table}[t]
\centering
\caption{Post-hoc per-language accuracy comparison across MCQ tracks.}
\label{tab:posthoc_lang}
\begin{tabular}{lcc}
\toprule
\textbf{Language} & \textbf{Visual MCQ} & \textbf{Textual MCQ} \\
\midrule""")
    for lang in lang_order:
        v = visual_analysis["lang_stats"].get(lang, {"c": 0, "t": 0})
        t = textual_analysis["lang_stats"].get(lang, {"c": 0, "t": 0})
        v_acc = v["c"] / v["t"] if v["t"] else 0
        t_acc = t["c"] / t["t"] if t["t"] else 0
        abbr = lang_abbr.get(lang, lang[:2].upper())
        if v["t"] > 0 or t["t"] > 0:
            v_str = f"{v_acc:.3f}" if v["t"] > 0 else "---"
            t_str = f"{t_acc:.3f}" if t["t"] > 0 else "---"
            print(f"{lang} ({abbr}) & {v_str} & {t_str} \\\\")

    v_overall = visual_analysis["overall_acc"]
    t_overall = textual_analysis["overall_acc"]
    print(r"""\midrule
\textbf{Overall} & \textbf{""" + f"{v_overall:.3f}" + r"""} & \textbf{""" + f"{t_overall:.3f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}""")

    # ── Hardest/easiest subjects narrative ────────────────────────
    print("\n% ── Narrative for Section 6.5 ──")

    v_hard = [(s, d["c"]/d["t"], d["t"]) for s, d in v_subj.items() if d["t"] >= 5]
    v_hard.sort(key=lambda x: x[1])
    t_hard = [(s, d["c"]/d["t"], d["t"]) for s, d in t_subj.items() if d["t"] >= 5]
    t_hard.sort(key=lambda x: x[1])

    print(f"% Visual MCQ hardest subjects: {', '.join(f'{s} ({a:.1%})' for s, a, _ in v_hard[:5])}")
    print(f"% Visual MCQ easiest subjects: {', '.join(f'{s} ({a:.1%})' for s, a, _ in sorted(v_hard, key=lambda x: -x[1])[:5])}")
    print(f"% Textual MCQ hardest subjects: {', '.join(f'{s} ({a:.1%})' for s, a, _ in t_hard[:5])}")
    print(f"% Textual MCQ easiest subjects: {', '.join(f'{s} ({a:.1%})' for s, a, _ in sorted(t_hard, key=lambda x: -x[1])[:5])}")

    # ── Gap analysis ─────────────────────────────────────────────
    gap = t_overall - v_overall
    print(f"\n% Overall gap: Textual - Visual = {gap:.3f} ({gap*100:.1f} pp)")
    print(f"% Visual accuracy: {v_overall:.4f}")
    print(f"% Textual accuracy: {t_overall:.4f}")


def save_analysis_json(visual_analysis: dict, textual_analysis: dict, output_path: Path):
    """Save full analysis results as JSON for later use."""
    # Strip per-question results to keep file manageable
    out = {
        "visual_mcq": {
            k: v for k, v in visual_analysis.items() if k != "results"
        },
        "textual_mcq": {
            k: v for k, v in textual_analysis.items() if k != "results"
        },
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nAnalysis saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc MCQ error analysis")
    parser.add_argument("--visual_preds", type=str,
                        help="Path to Visual MCQ predictions JSON")
    parser.add_argument("--textual_preds", type=str,
                        help="Path to Textual MCQ predictions JSON")
    parser.add_argument("--gold-only", action="store_true",
                        help="Only show gold label distribution (no predictions needed)")
    parser.add_argument("--latex", action="store_true",
                        help="Generate LaTeX tables for the paper")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for gated datasets")
    parser.add_argument("--gold_dir", type=str, default=None,
                        help="Local dir with gold labels (e.g. ./data with mcq_visual/test_metadata.json)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--output", type=str, default="analysis_results.json",
                        help="Path to save analysis JSON")
    args = parser.parse_args()

    if args.gold_dir:
        gold = load_gold_from_local(Path(args.gold_dir))
    else:
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        gold = download_gold_labels(cache_dir, token=args.hf_token)

    if args.gold_only:
        for name, labels in gold.items():
            gold_label_analysis(labels, name)
        return

    if not args.visual_preds or not args.textual_preds:
        print("\nNo prediction files provided. Showing gold-label distribution only.")
        print("To run full analysis, provide --visual_preds and --textual_preds")
        print("\nExpected prediction format (our submission JSON):")
        print('  [{"question_id": "...", "answer_key": "A"}, ...]')
        print("\nYour predictions are on DataLab at:")
        print("  /home/jovyan/vault/CLEF/ImageCLEF/outputs/mcq_visual_final/predictions.json")
        print("  /home/jovyan/vault/CLEF/ImageCLEF/outputs/mcq_textual_final/predictions.json")
        print("\nCopy them locally, then re-run with --visual_preds and --textual_preds")
        for name, labels in gold.items():
            gold_label_analysis(labels, name)
        return

    visual_preds = load_predictions(args.visual_preds)
    textual_preds = load_predictions(args.textual_preds)

    visual_analysis = analyse_track(gold["mcq_visual"], visual_preds, "Visual MCQ")
    textual_analysis = analyse_track(gold["mcq_textual"], textual_preds, "Textual MCQ")

    save_analysis_json(visual_analysis, textual_analysis, Path(args.output))

    if args.latex:
        generate_latex(visual_analysis, textual_analysis)


if __name__ == "__main__":
    main()
