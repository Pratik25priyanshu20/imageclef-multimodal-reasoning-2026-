"""
Run this on DataLab to export gold labels + your predictions as JSON.
Then copy the output files to your local machine for error_analysis.py.

Usage (on DataLab):
    python export_gold_labels.py

This creates:
    gold_and_preds/mcq_visual_gold.json
    gold_and_preds/mcq_textual_gold.json
    gold_and_preds/mcq_visual_preds.json   (if predictions exist)
    gold_and_preds/mcq_textual_preds.json  (if predictions exist)

Then copy the gold_and_preds/ folder to your local machine.
"""

import json
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("gold_and_preds")
OUTPUT.mkdir(exist_ok=True)

VAULT = Path("/home/jovyan/vault/CLEF/ImageCLEF")
PRED_DIRS = [
    ("mcq_visual", [
        VAULT / "outputs" / "mcq_visual_final" / "predictions.json",
        VAULT / "outputs" / "mcq_visual_sc3" / "predictions.json",
    ]),
    ("mcq_textual", [
        VAULT / "outputs" / "mcq_textual_final" / "predictions.json",
        VAULT / "outputs" / "mcq_textual_sc3" / "predictions.json",
    ]),
]

# ── Export gold labels ────────────────────────────────────────
for name, hf_id in [
    ("mcq_visual", "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual"),
    ("mcq_textual", "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Textual"),
]:
    print(f"Downloading {name}...")
    ds = load_dataset(hf_id)
    test = ds.get("test", list(ds.values())[0])

    gold = []
    for item in test:
        gold.append({
            "question_id": item["question_id"],
            "answer_key": item.get("answer_key", ""),
            "language": item.get("language", ""),
            "subject": item.get("subject", ""),
            "type": item.get("type", ""),
        })

    out_path = OUTPUT / f"{name}_gold.json"
    with open(out_path, "w") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(gold)} gold labels -> {out_path}")

# ── Export predictions ────────────────────────────────────────
for name, pred_paths in PRED_DIRS:
    for pp in pred_paths:
        if pp.exists():
            with open(pp) as f:
                preds = json.load(f)
            out_path = OUTPUT / f"{name}_preds.json"
            with open(out_path, "w") as f:
                json.dump(preds, f, indent=2, ensure_ascii=False)
            print(f"  {name} preds: {len(preds)} from {pp} -> {out_path}")
            break
    else:
        print(f"  {name} preds: NOT FOUND at any expected path")

print(f"\nDone! Copy {OUTPUT}/ to your local machine, then run:")
print(f"  python3 src/error_analysis.py \\")
print(f"    --gold_dir gold_and_preds \\")
print(f"    --visual_preds gold_and_preds/mcq_visual_preds.json \\")
print(f"    --textual_preds gold_and_preds/mcq_textual_preds.json \\")
print(f"    --latex")
