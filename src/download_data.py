"""
Download all ImageCLEF 2026 Multimodal Reasoning datasets.
Run once on GPU server to cache data locally.

Usage:
    python src/download_data.py --data_dir ./data
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset


def download_exams_v(data_dir: Path):
    """Download EXAMS-V MCQ dataset (16K train / 4.6K val / 3.5K test)."""
    print("Downloading EXAMS-V...")
    ds = load_dataset("MBZUAI/EXAMS-V")

    for split in ["train", "validation", "test"]:
        split_dir = data_dir / "exams_v" / split / "images"
        split_dir.mkdir(parents=True, exist_ok=True)

        metadata = []
        for i, item in enumerate(ds[split]):
            sample_id = item["sample_id"]
            img = item["image"]
            img_path = split_dir / f"{sample_id}.png"
            if not img_path.exists():
                img.save(img_path)

            metadata.append({
                "id": sample_id,
                "answer_key": item["answer_key"],
                "type": item["type"],
                "subject": item["subject"],
                "language": item["language"],
                "grade": item.get("grade", ""),
                "chemical_structure": item.get("chemical_structure", 0),
                "table": item.get("table", 0),
                "figure": item.get("figure", 0),
                "graph": item.get("graph", 0),
            })

        meta_path = data_dir / "exams_v" / f"{split}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"  {split}: {len(metadata)} samples, images in {split_dir}")

    print("EXAMS-V done.\n")


def download_openqa(data_dir: Path):
    """Download OpenQA Visual + Textual datasets."""
    for name, hf_id in [
        ("openqa_visual", "SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"),
        ("openqa_textual", "SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual"),
    ]:
        print(f"Downloading {name}...")
        ds = load_dataset(hf_id)

        for split in ds.keys():
            split_dir = data_dir / name / split / "images"
            split_dir.mkdir(parents=True, exist_ok=True)

            metadata = []
            for item in ds[split]:
                qid = item["question_id"]

                if "image" in item and item["image"] is not None:
                    img_path = split_dir / f"{qid}.png"
                    if not img_path.exists():
                        item["image"].save(img_path)

                entry = {
                    "question_id": qid,
                    "type": item.get("type", ""),
                    "subject": item.get("subject", ""),
                    "language": item.get("language", ""),
                }
                if "answer" in item and item["answer"]:
                    entry["answer"] = item["answer"]
                if "answers" in item and item["answers"]:
                    entry["answers"] = item["answers"]

                metadata.append(entry)

            meta_path = data_dir / name / f"{split}_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"  {split}: {len(metadata)} samples")

        print(f"{name} done.\n")


def print_summary(data_dir: Path):
    """Print dataset summary."""
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    for ds_name in ["exams_v", "openqa_visual", "openqa_textual"]:
        ds_dir = data_dir / ds_name
        if not ds_dir.exists():
            continue
        print(f"\n{ds_name}:")
        for meta_file in sorted(ds_dir.glob("*_metadata.json")):
            with open(meta_file) as f:
                data = json.load(f)
            split = meta_file.stem.replace("_metadata", "")

            langs = {}
            subjects = {}
            for item in data:
                lang = item.get("language", "unknown")
                subj = item.get("subject", "unknown")
                langs[lang] = langs.get(lang, 0) + 1
                subjects[subj] = subjects.get(subj, 0) + 1

            print(f"  {split}: {len(data)} samples, {len(langs)} languages, {len(subjects)} subjects")
            top_langs = sorted(langs.items(), key=lambda x: -x[1])[:5]
            print(f"    Top languages: {', '.join(f'{l}({n})' for l, n in top_langs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    download_exams_v(data_dir)
    download_openqa(data_dir)
    print_summary(data_dir)
    print("\nAll datasets downloaded!")
