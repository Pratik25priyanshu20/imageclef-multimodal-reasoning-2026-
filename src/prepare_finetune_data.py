"""
Convert EXAMS-V training data to LLaMA-Factory ShareGPT format for QLoRA fine-tuning.

Usage:
    python src/prepare_finetune_data.py \
        --data_dir ./data \
        --output ./data/exams_v_train_llamafactory.json \
        --prompt_style thinking

Output format (ShareGPT):
[
  {
    "messages": [
      {"role": "user", "content": "<image>prompt text"},
      {"role": "assistant", "content": "B"}
    ],
    "images": ["path/to/image.png"]
  },
  ...
]
"""

import argparse
import json
import random
from pathlib import Path


PROMPT_STYLES = {
    "thinking": (
        "Look at this exam question image carefully.\n"
        "1. Read the question and ALL answer options.\n"
        "2. If there are diagrams, graphs, tables, or visual elements, analyze them.\n"
        "3. Think step by step about the correct answer.\n"
        "4. Select exactly ONE answer.\n\n"
        "Answer with a single letter: A, B, C, D, or E."
    ),
    "direct": (
        "Look at this exam question image. Identify the question and all answer options. "
        "Select the correct answer. Reply with ONLY the letter (A, B, C, D, or E)."
    ),
}


def prepare_data(args):
    data_dir = Path(args.data_dir)
    meta_path = data_dir / "exams_v" / "train_metadata.json"
    image_dir = data_dir / "exams_v" / "train" / "images"

    with open(meta_path) as f:
        metadata = json.load(f)

    prompt_text = PROMPT_STYLES[args.prompt_style]
    samples = []

    for item in metadata:
        image_path = str(image_dir / f"{item['id']}.png")
        if not Path(image_path).exists():
            continue

        sample = {
            "messages": [
                {"role": "user", "content": f"<image>{prompt_text}"},
                {"role": "assistant", "content": item["answer_key"]},
            ],
            "images": [image_path],
        }
        samples.append(sample)

    # Shuffle for training
    random.seed(42)
    random.shuffle(samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # Print stats
    print(f"Prepared {len(samples)} training samples")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Output: {output_path}")

    # Language distribution
    langs = {}
    for item in metadata:
        lang = item.get("language", "unknown")
        langs[lang] = langs.get(lang, 0) + 1
    print(f"\nLanguage distribution:")
    for lang in sorted(langs, key=lambda l: -langs[l]):
        print(f"  {lang:<15} {langs[lang]}")

    # Also create a 500-sample stratified subset for prompt engineering
    if args.create_val_subset:
        val_meta_path = data_dir / "exams_v" / "validation_metadata.json"
        with open(val_meta_path) as f:
            val_metadata = json.load(f)

        # Stratify by language
        by_lang = {}
        for item in val_metadata:
            lang = item.get("language", "unknown")
            by_lang.setdefault(lang, []).append(item)

        subset = []
        target_per_lang = max(1, 500 // len(by_lang))
        for lang, items in by_lang.items():
            random.shuffle(items)
            subset.extend(items[:target_per_lang])

        # Trim to exactly 500
        random.shuffle(subset)
        subset = subset[:500]

        subset_path = data_dir / "exams_v" / "val_subset_500.json"
        with open(subset_path, "w") as f:
            json.dump(subset, f, indent=2, ensure_ascii=False)
        print(f"\nVal subset: {len(subset)} samples → {subset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./data/exams_v_train_llamafactory.json")
    parser.add_argument("--prompt_style", choices=PROMPT_STYLES.keys(), default="thinking")
    parser.add_argument("--create_val_subset", action="store_true", default=True)
    args = parser.parse_args()
    prepare_data(args)
