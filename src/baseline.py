"""
Zero-shot baseline: Run Qwen3-VL on EXAMS-V MCQ via vLLM.

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-VL-8B-Thinking --port 8000 \
        --max-model-len 4096 --dtype bfloat16

    # Then run baseline:
    python src/baseline.py \
        --data_dir ./data \
        --split validation \
        --model_name Qwen/Qwen3-VL-8B-Thinking \
        --port 8000 \
        --prompt_style thinking \
        --output_dir ./outputs/baseline_8b_thinking
"""

import argparse
import base64
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from openai import OpenAI


PROMPT_STYLES = {
    "direct": (
        "Look at this exam question image. Identify the question and all answer options. "
        "Select the correct answer. Reply with ONLY the letter (A, B, C, D, or E)."
    ),
    "thinking": (
        "Look at this exam question image carefully.\n"
        "1. Read the question and ALL answer options.\n"
        "2. If there are diagrams, graphs, tables, or visual elements, analyze them.\n"
        "3. Think step by step about the correct answer.\n"
        "4. Select exactly ONE answer.\n\n"
        "Answer with a single letter: A, B, C, D, or E."
    ),
    "decomposed": (
        "Look at this exam question image.\n"
        "First, describe what you see: the question text, answer options, and any "
        "visual elements (diagrams, graphs, tables, chemical structures).\n"
        "Then, reason step by step to determine the correct answer.\n"
        "Finally, state your answer as a single letter: A, B, C, D, or E."
    ),
}


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def predict_mcq(client, model_name, image_path, prompt_text, use_guided=True):
    """Predict a single MCQ answer."""
    image_b64 = image_to_base64(image_path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_b64}},
        ],
    }]

    kwargs = dict(
        model=model_name,
        messages=messages,
        temperature=0,
        max_completion_tokens=1 if use_guided else 50,
    )
    if use_guided:
        kwargs["extra_body"] = {"guided_choice": ["A", "B", "C", "D", "E"]}

    response = client.chat.completions.create(**kwargs)
    answer = response.choices[0].message.content.strip().upper()

    if answer not in {"A", "B", "C", "D", "E"}:
        for ch in answer:
            if ch in {"A", "B", "C", "D", "E"}:
                answer = ch
                break
        else:
            answer = "A"

    return answer


def run_baseline(args):
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1")
    prompt_text = PROMPT_STYLES[args.prompt_style]

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "exams_v" / f"{args.split}_metadata.json"
    image_dir = data_dir / "exams_v" / args.split / "images"

    with open(meta_path) as f:
        metadata = json.load(f)

    if args.limit:
        metadata = metadata[:args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    results_path = output_dir / "predictions.json"
    done_ids = set()
    results = []
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} already done")

    remaining = [m for m in metadata if m["id"] not in done_ids]
    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)

    print(f"Model: {args.model_name}")
    print(f"Prompt: {args.prompt_style}")
    print(f"Split: {args.split} ({len(remaining)} remaining of {len(metadata)})")

    t0 = time.time()
    for i, item in enumerate(remaining):
        image_path = str(image_dir / f"{item['id']}.png")
        if not os.path.exists(image_path):
            print(f"  SKIP {item['id']}: image not found")
            continue

        prediction = predict_mcq(
            client, args.model_name, image_path, prompt_text, args.guided
        )
        is_correct = prediction == item["answer_key"]
        correct += is_correct
        total += 1

        results.append({
            "id": item["id"],
            "answer_key": prediction,
            "gold": item["answer_key"],
            "correct": is_correct,
            "language": item["language"],
            "subject": item["subject"],
            "type": item["type"],
        })

        # Checkpoint every 100
        if (i + 1) % 100 == 0:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate / 60
            acc = correct / total * 100
            print(f"  [{i+1}/{len(remaining)}] acc={acc:.1f}% rate={rate:.1f}/s ETA={eta:.0f}min")

    # Final save
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print analysis
    acc = correct / total * 100 if total else 0
    print(f"\n{'='*60}")
    print(f"Overall: {correct}/{total} = {acc:.2f}%")

    # Per-language
    lang_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        lang_stats[r["language"]]["total"] += 1
        if r["correct"]:
            lang_stats[r["language"]]["correct"] += 1

    print(f"\nPer-language accuracy:")
    for lang in sorted(lang_stats, key=lambda l: -lang_stats[l]["total"]):
        s = lang_stats[lang]
        print(f"  {lang:<15} {s['correct']}/{s['total']} = {s['correct']/s['total']*100:.1f}%")

    # Per-subject (top 10)
    subj_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        subj_stats[r["subject"]]["total"] += 1
        if r["correct"]:
            subj_stats[r["subject"]]["correct"] += 1

    print(f"\nPer-subject accuracy (top 10):")
    for subj in sorted(subj_stats, key=lambda s: -subj_stats[s]["total"])[:10]:
        s = subj_stats[subj]
        print(f"  {subj:<25} {s['correct']}/{s['total']} = {s['correct']/s['total']*100:.1f}%")

    # Per-type
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        type_stats[r["type"]]["total"] += 1
        if r["correct"]:
            type_stats[r["type"]]["correct"] += 1

    print(f"\nPer-type accuracy:")
    for t in sorted(type_stats):
        s = type_stats[t]
        print(f"  {t:<15} {s['correct']}/{s['total']} = {s['correct']/s['total']*100:.1f}%")

    # Save analysis
    analysis = {
        "model": args.model_name,
        "prompt_style": args.prompt_style,
        "split": args.split,
        "overall_accuracy": acc,
        "total": total,
        "per_language": {l: {"accuracy": s["correct"]/s["total"]*100, "n": s["total"]}
                         for l, s in lang_stats.items()},
        "per_subject": {s: {"accuracy": v["correct"]/v["total"]*100, "n": v["total"]}
                        for s, v in subj_stats.items()},
        "per_type": {t: {"accuracy": s["correct"]/s["total"]*100, "n": s["total"]}
                     for t, s in type_stats.items()},
    }
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt_style", choices=PROMPT_STYLES.keys(), default="thinking")
    parser.add_argument("--output_dir", type=str, default="./outputs/baseline")
    parser.add_argument("--guided", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for testing")
    args = parser.parse_args()
    run_baseline(args)
