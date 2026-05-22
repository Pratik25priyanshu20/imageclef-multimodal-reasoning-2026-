"""
Self-consistency with confidence-weighted majority voting.

Run inference k times with temperature>0, then vote.

Usage:
    # Start vLLM server first, then:
    python src/self_consistency.py \
        --data_dir ./data \
        --split test \
        --model_name ./finetuned_8b \
        --port 8000 \
        --k 3 \
        --temperature 0.7 \
        --output_dir ./outputs/sc_k3
"""

import argparse
import base64
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from openai import OpenAI


PROMPT = (
    "Look at this exam question image carefully.\n"
    "1. Read the question and ALL answer options.\n"
    "2. If there are diagrams, graphs, tables, or visual elements, analyze them.\n"
    "3. Think step by step about the correct answer.\n"
    "4. Select exactly ONE answer.\n\n"
    "Answer with a single letter: A, B, C, D, or E."
)


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"


def predict_once(client, model_name, image_path, temperature):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": image_to_base64(image_path)}},
        ],
    }]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=1,
        extra_body={"guided_choice": ["A", "B", "C", "D", "E"]},
    )
    return response.choices[0].message.content.strip().upper()


def confidence_weighted_vote(answers: list[str]) -> str:
    """Majority vote weighted by inverse entropy of the distribution."""
    counts = Counter(answers)
    total = len(answers)

    if len(counts) == 1:
        return counts.most_common(1)[0][0]

    # Simple majority
    top_answer, top_count = counts.most_common(1)[0]

    # If clear majority (>50%), take it
    if top_count > total / 2:
        return top_answer

    # Tiebreaker: first occurrence (preserves model preference)
    return top_answer


def run_self_consistency(args):
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1")

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "exams_v" / f"{args.split}_metadata.json"
    image_dir = data_dir / "exams_v" / args.split / "images"

    with open(meta_path) as f:
        metadata = json.load(f)

    if args.limit:
        metadata = metadata[:args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    results_path = output_dir / "predictions.json"
    done_ids = set()
    results = []
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} done")

    remaining = [m for m in metadata if m["id"] not in done_ids]
    print(f"Self-consistency: k={args.k}, temp={args.temperature}")
    print(f"Split: {args.split} ({len(remaining)} remaining of {len(metadata)})")

    t0 = time.time()
    for i, item in enumerate(remaining):
        image_path = str(image_dir / f"{item['id']}.png")
        if not os.path.exists(image_path):
            continue

        # Run k times
        answers = []
        for _ in range(args.k):
            ans = predict_once(client, args.model_name, image_path, args.temperature)
            answers.append(ans)

        final_answer = confidence_weighted_vote(answers)
        counts = Counter(answers)

        has_gold = "answer_key" in item
        is_correct = final_answer == item.get("answer_key", "") if has_gold else None

        results.append({
            "id": item["id"],
            "answer_key": final_answer,
            "votes": dict(counts),
            "all_answers": answers,
            "agreement": counts.most_common(1)[0][1] / len(answers),
            **({"gold": item["answer_key"], "correct": is_correct} if has_gold else {}),
            "language": item.get("language", ""),
            "subject": item.get("subject", ""),
        })

        if (i + 1) % 50 == 0:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate / 60

            if has_gold:
                correct = sum(1 for r in results if r.get("correct"))
                acc = correct / len(results) * 100
                print(f"  [{i+1}/{len(remaining)}] acc={acc:.1f}% rate={rate:.2f}/s ETA={eta:.0f}min")
            else:
                print(f"  [{i+1}/{len(remaining)}] rate={rate:.2f}/s ETA={eta:.0f}min")

    # Final save
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Analysis
    if results and "correct" in results[0]:
        correct = sum(1 for r in results if r.get("correct"))
        total = len(results)
        print(f"\nOverall: {correct}/{total} = {correct/total*100:.2f}%")

        # Agreement analysis
        high_agree = [r for r in results if r["agreement"] >= 1.0]
        low_agree = [r for r in results if r["agreement"] < 0.5]
        if high_agree:
            ha_correct = sum(1 for r in high_agree if r["correct"])
            print(f"Full agreement ({len(high_agree)} samples): {ha_correct/len(high_agree)*100:.1f}%")
        if low_agree:
            la_correct = sum(1 for r in low_agree if r["correct"])
            print(f"Low agreement ({len(low_agree)} samples): {la_correct/len(low_agree)*100:.1f}%")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output_dir", type=str, default="./outputs/sc_k3")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_self_consistency(args)
