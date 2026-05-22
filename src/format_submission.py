"""
Format predictions into competition submission JSON.

Usage:
    # MCQ submission
    python src/format_submission.py \
        --predictions ./outputs/sc_k3/predictions.json \
        --task mcq \
        --output ./submissions/mcq_normal_sc3.json

    # OpenQA submission
    python src/format_submission.py \
        --predictions ./outputs/openqa/predictions.json \
        --task openqa \
        --output ./submissions/openqa_visual_normal.json
"""

import argparse
import json
from pathlib import Path


def format_mcq(predictions: list) -> list:
    """Format MCQ predictions for submission.

    Required format:
    [{"question_id": "uuid", "answer_key": "A"}, ...]
    """
    submission = []
    for pred in predictions:
        submission.append({
            "question_id": pred["id"],
            "answer_key": pred["answer_key"],
        })
    return submission


def format_openqa(predictions: list) -> list:
    """Format OpenQA predictions for submission.

    Required format:
    [{"question_id": "uuid", "answers": ["text"], "language": "English"}, ...]
    """
    submission = []
    for pred in predictions:
        answers = pred.get("answers", [pred.get("answer", "")])
        if isinstance(answers, str):
            answers = [answers]
        submission.append({
            "question_id": pred["question_id"],
            "answers": answers,
            "language": pred.get("language", ""),
        })
    return submission


def validate_mcq(submission: list):
    """Validate MCQ submission format."""
    ids = set()
    for item in submission:
        assert "question_id" in item, f"Missing question_id: {item}"
        assert "answer_key" in item, f"Missing answer_key: {item}"
        assert item["answer_key"] in {"A", "B", "C", "D", "E"}, \
            f"Invalid answer_key: {item['answer_key']}"
        assert item["question_id"] not in ids, f"Duplicate id: {item['question_id']}"
        ids.add(item["question_id"])
    print(f"MCQ validation passed: {len(submission)} predictions, all valid")


def validate_openqa(submission: list):
    """Validate OpenQA submission format."""
    ids = set()
    for item in submission:
        assert "question_id" in item, f"Missing question_id"
        assert "answers" in item, f"Missing answers"
        assert isinstance(item["answers"], list), f"answers must be a list"
        assert item["question_id"] not in ids, f"Duplicate id: {item['question_id']}"
        ids.add(item["question_id"])
    print(f"OpenQA validation passed: {len(submission)} predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--task", choices=["mcq", "openqa"], required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    if args.task == "mcq":
        submission = format_mcq(predictions)
        validate_mcq(submission)
    else:
        submission = format_openqa(predictions)
        validate_openqa(submission)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    print(f"Submission saved: {output_path} ({len(submission)} items)")
