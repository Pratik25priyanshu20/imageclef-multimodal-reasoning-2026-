#!/bin/bash
# ============================================================
# ImageCLEF 2026 — Final Submission Inference Script
# Must run on A40 (40GB). This is what organizers execute.
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODEL_DIR="${1:-./finetuned_8b_merged}"
TEST_DATA="${2:-./data/exams_v/test}"
OUTPUT_DIR="${3:-./submissions}"
PORT=8000

mkdir -p "$OUTPUT_DIR"

echo "Starting vLLM server with fine-tuned model..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --port $PORT \
    --max-model-len 4096 \
    --dtype bfloat16 &
VLLM_PID=$!
sleep 60

echo "Running MCQ inference with self-consistency k=3..."
python src/self_consistency.py \
    --data_dir ./data \
    --split test \
    --model_name "$MODEL_DIR" \
    --port $PORT \
    --k 3 \
    --temperature 0.7 \
    --output_dir "$OUTPUT_DIR/mcq_sc3"

echo "Formatting MCQ submission..."
python src/format_submission.py \
    --predictions "$OUTPUT_DIR/mcq_sc3/predictions.json" \
    --task mcq \
    --output "$OUTPUT_DIR/mcq_submission.json"

echo "Running OpenQA inference..."
# TODO: Add OpenQA inference script

kill $VLLM_PID 2>/dev/null || true
echo "Done! Submissions in $OUTPUT_DIR"
