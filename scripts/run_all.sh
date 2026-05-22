#!/bin/bash
# ============================================================
# ImageCLEF 2026 Multimodal Reasoning — Full Pipeline
# Run on H200 (80GB). Total: ~30 GPU hours.
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DATA_DIR="./data"
MODEL_8B="Qwen/Qwen3-VL-8B-Thinking"
MODEL_4B="Qwen/Qwen3-VL-4B-Thinking"
VLLM_PORT=8000

echo "============================================"
echo "Phase 1: Download Data"
echo "============================================"
python src/download_data.py --data_dir "$DATA_DIR"

echo "============================================"
echo "Phase 2: Prepare Fine-tuning Data"
echo "============================================"
python src/prepare_finetune_data.py \
    --data_dir "$DATA_DIR" \
    --output "$DATA_DIR/exams_v_train_llamafactory.json" \
    --prompt_style thinking \
    --create_val_subset

echo "============================================"
echo "Phase 3: Zero-shot Baseline (8B)"
echo "============================================"
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_8B" \
    --port $VLLM_PORT \
    --max-model-len 4096 \
    --dtype bfloat16 &
VLLM_PID=$!
sleep 60  # Wait for model to load

echo "Running baseline on 500-sample val subset..."
python src/baseline.py \
    --data_dir "$DATA_DIR" \
    --split validation \
    --model_name "$MODEL_8B" \
    --port $VLLM_PORT \
    --prompt_style thinking \
    --output_dir ./outputs/baseline_8b_thinking \
    --limit 500

# Kill vLLM server
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "============================================"
echo "Phase 3 complete. Check baseline accuracy."
echo "Next: Run prompt ablation, then QLoRA fine-tuning."
echo "See PLAN.md for QLoRA config details."
echo "============================================"
echo ""
echo "To fine-tune with LLaMA-Factory:"
echo "  llamafactory-cli train configs/qlora_8b.yaml"
echo ""
echo "Or with HF TRL:"
echo "  python src/finetune_qlora.py --model $MODEL_8B --data $DATA_DIR/exams_v_train_llamafactory.json"
