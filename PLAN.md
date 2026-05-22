# ImageCLEF 2026 — Multimodal Reasoning

## Deadline: May 7, 2026 | Registration closes: Apr 23, 2026

---

## Task Summary

Answer exam questions from images across 17 languages, 4 tracks:

| Track | Type | Metric |
|-------|------|--------|
| Visual MCQ | Image with diagrams/graphs → A/B/C/D/E | Accuracy |
| Text MCQ | Image with text-only question → A/B/C/D/E | Accuracy |
| Visual OpenQA | Image question → free-form answer | BLEU/ROUGE/METEOR/COMET |
| Text OpenQA | Text question → free-form answer | BLEU/ROUGE/METEOR/COMET |

**Constraints**: Open-weights only, Tiny (≤7B) vs Normal (≥8B), must run on A40 (40GB), vLLM serving, `bash inference.sh`

**Dataset**: EXAMS-V (16,300 train / 4,650 val / 3,570 test), OpenQA-Visual (528 train / 240 dev / 439 test), OpenQA-Textual (~300 train / 325 dev)

**Submissions**: Max 20/day, 200 total, JSON format

---

## Model Selection

| Category | Model | VRAM | Benchmark | Why |
|----------|-------|------|-----------|-----|
| **Normal (≥8B)** | Qwen3-VL-8B-Thinking | ~18GB | MMMU 74.1, MathVista 81.4 | Best open VLM per VRAM, native CoT, fits A40 easily |
| **Tiny (≤7B)** | Qwen3-VL-4B-Instruct | ~10GB | TBD | Fine-tune to compensate for size. Fewer teams invest in Tiny |
| Backup Normal | InternVL3.5-14B | ~30GB | MMMU 73.3 | If Qwen3-VL has vLLM issues |

---

## Strategy: The Accuracy Stack

| Layer | Normal (8B) | Tiny (4B) | GPU Cost |
|-------|-------------|-----------|----------|
| Zero-shot baseline | ~60-65% | ~45-50% | 4h |
| + Thinking mode CoT | ~67-72% | — | 0 (prompt) |
| + Prompt engineering | ~70-75% | ~50-55% | 2h |
| + QLoRA fine-tuning (16K EXAMS-V) | ~78-83% | ~65-72% | 14h |
| + Self-consistency k=3 | ~82-86% | ~68-75% | 7h |
| + Confidence-weighted voting | ~83-87% | ~69-76% | 0 (post-proc) |

2025 winner: 81.4%. **Target: 85%+ Normal, 70%+ Tiny.**

### Zero-Cost Prompt Improvements (included in all prompts)

Baked into the pipeline at no extra GPU cost:

1. **Subject-aware prompting** — test data includes subject field (32 categories). Prepend: "You are an expert in {subject}." to every prompt.
2. **Language-matched prompting** — detect question language from metadata, prompt in that language instead of English. Qwen3-VL is multilingual and performs better when prompt matches content.

---

## GPU Budget: ~30h on H200 (single GPU, sequential)

| Phase | H200 Hours | Running Total |
|-------|-----------|---------------|
| Setup + zero-shot baseline | 4h | 4h |
| Prompt engineering (3 variants × 500 samples) | 2h | 6h |
| QLoRA fine-tune 8B | 9h | 15h |
| QLoRA fine-tune 4B | 5h | 20h |
| Val evaluation (single pass) | 1h | 21h |
| Test submissions with SC k=3 | 7h | 28h |
| OpenQA runs | 1h | 29h |
| Buffer | 1h | 30h |

**Key optimizations applied:**
- QLoRA (nf4, bf16 compute) instead of LoRA: saves ~7h, <1% accuracy loss
- Prompt engineering on 500 stratified samples instead of full 4,650: saves ~4h
- Self-consistency only on test submissions, not val: saves ~4h

---

## Technical Details

### Prompting Strategy (ablate on 500-sample stratified val subset)

- **P1: Direct** — "Answer with one letter: A/B/C/D/E"
- **P2: Thinking mode** — Enable thinking tokens for internal CoT reasoning
- **P3: Decomposed** — "First describe what you see. Then identify question and options. Then reason step by step."

### QLoRA Fine-Tuning Config

| Setting | Value |
|---------|-------|
| Framework | LLaMA-Factory or HF TRL |
| Quantization | nf4 (QLoRA) |
| Compute dtype | bfloat16 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | all linear layers |
| Learning rate | 2e-4 |
| Epochs | 3 (reduce to 2 if overfitting) |
| Batch size | 1, grad accum 8 |
| LR scheduler | cosine, warmup_ratio=0.1 |
| Data | 16,300 EXAMS-V train |
| Time (8B on H200) | ~9h |
| Time (4B on H200) | ~5h |

### Self-Consistency Config

- temperature=0.7, k=3 samples per question
- Confidence-weighted majority voting (inverse entropy weighting)
- Applied only on test submissions (skip val SC to save GPU)

### Submission Format

**MCQ:**
```json
[{"question_id": "uuid", "answer_key": "B"}]
```

**OpenQA:**
```json
[{"question_id": "uuid", "answers": ["answer text"], "language": "English"}]
```

### inference.sh Template
```bash
#!/bin/bash
python -m vllm.entrypoints.openai.api_server \
    --model ./qwen3-vl-8b-thinking-finetuned \
    --port 8000 --max-model-len 4096 &
sleep 30
python predict_mcq.py --test_data $1 --output predictions.json
python predict_openqa.py --test_data $1 --output predictions_qa.json
```

---

## Execution Plan

### Phase 1: Foundation (Apr 21-22) — 4 GPU hours

- [ ] **Register on all 4 tracks (DEADLINE: Apr 23!)**
- [ ] Download EXAMS-V + OpenQA datasets from HuggingFace
- [ ] Download Qwen3-VL-8B-Thinking + Qwen3-VL-4B
- [ ] Verify Qwen3-VL support in vLLM (if not: fall back to InternVL3.5-14B)
- [ ] Zero-shot Qwen3-VL-8B on val (4,650 MCQ) → baseline accuracy
- [ ] Analyze: accuracy by language, subject, visual type

### Phase 2: Prompt Engineering (Apr 23) — 2 GPU hours

- [ ] Create 500-sample stratified val subset (balanced by language, subject, visual type)
- [ ] Test P1-P3 prompt variants on subset with 8B-Thinking
- [ ] Pick best prompt
- [ ] Analyze which subjects/languages benefit from thinking mode vs direct

### Phase 3: QLoRA Fine-Tuning — 14 GPU hours

- [ ] Prepare EXAMS-V training data in framework format (image + prompt → A/B/C/D/E)
- [ ] QLoRA fine-tune Qwen3-VL-8B-Thinking: ~9h on H200
- [ ] Evaluate fine-tuned 8B on val (single pass): ~0.5h
- [ ] QLoRA fine-tune Qwen3-VL-4B: ~5h on H200
- [ ] Evaluate fine-tuned 4B on val: ~0.5h
- [ ] If overfitting: reduce to 2 epochs, re-run

### Phase 4: Test Submissions with Self-Consistency (Apr 27-30) — 7 GPU hours

- [ ] Run fine-tuned 8B on test with SC k=3 (temp=0.7): ~4h
- [ ] Run fine-tuned 4B on test with SC k=3: ~2h
- [ ] Implement confidence-weighted voting (post-processing, no GPU)
- [ ] Format submissions (JSON) for all 4 tracks × 2 sizes
- [ ] Submit to leaderboard
- [ ] Analyze results, iterate if needed

### Phase 5: OpenQA (May 1) — 1 GPU hour

- [ ] Run fine-tuned 8B on OpenQA-Visual + OpenQA-Textual test sets
- [ ] Run fine-tuned 4B on same
- [ ] Format submissions, submit

### Phase 6: Iteration (May 2-5) — use buffer hours

- [ ] Based on leaderboard: identify weak languages/subjects
- [ ] Targeted improvements (prompt tweaks, not re-training)
- [ ] Final submissions

### Phase 7: Reproducibility (May 5-7) — no GPU needed

- [ ] Create inference.sh script
- [ ] Prepare public GitHub repo with model weights
- [ ] Verify format with organizer's format_checker.py
- [ ] Final submission by May 7

---

## Key Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Qwen3-VL not in vLLM | Medium | Fall back to InternVL3.5-14B or use transformers directly |
| Fine-tuning overfits | Low | Monitor val loss, reduce to 2 epochs |
| Someone uses 72B model | Possible | Fine-tuned 8B + SC beats zero-shot 72B |
| A40 can't fit model + KV cache | Low | 8B = 18GB, leaves 22GB for KV cache |
| Registration missed | Critical | Register TODAY |

---

## Paper Contribution

**"Fine-Tuned Thinking VLMs with Self-Consistency for Multilingual Exam Reasoning"**

1. First evaluation of thinking-mode VLMs on multilingual exam VQA
2. QLoRA fine-tuning ablation on EXAMS-V showing accuracy gains
3. Self-consistency voting adapted for VLM exam solving
4. Per-language, per-subject analysis
5. Tiny vs Normal comparison

### cMSCI Citation (Related Work only)

> "While geometric coherence metrics such as cMSCI (Priyanshu & Chandna, 2026) have shown
> promise for multimodal evaluation, we find that VLM-internal confidence signals outperform
> external embedding-based metrics for answer selection, as the VLM's representations are
> conditioned on the full visual context that embedding-only approaches cannot access."

---

## Reference: 2025 Winner Analysis

MSA team achieved 81.4% with an OCR-VLM ensemble pipeline using older models.
Our approach improves on this by:
- Using a thinking-mode VLM (native CoT, simpler pipeline)
- QLoRA fine-tuning on 16K task-specific examples (they used zero-shot)
- Self-consistency voting (they didn't)

## Key Resources

- Dataset: `MBZUAI/EXAMS-V` on HuggingFace
- OpenQA Visual: `SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual`
- OpenQA Textual: `SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual`
- Starter kit: `github.com/mbzuai-nlp/ImageCLEF-MultimodalReasoning` (2026/)
- Eval scripts: `2026/src/evaluation/evaluate_mcq.py` and `evaluate_qa.py`
- Competition: `https://mbzuai-nlp.github.io/ImageCLEF-MultimodalReasoning/2026/`
