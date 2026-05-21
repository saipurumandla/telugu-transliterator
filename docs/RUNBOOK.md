# TeluguTransliterator — Release Runbook

Operational guide for running the pipeline, retraining, and releasing new versions.

---

## Prerequisites

- Python 3.11+ (3.12 for Phase 6+)
- WSL2 with CUDA drivers (dual RTX 3090 tested)
- Ollama running locally for scoring and synthetic generation
- ~300 GB free disk for raw data + interim files + checkpoints

```bash
cd /mnt/c/Users/harin/source/repos/tracked-repos/telugu-transliterator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Full Pipeline Run

### 1. Ingest new data

```bash
python3 -m data.ingest.ingest_dakshina
python3 -m data.ingest.ingest_licensed_corpus
python3 -m data.ingest.ingest_wikipedia_te
python3 -m data.ingest.ingest_synthetic_gemma4   # requires Ollama + gemma4
```

Raw snapshots written to `data/raw/`.

### 2. Run curation pipeline

```bash
python3 -m data.curate.run_pipeline
```

Processes all raw snapshots through: normalize → filter → build_pairs → romanize → improve → augment → dedup → score.
Intermediate files in `data/interim/`. Scored files: `data/interim/scored_*.jsonl`.

### 3. Check for drift (optional, for new pulls)

```bash
python3 -m data.curate.drift_check \
    data/raw/new_snapshot.jsonl \
    data/interim/scored_aksharantar_20260515.jsonl
```

Alert if `status: DRIFT_DETECTED`. Investigate before proceeding.

### 4. Rebuild dataset splits

```bash
python3 -m data.curate.rebuild_v3
```

Output: `data/processed_v3/train.jsonl`, `val.jsonl`, `test.jsonl`.

### 5. Train

Pause Windows Update before starting (Settings → Windows Update → Pause 1 week).

```bash
# WSL2 — activate venv first
source .venv/bin/activate
accelerate launch --num_processes 2 -m train.train_byt5 train/config.yaml
```

Expected: ~10 hours on dual RTX 3090, bf16. Checkpoints every 10,000 steps in `train/checkpoints_v3/`.

### 6. Evaluate

```bash
python3 -m eval.evaluate \
    --model-dir train/checkpoints_v3/checkpoint-80000 \
    --test-file data/processed_v3/test_sample.jsonl \
    --output eval/v3_eval.json \
    --batch-size 256 --gpu 0
```

Compare against `reports/benchmarks/baseline_report.md`.

### 7. Export and release

```bash
python3 export/make_model_card.py
python3 export/make_dataset_card.py
python3 export/push_hub.py --checkpoint train/checkpoints_v3/checkpoint-80000
```

Requires `huggingface-cli login` first.

---

## Smoke Tests

After any dependency update or environment change:

```bash
pytest tests/smoke/ -v
```

All tests must pass before releasing.

---

## Troubleshooting

### Training OOM (out of memory)

Reduce `per_device_train_batch_size` in `train/config.yaml` from 32 to 16.

### Ollama not available during scoring

Pipeline degrades gracefully — Ollama scoring is skipped, rule-based scores used only.
Check `data/interim/scored_*.jsonl` for `ollama_score: null` entries.

### Windows Update kills training

- Pause Windows Update for 1 week before starting
- Checkpoints save every 10,000 steps — resume automatically from latest checkpoint
- Resume: just rerun the same `accelerate launch` command

### UnicodeDecodeError on Windows

All file opens use `encoding="utf-8"`. If you see cp1252 errors, check that no script
is using `open(path)` without the encoding argument.

### Tokenizer vocab_size error during eval

Known issue with ByT5 tokenizer in transformers ≥4.40 when decoding predictions.
Fixed by running standalone `eval/evaluate.py` rather than Trainer's built-in eval.

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| v0.1 | 2025-07-06 | Phase 3 baseline — 4.49% overall CER |
| v1.0 | 2026-01-11 | Phase 5 release — HuggingFace publish |
| v1.1 | 2026-04-25 | Phase 6 — long sentence CER 34% → 16%, 6.1M pairs |
