"""
Phase 6 pipeline: rebuild dataset v3 incorporating all new sources.
New data:
  - Wikipedia full (87k articles, ~1.47M sentences)
  - Synthetic Gemma4/qwen3 (Urdu loanwords + code-mix + long sentences)
  - Existing: Aksharantar, Dakshina, Samanantar (already processed)

Run this after all ingest scripts have completed.
"""

import json
import shutil
from pathlib import Path

from data.curate.normalize import normalize_snapshot
from data.curate.filter_junk import filter_snapshot
from data.curate.build_pairs import build_pairs_from_snapshot
from data.curate.romanize_rules import fill_pending_pairs
from data.curate.improve_pairs import improve_wikipedia_pairs
from data.curate.augment_colloquial import augment_snapshot
from data.curate.dedup import dedup_snapshot
from data.curate.quality_score import score_snapshot
from data.curate.split_dataset import split_files


RAW = Path("data/raw")
INTERIM = Path("data/interim")
REVIEW = Path("data/review")
PROCESSED = Path("data/processed")
PROCESSED_V3 = Path("data/processed_v3")

# New sources to process — already-processed sources are reused from scored_* files
NEW_SOURCES = [
    "wikipedia_te_20260517",
    "synthetic_gemma4_20260517",
]

# Already scored from Phase 2 + Phase 4 — reuse directly
EXISTING_SCORED = [
    "scored_aksharantar_20260515.jsonl",
    "scored_dakshina_20260515.jsonl",
    "scored_samanantar_20260517.jsonl",
]


def process_source(name: str) -> str | None:
    raw_file = RAW / f"{name}.jsonl"
    if not raw_file.exists():
        print(f"  SKIP {name} — raw file not found")
        return None

    REVIEW.mkdir(exist_ok=True)
    scored_file = INTERIM / f"scored_{name}.jsonl"

    if scored_file.exists():
        print(f"  {name}: already processed ({sum(1 for _ in scored_file.open(encoding='utf-8'))} pairs)")
        return str(scored_file)

    print(f"  Processing {name} ...")

    norm = INTERIM / f"normalized_{name}.jsonl"
    filt = INTERIM / f"filtered_{name}.jsonl"
    pairs = INTERIM / f"pairs_{name}.jsonl"
    roman = INTERIM / f"romanized_{name}.jsonl"
    improved = INTERIM / f"improved_{name}.jsonl"
    augmented = INTERIM / f"augmented_{name}.jsonl"
    deduped = INTERIM / f"deduped_{name}.jsonl"

    r = normalize_snapshot(raw_file, norm)
    print(f"    normalize: {r['output_count']:,}")
    r = filter_snapshot(norm, filt)
    print(f"    filter: {r['output_count']:,}")
    r = build_pairs_from_snapshot(filt, pairs)
    print(f"    build_pairs: {r['pair_count']:,}")
    r = fill_pending_pairs(pairs, roman)
    print(f"    romanize: {r['filled']:,} filled")
    r = improve_wikipedia_pairs(roman, improved)
    print(f"    improve: {r['total_out']:,}")
    r = augment_snapshot(improved, augmented)
    print(f"    augment: {r['original_pairs'] + r['augmented_pairs']:,}")
    r = dedup_snapshot(augmented, deduped)
    print(f"    dedup: {r['output_count']:,}")
    r = score_snapshot(deduped, scored_file, REVIEW / f"review_{name}.jsonl")
    print(f"    score: approved={r['approved']:,}")

    return str(scored_file)


def main() -> None:
    print("=== Phase 6 Dataset v3 Rebuild ===")
    print()

    all_scored = []

    # Process new sources
    print("New sources:")
    for name in NEW_SOURCES:
        path = process_source(name)
        if path:
            all_scored.append(Path(path))

    # Add existing scored files
    print("\nExisting sources:")
    for fname in EXISTING_SCORED:
        p = INTERIM / fname
        if p.exists():
            n = sum(1 for _ in p.open(encoding="utf-8"))
            print(f"  {fname}: {n:,} pairs")
            all_scored.append(p)
        else:
            print(f"  SKIP {fname} — not found")

    if not all_scored:
        print("No scored files found. Run ingest scripts first.")
        return

    print(f"\nSplitting {len(all_scored)} sources into train/val/test ...")
    result = split_files(all_scored, PROCESSED_V3)
    print(f"  train: {result['counts']['train']:,}")
    print(f"  val:   {result['counts']['val']:,}")
    print(f"  test:  {result['counts']['test']:,}")
    print(f"  total: {result['total']:,}")
    print(f"  leakage: {result['leakage_violations'] or 'clean'}")

    print(f"\nDataset v3 ready at {PROCESSED_V3}/")
    print("Update train/config.yaml to point to data/processed_v3/ and retrain.")


if __name__ == "__main__":
    main()
