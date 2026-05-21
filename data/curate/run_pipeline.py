"""Run the full curation pipeline on all new raw snapshots."""

import sys
from pathlib import Path

from data.curate.normalize import normalize_snapshot
from data.curate.filter_junk import filter_snapshot
from data.curate.build_pairs import build_pairs_from_snapshot
from data.curate.romanize_rules import fill_pending_pairs
from data.curate.improve_pairs import improve_wikipedia_pairs
from data.curate.augment_colloquial import augment_snapshot
from data.curate.dedup import dedup_snapshot
from data.curate.quality_score import score_snapshot

RAW = Path("data/raw")
INTERIM = Path("data/interim")
REVIEW = Path("data/review")
REVIEW.mkdir(exist_ok=True)


def process(name: str) -> bool:
    raw = RAW / f"{name}.jsonl"
    if not raw.exists():
        return False

    scored = INTERIM / f"scored_{name}.jsonl"
    if scored.exists():
        n = sum(1 for _ in scored.open(encoding="utf-8"))
        print(f"  SKIP {name} — already scored ({n:,} pairs)")
        return True

    print(f"\n{'='*50}")
    print(f"Processing: {name}")
    print(f"{'='*50}")

    steps = [
        ("normalize",    normalize_snapshot,      raw,                             INTERIM / f"normalized_{name}.jsonl"),
        ("filter",       filter_snapshot,          INTERIM / f"normalized_{name}.jsonl", INTERIM / f"filtered_{name}.jsonl"),
        ("build_pairs",  build_pairs_from_snapshot, INTERIM / f"filtered_{name}.jsonl",  INTERIM / f"pairs_{name}.jsonl"),
        ("romanize",     fill_pending_pairs,       INTERIM / f"pairs_{name}.jsonl",      INTERIM / f"romanized_{name}.jsonl"),
        ("improve",      improve_wikipedia_pairs,  INTERIM / f"romanized_{name}.jsonl",  INTERIM / f"improved_{name}.jsonl"),
        ("augment",      augment_snapshot,         INTERIM / f"improved_{name}.jsonl",   INTERIM / f"augmented_{name}.jsonl"),
        ("dedup",        dedup_snapshot,           INTERIM / f"augmented_{name}.jsonl",  INTERIM / f"deduped_{name}.jsonl"),
    ]

    for label, fn, src, dst in steps:
        if dst.exists():
            print(f"  {label}: already done")
            continue
        print(f"  {label}...", end=" ", flush=True)
        r = fn(src, dst)
        key = next((k for k in ("output_count","pair_count","filled","total_out","output_count") if k in r), None)
        print(f"{r.get(key, ''):,}" if key else "done")

    print(f"  score...", end=" ", flush=True)
    r = score_snapshot(
        INTERIM / f"deduped_{name}.jsonl",
        scored,
        REVIEW / f"review_{name}.jsonl",
    )
    print(f"approved:{r['approved']:,}  review:{r['review_bucket']:,}  rejected:{r['rejected']:,}")
    return True


def main() -> None:
    snapshots = sorted(RAW.glob("*.jsonl"))
    if not snapshots:
        print("No raw snapshots found.")
        sys.exit(1)

    print(f"Found {len(snapshots)} raw snapshots:")
    for s in snapshots:
        n = sum(1 for _ in s.open(encoding="utf-8", errors="replace"))
        print(f"  {s.name}  ({n:,} records)")

    print()
    for snap in snapshots:
        name = snap.stem
        process(name)

    print("\nPipeline complete. Run rebuild_v3.py to split and merge.")


if __name__ == "__main__":
    main()
