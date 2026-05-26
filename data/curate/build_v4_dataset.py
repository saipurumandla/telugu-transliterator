"""
Build processed_v4 dataset.

Combines v3 train/val with synthetic Gemma4 pairs (base + augmented variants).
Streams v3 data to avoid loading 2.3 GB into memory.

Usage:
    python -m data.curate.build_v4_dataset
    python -m data.curate.build_v4_dataset --v3-dir data/processed_v3 --out-dir data/processed_v4
"""

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SYNTHETIC_RAW_FILES = [
    "data/raw/synthetic_gemma4_20260517.jsonl",
    "data/raw/synthetic_gemma4_20260521.jsonl",
    "data/raw/synthetic_gemma4_20260522.jsonl",
]

# UUID pattern embedded in source_doc_id: syn_<set>_<uuid>_ro / _te
_UUID_RE = re.compile(r"syn_.+?_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_(ro|te)$")


def _load_synthetic_pairs(raw_files: list[str]) -> list[dict]:
    """Read all synthetic raw files and pair roman+telugu records by UUID."""
    roman: dict[str, dict] = {}
    telugu: dict[str, dict] = {}

    for path in raw_files:
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                doc_id = r.get("source_doc_id", "")
                m = _UUID_RE.match(doc_id)
                if not m:
                    continue
                uid, side = m.group(1), m.group(2)
                if side == "ro":
                    roman[uid] = r
                else:
                    telugu[uid] = r

    pairs = []
    for uid, ro in roman.items():
        te = telugu.get(uid)
        if te is None:
            continue
        pairs.append({
            "roman_text": ro["text_raw"],
            "telugu_text": te["text_raw"],
            "source_name": ro.get("source_name", "ollama_synthetic"),
            "license_tag": ro.get("license_tag", "Apache-2.0"),
            "created_at": ro.get("pull_timestamp_utc", ""),
            "model": ro.get("model", "gemma4"),
        })

    return pairs


def _make_pair(
    roman_text: str,
    telugu_text: str,
    source_name: str,
    license_tag: str,
    created_at: str,
    pair_source: str,
    augmentation_variant: str | None,
) -> dict:
    return {
        "pair_id": str(uuid.uuid4()),
        "roman_text": roman_text,
        "telugu_text": telugu_text,
        "source_name": source_name,
        "license_tag": license_tag,
        "pair_source": pair_source,
        "quality_score": 0.85,
        "confidence": 0.85,
        "review_status": "approved",
        "created_at": created_at,
        "augmentation_variant": augmentation_variant,
        "score_reasons": [],
    }


def build(v3_dir: str, out_dir: str) -> None:
    from data.curate.augment_colloquial import generate_variants

    v3 = Path(v3_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and expand synthetic pairs ─────────────────────────────────────
    print("Loading synthetic raw files ...")
    raw_pairs = _load_synthetic_pairs(SYNTHETIC_RAW_FILES)
    print(f"  Matched {len(raw_pairs):,} synthetic base pairs")

    synthetic_records: list[dict] = []
    for p in raw_pairs:
        base = _make_pair(
            roman_text=p["roman_text"],
            telugu_text=p["telugu_text"],
            source_name=p["source_name"],
            license_tag=p["license_tag"],
            created_at=p["created_at"],
            pair_source="synthetic",
            augmentation_variant=None,
        )
        synthetic_records.append(base)

        for variant in generate_variants(p["roman_text"], max_variants=5):
            synthetic_records.append(_make_pair(
                roman_text=variant,
                telugu_text=p["telugu_text"],
                source_name=p["source_name"],
                license_tag=p["license_tag"],
                created_at=p["created_at"],
                pair_source="augmented",
                augmentation_variant=variant,
            ))

    print(f"  Expanded to {len(synthetic_records):,} records (base + augmented variants)")

    # ── 2. Stream v3 train + append synthetic → v4 train ───────────────────────
    v3_train = v3 / "train.jsonl"
    out_train = out / "train.jsonl"
    print(f"Streaming {v3_train} → {out_train} ...")

    written = 0
    with open(v3_train, encoding="utf-8") as src, open(out_train, "w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
            written += 1
            if written % 500_000 == 0:
                print(f"  ... {written:,} v3 records written")

    print(f"  v3 train: {written:,} records")

    with open(out_train, "a", encoding="utf-8") as dst:
        for rec in synthetic_records:
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_train = written + len(synthetic_records)
    print(f"  synthetic appended: {len(synthetic_records):,}")
    print(f"  total train: {total_train:,}")

    # ── 3. Copy v3 val unchanged ────────────────────────────────────────────────
    v3_val = v3 / "val.jsonl"
    out_val = out / "val.jsonl"
    print(f"Copying {v3_val} → {out_val} ...")

    val_lines = 0
    with open(v3_val, encoding="utf-8") as src, open(out_val, "w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
            val_lines += 1

    print(f"  val: {val_lines:,} records (unchanged)")

    # ── 4. Summary ──────────────────────────────────────────────────────────────
    print()
    print("processed_v4 build complete:")
    print(f"  train: {total_train:,}")
    print(f"  val:   {val_lines:,}")
    print(f"  out:   {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-dir", default="data/processed_v3")
    parser.add_argument("--out-dir", default="data/processed_v4")
    args = parser.parse_args()
    build(args.v3_dir, args.out_dir)


if __name__ == "__main__":
    main()
