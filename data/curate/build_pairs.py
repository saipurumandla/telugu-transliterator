import json
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

LENGTH_RATIO_MIN = 0.3
LENGTH_RATIO_MAX = 4.0


@dataclass
class TranslitPair:
    pair_id: str
    roman_text: str | None
    telugu_text: str
    source_name: str
    license_tag: str
    pair_source: str       # direct | synthetic | wikipedia_pending
    quality_score: float
    confidence: float
    review_status: str     # approved | review | rejected
    created_at: str
    augmentation_variant: str | None


def _length_ratio_ok(roman: str, telugu: str) -> bool:
    if not telugu:
        return False
    ratio = len(roman) / len(telugu)
    return LENGTH_RATIO_MIN <= ratio <= LENGTH_RATIO_MAX


def _aksharantar_confidence(record: dict) -> float:
    score = record.get("aksharantar_score")
    source = record.get("aksharantar_source", "")
    if source in ("Dakshina", "Existing", "Wikidata"):
        return 0.90
    if source == "Samanantar":
        return 0.85
    if score is not None:
        # IndicCorp entries have log-prob quality scores (negative, closer to 0 = better)
        normalized = max(0.0, min(1.0, 1.0 + score))
        return round(normalized, 4)
    return 0.75


def _build_dakshina_pairs(records: list[dict], created_at: str) -> list[TranslitPair]:
    # Group by line number: source_doc_id = "te_lex_{line}_{variant}"
    by_line: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        parts = r["source_doc_id"].split("_")
        if len(parts) >= 4:
            line_key = parts[2]
            by_line[line_key].append(r)

    pairs = []
    for line_key, recs in by_line.items():
        telugu_recs = [r for r in recs if r["script_hint"] == "telugu"]
        roman_recs = [r for r in recs if r["script_hint"] == "roman"]
        if not telugu_recs or not roman_recs:
            continue

        telugu_text = telugu_recs[0]["text_normalized"]
        for roman_rec in roman_recs:
            roman_text = roman_rec["text_normalized"]
            ok_ratio = _length_ratio_ok(roman_text, telugu_text)
            pairs.append(TranslitPair(
                pair_id=str(uuid.uuid4()),
                roman_text=roman_text,
                telugu_text=telugu_text,
                source_name="dakshina",
                license_tag="CC-BY-SA-4.0",
                pair_source="direct",
                quality_score=0.90,
                confidence=0.90,
                review_status="approved" if ok_ratio else "review",
                created_at=created_at,
                augmentation_variant=None,
            ))
    return pairs


def _build_aksharantar_pairs(records: list[dict], created_at: str) -> list[TranslitPair]:
    by_id = {r["source_doc_id"]: r for r in records}
    seen_pairs: set[str] = set()
    pairs = []

    for rec in records:
        if rec["script_hint"] != "telugu":
            continue

        partner_id = rec.get("paired_with")
        if not partner_id or partner_id not in by_id:
            continue

        # Deduplicate — each te+ro pair appears once
        pair_key = f"{rec['source_doc_id']}|{partner_id}"
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        roman_rec = by_id[partner_id]
        telugu_text = rec["text_normalized"]
        roman_text = roman_rec["text_normalized"]
        ok_ratio = _length_ratio_ok(roman_text, telugu_text)
        confidence = _aksharantar_confidence(rec)

        pairs.append(TranslitPair(
            pair_id=str(uuid.uuid4()),
            roman_text=roman_text,
            telugu_text=telugu_text,
            source_name="aksharantar",
            license_tag="CC-BY-4.0",
            pair_source="direct",
            quality_score=confidence,
            confidence=confidence,
            review_status="approved" if ok_ratio and confidence >= 0.4 else "review",
            created_at=created_at,
            augmentation_variant=None,
        ))
    return pairs


def _build_wikipedia_pairs(records: list[dict], created_at: str) -> list[TranslitPair]:
    pairs = []
    for rec in records:
        if rec["script_hint"] != "telugu":
            continue
        pairs.append(TranslitPair(
            pair_id=str(uuid.uuid4()),
            roman_text=None,
            telugu_text=rec["text_normalized"],
            source_name=rec.get("source_name", "wikipedia_te"),
            license_tag=rec.get("license_tag", "CC-BY-SA-3.0"),
            pair_source="wikipedia_pending",
            quality_score=0.0,
            confidence=0.0,
            review_status="review",
            created_at=created_at,
            augmentation_variant=None,
        ))
    return pairs


def _build_synthetic_pairs(records: list[dict], created_at: str) -> list[TranslitPair]:
    # Synthetic records come in roman+telugu pairs linked by shared UUID in source_doc_id
    # roman: syn_{cat}_{uuid}_ro  telugu: syn_{cat}_{uuid}_te
    by_uuid: dict[str, dict] = {}
    for rec in records:
        doc_id = rec.get("source_doc_id", "")
        if doc_id.endswith("_ro"):
            key = doc_id[:-3]
            by_uuid.setdefault(key, {})["roman"] = rec
        elif doc_id.endswith("_te"):
            key = doc_id[:-3]
            by_uuid.setdefault(key, {})["telugu"] = rec

    pairs = []
    for key, group in by_uuid.items():
        roman_rec = group.get("roman")
        telugu_rec = group.get("telugu")
        if not roman_rec or not telugu_rec:
            continue
        roman_text = roman_rec["text_normalized"]
        telugu_text = telugu_rec["text_normalized"]
        ok_ratio = _length_ratio_ok(roman_text, telugu_text)
        pairs.append(TranslitPair(
            pair_id=str(uuid.uuid4()),
            roman_text=roman_text,
            telugu_text=telugu_text,
            source_name="ollama_synthetic",
            license_tag="internal",
            pair_source="synthetic",
            quality_score=0.80,
            confidence=0.80,
            review_status="approved" if ok_ratio else "review",
            created_at=created_at,
            augmentation_variant=None,
        ))
    return pairs


def build_pairs_from_snapshot(input_path: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(tz=timezone.utc).isoformat()

    records = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"input_file": str(input_path), "pair_count": 0}

    source_name = records[0].get("source_name", "unknown")

    if source_name == "dakshina":
        pairs = _build_dakshina_pairs(records, created_at)
    elif source_name == "aksharantar":
        pairs = _build_aksharantar_pairs(records, created_at)
    elif source_name in ("wikipedia_te", "samanantar"):
        pairs = _build_wikipedia_pairs(records, created_at)
    elif source_name == "ollama_synthetic":
        pairs = _build_synthetic_pairs(records, created_at)
    else:
        print(f"  Unknown source '{source_name}' — skipping.")
        return {"input_file": str(input_path), "pair_count": 0}

    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")

    status_counts: dict[str, int] = {}
    for p in pairs:
        status_counts[p.review_status] = status_counts.get(p.review_status, 0) + 1

    return {
        "stage": "build_pairs",
        "source_name": source_name,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_records": len(records),
        "pair_count": len(pairs),
        "review_status_counts": status_counts,
    }


def main() -> None:
    interim_dir = Path("data/interim")
    inputs = sorted(interim_dir.glob("filtered_*.jsonl"))

    if not inputs:
        print("No filtered snapshots found in data/interim/")
        return

    total_pairs = 0
    for inp in inputs:
        out = interim_dir / inp.name.replace("filtered_", "pairs_")
        print(f"Building pairs from {inp.name} ...")
        result = build_pairs_from_snapshot(inp, out)
        pairs = result.get("pair_count", 0)
        total_pairs += pairs
        print(f"  {result.get('input_records', 0):,} records -> {pairs:,} pairs")
        for status, count in result.get("review_status_counts", {}).items():
            print(f"    {status}: {count:,}")

    print(f"\nTotal pairs built: {total_pairs:,}")


if __name__ == "__main__":
    main()
