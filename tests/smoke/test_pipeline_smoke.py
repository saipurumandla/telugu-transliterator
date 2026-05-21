"""
Smoke tests — verify the pipeline is deterministically reproducible.
Runs key stages on a tiny fixture and checks output hashes are stable.
Not a unit test suite; run after dependency updates or environment changes.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from data.curate.normalize import normalize_snapshot
from data.curate.filter_junk import filter_snapshot
from data.curate.quality_score import score_snapshot


FIXTURE_RECORDS = [
    {
        "source_name": "dakshina",
        "source_doc_id": "te_lex_001_0",
        "source_url": "https://github.com/google-research-datasets/dakshina",
        "license_tag": "CC-BY-SA-4.0",
        "pull_timestamp_utc": "2026-05-15T10:00:00+00:00",
        "text_raw": "nenu Telugu maatladutaanu",
        "text_normalized": "nenu Telugu maatladutaanu",
        "script_hint": "roman",
        "lang_hint": "te",
        "row_hash": "abc123",
    },
    {
        "source_name": "dakshina",
        "source_doc_id": "te_lex_001_1",
        "source_url": "https://github.com/google-research-datasets/dakshina",
        "license_tag": "CC-BY-SA-4.0",
        "pull_timestamp_utc": "2026-05-15T10:00:00+00:00",
        "text_raw": "నేను తెలుగు మాట్లాడుతాను",
        "text_normalized": "నేను తెలుగు మాట్లాడుతాను",
        "script_hint": "telugu",
        "lang_hint": "te",
        "row_hash": "def456",
    },
]


def _write_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture.jsonl"
    with fixture.open("w", encoding="utf-8") as f:
        for r in FIXTURE_RECORDS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return fixture


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_normalize_is_deterministic(tmp_path):
    fixture = _write_fixture(tmp_path)
    out1 = tmp_path / "norm1.jsonl"
    out2 = tmp_path / "norm2.jsonl"
    normalize_snapshot(fixture, out1)
    normalize_snapshot(fixture, out2)
    assert _sha256(out1) == _sha256(out2), "normalize_snapshot is not deterministic"


def test_normalize_output_count(tmp_path):
    fixture = _write_fixture(tmp_path)
    out = tmp_path / "norm.jsonl"
    result = normalize_snapshot(fixture, out)
    assert result["output_count"] == len(FIXTURE_RECORDS)


def test_filter_passes_valid_records(tmp_path):
    fixture = _write_fixture(tmp_path)
    norm = tmp_path / "norm.jsonl"
    normalize_snapshot(fixture, norm)
    filtered = tmp_path / "filtered.jsonl"
    result = filter_snapshot(norm, filtered)
    assert result["output_count"] > 0, "All records filtered out — check filter thresholds"


def test_score_snapshot_produces_approved(tmp_path):
    fixture = _write_fixture(tmp_path)
    norm = tmp_path / "norm.jsonl"
    normalize_snapshot(fixture, norm)
    filt = tmp_path / "filt.jsonl"
    filter_snapshot(norm, filt)

    from data.curate.build_pairs import build_pairs_from_snapshot
    from data.curate.romanize_rules import fill_pending_pairs

    pairs = tmp_path / "pairs.jsonl"
    build_pairs_from_snapshot(filt, pairs)
    roman = tmp_path / "roman.jsonl"
    fill_pending_pairs(pairs, roman)

    scored = tmp_path / "scored.jsonl"
    review = tmp_path / "review.jsonl"
    result = score_snapshot(roman, scored, review)
    assert result["approved"] + result["review_bucket"] + result["rejected"] > 0


def test_processed_v3_splits_exist():
    for split in ("train", "val", "test"):
        p = Path(f"data/processed_v3/{split}.jsonl")
        assert p.exists(), f"Missing {p} — run rebuild_v3.py first"


def test_checkpoint_exists():
    ckpt = Path("train/checkpoints_v3/checkpoint-80000")
    assert ckpt.exists(), "v3 checkpoint not found — run training first"
    assert (ckpt / "model.safetensors").exists(), "model.safetensors missing from checkpoint"
