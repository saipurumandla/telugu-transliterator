import json
import pytest
from unittest.mock import patch, mock_open

from data.manifests.validate import validate_raw_record, validate_pull_manifest


VALID_SCHEMA = json.dumps({
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["run_id", "source_name", "pull_timestamp_utc", "record_count", "output_file", "license_tag"],
    "properties": {
        "run_id": {"type": "string"},
        "source_name": {"type": "string"},
        "pull_timestamp_utc": {"type": "string"},
        "record_count": {"type": "integer", "minimum": 0},
        "output_file": {"type": "string"},
        "license_tag": {"type": "string", "enum": ["CC0", "CC-BY-4.0", "CC-BY-SA-4.0", "Apache-2.0", "MIT", "internal", "CC-BY-NC-4.0"]},
        "takedown_contact": {"type": "string"},
    },
    "additionalProperties": False,
})


@pytest.fixture
def valid_record():
    return {
        "source_name": "dakshina",
        "source_doc_id": "lex_001",
        "source_url": "https://github.com/google-research-datasets/dakshina",
        "license_tag": "CC-BY-SA-4.0",
        "pull_timestamp_utc": "2025-01-19T10:21:37+00:00",
        "text_raw": "nenu Telugu maatladutaanu",
        "script_hint": "roman",
        "lang_hint": "te",
        "row_hash": "abc123def456",
    }


def test_valid_record_passes(valid_record):
    result = validate_raw_record(valid_record)
    assert result.passed is True
    assert result.action == ""


def test_missing_license_tag_rejected(valid_record):
    del valid_record["license_tag"]
    result = validate_raw_record(valid_record)
    assert result.passed is False
    assert "license_tag" in result.reason
    assert result.action == "reject"


def test_disallowed_license_rejected(valid_record):
    valid_record["license_tag"] = "UNKNOWN-LICENSE"
    result = validate_raw_record(valid_record)
    assert result.passed is False
    assert result.action == "reject"


def test_empty_text_raw_rejected(valid_record):
    valid_record["text_raw"] = ""
    result = validate_raw_record(valid_record)
    assert result.passed is False
    assert result.action == "reject"


def test_invalid_script_hint_rejected(valid_record):
    valid_record["script_hint"] = "japanese"
    result = validate_raw_record(valid_record)
    assert result.passed is False
    assert result.action == "reject"


def test_tier_b_record_with_takedown_passes(valid_record):
    valid_record["license_tag"] = "CC-BY-NC-4.0"
    valid_record["takedown_contact"] = "contact@tdil-dc.in"
    result = validate_raw_record(valid_record)
    assert result.passed is True


def test_valid_manifest_passes():
    manifest = {
        "run_id": "550e8400-e29b-41d4-a716-446655440000",
        "source_name": "dakshina",
        "pull_timestamp_utc": "2025-01-19T10:21:37+00:00",
        "record_count": 10420,
        "output_file": "data/raw/dakshina_20250119.jsonl",
        "license_tag": "CC-BY-SA-4.0",
    }
    with patch("builtins.open", mock_open(read_data=VALID_SCHEMA)):
        result = validate_pull_manifest(manifest)
    assert result.passed is True


def test_manifest_missing_source_name_rejected():
    manifest = {
        "run_id": "550e8400-e29b-41d4-a716-446655440000",
        "pull_timestamp_utc": "2025-01-19T10:21:37+00:00",
        "record_count": 10420,
        "output_file": "data/raw/dakshina_20250119.jsonl",
        "license_tag": "CC-BY-SA-4.0",
    }
    with patch("builtins.open", mock_open(read_data=VALID_SCHEMA)):
        result = validate_pull_manifest(manifest)
    assert result.passed is False
    assert result.action == "reject"
