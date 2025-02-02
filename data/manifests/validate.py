import json
from dataclasses import dataclass
from pathlib import Path

from jsonschema import validate, ValidationError


APPROVED_LICENSES = {
    "CC0", "CC-BY-4.0", "CC-BY-SA-4.0",
    "Apache-2.0", "MIT", "internal", "CC-BY-NC-4.0",
}

REQUIRED_RECORD_FIELDS = {
    "source_name", "source_doc_id", "source_url", "license_tag",
    "pull_timestamp_utc", "text_raw", "script_hint", "lang_hint", "row_hash",
}

VALID_SCRIPT_HINTS = {"roman", "telugu", "mixed"}

SCHEMA_PATH = Path(__file__).parent / "pull_manifest.schema.json"


@dataclass
class GuardrailResult:
    passed: bool
    reason: str
    action: str


def validate_raw_record(record: dict) -> GuardrailResult:
    missing = REQUIRED_RECORD_FIELDS - set(record.keys())
    if missing:
        return GuardrailResult(
            passed=False,
            reason=f"Missing required fields: {', '.join(sorted(missing))}",
            action="reject",
        )

    if record["license_tag"] not in APPROVED_LICENSES:
        return GuardrailResult(
            passed=False,
            reason=f"license_tag '{record['license_tag']}' is not in approved tier",
            action="reject",
        )

    if not isinstance(record["text_raw"], str) or not record["text_raw"].strip():
        return GuardrailResult(passed=False, reason="text_raw must be a non-empty string", action="reject")

    if record["script_hint"] not in VALID_SCRIPT_HINTS:
        return GuardrailResult(
            passed=False,
            reason=f"script_hint '{record['script_hint']}' must be one of {VALID_SCRIPT_HINTS}",
            action="reject",
        )

    return GuardrailResult(passed=True, reason="", action="")


def validate_pull_manifest(manifest: dict) -> GuardrailResult:
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        validate(instance=manifest, schema=schema)
        return GuardrailResult(passed=True, reason="", action="")
    except (FileNotFoundError, ValidationError) as e:
        return GuardrailResult(passed=False, reason=str(e), action="reject")
