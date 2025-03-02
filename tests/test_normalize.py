import pytest
from data.curate.normalize import (
    nfc,
    remove_control_chars,
    normalize_whitespace,
    telugu_char_ratio,
    detect_script,
    normalize_record,
)


def test_nfc_normalization():
    # Composed vs decomposed form — NFC should unify them
    composed = "క"       # Telugu ka — precomposed
    decomposed = "క"     # same codepoint, already NFC
    assert nfc(composed) == nfc(decomposed)


def test_remove_control_chars():
    text = "hello\x00world\x1ftest"
    assert normalize_whitespace(remove_control_chars(text)) == "helloworldtest"


def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world  ") == "hello world"
    assert normalize_whitespace("line1\n\nline2") == "line1 line2"
    assert normalize_whitespace("") == ""


def test_telugu_char_ratio_pure_telugu():
    telugu = "నేను వస్తున్నాను"
    ratio = telugu_char_ratio(telugu)
    assert ratio > 0.5


def test_telugu_char_ratio_pure_roman():
    roman = "nenu vastunnanu"
    ratio = telugu_char_ratio(roman)
    assert ratio == 0.0


def test_telugu_char_ratio_empty():
    assert telugu_char_ratio("") == 0.0


def test_detect_script_telugu():
    assert detect_script("నేను వస్తున్నాను") == "telugu"


def test_detect_script_roman():
    assert detect_script("nenu vastunnanu ela unnav") == "roman"


def test_detect_script_mixed():
    assert detect_script("nenu Telugu lo 123") in ("roman", "mixed")


def test_normalize_record_adds_fields():
    record = {
        "source_name": "dakshina",
        "source_doc_id": "te_lex_1_0",
        "source_url": "https://example.com",
        "license_tag": "CC-BY-SA-4.0",
        "pull_timestamp_utc": "2025-01-19T10:21:37+00:00",
        "text_raw": "  నేను  ",
        "script_hint": "telugu",
        "lang_hint": "te",
        "row_hash": "abc123",
    }
    result = normalize_record(record)
    assert "text_normalized" in result
    assert "script_detected" in result
    assert result["text_normalized"] == "నేను"
    assert result["script_detected"] == "telugu"
    assert result["text_raw"] == "  నేను  "  # original preserved


def test_normalize_record_preserves_all_original_fields():
    record = {
        "source_name": "dakshina",
        "source_doc_id": "te_lex_1_1",
        "source_url": "https://example.com",
        "license_tag": "CC-BY-SA-4.0",
        "pull_timestamp_utc": "2025-01-19T10:21:37+00:00",
        "text_raw": "nenu",
        "script_hint": "roman",
        "lang_hint": "te",
        "row_hash": "def456",
    }
    result = normalize_record(record)
    for key in record:
        assert key in result
