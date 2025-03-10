import pytest
from data.curate.filter_junk import (
    is_url_only,
    is_emoji_only,
    is_too_short,
    is_too_long,
    is_high_digit_ratio,
    is_high_repeat,
    check_record,
)


def _record(text):
    return {"text_normalized": text, "script_hint": "roman"}


def test_url_only_rejected():
    assert is_url_only("https://example.com/some/path") is True
    assert is_url_only("www.google.com") is True


def test_url_embedded_in_text_allowed():
    assert is_url_only("check https://example.com for more") is False


def test_emoji_only_rejected():
    assert is_emoji_only("😀🎉🔥") is True


def test_emoji_with_text_allowed():
    assert is_emoji_only("hello 😀") is False


def test_too_short_rejected():
    assert is_too_short("a") is True
    assert is_too_short("") is True


def test_min_length_allowed():
    assert is_too_short("ok") is False


def test_too_long_rejected():
    assert is_too_long("a" * 513) is True


def test_max_length_allowed():
    assert is_too_long("a" * 512) is False


def test_high_digit_ratio_rejected():
    assert is_high_digit_ratio("1234567890") is True
    assert is_high_digit_ratio("abc 12345678") is True


def test_low_digit_ratio_allowed():
    assert is_high_digit_ratio("nenu 1 vastanu") is False


def test_high_repeat_rejected():
    assert is_high_repeat("aaaaaaaaaa") is True
    assert is_high_repeat("          ") is True


def test_low_repeat_allowed():
    assert is_high_repeat("nenu vastanu") is False


def test_check_record_clean_roman():
    assert check_record(_record("nenu vastanu ela unnav")) is None


def test_check_record_clean_telugu():
    record = {"text_normalized": "నేను వస్తున్నాను", "script_hint": "telugu"}
    assert check_record(record) is None


def test_check_record_single_char():
    assert check_record(_record("a")) == "too_short"


def test_check_record_url():
    assert check_record(_record("https://example.com")) == "url_only"


def test_check_record_pure_number():
    assert check_record(_record("1")) == "too_short"


def test_check_record_numeric_string():
    assert check_record(_record("12345678901")) == "high_digit_ratio"


@pytest.mark.parametrize("text,expected", [
    ("nenu", None),
    ("a", "too_short"),
    ("https://x.com", "url_only"),
    ("😂😂😂", "emoji_only"),
    ("1234567890", "high_digit_ratio"),
    ("aaaaaaaaaa", "high_repeat"),
])
def test_check_record_parametrized(text, expected):
    assert check_record(_record(text)) == expected
