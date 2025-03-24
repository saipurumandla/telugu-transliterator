import pytest
from data.curate.romanize_rules import romanize, romanize_variants


def test_romanize_basic_word():
    result = romanize("నేను")
    assert result == "nenu"


def test_romanize_long_vowels_simplified():
    # ITRANS produces vastunnAnu — should become vastunnanu
    result = romanize("వస్తున్నాను")
    assert "A" not in result
    assert result.islower()


def test_romanize_lowercase():
    result = romanize("తెలుగు")
    assert result == result.lower()


def test_romanize_sentence():
    result = romanize("ఎలా ఉన్నావ్")
    assert isinstance(result, str)
    assert len(result) > 0
    assert result == result.lower()


def test_romanize_empty():
    result = romanize("")
    assert result == ""


def test_romanize_variants_returns_list():
    variants = romanize_variants("నేను")
    assert isinstance(variants, list)
    assert len(variants) >= 1
    assert "nenu" in variants


def test_romanize_variants_no_duplicates():
    variants = romanize_variants("గుంటూరు")
    assert len(variants) == len(set(variants))


def test_romanize_preserves_spaces():
    result = romanize("నేను వస్తున్నాను")
    assert " " in result


@pytest.mark.parametrize("telugu,expected_contains", [
    ("నేను", "nenu"),
    ("మీరు", "miru"),
    ("తెలుగు", "telugu"),
])
def test_romanize_known_words(telugu, expected_contains):
    result = romanize(telugu)
    assert expected_contains in result
