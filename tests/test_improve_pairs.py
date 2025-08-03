import pytest
from data.curate.improve_pairs import (
    is_telugu,
    romanize_natural,
    split_into_clauses,
)


def test_is_telugu_pure():
    assert is_telugu('నేను వస్తాను') is True


def test_is_telugu_roman():
    assert is_telugu('nenu vastanu') is False


def test_is_telugu_mixed_mostly_roman():
    assert is_telugu('nenu 123') is False


def test_romanize_natural_simple():
    result = romanize_natural('నేను')
    assert result == 'nenu'


def test_romanize_natural_lowercase():
    result = romanize_natural('వస్తాను')
    assert result == result.lower()


def test_romanize_natural_no_uppercase_vowels():
    result = romanize_natural('వస్తున్నాను')
    assert 'A' not in result
    assert 'I' not in result
    assert 'U' not in result


def test_split_into_clauses_long_sentence():
    text = 'గుంటూరు జిల్లా ఆంధ్రప్రదేశ్ లోని ఒక జిల్లా, దీని ముఖ్యపట్టణం గుంటూరు నగరం'
    clauses = split_into_clauses(text)
    assert len(clauses) >= 1
    for c in clauses:
        assert len(c) <= 120


def test_split_into_clauses_short_discarded():
    text = 'hi, hello'
    clauses = split_into_clauses(text)
    assert all(len(c) >= 15 for c in clauses)


def test_split_into_clauses_returns_list():
    result = split_into_clauses('నేను వస్తాను. మీరు ఎలా ఉన్నారు.')
    assert isinstance(result, list)
