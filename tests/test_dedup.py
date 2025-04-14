import pytest
from data.curate.dedup import _pair_key, _make_minhash


def _pair(te, ro):
    return {"telugu_text": te, "roman_text": ro, "pair_id": "x"}


def test_pair_key_same_pair():
    a = _pair("నేను", "nenu")
    b = _pair("నేను", "nenu")
    assert _pair_key(a) == _pair_key(b)


def test_pair_key_different_roman():
    a = _pair("నేను", "nenu")
    b = _pair("నేను", "neenu")
    assert _pair_key(a) != _pair_key(b)


def test_pair_key_different_telugu():
    a = _pair("నేను", "nenu")
    b = _pair("మీరు", "nenu")
    assert _pair_key(a) != _pair_key(b)


def test_pair_key_case_insensitive():
    a = _pair("నేను", "Nenu")
    b = _pair("నేను", "nenu")
    assert _pair_key(a) == _pair_key(b)


def test_minhash_similar_strings_high_jaccard():
    m1 = _make_minhash("nenu vastunna ela unnav")
    m2 = _make_minhash("nenu vastunna ela unav")
    jaccard = m1.jaccard(m2)
    assert jaccard > 0.5


def test_minhash_different_strings_low_jaccard():
    m1 = _make_minhash("nenu vastunna")
    m2 = _make_minhash("xyzkqw mnjprst")
    jaccard = m1.jaccard(m2)
    assert jaccard < 0.3


def test_minhash_identical_strings():
    m1 = _make_minhash("nenu cheppanu")
    m2 = _make_minhash("nenu cheppanu")
    jaccard = m1.jaccard(m2)
    assert jaccard > 0.95
