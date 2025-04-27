import pytest
from data.curate.split_dataset import assign_split, _split_key, _check_leakage


def _pair(te, ro, status="approved"):
    return {"telugu_text": te, "roman_text": ro, "review_status": status, "pair_id": "x"}


def test_assign_split_deterministic():
    key = "abcd1234"
    assert assign_split(key) == assign_split(key)


def test_all_splits_covered():
    results = set()
    for i in range(0, 0x10000, 256):
        key = f"{i:04x}{'0'*28}"
        results.add(assign_split(key))
    assert results == {"train", "val", "test"}


def test_split_key_same_telugu_same_key():
    a = _pair("నేను", "nenu")
    b = _pair("నేను", "neenu")
    assert _split_key(a) == _split_key(b)


def test_split_key_different_telugu_different_key():
    a = _pair("నేను", "nenu")
    b = _pair("మీరు", "meeru")
    assert _split_key(a) != _split_key(b)


def test_no_leakage_in_disjoint_splits():
    splits = {
        "train": [_pair("నేను", "nenu"), _pair("మీరు", "meeru")],
        "val": [_pair("తెలుగు", "telugu")],
        "test": [_pair("వస్తాను", "vastanu")],
    }
    violations = _check_leakage(splits)
    assert violations == []


def test_leakage_detected_when_same_telugu_in_multiple_splits():
    splits = {
        "train": [_pair("నేను", "nenu")],
        "val": [_pair("నేను", "neenu")],  # same Telugu, different roman
        "test": [],
    }
    violations = _check_leakage(splits)
    assert len(violations) > 0
    assert "train/val" in violations[0]


def test_train_gets_majority():
    # Most keys should land in train (90% ratio)
    train_count = sum(
        1 for i in range(1000)
        if assign_split(f"{i:04x}{'0'*28}") == "train"
    )
    assert train_count > 850
