import pytest
from data.curate.quality_score import rule_score, score_and_route


def _pair(te, ro, source="direct", confidence=0.85, pair_source="direct"):
    return {
        "telugu_text": te,
        "roman_text": ro,
        "source_name": source,
        "confidence": confidence,
        "pair_source": pair_source,
        "pair_id": "test-001",
    }


def test_clean_pair_scores_high():
    pair = _pair("నేను", "nenu")
    score, reasons = rule_score(pair)
    assert score >= 0.7
    assert reasons == []


def test_empty_roman_scores_zero():
    pair = _pair("నేను", "")
    score, reasons = rule_score(pair)
    assert score == 0.0
    assert "roman_too_short" in reasons


def test_no_telugu_script_scores_zero():
    pair = _pair("hello", "hello")
    score, reasons = rule_score(pair)
    assert score == 0.0
    assert "telugu_no_script" in reasons


def test_bad_length_ratio_penalised():
    pair = _pair("న", "this is a very long romanization that goes on and on")
    score, reasons = rule_score(pair)
    assert any("length_ratio" in r for r in reasons)
    assert score < 0.9  # penalised but not zeroed


def test_augmented_pair_slightly_lower():
    direct = _pair("నేను", "nenu", pair_source="direct", confidence=0.85)
    augmented = _pair("నేను", "neenu", pair_source="augmented", confidence=0.85)
    s_direct, _ = rule_score(direct)
    s_aug, _ = rule_score(augmented)
    assert s_direct > s_aug


def test_score_and_route_approved():
    pair = _pair("నేను", "nenu", confidence=0.9)
    result = score_and_route(pair)
    assert result["review_status"] == "approved"
    assert "quality_score" in result
    assert "score_reasons" in result


def test_score_and_route_review_bucket():
    # Zero confidence + bad length ratio → below threshold
    pair = _pair("న", "this is very long romanization indeed", confidence=0.0)
    result = score_and_route(pair)
    assert result["review_status"] == "review"


def test_score_preserves_all_original_fields():
    pair = _pair("నేను", "nenu")
    result = score_and_route(pair)
    for key in pair:
        assert key in result


@pytest.mark.parametrize("roman,telugu,expect_nonzero", [
    ("nenu", "నేను", True),
    ("", "నేను", False),      # empty roman → score 0
    ("nenu", "", False),       # empty telugu → score 0
    ("nenu", "hello", False),  # no Telugu script → score 0
])
def test_score_parametrized(roman, telugu, expect_nonzero):
    pair = _pair(telugu, roman)
    score, _ = rule_score(pair)
    if expect_nonzero:
        assert score > 0
    else:
        assert score == 0.0
