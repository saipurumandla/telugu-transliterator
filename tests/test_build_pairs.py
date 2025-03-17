import pytest
from data.curate.build_pairs import (
    TranslitPair,
    _length_ratio_ok,
    _aksharantar_confidence,
    _build_dakshina_pairs,
    _build_aksharantar_pairs,
    _build_wikipedia_pairs,
)

TS = "2025-03-16T21:49:34+00:00"


def _dakshina_rec(line, variant, script, text):
    return {
        "source_doc_id": f"te_lex_{line}_{variant}",
        "script_hint": script,
        "text_normalized": text,
        "source_name": "dakshina",
        "license_tag": "CC-BY-SA-4.0",
    }


def _aksharantar_rec(doc_id, script, text, paired_with, ak_source="Dakshina", score=None):
    return {
        "source_doc_id": doc_id,
        "script_hint": script,
        "text_normalized": text,
        "source_name": "aksharantar",
        "license_tag": "CC-BY-4.0",
        "paired_with": paired_with,
        "aksharantar_source": ak_source,
        "aksharantar_score": score,
    }


def _wiki_rec(text):
    return {
        "source_doc_id": "wiki_123_s0",
        "script_hint": "telugu",
        "text_normalized": text,
        "source_name": "wikipedia_te",
        "license_tag": "CC-BY-SA-3.0",
    }


# --- length ratio ---

def test_length_ratio_ok_valid():
    assert _length_ratio_ok("nenu", "నేను") is True


def test_length_ratio_ok_too_short():
    assert _length_ratio_ok("a", "నేను వస్తున్నాను") is False


def test_length_ratio_ok_too_long():
    assert _length_ratio_ok("a" * 100, "న") is False


# --- aksharantar confidence ---

def test_confidence_dakshina_source():
    rec = {"aksharantar_source": "Dakshina", "aksharantar_score": None}
    assert _aksharantar_confidence(rec) == 0.90


def test_confidence_indic_corp_good_score():
    rec = {"aksharantar_source": "IndicCorp", "aksharantar_score": -0.05}
    conf = _aksharantar_confidence(rec)
    assert 0.9 <= conf <= 1.0


def test_confidence_indic_corp_bad_score():
    rec = {"aksharantar_source": "IndicCorp", "aksharantar_score": -0.9}
    conf = _aksharantar_confidence(rec)
    assert conf < 0.5


# --- dakshina pairs ---

def test_dakshina_single_pair():
    records = [
        _dakshina_rec(1, 0, "telugu", "నేను"),
        _dakshina_rec(1, 1, "roman", "nenu"),
    ]
    pairs = _build_dakshina_pairs(records, TS)
    assert len(pairs) == 1
    assert pairs[0].telugu_text == "నేను"
    assert pairs[0].roman_text == "nenu"
    assert pairs[0].pair_source == "direct"
    assert pairs[0].source_name == "dakshina"


def test_dakshina_multiple_roman_variants():
    records = [
        _dakshina_rec(2, 0, "telugu", "అంకిత"),
        _dakshina_rec(2, 1, "roman", "ankita"),
        _dakshina_rec(2, 2, "roman", "ankitha"),
        _dakshina_rec(2, 3, "roman", "amkita"),
    ]
    pairs = _build_dakshina_pairs(records, TS)
    assert len(pairs) == 3
    roman_texts = {p.roman_text for p in pairs}
    assert "ankita" in roman_texts
    assert "ankitha" in roman_texts


def test_dakshina_pair_has_all_required_fields():
    records = [
        _dakshina_rec(3, 0, "telugu", "వస్తాను"),
        _dakshina_rec(3, 1, "roman", "vastanu"),
    ]
    pair = _build_dakshina_pairs(records, TS)[0]
    assert pair.pair_id is not None
    assert pair.license_tag == "CC-BY-SA-4.0"
    assert pair.created_at == TS
    assert pair.augmentation_variant is None


# --- aksharantar pairs ---

def test_aksharantar_builds_pair_via_paired_with():
    records = [
        _aksharantar_rec("tel1_te", "telugu", "పరిష్కరించబడలేదని", "tel1_ro"),
        _aksharantar_rec("tel1_ro", "roman", "parishkarinchabadaledani", "tel1_te"),
    ]
    pairs = _build_aksharantar_pairs(records, TS)
    assert len(pairs) == 1
    assert pairs[0].telugu_text == "పరిష్కరించబడలేదని"
    assert pairs[0].roman_text == "parishkarinchabadaledani"


def test_aksharantar_no_duplicate_pairs():
    records = [
        _aksharantar_rec("tel2_te", "telugu", "విధుల్లోకి", "tel2_ro"),
        _aksharantar_rec("tel2_ro", "roman", "vidhulloki", "tel2_te"),
    ]
    pairs = _build_aksharantar_pairs(records, TS)
    assert len(pairs) == 1


def test_aksharantar_missing_partner_skipped():
    records = [
        _aksharantar_rec("tel3_te", "telugu", "మియాందాద్", "tel3_ro"),
        # partner tel3_ro is missing
    ]
    pairs = _build_aksharantar_pairs(records, TS)
    assert len(pairs) == 0


# --- wikipedia pairs ---

def test_wikipedia_produces_pending_pair():
    records = [_wiki_rec("గుంటూరు జిల్లా ఆంధ్రప్రదేశ్ లోని ఒక జిల్లా.")]
    pairs = _build_wikipedia_pairs(records, TS)
    assert len(pairs) == 1
    assert pairs[0].roman_text is None
    assert pairs[0].pair_source == "wikipedia_pending"
    assert pairs[0].review_status == "review"


def test_wikipedia_skips_roman_records():
    records = [
        _wiki_rec("తెలుగు వాక్యం"),
        {"source_doc_id": "wiki_1_s1", "script_hint": "roman", "text_normalized": "telugu vakyam",
         "source_name": "wikipedia_te", "license_tag": "CC-BY-SA-3.0"},
    ]
    pairs = _build_wikipedia_pairs(records, TS)
    assert len(pairs) == 1
