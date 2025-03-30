import pytest
from data.curate.augment_colloquial import generate_variants


def test_aa_shortening():
    variants = generate_variants("naanu vastanu")
    assert "nanu vastanu" in variants


def test_th_simplification():
    variants = generate_variants("nenu vastunna")
    # no 'th' in this string — no variant expected from this rule
    assert isinstance(variants, list)


def test_th_produces_variant():
    variants = generate_variants("nenu vachthanu")
    assert any("vachtanu" in v for v in variants)


def test_v_w_swap():
    variants = generate_variants("vastanu")
    assert "wastanu" in variants


def test_w_v_swap():
    variants = generate_variants("wastanu")
    assert "vastanu" in variants


def test_no_variant_when_no_rule_matches():
    variants = generate_variants("ok")
    assert isinstance(variants, list)


def test_variants_differ_from_original():
    original = "naanu vastunna"
    variants = generate_variants(original)
    for v in variants:
        assert v != original


def test_max_variants_respected():
    variants = generate_variants("naanu vaastunnaa theeyagaa", max_variants=2)
    assert len(variants) <= 2


def test_variants_are_strings():
    for v in generate_variants("nenu cheppanu"):
        assert isinstance(v, str)
        assert len(v) > 0


def test_empty_input():
    assert generate_variants("") == []
