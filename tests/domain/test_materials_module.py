from cad_quoter.domain_models import (
    DEFAULT_MATERIAL_DISPLAY,
    DEFAULT_MATERIAL_KEY,
    MATERIAL_DISPLAY_BY_KEY,
    MATERIAL_DROPDOWN_OPTIONS,
    MATERIAL_KEYWORDS,
    normalize_material_key,
)


def test_normalize_material_key_handles_punctuation() -> None:
    assert normalize_material_key("Stainless-Steel 304") == "stainless steel 304"


def test_material_keywords_include_display_synonyms() -> None:
    keywords = MATERIAL_KEYWORDS[normalize_material_key("Stainless Steel")]
    assert "stainless steel" in keywords
    assert "stainless" in keywords


def test_display_lookup_uses_default_key() -> None:
    assert MATERIAL_DISPLAY_BY_KEY[DEFAULT_MATERIAL_KEY] == DEFAULT_MATERIAL_DISPLAY
    assert DEFAULT_MATERIAL_DISPLAY in MATERIAL_DROPDOWN_OPTIONS
