from cad_quoter.geo_dump import _smart_clause_split


def test_smart_clause_split_skips_non_operations() -> None:
    text = "\"A\" (Ø1.78) C'BORE X .38 DEEP FROM BACK; 1.78∅"
    parts = _smart_clause_split(text)
    assert parts == ["\"A\" (Ø1.78) C'BORE X .38 DEEP FROM BACK"]


def test_smart_clause_split_preserves_leading_diameter() -> None:
    text = "Ø.38 C'BORE X .38 DEEP FROM BACK"
    parts = _smart_clause_split(text)
    assert parts == ["Ø.38 C'BORE X .38 DEEP FROM BACK"]
