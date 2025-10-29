from cad_quoter.geo_dump import _parse_clause_to_ops, _smart_clause_split


def test_smart_clause_split_skips_non_operations() -> None:
    text = "\"A\" (Ø1.78) C'BORE X .38 DEEP FROM BACK; 1.78∅"
    parts = _smart_clause_split(text)
    assert parts == ["\"A\" (Ø1.78) C'BORE X .38 DEEP FROM BACK"]


def test_smart_clause_split_preserves_leading_diameter() -> None:
    text = "Ø.38 C'BORE X .38 DEEP FROM BACK"
    parts = _smart_clause_split(text)
    assert parts == ["Ø.38 C'BORE X .38 DEEP FROM BACK"]


def test_parse_clause_keeps_tap_on_base_diameter() -> None:
    ops = _parse_clause_to_ops(
        hole_idx=0,
        base_diam="Ø0.2010",
        qtys=[2],
        text="Ø0.250 TAP 1/4-20",
        hole_letters=["A"],
        diam_list=["Ø0.2010"],
    )

    assert ops == [(0, "Ø0.2010", 2, "TAP 1/4-20")]
