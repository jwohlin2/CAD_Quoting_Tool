
from appV5 import _aggregate_hole_entries, _dedupe_hole_entries, _parse_hole_line


def test_leader_entries_duplicate_table_rows_are_ignored() -> None:
    table_entry = _parse_hole_line("QTY 8 Ø0.201 THRU", 1.0, source="TABLE")
    assert table_entry is not None
    leader_entry = _parse_hole_line("qty 8 0.201 thru", 1.0, source="LEADER")
    assert leader_entry is not None

    unique_leaders = _dedupe_hole_entries([table_entry], [leader_entry])

    assert unique_leaders == []


def test_leader_entries_with_new_information_are_preserved() -> None:
    table_entry = _parse_hole_line("QTY 4 1/4-20 TAP", 1.0, source="TABLE")
    assert table_entry is not None
    leader_entry = _parse_hole_line("QTY 2 C'BORE .500", 1.0, source="LEADER")
    assert leader_entry is not None

    unique_leaders = _dedupe_hole_entries([table_entry], [leader_entry])

    assert len(unique_leaders) == 1
    agg = _aggregate_hole_entries([table_entry] + unique_leaders)

    assert agg["hole_count"] == 6
    assert agg["cbore_qty"] == 2


def test_from_back_hint_sets_side() -> None:
    entry = _parse_hole_line("TAP THRU (FROM BACK)", 1.0, source="LEADER")
    assert entry is not None
    assert entry["side"] == "BACK"


def test_front_and_back_marks_double_sided() -> None:
    entry = _parse_hole_line("0.250 C'BORE FRONT & BACK", 1.0, source="LEADER")
    assert entry is not None
    assert entry["double_sided"] is True


def test_aggregate_flags_back_side_from_hint() -> None:
    entry = _parse_hole_line("QTY 2 TAP THRU (FROM BACK)", 1.0, source="LEADER")
    assert entry is not None
    agg = _aggregate_hole_entries([entry])
    assert agg["from_back"] is True
