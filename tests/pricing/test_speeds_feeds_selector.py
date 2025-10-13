"""Tests for the speeds/feeds selector helpers."""

from cad_quoter.pricing.speeds_feeds_selector import pick_speeds_row


def _material_group(row: dict[str, object] | None) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("material_group") or "").strip().upper()


def test_pick_speeds_row_stainless_303_prefers_m_group() -> None:
    row = pick_speeds_row("Stainless Steel 303", operation="Drill")
    assert row is not None
    assert _material_group(row).startswith("M")


def test_pick_speeds_row_stainless_303_overrides_non_m_group() -> None:
    row = pick_speeds_row(
        "Stainless Steel 303",
        operation="Drill",
        material_group="N1",
    )
    assert row is not None
    assert _material_group(row).startswith("M")
