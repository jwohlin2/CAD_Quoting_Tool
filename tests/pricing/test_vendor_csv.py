from pathlib import Path

from cad_quoter.pricing import vendor_csv


def test_vendor_csv_skips_header_and_notes(tmp_path: Path) -> None:
    csv_content = """symbol,usd_per_kg\n6061,5.5\nnotes,not a number\n7075,8.9\n"""
    csv_file = tmp_path / "vendor_prices.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    provider = vendor_csv.VendorCSV(str(csv_file))
    provider._load()

    assert provider._rows == {"6061": 5.5, "7075": 8.9}


def test_pick_plate_from_mcmaster_forwards_to_helper(monkeypatch) -> None:
    rows = [
        {
            "material": "Aluminum MIC6",
            "thickness_in": "1",
            "length_in": "12",
            "width_in": "6",
            "part": "86825K111",
            "price_usd": "42.5",
        }
    ]

    calls: list[tuple[float, float, float, str]] = []

    def fake_pick(L, W, T, *, material_key, catalog_rows):
        calls.append((L, W, T, material_key))
        assert catalog_rows is rows
        return {
            "len_in": 12.0,
            "wid_in": 6.0,
            "thk_in": 1.0,
            "mcmaster_part": "86825K111",
        }

    monkeypatch.setattr(vendor_csv, "load_mcmaster_catalog_rows", lambda _path=None: rows)
    monkeypatch.setattr(vendor_csv, "pick_mcmaster_plate_sku", fake_pick)

    choice = vendor_csv.pick_plate_from_mcmaster(
        "Aluminum MIC6",
        10.0,
        5.0,
        1.0,
        scrap_fraction=0.0,
        allow_thickness_upsize=False,
    )

    assert calls, "helper should be invoked"
    assert choice == {
        "vendor": "McMaster",
        "part_no": "86825K111",
        "len_in": 12.0,
        "wid_in": 6.0,
        "thk_in": 1.0,
        "thickness_diff_in": 0.0,
        "price_usd": 42.5,
        "min_charge_usd": None,
    }
