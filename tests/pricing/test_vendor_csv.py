from pathlib import Path

from cad_quoter.pricing.vendor_csv import VendorCSV


def test_vendor_csv_skips_header_and_notes(tmp_path: Path) -> None:
    csv_content = """symbol,usd_per_kg\n6061,5.5\nnotes,not a number\n7075,8.9\n"""
    csv_file = tmp_path / "vendor_prices.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    provider = VendorCSV(str(csv_file))
    provider._load()

    assert provider._rows == {"6061": 5.5, "7075": 8.9}
