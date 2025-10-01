import math
import os

import pytest

from cad_quoter.pricing import wieland_scraper


SAMPLE_HTML = """
<html>
  <head>
    <script id="__NEXT_DATA__" type="application/json">
      {"props": {"pageProps": {"data": {"metalInformation": {
        "lastUpdate": "2024-10-05",
        "currencyRates": [
          {"pair": "EUR/USD", "value": "1,0965"},
          {"pair": "EUR/GBP", "value": "0,8564"},
          {"baseCurrency": "GBP", "targetCurrency": "USD", "rate": "1.2819"}
        ],
        "lme": [
          {"symbol": "CU", "value": "9,862.00", "currency": "USD", "unit": "t"},
          {"name": "Aluminum", "value": "2.310,5", "currency": "USD", "unit": "t"}
        ],
        "metalPricesEurope": [
          {"name": "Wieland Kupfer", "value": "8 120,00", "currency": "EUR", "unit": "100 kg"},
          {"name": "MS 58I", "value": "430,20", "currency": "EUR", "unit": "100 kg"}
        ],
        "metalPricesEngland": [
          {"name": "Copper Rod", "value": "8 200,0", "currency": "GBP", "unit": "t", "region": "England"},
          {"name": "Brass", "value": "410", "currency": "GBP", "unit": "100 kg", "region": "United Kingdom"}
        ],
        "usdList": [
          {"name": "Direct USD", "value": "6.50", "currency": "USD", "unit": "kg"}
        ]
      }}}}}
    </script>
  </head>
  <body>
    <main>Metal information snapshot</main>
  </body>
</html>
"""


def _make_doc():
    return wieland_scraper.SoupDocument(SAMPLE_HTML)


def test_parse_fx_from_structured_payload() -> None:
    doc = _make_doc()
    fx = wieland_scraper._parse_fx(doc)
    assert pytest.approx(fx["EURUSD"], rel=1e-6) == 1.0965
    assert pytest.approx(fx["EURGBP"], rel=1e-6) == 0.8564
    # GBPUSD is pulled directly from payload (and falls back to cross if missing)
    assert pytest.approx(fx["GBPUSD"], rel=1e-6) == 1.2819


def test_scrape_wieland_prices_from_structured_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wieland_scraper, "_get_soup", lambda debug=False: _make_doc())
    monkeypatch.setattr(wieland_scraper, "_write_temp_cache", lambda data: None)
    monkeypatch.setattr(wieland_scraper, "_read_temp_cache", lambda: None)
    # clear caches so we exercise the parsing path each time
    wieland_scraper._MEM_CACHE.clear()
    try:
        data = wieland_scraper.scrape_wieland_prices(force=True)

        assert data["fx"]["EURUSD"] == pytest.approx(1.0965)
        assert data["fx"]["GBPUSD"] == pytest.approx(1.2819)
        assert data["asof"] == "2024-10-05"

        assert data["wieland_eur100kg"]["Wieland Kupfer"] == pytest.approx(8120.0)
        assert data["wieland_usd_per_kg"]["Wieland Kupfer"] == pytest.approx(89.0358, rel=1e-5)

        assert data["wieland_usd_per_kg"]["Direct USD"] == pytest.approx(6.5)

        assert data["england_gbp_t"]["Copper Rod"] == pytest.approx(8200.0)
        assert data["england_usd_per_kg"]["Copper Rod"] == pytest.approx(10.51158, rel=1e-5)

        # LME conversions converted to USD/kg
        assert data["lme_usd_per_kg"]["CU"] == pytest.approx(9.862)
        assert data["lme_usd_per_kg"]["AL"] == pytest.approx(2.3105)

        # USD/lb conversion sanity check
        assert data["england_usd_per_lb"]["Copper Rod"] == pytest.approx(
            wieland_scraper._usdkg_to_usdlb(data["england_usd_per_kg"]["Copper Rod"]), rel=1e-6
        )
    finally:
        wieland_scraper._MEM_CACHE.clear()
        cache_path = wieland_scraper._cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)


def test_to_float_handles_localized_formats() -> None:
    cases = {
        "8 120,00": 8120.0,
        "2.310,5": 2310.5,
        "9 862.00": 9862.0,
        "1 234,56": 1234.56,
        "6.50": 6.5,
    }
    for raw, expected in cases.items():
        assert wieland_scraper._to_float(raw) == pytest.approx(expected)


def test_get_live_material_price_steel_prefers_wieland(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_data = {
        "asof": "2024-10-05",
        "wieland_usd_per_kg": {"Direct USD": 6.5},
        "england_usd_per_kg": {},
        "lme_usd_per_kg": {"NI": 25.0},
    }

    monkeypatch.setattr(wieland_scraper, "scrape_wieland_prices", lambda force=False: sample_data)

    price, src = wieland_scraper.get_live_material_price_usd_per_kg("A36", fallback_usd_per_kg=12.34)

    assert math.isfinite(price)
    assert price == pytest.approx(6.5)
    assert "Wieland Direct USD" in src
