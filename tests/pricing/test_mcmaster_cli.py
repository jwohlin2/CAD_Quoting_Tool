import pytest

from cad_quoter.vendors import mcmaster_stock


class _FakeAPI:
    def __init__(self, username: str, password: str, pfx_path: str, pfx_password: str):
        self.username = username
        self.password = password
        self.pfx_path = pfx_path
        self.pfx_password = pfx_password
        self.logged_in = False

    def login(self) -> None:
        self.logged_in = True

    def get_price_tiers(self, part: str):
        assert self.logged_in
        assert part == "86825K956"
        return [
            {"MinimumQuantity": 1, "Amount": 456.78, "UnitOfMeasure": "Each"},
            {"MinimumQuantity": 5, "Amount": 430.00, "UnitOfMeasure": "Each"},
        ]


def test_main_part_lookup(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(mcmaster_stock, "_get_env_or_prompt", lambda: ("user", "pass", "cert.pfx", ""))
    monkeypatch.setattr(mcmaster_stock, "McMasterAPI", _FakeAPI)

    mcmaster_stock.main(["--part", "86825K956"])

    captured = capsys.readouterr()
    assert "Fetched 2 price tier(s) for part 86825K956." in captured.out
    assert "Price @ qty=1: $456.78 Each" in captured.out
