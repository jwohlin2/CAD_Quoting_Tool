"""Provider that sources prices from a user-supplied CSV file."""
from __future__ import annotations

import csv
import os
from typing import Dict

from .base import PriceProvider


class VendorCSV(PriceProvider):
    name = "vendor_csv"
    quote_basis = "usd_per_kg"

    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError("VendorCSV requires a file path")
        self.path = path
        self._rows: Dict[str, float] | None = None

    def _load(self) -> None:
        if self._rows is not None:
            return
        self._rows = {}
        if os.path.isfile(self.path):
            with open(self.path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if not row:
                        continue
                    symbol, usd_per_kg, *_ = row
                    self._rows[symbol.strip().upper()] = float(usd_per_kg)

    def get(self, symbol: str) -> tuple[float, str]:
        self._load()
        assert self._rows is not None
        price = self._rows.get(symbol.upper())
        if price is None:
            raise KeyError(symbol)
        return float(price), os.path.basename(self.path)


__all__ = ["VendorCSV"]
