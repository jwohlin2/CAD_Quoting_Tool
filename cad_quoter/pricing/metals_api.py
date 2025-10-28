"""Provider for Metals-API / Commodities-API services."""
from __future__ import annotations

import json
import os
import time
import urllib.request

from .base import PriceProvider


class MetalsAPI(PriceProvider):
    name = "metals_api"
    base_url = "https://api.metals-api.com/v1/latest"

    def __init__(self, base_url: str | None = None) -> None:
        if base_url:
            self.base_url = base_url

    def get(self, symbol: str) -> tuple[float, str]:
        api_key = os.getenv("METALS_API_KEY")
        if not api_key:
            raise RuntimeError("METALS_API_KEY not set")
        url = f"{self.base_url}?access_key={api_key}&base=USD&symbols={symbol}"
        with urllib.request.urlopen(url, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not data.get("success", True) and "rates" not in data:
            raise RuntimeError(str(data)[:200])
        rate = float(data["rates"][symbol])
        ts = data.get("timestamp")
        asof = time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts)) if ts else "now"
        return rate, asof


__all__ = ["MetalsAPI"]
