"""Simple in-memory cache for material pricing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import time


@dataclass
class PriceCacheEntry:
    timestamp: float
    usd_per_kg: float
    source: str
    basis: str


class PriceCache:
    """LRU-less TTL cache for normalised material prices."""

    def __init__(self, ttl_s: float = 60 * 30) -> None:
        self.ttl_s = float(ttl_s)
        self._entries: Dict[str, PriceCacheEntry] = {}

    def get(self, symbol: str) -> PriceCacheEntry | None:
        entry = self._entries.get(symbol.upper())
        if not entry:
            return None
        if (time.time() - entry.timestamp) >= self.ttl_s:
            self._entries.pop(symbol.upper(), None)
            return None
        return entry

    def set(self, symbol: str, usd_per_kg: float, source: str, basis: str) -> None:
        key = symbol.upper()
        self._entries[key] = PriceCacheEntry(time.time(), float(usd_per_kg), str(source), str(basis))

    def clear(self) -> None:
        self._entries.clear()


__all__ = ["PriceCache", "PriceCacheEntry"]
