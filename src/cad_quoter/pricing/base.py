"""Common types for pricing providers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple


class PriceProvider:
    """Abstract base for spot price providers."""

    name: str = "base"
    #: Optional basis for the returned quote. When ``None`` the caller is
    #: expected to supply the desired basis for conversion.
    quote_basis: str | None = None

    def get(self, symbol: str) -> Tuple[float, str]:
        """Return a numeric quote and an ``as-of`` description."""

        raise NotImplementedError


class ProviderFactory(Protocol):
    """Factory protocol for constructing :class:`PriceProvider` instances."""

    def __call__(self, **config: object) -> PriceProvider:
        ...


@dataclass(frozen=True)
class PriceQuote:
    """Normalised price information in USD/kg."""

    usd_per_kg: float
    source: str
    basis: str


__all__ = ["PriceProvider", "ProviderFactory", "PriceQuote"]
