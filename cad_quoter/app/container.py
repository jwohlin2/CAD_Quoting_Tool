"""Simple dependency-injection helpers for the desktop application."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from cad_quoter.config import load_default_params, load_default_rates
from cad_quoter.pricing import PriceQuote, PricingEngine, create_default_registry


class SupportsPricingEngine(Protocol):
    """Protocol describing the subset of :class:`PricingEngine` used by the UI."""

    def clear_cache(self) -> None:  # pragma: no cover - structural typing helper
        ...

    def get_usd_per_kg(
        self,
        symbol: str,
        basis: str,
        *,
        vendor_csv: str | None = None,
        providers: Sequence[Any] | None = None,
    ) -> PriceQuote:  # pragma: no cover - structural typing helper
        ...


@dataclass(slots=True)
class ServiceContainer:
    """Bundle callables used to construct core application services."""

    load_params: Callable[[], dict[str, Any]]
    load_rates: Callable[[], dict[str, dict[str, float]]]
    pricing_engine_factory: Callable[[], SupportsPricingEngine]
    _pricing_engine_cache: SupportsPricingEngine | None = field(default=None, init=False, repr=False)

    def create_pricing_engine(self) -> SupportsPricingEngine:
        """Return a fresh pricing engine instance."""

        return self.pricing_engine_factory()

    def get_pricing_engine(self) -> SupportsPricingEngine:
        """Return a cached pricing engine instance, creating one if required."""

        if self._pricing_engine_cache is None:
            self._pricing_engine_cache = self.pricing_engine_factory()
        return self._pricing_engine_cache


def create_default_container() -> ServiceContainer:
    """Create a :class:`ServiceContainer` wired to the production defaults."""

    return ServiceContainer(
        load_params=load_default_params,
        load_rates=load_default_rates,
        pricing_engine_factory=lambda: PricingEngine(create_default_registry()),
    )


__all__ = ["ServiceContainer", "create_default_container"]
