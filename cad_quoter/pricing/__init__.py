"""Pricing provider registry and utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .base import PriceProvider, ProviderFactory, PriceQuote
from .cache import PriceCache, PriceCacheEntry
from .materials import (
    BACKUP_CSV_NAME,
    LB_PER_KG,
    ensure_material_backup_csv,
    get_mcmaster_unit_price,
    load_backup_prices_csv,
    price_value_to_per_gram,
    resolve_material_unit_price,
    usdkg_to_usdlb,
)
from .metals_api import MetalsAPI
from .vendor_csv import VendorCSV
from .speeds_feeds_selector import (
    load_csv_as_records,
    pick_speeds_row,
    unit_hp_cap,
)


def _usd_per_kg_from_quote(quote: float, basis: str) -> float:
    if basis == "index_usd_per_tonne":
        return float(quote) / 1000.0
    if basis == "usd_per_troy_oz":
        return float(quote) / 31.1034768 * 1000.0
    if basis == "usd_per_lb":
        return float(quote) / 2.2046226218
    return float(quote)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    config: Dict[str, Any]


def _normalise_spec(spec: Any) -> ProviderSpec:
    if isinstance(spec, ProviderSpec):
        return spec
    if isinstance(spec, str):
        return ProviderSpec(spec, {})
    if isinstance(spec, dict):
        spec = dict(spec)
        name = spec.pop("type", spec.pop("name", None))
        if not name:
            raise ValueError("Provider specification requires a 'type' or 'name'")
        return ProviderSpec(str(name), spec)
    raise TypeError(f"Unsupported provider specification: {spec!r}")


class ProviderRegistry:
    """Registry of named :class:`PriceProvider` factories."""

    def __init__(self) -> None:
        self._factories: Dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        self._factories[name] = factory

    def create(self, name: str, **config: Any) -> PriceProvider:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"Unknown price provider '{name}'") from exc
        provider = factory(**config)
        if not isinstance(provider, PriceProvider):
            raise TypeError(f"Factory for '{name}' returned {type(provider)!r}")
        return provider

    def available(self) -> Sequence[str]:
        return tuple(sorted(self._factories))


class PricingEngine:
    """Resolve prices using a registry of providers with caching."""

    def __init__(self,
                 registry: ProviderRegistry | None = None,
                 cache: PriceCache | None = None,
                 default_providers: Sequence[Any] | None = None) -> None:
        self.registry = registry or create_default_registry()
        self.cache = cache or PriceCache()
        specs = default_providers or ("metals_api",)
        self.default_specs: Tuple[ProviderSpec, ...] = tuple(_normalise_spec(spec) for spec in specs)
        self._provider_cache: Dict[Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], ...], List[PriceProvider]] = {}

    def clear_cache(self) -> None:
        self.cache.clear()

    def _spec_key(self, specs: Sequence[ProviderSpec]) -> Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], ...]:
        key: List[Tuple[str, Tuple[Tuple[str, Any], ...]]] = []
        for spec in specs:
            items = tuple(sorted((str(k), v) for k, v in spec.config.items()))
            key.append((spec.name, items))
        return tuple(key)

    def _providers_for(self, specs: Sequence[ProviderSpec]) -> List[PriceProvider]:
        key = self._spec_key(specs)
        cached = self._provider_cache.get(key)
        if cached is not None:
            return cached
        providers: List[PriceProvider] = []
        for spec in specs:
            config = dict(spec.config)
            provider = self.registry.create(spec.name, **config)
            providers.append(provider)
        self._provider_cache[key] = providers
        return providers

    def _prepare_specs(self,
                       vendor_csv: str | None,
                       overrides: Sequence[Any] | None) -> List[ProviderSpec]:
        specs: List[ProviderSpec] = []
        seen_vendor = False
        if overrides is not None:
            for raw_spec in overrides:
                spec = _normalise_spec(raw_spec)
                config = dict(spec.config)
                if spec.name == "vendor_csv":
                    seen_vendor = True
                    if vendor_csv:
                        config.setdefault("path", vendor_csv)
                    else:
                        continue
                specs.append(ProviderSpec(spec.name, config))
        else:
            if vendor_csv:
                specs.append(ProviderSpec("vendor_csv", {"path": vendor_csv}))
            specs.extend(self.default_specs)
            return specs

        if vendor_csv and not seen_vendor:
            specs.insert(0, ProviderSpec("vendor_csv", {"path": vendor_csv}))
        return specs

    def get_usd_per_kg(self,
                       symbol: str,
                       basis: str,
                       *,
                       vendor_csv: str | None = None,
                       providers: Sequence[Any] | None = None) -> PriceQuote:
        cache_entry = self.cache.get(symbol)
        if cache_entry:
            return PriceQuote(cache_entry.usd_per_kg, cache_entry.source, cache_entry.basis)

        specs = self._prepare_specs(vendor_csv, providers)
        last_err: Exception | None = None
        for provider in self._providers_for(specs):
            try:
                quote, asof = provider.get(symbol)
                provider_basis = provider.quote_basis or basis
                usd_per_kg = _usd_per_kg_from_quote(quote, provider_basis)
                source = provider.name
                if asof:
                    source = f"{source}@{asof}"
                self.cache.set(symbol, usd_per_kg, source, provider_basis)
                return PriceQuote(usd_per_kg, source, provider_basis)
            except Exception as err:
                last_err = err
                continue

        raise RuntimeError(f"All price providers failed for {symbol}: {last_err}")


def create_default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("metals_api", lambda **cfg: MetalsAPI(**cfg))
    registry.register("vendor_csv", lambda **cfg: VendorCSV(**cfg))
    return registry


__all__ = [
    "BACKUP_CSV_NAME",
    "LB_PER_KG",
    "get_mcmaster_unit_price",
    "PriceProvider",
    "ProviderFactory",
    "PriceQuote",
    "PriceCache",
    "PriceCacheEntry",
    "ProviderRegistry",
    "PricingEngine",
    "ensure_material_backup_csv",
    "load_backup_prices_csv",
    "price_value_to_per_gram",
    "resolve_material_unit_price",
    "usdkg_to_usdlb",
    "create_default_registry",
    "load_csv_as_records",
    "pick_speeds_row",
    "unit_hp_cap",
]
