from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from cad_quoter.pricing.process_buckets import (
    RATE_ALIAS_KEYS,
    canonical_bucket_key,
    normalize_bucket_key,
)
from cad_quoter.pricing.process_rates import machine_rate
from cad_quoter.utils import _dict


__all__ = [
    "DEFAULT_RATE_PER_HOUR",
    "RateBucket",
    "RATE_BUCKETS",
    "bucket_cost_breakdown",
]


DEFAULT_RATE_PER_HOUR: float = machine_rate("milling")


@dataclass(frozen=True)
class RateBucket:
    """Describe how an operation maps to a two-bucket rate structure."""

    key: str
    label: str
    bucket: str
    canonical_key: str | None = None
    extra_rate_keys: tuple[str, ...] = ()

    def normalized_label(self) -> str:
        return normalize_bucket_key(self.label)

    @property
    def rate_keys(self) -> tuple[str, ...]:
        """Return the configured rate aliases for this bucket."""

        aliases: tuple[str, ...] = ()
        if self.canonical_key:
            canon = canonical_bucket_key(self.canonical_key, default=self.canonical_key)
            if canon:
                aliases = RATE_ALIAS_KEYS.get(canon, ())
        if not aliases and self.canonical_key:
            aliases = RATE_ALIAS_KEYS.get(self.canonical_key, ())
        if self.extra_rate_keys:
            aliases = tuple(
                dict.fromkeys((*aliases, *self.extra_rate_keys))
            )
        if aliases:
            return aliases
        normalized = normalize_bucket_key(self.label)
        return (self.label, normalized) if normalized and normalized != self.label else (self.label,)

    @property
    def minute_keys(self) -> tuple[str, ...]:
        """Return normalized minute labels that map planner minutes to this bucket."""

        keys: list[str] = []
        for candidate in (self.key, self.label, self.canonical_key, self.bucket):
            if not candidate:
                continue
            norm = normalize_bucket_key(candidate)
            if norm and norm not in keys:
                keys.append(norm)
        return tuple(keys)


RATE_BUCKETS: tuple[RateBucket, ...] = (
    RateBucket("Inspection", "labor", "inspection"),
    RateBucket("Fixture Build (amortized)", "labor", "fixture_build_amortized"),
    RateBucket("Programming (per part)", "labor", "programming_amortized"),
    RateBucket("Deburr", "labor", "finishing_deburr"),
    RateBucket("Lapping/Honing", "labor", "grinding"),
    RateBucket("Drilling", "machine", "drilling"),
    RateBucket("Milling", "machine", "milling"),
    RateBucket("Wire EDM", "machine", "wire_edm"),
    RateBucket("Grinding", "machine", "grinding"),
    RateBucket("Saw/Waterjet", "machine", "saw_waterjet"),
    RateBucket("Sinker EDM", "machine", "sinker_edm"),
    RateBucket("Abrasive Flow", "machine", "abrasive_flow", extra_rate_keys=("AbrasiveFlowRate",)),
)


def _iter_rate_candidates(rate_keys: Sequence[str]) -> Iterable[str]:
    for key in rate_keys:
        if not key:
            continue
        yield key
        norm = normalize_bucket_key(key)
        if norm and norm != key:
            yield norm
        camel = "".join(part.title() for part in norm.split("_")) if norm else ""
        if camel:
            yield camel
            yield f"{camel}Rate"
        snake = norm.replace("__", "_") if norm else ""
        if snake and snake != norm:
            yield snake


def _bucket_rates(two_bucket_rates: Mapping[str, Mapping[str, float]] | None, bucket: str) -> Mapping[str, float]:
    if not isinstance(two_bucket_rates, Mapping):
        return {}
    for candidate in (bucket, bucket.lower(), bucket.upper(), bucket.capitalize()):
        mapping = _dict(two_bucket_rates.get(candidate))
        if mapping:
            return mapping
    return {}


def _coerce_rate(value: object, default: float) -> float:
    try:
        rate = float(value)  # type: ignore[arg-type]
    except Exception:
        return default
    return rate if rate > 0 else default


def _lookup_rate(two_bucket_rates: Mapping[str, Mapping[str, float]] | None, spec: RateBucket, default: float) -> float:
    bucket_rates = _bucket_rates(two_bucket_rates, spec.bucket)
    for candidate in _iter_rate_candidates(spec.rate_keys):
        if candidate in bucket_rates:
            return _coerce_rate(bucket_rates[candidate], default)
    return default


def bucket_cost_breakdown(
    minutes: Mapping[str, float] | None,
    two_bucket_rates: Mapping[str, Mapping[str, float]] | None,
    *,
    default_rate: float = DEFAULT_RATE_PER_HOUR,
) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    """Convert planner minutes into dollars using shared rate metadata."""

    minute_source = minutes or {}
    line_items: list[dict[str, float | str]] = []

    labor_min = machine_min = labor_cost = machine_cost = total_minutes = 0.0

    minute_lookup: dict[str, float] = {}
    for name, raw in minute_source.items():
        try:
            mins = float(raw or 0.0)
        except Exception:
            continue
        norm = normalize_bucket_key(name)
        if not norm:
            continue
        minute_lookup[norm] = minute_lookup.get(norm, 0.0) + mins

    for spec in RATE_BUCKETS:
        mins = 0.0
        matched_keys: list[str] = []
        for key in spec.minute_keys:
            value = minute_lookup.get(key)
            if value is None or value <= 0:
                continue
            mins += value
            matched_keys.append(key)
        if mins <= 0:
            continue
        for key in matched_keys:
            minute_lookup[key] = 0.0
        rate = _lookup_rate(two_bucket_rates, spec, default_rate)
        cost = (mins / 60.0) * rate
        entry: dict[str, float | str] = {
            "op": spec.label,
            "name": spec.label,
            "minutes": round(mins, 2),
            f"{spec.bucket}_cost": round(cost, 2),
        }
        line_items.append(entry)
        total_minutes += mins
        if spec.bucket == "labor":
            labor_min += mins
            labor_cost += cost
        else:
            machine_min += mins
            machine_cost += cost

    totals = {
        "labor_minutes": round(labor_min, 2),
        "machine_minutes": round(machine_min, 2),
        "minutes": round(total_minutes, 2),
        "labor_cost": round(labor_cost, 2),
        "machine_cost": round(machine_cost, 2),
        "total_cost": round(labor_cost + machine_cost, 2),
    }

    return line_items, totals

