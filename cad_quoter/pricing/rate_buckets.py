from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from cad_quoter.pricing.process_buckets import RATE_BUCKET_META, normalize_bucket_key
from cad_quoter.utils import _dict


__all__ = [
    "DEFAULT_RATE_PER_HOUR",
    "RateBucket",
    "RATE_BUCKETS",
    "bucket_cost_breakdown",
]


DEFAULT_RATE_PER_HOUR: float = 90.0


@dataclass(frozen=True)
class RateBucket:
    """Describe how an operation maps to a two-bucket rate structure."""

    key: str
    label: str
    bucket: str
    rate_keys: tuple[str, ...]
    minute_keys: tuple[str, ...]

    def normalized_label(self) -> str:
        return normalize_bucket_key(self.label)


def _build_rate_buckets() -> tuple[RateBucket, ...]:
    buckets: list[RateBucket] = []
    for meta in RATE_BUCKET_META:
        buckets.append(
            RateBucket(
                key=meta.key,
                label=meta.label,
                bucket=meta.bucket,
                rate_keys=meta.rate_aliases,
                minute_keys=meta.minute_keys,
            )
        )
    return tuple(buckets)


RATE_BUCKETS: tuple[RateBucket, ...] = _build_rate_buckets()


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

