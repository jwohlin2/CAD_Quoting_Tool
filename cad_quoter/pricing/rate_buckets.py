from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

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

    label: str
    bucket: str
    rate_keys: tuple[str, ...]

    def normalized_label(self) -> str:
        return _normalize_key(self.label)


RATE_BUCKETS: tuple[RateBucket, ...] = (
    RateBucket("Inspection", "labor", ("InspectionRate",)),
    RateBucket("Fixture Build (amortized)", "labor", ("FixtureBuildRate",)),
    RateBucket("Programming (per part)", "labor", ("ProgrammingRate",)),
    RateBucket("Deburr", "labor", ("DeburrRate", "FinishingRate")),
    RateBucket("Lapping/Honing", "labor", ("SurfaceGrindRate", "GrindingRate")),
    RateBucket(
        "Drilling",
        "machine",
        ("DrillingRate", "CncVertical", "CncVerticalRate", "cnc_vertical"),
    ),
    RateBucket("Milling", "machine", ("MillingRate",)),
    RateBucket("Wire EDM", "machine", ("WireEDMRate", "EDMRate")),
    RateBucket(
        "Grinding",
        "machine",
        ("SurfaceGrindRate", "GrindingRate", "ODIDGrindRate", "JigGrindRate"),
    ),
    RateBucket("Saw/Waterjet", "machine", ("SawWaterjetRate", "SawRate", "WaterjetRate")),
    RateBucket("Sinker EDM", "machine", ("SinkerEDMRate", "EDMRate")),
    RateBucket("Abrasive Flow", "machine", ("AbrasiveFlowRate",)),
)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _iter_rate_candidates(rate_keys: Sequence[str]) -> Iterable[str]:
    for key in rate_keys:
        if not key:
            continue
        yield key
        norm = _normalize_key(key)
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

    for spec in RATE_BUCKETS:
        mins = float(minute_source.get(spec.label, 0.0) or 0.0)
        if mins <= 0:
            continue
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

