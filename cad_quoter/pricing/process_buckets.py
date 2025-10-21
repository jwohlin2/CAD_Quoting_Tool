"""Shared helpers for process cost bucket normalisation and labelling."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
import re
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class _BucketSpec:
    """Configuration for a canonical process bucket."""

    key: str
    label: str
    rate_aliases: tuple[str, ...] = ()
    hide_in_cost: bool = False


_BASE_BUCKET_SPECS: tuple[_BucketSpec, ...] = (
    _BucketSpec("milling", "Milling", ("MillingRate",)),
    _BucketSpec(
        "drilling",
        "Drilling",
        ("DrillingRate", "CncVertical", "CncVerticalRate", "cnc_vertical"),
    ),
    _BucketSpec("counterbore", "Counterbore", ("CounterboreRate", "DrillingRate")),
    _BucketSpec("tapping", "Tapping", ("TappingRate", "DrillingRate")),
    _BucketSpec(
        "grinding",
        "Grinding",
        ("GrindingRate", "SurfaceGrindRate", "ODIDGrindRate", "JigGrindRate"),
    ),
    _BucketSpec("wire_edm", "Wire EDM", ("WireEDMRate", "EDMRate")),
    _BucketSpec("sinker_edm", "Sinker EDM", ("SinkerEDMRate", "EDMRate")),
    _BucketSpec("finishing_deburr", "Finishing/Deburr", ("FinishingRate", "DeburrRate")),
    _BucketSpec("saw_waterjet", "Saw/Waterjet", ("SawWaterjetRate", "SawRate", "WaterjetRate")),
    _BucketSpec("inspection", "Inspection", ("InspectionRate",)),
    _BucketSpec(
        "toolmaker_support",
        "Toolmaker Support",
        ("ToolmakerRate", "ToolAndDieMakerRate", "LaborRate"),
    ),
    _BucketSpec("fixture_build_amortized", "Fixture Build (amortized)", ("FixtureBuildRate",)),
    _BucketSpec(
        "programming_amortized",
        "Programming (per part)",
        ("ProgrammingRate", "EngineerRate", "ProgrammerRate"),
    ),
    _BucketSpec(
        "misc",
        "Misc",
        ("LaborRate", "MachineRate", "DefaultLaborRate", "DefaultMachineRate"),
        hide_in_cost=True,
    ),
)


# Additional buckets are surfaced by the planner UI but are not part of the
# pricing renderer's default order.  They still benefit from the same labelling
# and rate heuristics.
_EXTRA_BUCKET_SPECS: dict[str, _BucketSpec] = {
    "programming": _BucketSpec(
        "programming",
        "Programming",
        ("ProgrammingRate", "EngineerRate", "ProgrammerRate"),
    ),
    "fixture_build": _BucketSpec("fixture_build", "Fixture Build", ("FixtureBuildRate",)),
    "countersink": _BucketSpec("countersink", "Countersink", ("CounterSinkRate", "DrillingRate")),
}


ORDER: tuple[str, ...] = tuple(spec.key for spec in _BASE_BUCKET_SPECS)
"""Canonical ordering for pricing process buckets."""


_LABEL_MAP: dict[str, str] = {spec.key: spec.label for spec in _BASE_BUCKET_SPECS}
_LABEL_MAP.update({key: spec.label for key, spec in _EXTRA_BUCKET_SPECS.items()})

_RATE_ALIAS_KEYS: dict[str, tuple[str, ...]] = {
    spec.key: spec.rate_aliases for spec in _BASE_BUCKET_SPECS if spec.rate_aliases
}
_RATE_ALIAS_KEYS.update(
    {
        key: spec.rate_aliases
        for key, spec in _EXTRA_BUCKET_SPECS.items()
        if spec.rate_aliases
    }
)

HIDE_IN_COST: frozenset[str] = frozenset(
    {"planner_total", "planner_labor", "planner_machine"}
    | {spec.key for spec in _BASE_BUCKET_SPECS if spec.hide_in_cost}
)
"""Buckets that should be hidden from rendered totals."""


_ALIAS_MAP: dict[str, str] = {
    "machining": "milling",
    "mill": "milling",
    "cnc_milling": "milling",
    "cnc": "milling",
    "turning": "misc",
    "cnc_turning": "misc",
    "wire_edm": "wire_edm",
    "wireedm": "wire_edm",
    "wire-edm": "wire_edm",
    "wire edm": "wire_edm",
    "wire_edm_windows": "wire_edm",
    "wire_edm_outline": "wire_edm",
    "wire_edm_open_id": "wire_edm",
    "wire_edm_cam_slot_or_profile": "wire_edm",
    "wire_edm_id_leave": "wire_edm",
    "wedm": "wire_edm",
    "edm": "wire_edm",
    "sinker_edm": "sinker_edm",
    "sinkeredm": "sinker_edm",
    "sinker-edm": "sinker_edm",
    "sinker edm": "sinker_edm",
    "ram_edm": "sinker_edm",
    "ramedm": "sinker_edm",
    "ram-edm": "sinker_edm",
    "sinker_edm_finish_burn": "sinker_edm",
    "lap": "grinding",
    "lapping": "grinding",
    "lapping_honing": "grinding",
    "honing": "grinding",
    "deburr": "finishing_deburr",
    "deburring": "finishing_deburr",
    "finishing": "finishing_deburr",
    "finishing_misc": "finishing_deburr",
    "finishing_deburr": "finishing_deburr",
    "saw": "saw_waterjet",
    "waterjet": "saw_waterjet",
    "saw_waterjet": "saw_waterjet",
    "inspection": "inspection",
    "inspect": "inspection",
    "quality": "inspection",
    "counter_bore": "counterbore",
    "counter_boring": "counterbore",
    "counterbore": "counterbore",
    "counter_sink": "drilling",
    "countersink": "drilling",
    "csk": "drilling",
    "tap": "tapping",
    "taps": "tapping",
    "tapping": "tapping",
    "drill": "drilling",
    "drilling": "drilling",
    "assembly": "misc",
    "packaging": "misc",
    "ehs_compliance": "misc",
    "machine": "misc",
    "labor": "misc",
    "planner_machine": "misc",
    "planner_labor": "misc",
    "planner_misc": "misc",
}


__all__ = [
    "ORDER",
    "HIDE_IN_COST",
    "bucket_label",
    "canonical_bucket_key",
    "flatten_rates",
    "lookup_rate",
    "normalize_bucket_key",
]


def normalize_bucket_key(name: Any) -> str:
    """Return a stable, lowercase/underscored representation of *name*."""

    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


def canonical_bucket_key(
    name: Any,
    *,
    allowed: Iterable[str] | None = None,
    default: str = "misc",
) -> str | None:
    """Return the canonical bucket key for *name*.

    When *allowed* is provided the normalised key is accepted before any alias
    lookups.  This lets callers opt into planner-specific buckets (such as
    countersink) without affecting the pricing renderer's canonicalisation.
    """

    norm = normalize_bucket_key(name)
    if not norm:
        return None
    if norm == "planner_total":
        return None

    allowed_keys: Tuple[str, ...]
    if allowed is None:
        allowed_keys = ORDER
    else:
        allowed_keys = tuple(allowed)
        if norm in allowed_keys:
            return norm

    alias = _ALIAS_MAP.get(norm)
    if alias:
        norm = alias

    if norm in allowed_keys:
        return norm

    if norm.startswith("planner_"):
        return default

    return default


def bucket_label(key: str) -> str:
    """Return the display label for a canonical bucket key."""

    if not key:
        return ""
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    return key.replace("_", " ").title()


def _rate_aliases_for(key: str) -> tuple[str, ...]:
    norm = normalize_bucket_key(key)
    if not norm:
        return ()
    aliases = _RATE_ALIAS_KEYS.get(norm)
    if aliases:
        return aliases
    return ()


def flatten_rates(
    rates: Mapping[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return flat and normalised rate lookups."""

    flat: dict[str, float] = {}
    normalized: dict[str, float] = {}

    if not isinstance(rates, Mapping):
        return flat, normalized

    def _walk(mapping: Mapping[str, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                _walk(value)
                continue
            try:
                amount = float(value)
            except Exception:
                continue
            key_str = str(key)
            flat[key_str] = amount
            norm = normalize_bucket_key(key_str)
            if norm and norm not in normalized:
                normalized[norm] = amount

    _walk(rates)
    return flat, normalized


def _rate_candidates(key: str) -> tuple[str, ...]:
    norm = normalize_bucket_key(key)
    if not norm:
        return ()
    pieces = [part for part in norm.split("_") if part]
    camel = "".join(piece.title() for piece in pieces)
    spaced = " ".join(piece.title() for piece in pieces)
    candidates = [
        key,
        norm,
        camel,
        f"{camel}Rate",
        f"{camel}_rate",
        spaced,
        f"{spaced} Rate",
        f"{spaced} rate",
    ]
    candidates.extend(_rate_aliases_for(norm))
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def lookup_rate(
    key: str,
    flat: Mapping[str, float],
    normalized: Mapping[str, float],
    *,
    fallbacks: Iterable[str] = ("labor", "labor_rate", "machine", "machine_rate"),
) -> float:
    """Return the rate for *key* using flattened/normalised rate maps."""

    for candidate in _rate_candidates(key):
        if candidate in flat:
            return flat[candidate]
        norm = normalize_bucket_key(candidate)
        if norm and norm in normalized:
            return normalized[norm]
    for fallback in fallbacks:
        norm = normalize_bucket_key(fallback)
        if norm and norm in normalized:
            return normalized[norm]
    return 0.0

