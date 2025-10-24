"""Shared rate defaults and fallback helpers for planner-aware pricing."""
from __future__ import annotations

import re
from types import MappingProxyType
from typing import Iterable, Mapping

__all__ = [
    "DEFAULT_ROLE_MODE_FALLBACKS",
    "LEGACY_MACHINE_RATES",
    "LEGACY_ROLE_RATES",
    "LEGACY_TWO_BUCKET_RATES",
    "PROCESS_RATE_DEFAULTS",
    "ROLE_BUCKETS",
    "default_labor_rate",
    "default_machine_rate",
    "default_process_rate",
    "fallback_keys_for_mode",
    "fallback_rate_for_bucket",
    "fallback_rate_for_role",
    "get_process_rate",
    "labor_rate",
    "machine_rate",
    "rate_for_machine",
    "rate_for_role",
]

# Canonical fallback rates keyed by the bucket role and rate mode.
_DEFAULT_ROLE = "_default"
_ROLE_MODE_RATES: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "labor_only": MappingProxyType({"machine": 45.0, "labor": 45.0}),
        "machine_only": MappingProxyType({"machine": 45.0, "labor": 45.0}),
        "split": MappingProxyType({"machine": 90.0, "labor": 45.0}),
        _DEFAULT_ROLE: MappingProxyType({"machine": 45.0, "labor": 45.0}),
    }
)

ROLE_BUCKETS: tuple[str, ...] = tuple(
    role for role in _ROLE_MODE_RATES.keys() if role != _DEFAULT_ROLE
)
"""Canonical bucket roles recognised by pricing helpers."""

DEFAULT_ROLE_MODE_FALLBACKS: Mapping[str, Mapping[str, float]] = _ROLE_MODE_RATES
"""Read-only mapping of fallback rates for each bucket role/mode pair."""

_FALLBACK_KEYS = MappingProxyType(
    {
        "machine": ("MachineRate", "machine_rate", "machine"),
        "labor": ("LaborRate", "labor_rate", "labor"),
    }
)

_LEGACY_MACHINE_RATES = MappingProxyType(
    {
        "CNC_Mill": 90.0,
        "MillingRate": 90.0,
        "CNC_Vertical": 90.0,
        "cnc_vertical": 90.0,
        "DrillingRate": 95.0,
        "DrillPress": 95.0,
        "SurfaceGrind": 95.0,
        "SurfaceGrindRate": 95.0,
        "GrindingRate": 95.0,
        "ODIDGrind": 95.0,
        "ODIDGrindRate": 95.0,
        "JigGrind": 95.0,
        "JigGrindRate": 95.0,
        "Blanchard": 95.0,
        "VisualContourGrind": 95.0,
        "ExtrudeHone": 110.0,
        "AbrasiveFlowRate": 110.0,
        "WireEDM": 130.0,
        "WireEDMRate": 130.0,
        "SinkerEDM": 130.0,
        "SinkerEDMRate": 130.0,
        "SawWaterjetRate": 90.0,
        "SawRate": 90.0,
        "WaterjetRate": 90.0,
        "Waterjet": 90.0,
        "MachineRate": 90.0,
        "DefaultMachineRate": 90.0,
    }
)

_LEGACY_LABOR_RATES = MappingProxyType(
    {
        "Programmer": 90.0,
        "ProgrammingRate": 90.0,
        "Engineer": 90.0,
        "ProjectManager": 90.0,
        "Toolmaker": 90.0,
        "FixtureBuilder": 75.0,
        "FixtureBuildRate": 75.0,
        "Finisher": 45.0,
        "FinishingRate": 45.0,
        "DeburrRate": 45.0,
        "Lapper": 60.0,
        "Inspector": 85.0,
        "InspectionRate": 85.0,
        "Assembler": 60.0,
        "Grinder": 55.0,
        "EDMOperator": 60.0,
        "Machinist": 45.0,
        "LaborRate": 45.0,
        "DefaultLaborRate": 45.0,
        "Support": 40.0,
    }
)

LEGACY_MACHINE_RATES: Mapping[str, float] = _LEGACY_MACHINE_RATES
"""Read-only mapping of canonical machine rate fallbacks keyed by legacy names."""

LEGACY_ROLE_RATES: Mapping[str, float] = _LEGACY_LABOR_RATES
"""Read-only mapping of canonical labor rate fallbacks keyed by legacy names."""

LEGACY_TWO_BUCKET_RATES: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "machine": _LEGACY_MACHINE_RATES,
        "labor": _LEGACY_LABOR_RATES,
    }
)

_PROCESS_MACHINE_RATES = MappingProxyType(
    {
        "milling": 90.0,
        "drilling": 95.0,
    }
)

_PROCESS_LABOR_RATES = MappingProxyType(
    {
        "programming": 45.0,
        "milling": 45.0,
        "drilling": 45.0,
        "inspection": 90.0,
    }
)

PROCESS_RATE_DEFAULTS: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "machine": _PROCESS_MACHINE_RATES,
        "labor": _PROCESS_LABOR_RATES,
    }
)


def _normalize_key(name: object | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


def _build_default_index() -> dict[str, dict[str, float]]:
    index: dict[str, dict[str, float]] = {}
    for kind, mapping in LEGACY_TWO_BUCKET_RATES.items():
        kind_index: dict[str, float] = {}
        for key, value in mapping.items():
            if value is None:
                continue
            key_text = str(key)
            kind_index[key_text] = float(value)
            normalized = _normalize_key(key_text)
            if normalized and normalized not in kind_index:
                kind_index[normalized] = float(value)
        index[kind] = kind_index
    return index


_DEFAULT_RATE_INDEX = _build_default_index()


def _build_process_index() -> dict[str, dict[str, float]]:
    index: dict[str, dict[str, float]] = {}
    for kind, mapping in PROCESS_RATE_DEFAULTS.items():
        kind_index: dict[str, float] = {}
        for key, value in mapping.items():
            key_text = str(key)
            kind_index[key_text] = float(value)
            normalized = _normalize_key(key_text)
            if normalized and normalized not in kind_index:
                kind_index[normalized] = float(value)
        index[kind] = kind_index
    return index


_PROCESS_RATE_INDEX = _build_process_index()


_PROCESS_RATE_ALIASES: Mapping[str, Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "machine": MappingProxyType(
            {
                "milling": ("CNC_Mill", "MillingRate", "CNC_Vertical", "cnc_vertical"),
                "drilling": ("DrillPress", "DrillingRate"),
                "grinding": (
                    "GrindingRate",
                    "SurfaceGrind",
                    "SurfaceGrindRate",
                    "Blanchard",
                    "JigGrind",
                    "JigGrindRate",
                    "ODIDGrind",
                    "ODIDGrindRate",
                ),
                "wire_edm": ("WireEDM", "WireEDMRate"),
                "sinker_edm": ("SinkerEDM", "SinkerEDMRate"),
                "saw_waterjet": ("SawWaterjetRate", "SawRate", "WaterjetRate", "Waterjet"),
                "abrasive_flow": ("ExtrudeHone", "AbrasiveFlowRate"),
            }
        ),
        "labor": MappingProxyType(
            {
                "milling": ("Machinist", "LaborRate", "DefaultLaborRate"),
                "drilling": ("Machinist", "LaborRate", "DefaultLaborRate"),
                "programming": ("Programmer", "ProgrammingRate"),
                "programming_amortized": ("Programmer", "ProgrammingRate"),
                "inspection": ("Inspector", "InspectionRate"),
                "fixture_build": ("FixtureBuilder", "FixtureBuildRate"),
                "fixture_build_amortized": ("FixtureBuilder", "FixtureBuildRate"),
                "finishing": ("Finisher", "DeburrRate", "FinishingRate"),
                "finishing_deburr": ("Finisher", "DeburrRate", "FinishingRate"),
                "grinding": ("Grinder", "Finisher", "DeburrRate", "FinishingRate"),
            }
        ),
    }
)


def get_process_rate(kind: str, process: str, *, default: float | None = None) -> float:
    """Return the configured rate for ``process`` under ``kind``."""

    kind_key = str(kind).strip().lower()
    mapping = _PROCESS_RATE_INDEX.get(kind_key)
    if not mapping:
        return float(default or 0.0)

    for candidate in _iter_lookup_keys(process, kind=kind_key):
        if candidate in mapping:
            return float(mapping[candidate])

    if default is not None:
        return float(default)
    return 0.0


def _iter_lookup_keys(name: str | None, *, kind: str | None = None) -> Iterable[str]:
    candidates = [str(name or "").strip()]
    normalized = _normalize_key(name)
    if normalized and normalized not in candidates:
        candidates.append(normalized)

    alias_mapping = None
    if kind:
        alias_mapping = _PROCESS_RATE_ALIASES.get(kind)
        if alias_mapping:
            alias_keys = alias_mapping.get(normalized) or alias_mapping.get(candidates[0])
            if alias_keys:
                candidates.extend(alias_keys)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate
        if alias_mapping:
            alias_norm = _normalize_key(candidate)
            if alias_norm and alias_norm not in seen:
                seen.add(alias_norm)
                yield alias_norm


def machine_rate(process: str, *, default: float | None = None) -> float:
    """Return the configured machine rate for ``process``."""

    return get_process_rate("machine", process, default=default)


def labor_rate(process: str, *, default: float | None = None) -> float:
    """Return the configured labor rate for ``process``."""

    return get_process_rate("labor", process, default=default)


def _lookup_named_rate(kind: str, name: str) -> float | None:
    kind_key = str(kind).strip().lower()
    mapping = _DEFAULT_RATE_INDEX.get(kind_key)
    if not mapping:
        return None

    for candidate in _iter_lookup_keys(name, kind=kind_key):
        if candidate in mapping:
            return float(mapping[candidate])
    return None


def default_process_rate(kind: str, name: str, *, default: float | None = None) -> float:
    """Return the canonical default rate for ``name`` within ``kind``."""

    rate = _lookup_named_rate(kind, name)
    if rate is not None:
        return rate
    if default is not None:
        return float(default)
    return 0.0


def default_machine_rate(name: str, *, default: float | None = None) -> float:
    """Return the canonical default machine rate for ``name``."""

    return default_process_rate("machine", name, default=default)


def default_labor_rate(name: str, *, default: float | None = None) -> float:
    """Return the canonical default labor rate for ``name``."""

    return default_process_rate("labor", name, default=default)


def rate_for_machine(rates: Mapping[str, Mapping[str, float]], machine: str) -> float:
    return float(rates["machine"][machine])


def rate_for_role(rates: Mapping[str, Mapping[str, float]], role: str) -> float:
    return float(rates["labor"][role])


def _normalise_role(role: str | None) -> str:
    if not role:
        return _DEFAULT_ROLE
    role_norm = str(role).strip().lower()
    if role_norm in _ROLE_MODE_RATES:
        return role_norm
    return _DEFAULT_ROLE


def fallback_rate_for_role(role: str | None, *, mode: str | None = None) -> float:
    """Return the numeric fallback rate for ``role`` under ``mode``.

    Parameters
    ----------
    role:
        Canonical bucket role such as ``"labor_only"`` or ``"split"``. ``None``
        is treated as the default role.
    mode:
        Either ``"machine"`` or ``"labor"`` to request the corresponding
        default rate. When omitted the machine fallback is used.
    """

    role_key = _normalise_role(role)
    mode_key = str(mode or "machine").strip().lower()
    role_rates = _ROLE_MODE_RATES.get(role_key, _ROLE_MODE_RATES[_DEFAULT_ROLE])

    if mode_key in role_rates:
        return float(role_rates[mode_key])

    # Gracefully fall back to the machine rate if an unknown mode is requested.
    return float(role_rates.get("machine", 0.0))


def fallback_rate_for_bucket(
    bucket: str | None,
    *,
    roles: Mapping[str, str] | None = None,
    mode: str | None = None,
    default_role: str | None = None,
) -> float:
    """Return the fallback rate for *bucket* using ``roles`` metadata."""

    role_value: str | None = None
    if roles:
        key_norm = str(bucket or "").strip().lower()
        role_value = roles.get(key_norm)
        if role_value is None and bucket is not None:
            role_value = roles.get(str(bucket))
        if role_value is None:
            role_value = roles.get(_DEFAULT_ROLE)
    if role_value is None:
        role_value = default_role
    return fallback_rate_for_role(role_value, mode=mode)


def fallback_keys_for_mode(mode: str | None) -> tuple[str, ...]:
    """Return canonical lookup keys for ``lookup_rate`` fallbacks."""

    if not mode:
        combined: list[str] = []
        for key in ("machine", "labor"):
            combined.extend(_FALLBACK_KEYS[key])
        # ``dict.fromkeys`` preserves order while removing duplicates.
        return tuple(dict.fromkeys(combined))

    mode_key = str(mode).strip().lower()
    keys = _FALLBACK_KEYS.get(mode_key)
    if keys is not None:
        return keys
    return _FALLBACK_KEYS["machine"]
