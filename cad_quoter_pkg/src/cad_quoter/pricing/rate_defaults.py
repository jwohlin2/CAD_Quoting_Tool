"""Shared rate defaults and fallback helpers for planner-aware pricing."""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

__all__ = [
    "DEFAULT_ROLE_MODE_FALLBACKS",
    "fallback_keys_for_mode",
    "fallback_rate_for_bucket",
    "fallback_rate_for_role",
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

DEFAULT_ROLE_MODE_FALLBACKS: Mapping[str, Mapping[str, float]] = _ROLE_MODE_RATES
"""Read-only mapping of fallback rates for each bucket role/mode pair."""

_FALLBACK_KEYS = MappingProxyType(
    {
        "machine": ("MachineRate", "machine_rate", "machine"),
        "labor": ("LaborRate", "labor_rate", "labor"),
    }
)


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
