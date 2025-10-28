"""Helpers for configuring quote rendering behaviour."""

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from typing import Any, MutableMapping as MutableMappingType, cast

from cad_quoter.ui.services import QuoteConfiguration

RenderOverrides = tuple[tuple[str, Any], ...]

_DEFAULT_OVERRIDES: RenderOverrides = (
    ("prefer_removal_drilling_hours", True),
    ("separate_machine_labor", True),
    ("machine_rate_per_hr", 45.0),
    ("labor_rate_per_hr", 45.0),
    ("milling_attended_fraction", 1.0),
)


def _clone_params(source: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a deep copy of ``source`` when provided."""

    if source is None:
        return None
    if isinstance(source, dict):
        return copy.deepcopy(source)
    try:
        return copy.deepcopy(dict(source))
    except Exception:
        try:
            return dict(source)
        except Exception:
            return None


def apply_render_overrides(
    cfg: QuoteConfiguration | None,
    *,
    default_params: Mapping[str, Any] | None = None,
) -> QuoteConfiguration:
    """Ensure ``cfg`` has the overrides expected by the renderer."""

    cfg_obj = cfg
    if cfg_obj is None:
        cfg_obj = QuoteConfiguration(default_params=_clone_params(default_params))

    for name, value in _DEFAULT_OVERRIDES:
        try:
            setattr(cfg_obj, name, value)
        except Exception:
            cfg_obj = QuoteConfiguration(default_params=_clone_params(default_params))
            for name2, value2 in _DEFAULT_OVERRIDES:
                setattr(cfg_obj, name2, value2)
            break

    return cfg_obj


def ensure_mutable_breakdown(
    breakdown: Mapping[str, Any] | MutableMapping[str, Any] | None,
) -> tuple[Mapping[str, Any], MutableMappingType[str, Any]]:
    """Return a mapping pair guaranteeing a mutable view of ``breakdown``."""

    if isinstance(breakdown, MutableMapping):
        mutable = cast(MutableMappingType[str, Any], breakdown)
        return breakdown, mutable

    try:
        mutable_dict: dict[str, Any] = dict(breakdown or {})
    except Exception:
        mutable_dict = {}

    return mutable_dict, mutable_dict

