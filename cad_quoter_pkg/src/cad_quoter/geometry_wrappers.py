"""Wrappers around geometry helper utilities."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from cad_quoter.utils.geo_fallbacks import (
    collect_geo_features_from_df as _collect_geo_features_from_df_helper,
    map_geo_to_double_underscore as _map_geo_to_double_underscore_helper,
    update_variables_df_with_geo as _update_variables_df_with_geo_helper,
)


@lru_cache(maxsize=1)
def _load_app_module():  # pragma: no cover - thin wrapper around import machinery
    import importlib

    return importlib.import_module("appV5")


def _get_app_attr(name: str):
    try:
        module = _load_app_module()
    except Exception:
        return None
    return getattr(module, name, None)


_app_map_geo = _get_app_attr("_map_geo_to_double_underscore")
_app_collect_geo = _get_app_attr("_collect_geo_features_from_df")
_app_update_geo = _get_app_attr("update_variables_df_with_geo")

__all__ = [
    "map_geo_to_double_underscore",
    "collect_geo_features_from_df",
    "update_variables_df_with_geo",
]


def map_geo_to_double_underscore(geo: Mapping[str, Any] | None) -> dict[str, float]:
    """Expose :func:`appV5._map_geo_to_double_underscore` with a fallback."""

    return _map_geo_to_double_underscore_helper(geo, mapper=_app_map_geo)


def collect_geo_features_from_df(df):
    """Return geometry features from a variable dataframe using a robust fallback."""

    return _collect_geo_features_from_df_helper(df, collector=_app_collect_geo)


def update_variables_df_with_geo(df, geo: dict):
    """Update a variables dataframe with GEO__ entries using a robust fallback."""

    return _update_variables_df_with_geo_helper(df, geo, updater=_app_update_geo)
