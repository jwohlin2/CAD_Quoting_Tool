"""Wrappers around geometry helper utilities."""

from __future__ import annotations

from appV5 import (
    _collect_geo_features_from_df as _collect_geo_features_from_df,
    _map_geo_to_double_underscore as _map_geo_to_double_underscore,
    update_variables_df_with_geo as _update_variables_df_with_geo,
)

__all__ = [
    "map_geo_to_double_underscore",
    "collect_geo_features_from_df",
    "update_variables_df_with_geo",
]


def map_geo_to_double_underscore(geo: dict) -> dict:
    """Expose :func:`appV5._map_geo_to_double_underscore`."""

    return _map_geo_to_double_underscore(geo)


def collect_geo_features_from_df(df):
    """Expose :func:`appV5._collect_geo_features_from_df`."""

    return _collect_geo_features_from_df(df)


def update_variables_df_with_geo(df, geo: dict):
    """Expose :func:`appV5.update_variables_df_with_geo`."""

    return _update_variables_df_with_geo(df, geo)
