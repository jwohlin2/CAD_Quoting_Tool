"""Formatting helpers shared across the desktop UI modules."""

from __future__ import annotations

from typing import Any

from cad_quoter.utils.rendering import fmt_hours, fmt_money, fmt_percent


def _coerce_user_value(raw: Any, kind: str) -> Any:
    """Best-effort coercion of user supplied override values."""

    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            return None
    try:
        if kind in {"float", "currency", "percent", "multiplier", "hours"}:
            return float(raw)
        if kind == "int":
            return int(round(float(raw)))
    except Exception:
        return None
    if kind == "text":
        return str(raw)
    return raw


def _format_value(value: Any, kind: str) -> str:
    """Render ``value`` for display in read-only widgets."""

    if value is None:
        return "–"
    try:
        if kind == "percent":
            return fmt_percent(float(value))
        if kind == "hours":
            return fmt_hours(float(value), include_unit=False, decimals=3)
        if kind in {"float", "multiplier"}:
            return f"{float(value):.3f}"
        if kind == "currency":
            return fmt_money(float(value), "$")
        if kind == "int":
            return f"{int(round(float(value)))}"
    except Exception:
        return str(value)
    return str(value)


def _format_entry_value(value: Any, kind: str) -> str:
    """Render ``value`` for use as the default text in entry widgets."""

    if value is None:
        return ""
    try:
        if kind == "int":
            return str(int(round(float(value))))
        if kind == "currency":
            return f"{float(value):.2f}"
        if kind in {"percent", "multiplier", "hours", "float"}:
            return f"{float(value):.3f}"
    except Exception:
        return str(value)
    return str(value)


__all__ = ["_coerce_user_value", "_format_value", "_format_entry_value"]

