from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, MutableSequence
from typing import Any, cast

from cad_quoter.app.hole_ops import COUNTERDRILL_MIN_PER_SIDE_MIN
from cad_quoter.ui.planner_render import (
    JIG_GRIND_MIN_PER_FEATURE,
    _lookup_bucket_rate,
    _set_bucket_minutes_cost,
)

__all__ = [
    "_normalize_extra_bucket_payload",
    "_extra_bucket_entry_key",
    "_merge_extra_bucket_entries",
    "_publish_extra_bucket_op",
    "_ensure_bucket_view_mapping",
    "_append_counterdrill_extra",
    "_append_jig_extra",
    "COUNTERDRILL_MIN_PER_SIDE_MIN",
    "JIG_GRIND_MIN_PER_FEATURE",
]


def _coerce_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number if number > 0 else None


def _normalize_extra_bucket_payload(
    payload: Mapping[str, Any] | Any,
    *,
    minutes: float | None = None,
) -> dict[str, Any] | None:
    if isinstance(payload, Mapping):
        data = dict(payload)
    else:
        try:
            data = dict(payload)  # type: ignore[arg-type]
        except Exception:
            return None

    name_text = str(data.get("name") or data.get("op") or "").strip()
    if not name_text:
        return None

    normalized: dict[str, Any] = {"name": name_text}

    side_raw = data.get("side")
    if side_raw is None:
        side_norm: str | None = None
    elif isinstance(side_raw, str):
        side_norm = side_raw.strip().lower() or None
    else:
        side_norm = str(side_raw).strip().lower() or None
    normalized["side"] = side_norm

    qty_raw = data.get("qty")
    qty_val = 0
    if qty_raw not in (None, ""):
        try:
            qty_val = int(round(float(qty_raw)))
        except Exception:
            try:
                qty_val = int(qty_raw)  # type: ignore[arg-type]
            except Exception:
                qty_val = 0
    normalized["qty"] = max(qty_val, 0)

    if minutes is None:
        minutes_raw = data.get("minutes")
        if minutes_raw in (None, ""):
            minutes_raw = data.get("mins")
        try:
            minutes_val = float(minutes_raw)
        except Exception:
            minutes_val = 0.0
    else:
        minutes_val = float(minutes)

    minutes_val = minutes_val if math.isfinite(minutes_val) else 0.0
    if minutes_val > 0.0:
        normalized["minutes"] = round(minutes_val, 3)

    def _extract_cost(*keys: str) -> float:
        for key in keys:
            cost_val = _coerce_positive_float(data.get(key))
            if cost_val is not None:
                return cost_val
        return 0.0

    machine_val = _extract_cost("machine", "machine_cost")
    labor_val = _extract_cost("labor", "labor_cost")
    total_val = _extract_cost("total", "total_cost")

    if machine_val > 0.0:
        normalized["machine"] = machine_val
    if labor_val > 0.0:
        normalized["labor"] = labor_val
    if total_val > 0.0:
        normalized["total"] = total_val

    return normalized


def _extra_bucket_entry_key(entry: Mapping[str, Any]) -> tuple[str, str | None]:
    name_text = str(entry.get("name") or entry.get("op") or "").strip().lower()
    side_raw = entry.get("side")
    if side_raw is None:
        side_norm: str | None = None
    elif isinstance(side_raw, str):
        side_norm = side_raw.strip().lower() or None
    else:
        side_norm = str(side_raw).strip().lower() or None
    return name_text, side_norm


def _merge_extra_bucket_entries(
    existing: MutableMapping[str, Any],
    incoming: Mapping[str, Any],
) -> None:
    def _as_int(value: Any) -> int:
        try:
            return int(round(float(value)))
        except Exception:
            try:
                return int(value)  # type: ignore[arg-type]
            except Exception:
                return 0

    def _as_float(value: Any) -> float:
        try:
            numeric = float(value)
        except Exception:
            return 0.0
        return numeric if math.isfinite(numeric) else 0.0

    existing["name"] = str(incoming.get("name") or existing.get("name") or "").strip()

    incoming_side = incoming.get("side")
    if incoming_side is not None:
        existing["side"] = incoming_side

    incoming_qty = _as_int(incoming.get("qty"))
    if incoming_qty > 0:
        existing_qty = _as_int(existing.get("qty"))
        if incoming_qty > existing_qty:
            existing["qty"] = incoming_qty

    incoming_minutes = _as_float(incoming.get("minutes"))
    if incoming_minutes > 0.0:
        existing_minutes = _as_float(existing.get("minutes") or existing.get("mins"))
        if incoming_minutes > existing_minutes:
            existing["minutes"] = round(incoming_minutes, 3)

    for field in ("machine", "labor", "total"):
        incoming_val = _as_float(incoming.get(field))
        if incoming_val <= 0.0:
            continue
        existing_val = _as_float(existing.get(field))
        if incoming_val > existing_val:
            existing[field] = incoming_val

    for alias in ("mins", "machine_cost", "labor_cost", "total_cost"):
        if alias in existing:
            try:
                del existing[alias]
            except Exception:
                pass


def _publish_extra_bucket_op(
    extra_bucket_ops: MutableMapping[str, Any] | Mapping[str, Any] | None,
    bucket: str,
    payload: Mapping[str, Any] | Any,
    *,
    minutes: float | None = None,
) -> None:
    if not isinstance(extra_bucket_ops, MutableMapping):
        return

    try:
        entries_obj = extra_bucket_ops.setdefault(bucket, [])
    except Exception:
        return

    if isinstance(entries_obj, list):
        entries: MutableSequence[Any] = cast(MutableSequence[Any], entries_obj)
    else:
        try:
            entries = cast(MutableSequence[Any], list(entries_obj))  # type: ignore[arg-type]
        except Exception:
            entries = []
        try:
            extra_bucket_ops[bucket] = entries  # type: ignore[index]
        except Exception:
            return

    normalized = _normalize_extra_bucket_payload(payload, minutes=minutes)
    if not normalized:
        return

    new_key = _extra_bucket_entry_key(normalized)
    for idx, existing in enumerate(entries):
        if isinstance(existing, MutableMapping):
            existing_map = cast(MutableMapping[str, Any], existing)
        elif isinstance(existing, Mapping):
            try:
                existing_map = dict(existing)
                entries[idx] = existing_map
            except Exception:
                continue
        else:
            continue

        if _extra_bucket_entry_key(existing_map) == new_key:
            _merge_extra_bucket_entries(existing_map, normalized)
            return

    entries.append(normalized)


def _ensure_bucket_view_mapping(
    owner: MutableMapping[str, Any] | Mapping[str, Any] | None,
) -> MutableMapping[str, Any] | Mapping[str, Any] | None:
    if isinstance(owner, dict):
        try:
            return owner.setdefault("bucket_view", {})
        except Exception:
            return owner.get("bucket_view")
    if isinstance(owner, MutableMapping):
        try:
            return cast(MutableMapping[str, Any], owner.setdefault("bucket_view", {}))
        except Exception:
            try:
                return owner.get("bucket_view")
            except Exception:
                return None
    return None


def _ensure_extra_bucket_ops(
    owner: MutableMapping[str, Any] | Mapping[str, Any] | None,
) -> MutableMapping[str, Any] | None:
    if isinstance(owner, dict):
        try:
            extra = owner.setdefault("extra_bucket_ops", {})
        except Exception:
            return None
    elif isinstance(owner, MutableMapping):
        try:
            extra = owner.setdefault("extra_bucket_ops", {})
        except Exception:
            return None
    elif isinstance(owner, Mapping):
        extra = owner.get("extra_bucket_ops")
    else:
        return None

    if isinstance(extra, MutableMapping):
        return extra
    try:
        converted = dict(extra)  # type: ignore[arg-type]
    except Exception:
        return None

    if isinstance(owner, (dict, MutableMapping)):
        try:
            owner["extra_bucket_ops"] = converted  # type: ignore[index]
        except Exception:
            pass
    return cast(MutableMapping[str, Any], converted)


def _append_counterdrill_extra(
    owner: MutableMapping[str, Any] | Mapping[str, Any] | None,
    qty: int,
    *,
    side: str | None = "front",
    rates: Mapping[str, Any] | None,
) -> float:
    try:
        qty_val = int(qty)
    except Exception:
        qty_val = 0
    if qty_val <= 0:
        return 0.0

    extra_ops = _ensure_extra_bucket_ops(owner)
    if extra_ops is not None:
        _publish_extra_bucket_op(
            extra_ops,
            "counterdrill",
            {"name": "Counterdrill", "qty": qty_val, "side": side},
        )

    minutes_per = float(
        globals().get("COUNTERDRILL_MIN_PER_SIDE_MIN")
        or COUNTERDRILL_MIN_PER_SIDE_MIN
        or 0.12
    )
    total_minutes = float(qty_val) * minutes_per

    bucket_view_obj = _ensure_bucket_view_mapping(owner)
    if bucket_view_obj is not None:
        machine_rate = (
            _lookup_bucket_rate("counterdrill", rates)
            or _lookup_bucket_rate("drilling", rates)
            or _lookup_bucket_rate("machine", rates)
            or 53.76
        )
        labor_rate = (
            _lookup_bucket_rate("drilling_labor", rates)
            or _lookup_bucket_rate("labor", rates)
            or 25.46
        )
        _set_bucket_minutes_cost(
            bucket_view_obj,
            "counterdrill",
            total_minutes,
            machine_rate,
            labor_rate,
        )

    return total_minutes


def _append_jig_extra(
    owner: MutableMapping[str, Any] | Mapping[str, Any] | None,
    qty: int,
    *,
    rates: Mapping[str, Any] | None,
) -> float:
    try:
        qty_val = int(qty)
    except Exception:
        qty_val = 0
    if qty_val <= 0:
        return 0.0

    extra_ops = _ensure_extra_bucket_ops(owner)
    if extra_ops is not None:
        _publish_extra_bucket_op(
            extra_ops,
            "jig-grind",
            {"name": "Jig-grind", "qty": qty_val, "side": None},
        )

    minutes_per = float(globals().get("JIG_GRIND_MIN_PER_FEATURE") or 0.75)
    total_minutes = float(qty_val) * minutes_per

    bucket_view_obj = _ensure_bucket_view_mapping(owner)
    if bucket_view_obj is not None:
        machine_rate = (
            _lookup_bucket_rate("grinding", rates)
            or _lookup_bucket_rate("machine", rates)
            or 53.76
        )
        labor_rate = _lookup_bucket_rate("labor", rates) or 25.46
        _set_bucket_minutes_cost(
            bucket_view_obj,
            "grinding",
            total_minutes,
            machine_rate,
            labor_rate,
        )

    return total_minutes
