from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping

from cad_quoter.domain_models.values import safe_float as _safe_float
from cad_quoter.pricing.process_cost_renderer import render_process_costs
from appkit.ui.planner_render import (
    _canonical_bucket_key,
    _display_rate_for_row,
)


__all__ = [
    "render_process_costs",
    "_ProcessCostTableRecorder",
    "_ProcessRowRecord",
    "_merge_process_meta",
    "_fold_process_meta",
    "_merge_applied_process_entries",
    "_fold_applied_process",
    "_lookup_process_meta",
]


@dataclass(slots=True)
class _ProcessRowRecord:
    name: str
    hours: float
    rate: float
    total: float
    canon_key: str | None = None


class _ProcessCostTableRecorder:
    def __init__(
        self,
        *,
        cfg: Any,
        bucket_state: Any,
        detail_lookup: MutableMapping[str, Any],
        label_to_canon: MutableMapping[str, str],
        canon_to_display_label: MutableMapping[str, str],
        process_cost_row_details: MutableMapping[str, tuple[float, float, float]],
        labor_costs_display: MutableMapping[str, float],
        add_labor_cost_line: Callable[..., Any],
        process_meta: Mapping[str, Any] | None,
    ) -> None:
        self.cfg = cfg
        self.bucket_state = bucket_state
        self.detail_lookup = detail_lookup
        self.label_to_canon = label_to_canon
        self.canon_to_display_label = canon_to_display_label
        self.process_cost_row_details = process_cost_row_details
        self.labor_costs_display = labor_costs_display
        self.add_labor_cost_line = add_labor_cost_line
        self.process_meta = process_meta

        self.had_rows = False
        self.rows: list[_ProcessRowRecord] = []
        self._rows: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}

    def _lookup_meta(self, key: str | None) -> Mapping[str, Any] | None:
        return _lookup_process_meta(self.process_meta, key)

    def add_row(
        self,
        label: str,
        hours: float,
        rate: float,
        cost: float,
    ) -> None:
        self.had_rows = True
        label_str = str(label or "").strip()
        canon_key = self.label_to_canon.get(label_str)
        if canon_key is None:
            canon_key = _canonical_bucket_key(label_str)
        display_label = label_str
        if canon_key:
            override_label = self.canon_to_display_label.get(canon_key)
            if override_label:
                display_label = override_label
                self.label_to_canon.setdefault(display_label, canon_key)
        try:
            hours_val = float(hours or 0.0)
        except Exception:
            hours_val = 0.0
        try:
            rate_val = float(rate or 0.0)
        except Exception:
            rate_val = 0.0
        try:
            cost_val = float(cost or 0.0)
        except Exception:
            cost_val = 0.0
        if canon_key:
            self.process_cost_row_details[canon_key] = (hours_val, rate_val, cost_val)
        record_canon = canon_key or _canonical_bucket_key(display_label)
        if not record_canon:
            record_canon = None
        record = _ProcessRowRecord(
            display_label,
            hours_val,
            rate_val,
            cost_val,
            record_canon,
        )
        self.rows.append(record)
        if record_canon:
            self._index[record_canon] = len(self.rows) - 1
        rate_display = _display_rate_for_row(
            record_canon or display_label,
            cfg=self.cfg,
            render_state=self.bucket_state,
            hours=hours_val,
        )
        detail_parts: list[str] = []
        if rate_display:
            detail_parts.append(str(rate_display))
        existing_detail = self.detail_lookup.get(display_label)
        if existing_detail not in (None, ""):
            for segment in str(existing_detail).split(";"):
                cleaned = segment.strip()
                if not cleaned or cleaned.startswith("-"):
                    continue
                if cleaned not in detail_parts:
                    detail_parts.append(cleaned)
        simple_hours_line: str | None = None
        if hours_val > 0.0 and cost_val > 0.0:
            rate_for_detail = cost_val / hours_val if hours_val else 0.0
            planner_rate_override: float | None = None
            if record_canon and self.bucket_state is not None:
                extra_payload = getattr(self.bucket_state, "extra", None)
                split_lookup: Mapping[str, Any] | None = None
                if isinstance(extra_payload, Mapping):
                    split_source = extra_payload.get("bucket_hour_split")
                    if isinstance(split_source, Mapping):
                        split_lookup = split_source
                if isinstance(split_lookup, Mapping):
                    split_entry = split_lookup.get(record_canon)
                    if not isinstance(split_entry, Mapping) and canon_key and canon_key != record_canon:
                        split_entry = split_lookup.get(canon_key)
                    if not isinstance(split_entry, Mapping):
                        alt_key = _canonical_bucket_key(record_canon)
                        if alt_key and alt_key != record_canon:
                            split_entry = split_lookup.get(alt_key)
                    if isinstance(split_entry, Mapping):
                        machine_split = _safe_float(split_entry.get("machine_hours"))
                        labor_split = _safe_float(split_entry.get("labor_hours"))
                        if machine_split > 0.0 and labor_split <= 0.0:
                            meta_entry = self._lookup_meta(record_canon) or self._lookup_meta(display_label)
                            if isinstance(meta_entry, Mapping):
                                base_extra_val = _safe_float(meta_entry.get("base_extra"))
                                meta_rate_val = _safe_float(meta_entry.get("rate"))
                                if base_extra_val > 0.0 and meta_rate_val > 0.0:
                                    planner_rate_override = meta_rate_val
            if planner_rate_override and planner_rate_override > 0.0:
                rate_for_detail = planner_rate_override
            if rate_for_detail > 0.0:
                simple_hours_line = f"{hours_val:.2f} hr @ ${rate_for_detail:.2f}/hr"
        if simple_hours_line and simple_hours_line not in detail_parts:
            detail_parts.append(simple_hours_line)
        self._rows.append(
            {
                "label": display_label,
                "hours": hours_val,
                "rate": rate_val,
                "cost": cost_val,
                "canon_key": record_canon,
                "rate_display": rate_display,
            }
        )
        self.add_labor_cost_line(
            display_label,
            cost,
            process_key=canon_key,
            detail_bits=detail_parts if detail_parts else None,
        )
        try:
            self.labor_costs_display[display_label] = float(cost or 0.0)
        except Exception:
            self.labor_costs_display[display_label] = 0.0
    def update_row(
        self,
        canon_key: str,
        *,
        hours: float | None = None,
        rate: float | None = None,
        cost: float | None = None,
    ) -> None:
        index = self._index.get(canon_key)
        if index is None or index < 0 or index >= len(self.rows):
            return
        record = self.rows[index]
        row_dict = self._rows[index]

        if hours is not None:
            try:
                hours_val = float(hours)
            except Exception:
                hours_val = 0.0
            record.hours = hours_val
            row_dict["hours"] = hours_val
        if rate is not None:
            try:
                rate_val = float(rate)
            except Exception:
                rate_val = 0.0
            record.rate = rate_val
            row_dict["rate"] = rate_val
        if cost is not None:
            try:
                cost_val = float(cost)
            except Exception:
                cost_val = 0.0
            record.total = cost_val
            row_dict["cost"] = cost_val
        hours_for_display = row_dict.get("hours", 0.0)
        rate_display = _display_rate_for_row(
            canon_key or record.name,
            cfg=self.cfg,
            render_state=self.bucket_state,
            hours=float(hours_for_display or 0.0),
        )
        row_dict["rate_display"] = rate_display


def _is_planner_meta(key: str) -> bool:
    canonical_key = _canonical_bucket_key(key)
    if not canonical_key:
        return False
    return canonical_key.startswith("planner_") or canonical_key == "planner_total"


def _merge_process_meta(
    existing: Mapping[str, Any] | None, incoming: Mapping[str, Any] | Any
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
    incoming_map: Mapping[str, Any]
    if isinstance(incoming, Mapping):
        incoming_map = incoming
    else:
        incoming_map = {}

    if not incoming_map:
        return merged

    existing_hr = _safe_float(merged.get("hr"))
    existing_minutes = _safe_float(merged.get("minutes"))
    existing_extra = _safe_float(merged.get("base_extra"))
    existing_cost = _safe_float(merged.get("cost"))
    existing_rate = _safe_float(merged.get("rate"))

    incoming_minutes = _safe_float(incoming_map.get("minutes"))
    incoming_hr = _safe_float(incoming_map.get("hr"))
    if incoming_hr <= 0 and incoming_minutes > 0:
        incoming_hr = incoming_minutes / 60.0

    incoming_extra = _safe_float(incoming_map.get("base_extra"))
    incoming_cost = _safe_float(incoming_map.get("cost"))
    incoming_rate = _safe_float(incoming_map.get("rate"))

    if incoming_cost <= 0 and incoming_rate > 0 and incoming_hr > 0:
        incoming_cost = incoming_rate * incoming_hr

    total_minutes = existing_minutes + incoming_minutes
    if total_minutes > 0:
        merged["minutes"] = total_minutes
    elif "minutes" in merged:
        merged.pop("minutes", None)

    total_hr = existing_hr + incoming_hr
    if total_hr > 0:
        merged["hr"] = total_hr
    elif "hr" in merged:
        merged.pop("hr", None)

    total_extra = existing_extra + incoming_extra
    if abs(total_extra) > 1e-9:
        merged["base_extra"] = total_extra
    elif "base_extra" in merged:
        merged.pop("base_extra", None)

    if existing_cost <= 0 and existing_rate > 0 and existing_hr > 0:
        existing_cost = existing_rate * existing_hr
    total_cost = existing_cost + incoming_cost
    if total_cost > 0:
        merged["cost"] = total_cost
    elif "cost" in merged:
        merged.pop("cost", None)

    if total_hr > 0:
        if total_cost > 0:
            merged["rate"] = total_cost / total_hr
        elif incoming_rate > 0:
            merged["rate"] = incoming_rate
        elif existing_rate > 0:
            merged["rate"] = existing_rate
        else:
            merged.pop("rate", None)
    elif incoming_rate > 0:
        merged["rate"] = incoming_rate

    def _collect_notes(value: Any, dest: list[str], seen: set[str]) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text and text not in seen:
                dest.append(text)
                seen.add(text)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                text = str(item).strip()
                if text and text not in seen:
                    dest.append(text)
                    seen.add(text)

    notes: list[str] = []
    seen_notes: set[str] = set()
    _collect_notes(merged.get("notes"), notes, seen_notes)
    _collect_notes(incoming_map.get("notes"), notes, seen_notes)
    if notes:
        merged["notes"] = notes
    elif "notes" in merged:
        merged.pop("notes", None)

    special_keys = {"hr", "minutes", "base_extra", "cost", "rate", "notes"}
    for key, value in incoming_map.items():
        if key in special_keys:
            continue
        merged[key] = value

    return merged


def _fold_process_meta(
    meta_source: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    folded: dict[str, dict[str, Any]] = {}
    alias_map: dict[str, str] = {}
    if not isinstance(meta_source, Mapping):
        return {}, {}

    for raw_key, raw_meta in meta_source.items():
        alias_key = str(raw_key).lower().strip()
        if not alias_key:
            continue
        if _is_planner_meta(alias_key):
            folded[alias_key] = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
            continue

        canon_key = _canonical_bucket_key(raw_key) or alias_key
        alias_map.setdefault(alias_key, canon_key)
        existing = folded.get(canon_key)
        folded[canon_key] = _merge_process_meta(existing, raw_meta)

    result: dict[str, dict[str, Any]] = {key: value for key, value in folded.items()}
    for alias_key, canon_key in alias_map.items():
        result[alias_key] = result.get(canon_key, {})

    return result, alias_map


def _merge_applied_process_entries(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    notes: list[str] = []
    seen_notes: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        value_notes = entry.get("notes")
        if isinstance(value_notes, str):
            text = value_notes.strip()
            if text and text not in seen_notes:
                notes.append(text)
                seen_notes.add(text)
        elif isinstance(value_notes, (list, tuple, set)):
            for item in value_notes:
                text = str(item).strip()
                if text and text not in seen_notes:
                    notes.append(text)
                    seen_notes.add(text)
        for key, value in entry.items():
            if key == "notes":
                continue
            merged.setdefault(key, value)
    if notes:
        merged["notes"] = notes
    return merged


def _fold_applied_process(
    applied_source: Mapping[str, Any] | None, alias_map: Mapping[str, str]
) -> dict[str, Any]:
    base: dict[str, Any] = {}
    if isinstance(applied_source, Mapping):
        base = {str(k).lower().strip(): (v or {}) for k, v in applied_source.items()}
    if not alias_map:
        return base

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for alias_key, canon_key in alias_map.items():
        entry = base.get(alias_key)
        if isinstance(entry, Mapping):
            grouped.setdefault(canon_key, []).append(entry)

    for canon_key, entries in grouped.items():
        merged_entry = _merge_applied_process_entries(entries)
        base[canon_key] = merged_entry
        for alias_key, alias_canon in alias_map.items():
            if alias_canon == canon_key:
                base[alias_key] = merged_entry

    return base


def _lookup_process_meta(
    process_meta: Mapping[str, Any] | None, key: str | None
) -> Mapping[str, Any] | None:
    if not isinstance(process_meta, Mapping):
        return None
    candidates: list[str] = []
    base = str(key or "").lower()
    if base:
        candidates.append(base)
    canon = _canonical_bucket_key(key)
    if canon and canon not in candidates:
        candidates.append(canon)
    variants: list[str] = []
    for candidate in list(candidates):
        if "_" in candidate:
            variants.append(candidate.replace("_", " "))
        if " " in candidate:
            variants.append(candidate.replace(" ", "_"))
    seen: set[str] = set()
    for candidate in candidates + variants:
        candidate_key = candidate.strip()
        if not candidate_key or candidate_key in seen:
            continue
        seen.add(candidate_key)
        meta_entry = process_meta.get(candidate_key)
        if isinstance(meta_entry, Mapping):
            return meta_entry
    return None
