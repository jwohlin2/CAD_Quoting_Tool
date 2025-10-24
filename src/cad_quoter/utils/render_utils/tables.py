"""ASCII table rendering helpers used across quote output builders."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import math

from cad_quoter.domain_models.values import safe_float as _safe_float

from . import ellipsize

DEFAULT_WIDTH = 114


@dataclass(frozen=True)
class ColumnSpec:
    """Describe a table column."""

    width: int
    align: str = "L"
    header_align: str | None = None


def _coerce_alignment(value: str) -> str:
    upper = (value or "L").upper()
    if upper not in {"L", "C", "R"}:
        upper = "L"
    return upper


def _pad(text: str, width: int, align: str) -> str:
    truncated = ellipsize(text, width)
    pad = max(width - len(truncated), 0)
    if align == "R":
        return " " * pad + truncated
    if align == "C":
        left = pad // 2
        right = pad - left
        return " " * left + truncated + " " * right
    return truncated + " " * pad


def draw_boxed_table(
    headers: Sequence[str] | None,
    rows: Sequence[Sequence[str]],
    colspecs: Sequence[ColumnSpec],
) -> str:
    """Render a fixed-width ASCII table with box-drawing borders."""

    if any(spec.width <= 0 for spec in colspecs):
        raise ValueError("column widths must be positive")
    column_count = len(colspecs)
    if headers and len(headers) != column_count:
        raise ValueError("header count must match column specification")
    for row in rows:
        if len(row) != column_count:
            raise ValueError("row does not match column specification")

    horizontal = "+" + "+".join("-" * spec.width for spec in colspecs) + "+"

    def _render_row(cells: Sequence[str], *, header: bool = False) -> str:
        formatted: list[str] = []
        for idx, cell in enumerate(cells):
            spec = colspecs[idx]
            align = spec.header_align if header and spec.header_align else spec.align
            align = _coerce_alignment(align)
            formatted.append(_pad(str(cell), spec.width, align))
        return "|" + "|".join(formatted) + "|"

    output: list[str] = [horizontal]
    if headers:
        output.append(_render_row(headers, header=True))
        output.append(horizontal)
    for row in rows:
        output.append(_render_row(row))
    output.append(horizontal)
    return "\n".join(output)


def draw_kv_table(
    pairs: Iterable[tuple[str, str]],
    left_width: int,
    right_width: int,
    *,
    left_align: str = "L",
    right_align: str = "R",
) -> str:
    """Convenience wrapper for two-column key/value tables."""

    colspecs = (
        ColumnSpec(left_width, left_align),
        ColumnSpec(right_width, right_align),
    )
    rows = list(pairs)
    return draw_boxed_table(None, rows, colspecs)


def ascii_table(
    headers: Sequence[str] | None,
    rows: Sequence[Sequence[object]],
    *,
    col_widths: Sequence[int],
    col_aligns: Sequence[str] | None = None,
    header_aligns: Sequence[str] | None = None,
) -> str:
    """Render an ASCII table with optional headers and automatic wrapping."""

    column_count = len(col_widths)
    if column_count == 0:
        raise ValueError("at least one column is required")
    if any(width <= 0 for width in col_widths):
        raise ValueError("column widths must be positive")

    def _normalize_alignments(values: Sequence[str] | None, fallback: str) -> list[str]:
        result: list[str] = []
        for idx in range(column_count):
            raw = fallback
            if values and idx < len(values) and values[idx]:
                raw = str(values[idx])
            token = raw.strip().upper()[0] if raw else "L"
            result.append(token if token in {"L", "C", "R"} else "L")
        return result

    body_aligns = _normalize_alignments(col_aligns, "L")
    header_aligns = _normalize_alignments(header_aligns, "C")

    def _wrap_cell(value: object, width: int) -> list[str]:
        text = "" if value is None else str(value)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines: list[str] = []
        segments = text.split("\n") or [""]
        for segment in segments:
            clean = segment.strip()
            wrapped = textwrap.wrap(
                clean,
                width=width,
                break_long_words=True,
                break_on_hyphens=False,
                drop_whitespace=False,
            )
            if not wrapped:
                lines.append("")
            else:
                lines.extend(wrapped)
        return lines or [""]

    def _pad(text: str, width: int, align: str) -> str:
        pad = max(width - len(text), 0)
        if align == "R":
            return " " * pad + text
        if align == "C":
            left = pad // 2
            right = pad - left
            return " " * left + text + " " * right
        return text + " " * pad

    def _render_row(cells: Sequence[object], aligns: Sequence[str]) -> list[str]:
        wrapped_cells = [_wrap_cell(cell, col_widths[idx]) for idx, cell in enumerate(cells)]
        height = max((len(cell_lines) for cell_lines in wrapped_cells), default=1)
        lines: list[str] = []
        for line_idx in range(height):
            pieces: list[str] = []
            for col_idx in range(column_count):
                cell_lines = wrapped_cells[col_idx]
                segment = cell_lines[line_idx] if line_idx < len(cell_lines) else ""
                pieces.append(_pad(segment, col_widths[col_idx], aligns[col_idx]))
            lines.append("|" + "|".join(pieces) + "|")
        return lines

    horizontal = "+" + "+".join("-" * width for width in col_widths) + "+"
    output: list[str] = [horizontal]

    if headers:
        if len(headers) != column_count:
            raise ValueError("header count must match column specification")
        output.extend(_render_row(headers, header_aligns))
        output.append(horizontal)

    for row in rows:
        if len(row) != column_count:
            raise ValueError("row does not match column specification")
        output.extend(_render_row(row, body_aligns))

    output.append(horizontal)
    return "\n".join(output)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def render_process_sections(
    bucket_view_obj: Mapping[str, Any] | None,
    *,
    process_meta: Mapping[str, Any] | None,
    rates: Mapping[str, Any] | None,
    cfg: Any | None,
    label_overrides: Mapping[str, Any] | None,
    format_money: Callable[[Any], str],
    add_process_notes: Callable[[str], None],
    config_sources: Sequence[Mapping[str, Any]] | None = None,
    labor_cost_totals: Mapping[str, Any] | None = None,
    programming_minutes: Any | None = None,
    page_width: int = 74,
    canonical_bucket_key: Callable[[str | None], str | None] | None = None,
    normalize_bucket_key: Callable[[str | None], str] | None = None,
    display_bucket_label: Callable[[str, Mapping[str, Any] | None], str] | None = None,
    programming_per_part_label: str = "Programming (per part)",
    programming_amortized_label: str | None = None,
) -> tuple[
    list[str],
    float,
    float,
    list[tuple[str, float, float, float, float]],
]:
    """Render process buckets into text rows and return diagnostics.

    The helper mirrors the legacy behaviour used inside ``render_quote`` while
    isolating the ASCII-table construction so it can be unit-tested.  It
    returns the rendered lines for the section, the summed total cost,
    aggregated minutes, and the raw bucket rows for optional diagnostics.
    """

    try:
        buckets_candidate = bucket_view_obj.get("buckets") if bucket_view_obj else None
    except Exception:
        buckets_candidate = None
    if isinstance(buckets_candidate, dict):
        buckets: Mapping[str, Any] = buckets_candidate
    elif isinstance(buckets_candidate, Mapping):
        try:
            buckets = dict(buckets_candidate)
        except Exception:
            buckets = {}
    else:
        buckets = {}

    canonical_bucket_key_fn = (
        canonical_bucket_key if canonical_bucket_key is not None else lambda key: str(key or "").lower()
    )
    normalize_bucket_key_fn = (
        normalize_bucket_key if normalize_bucket_key is not None else lambda key: str(key or "").lower()
    )
    display_bucket_label_fn = (
        display_bucket_label if display_bucket_label is not None else lambda key, overrides: key or ""
    )
    programming_amortized_label = programming_amortized_label or programming_per_part_label

    def _lookup_process_meta_local(key: str | None) -> Mapping[str, Any] | None:
        if not isinstance(process_meta, Mapping):
            return None
        candidates: list[str] = []
        base = str(key or "").lower()
        if base:
            candidates.append(base)
        canon = canonical_bucket_key_fn(key)
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

    order = [
        "programming",
        "programming_amortized",
        "milling",
        "turning",
        "drilling",
        "tapping",
        "counterbore",
        "countersink",
        "spot_drill",
        "grinding",
        "jig_grind",
        "finishing_deburr",
        "saw_waterjet",
        "wire_edm",
        "sinker_edm",
        "assembly",
        "inspection",
    ]

    canonical_entries: dict[str, dict[str, float]] = {}
    if isinstance(buckets, Mapping):
        for raw_key, raw_entry in buckets.items():
            if not isinstance(raw_entry, Mapping):
                continue
            key_str = str(raw_key)
            canon_key = (
                canonical_bucket_key_fn(key_str)
                or normalize_bucket_key_fn(key_str)
                or key_str
            )
            minutes_val = max(0.0, _as_float(raw_entry.get("minutes"), 0.0))
            machine_val = max(0.0, _as_float(raw_entry.get("machine$"), 0.0))
            labor_val = max(0.0, _as_float(raw_entry.get("labor$"), 0.0))
            total_val = max(0.0, _as_float(raw_entry.get("total$"), 0.0))
            if total_val <= 0.0:
                total_val = round(machine_val + labor_val, 2)
            canonical_entries[canon_key] = {
                "minutes": minutes_val,
                "machine$": machine_val,
                "labor$": labor_val,
                "total$": total_val,
            }

    milling_entry = canonical_entries.get("milling")
    if milling_entry:
        milling_meta = _lookup_process_meta_local("milling") or {}

        def _maybe_float(value: Any) -> float | None:
            try:
                number = float(value)
            except Exception:
                return None
            if not math.isfinite(number):
                return None
            return number

        milling_minutes = _safe_float(milling_entry.get("minutes"), default=0.0)
        meta_minutes = _safe_float(milling_meta.get("minutes"), default=0.0)
        meta_hours = _safe_float(milling_meta.get("hr"), default=0.0)
        if meta_minutes > 0.0:
            milling_minutes = meta_minutes
        elif meta_hours > 0.0:
            milling_minutes = meta_hours * 60.0

        if milling_minutes > 0.0:
            milling_hours = milling_minutes / 60.0

            def _rate_from_candidates(
                mapping: Mapping[str, Any] | None,
                keys: Sequence[str],
                default: float,
            ) -> float:
                if not isinstance(mapping, Mapping):
                    mapping = {}
                for key in keys:
                    if not key:
                        continue
                    try:
                        raw = mapping.get(key)  # type: ignore[index]
                    except Exception:
                        raw = None
                    rate_val = _maybe_float(raw)
                    if rate_val is not None and rate_val > 0.0:
                        return rate_val
                return default

            machine_rate = _rate_from_candidates(
                rates,
                (
                    "machine_per_hour",
                    "machine_rate",
                    "milling_rate",
                    "MachineRate",
                    "MillingRate",
                    "ShopMachineRate",
                    "ShopRate",
                ),
                90.0,
            )
            labor_rate = _rate_from_candidates(
                rates,
                (
                    "labor_per_hour",
                    "labor_rate",
                    "milling_labor_rate",
                    "LaborRate",
                    "ShopLaborRate",
                ),
                45.0,
            )

            if cfg is not None:
                try:
                    cfg_machine = _maybe_float(getattr(cfg, "machine_rate_per_hr", None))
                except Exception:
                    cfg_machine = None
                if cfg_machine is not None and cfg_machine > 0.0:
                    machine_rate = cfg_machine
                try:
                    cfg_labor = _maybe_float(getattr(cfg, "labor_rate_per_hr", None))
                except Exception:
                    cfg_labor = None
                if cfg_labor is not None and cfg_labor > 0.0:
                    labor_rate = cfg_labor

            resolved_config_sources: list[Mapping[str, Any]] = []

            def _add_config_source(candidate: Any) -> None:
                if isinstance(candidate, dict):
                    resolved_config_sources.append(candidate)
                elif isinstance(candidate, Mapping):
                    resolved_config_sources.append(dict(candidate))

            if config_sources:
                for source in config_sources:
                    _add_config_source(source)

            attended_frac: float | None = None
            try:
                cfg_frac = _maybe_float(getattr(cfg, "milling_attended_fraction", None))
            except Exception:
                cfg_frac = None
            if cfg_frac is not None:
                attended_frac = cfg_frac

            for source in resolved_config_sources:
                try:
                    candidate = source.get("milling_attended_fraction")
                except Exception:
                    candidate = None
                frac_val = _maybe_float(candidate)
                if frac_val is not None:
                    attended_frac = frac_val
                    break

            if attended_frac is None:
                attended_frac = 1.0
            attended_frac = max(0.0, min(attended_frac, 1.0))
            milling_labor_hours = milling_hours * attended_frac

            machine_cost = milling_hours * machine_rate
            labor_cost = milling_labor_hours * labor_rate
            total_cost = machine_cost + labor_cost

            milling_entry["minutes"] = round(milling_minutes, 2)
            milling_entry["machine$"] = round(machine_cost, 2)
            milling_entry["labor$"] = round(labor_cost, 2)
            milling_entry["total$"] = round(total_cost, 2)
            canonical_entries["milling"] = milling_entry

    section_lines: list[str] = []
    section_lines.append("Process & Labor Costs")
    divider_width = max(0, page_width)
    if divider_width <= 0:
        divider_width = 74
    section_lines.append("-" * divider_width)
    rows: list[tuple[str, float, float, float, float]] = []

    programming_entry: dict[str, float] | None = None
    programming_entry_label = programming_per_part_label
    for candidate in ("programming_amortized", "programming"):
        entry = canonical_entries.pop(candidate, None)
        if entry is not None:
            programming_entry = entry
            programming_entry_label = (
                programming_amortized_label
                if candidate == "programming_amortized"
                else programming_per_part_label
            )
            break

    prog_minutes = 0.0
    prog_total = 0.0
    if programming_entry is not None:
        prog_minutes = programming_entry.get("minutes", 0.0)
        prog_total = programming_entry.get("total$", 0.0)
        if prog_total <= 0.0:
            prog_total = programming_entry.get("labor$", 0.0)
    if prog_total <= 0.0 and isinstance(labor_cost_totals, Mapping):
        prog_total = max(
            0.0,
            _as_float(labor_cost_totals.get(programming_per_part_label), 0.0),
        )
    if prog_minutes <= 0.0:
        prog_minutes = max(0.0, _as_float(programming_minutes, 0.0))
    if prog_total > 0.0 or prog_minutes > 0.0:
        rows.append(
            (
                programming_entry_label,
                prog_minutes,
                0.0,
                prog_total,
                prog_total,
            )
        )

    def _append_process_row(
        rows_list: list[tuple[str, float, float, float, float]],
        label: str,
        minutes_val: float,
        machine_val: float,
        labor_val: float,
        total_val: float,
    ) -> None:
        minutes_clean = max(0.0, _as_float(minutes_val, 0.0))
        machine_clean = max(0.0, _as_float(machine_val, 0.0))
        labor_clean = max(0.0, _as_float(labor_val, 0.0))
        total_clean = max(0.0, _as_float(total_val, 0.0))
        if total_clean <= 0.0:
            total_clean = round(machine_clean + labor_clean, 2)
        if (
            total_clean <= 0.0
            and machine_clean <= 0.0
            and labor_clean <= 0.0
            and minutes_clean <= 0.0
        ):
            return
        rows_list.append(
            (
                str(label),
                minutes_clean,
                machine_clean,
                labor_clean,
                total_clean,
            )
        )

    def _label_for_bucket(canon_key: str) -> str:
        display_label = display_bucket_label_fn(canon_key, label_overrides)
        if display_label:
            return display_label
        return canon_key or ""

    def _consume_entry(canon_key: str) -> None:
        entry = canonical_entries.pop(canon_key, None)
        if not entry:
            return
        minutes_val = entry.get("minutes", 0.0)
        machine_val = entry.get("machine$", 0.0)
        labor_val = entry.get("labor$", 0.0)
        total_val = entry.get("total$", 0.0)
        _append_process_row(
            rows,
            _label_for_bucket(canon_key),
            minutes_val,
            machine_val,
            labor_val,
            total_val,
        )

    for bucket_key in order:
        _consume_entry(bucket_key)

    if canonical_entries:
        for canon_key, entry in sorted(
            canonical_entries.items(),
            key=lambda item: _label_for_bucket(item[0]).lower(),
        ):
            minutes_val = entry.get("minutes", 0.0)
            machine_val = entry.get("machine$", 0.0)
            labor_val = entry.get("labor$", 0.0)
            total_val = entry.get("total$", 0.0)
            _append_process_row(
                rows,
                _label_for_bucket(canon_key),
                minutes_val,
                machine_val,
                labor_val,
                total_val,
            )

    row_canon_keys = {
        canonical_bucket_key_fn(row_label) or normalize_bucket_key_fn(row_label)
        for row_label, *_ in rows
    }
    if "tapping" not in row_canon_keys and isinstance(buckets, Mapping):
        tapping_bucket: Mapping[str, Any] | None = None
        for candidate_key in ("tapping", "Tapping"):
            entry = buckets.get(candidate_key)
            if isinstance(entry, Mapping):
                tapping_bucket = entry
                break
        if tapping_bucket is None:
            tapping_bucket = {}
        tapping_minutes = _as_float(tapping_bucket.get("minutes", 0.0), 0.0)
        tapping_machine = _as_float(tapping_bucket.get("machine$", 0.0), 0.0)
        tapping_labor = _as_float(tapping_bucket.get("labor$", 0.0), 0.0)
        tapping_total = _as_float(tapping_bucket.get("total$", 0.0), 0.0)
        if (
            tapping_minutes > 0.0
            or tapping_machine > 0.0
            or tapping_labor > 0.0
            or tapping_total > 0.0
        ):
            _append_process_row(
                rows,
                _label_for_bucket("tapping"),
                tapping_minutes,
                tapping_machine,
                tapping_labor,
                tapping_total,
            )

    total_cost = sum(row[4] for row in rows)
    total_minutes = sum(row[1] for row in rows)

    if rows:
        headers = ("Process", "Minutes", "Machine $", "Labor $", "Total $")
        display_rows: list[tuple[str, str, str, str, str]] = []
        for name, minutes_val, machine_val, labor_val, total_val in rows:
            display_rows.append(
                (
                    str(name),
                    f"{minutes_val:,.2f}",
                    format_money(machine_val),
                    format_money(labor_val),
                    format_money(total_val),
                )
            )

        total_row = ("Total", "", "", "", format_money(total_cost))
        width_candidates = display_rows + [total_row]
        col_widths = [len(header) for header in headers]
        for row_values in width_candidates:
            for idx, value in enumerate(row_values):
                col_widths[idx] = max(col_widths[idx], len(value))

        def _format_row(values: Sequence[str]) -> str:
            pieces: list[str] = []
            for idx, value in enumerate(values):
                align = "L" if idx == 0 else "R"
                width = col_widths[idx]
                if align == "L":
                    pieces.append(value.ljust(width))
                else:
                    pieces.append(value.rjust(width))
            return "  " + "  ".join(pieces)

        section_lines.append(_format_row(headers))
        section_lines.append("  " + "  ".join("-" * width for width in col_widths))
        for row in display_rows:
            section_lines.append(_format_row(row))
        section_lines.append("  " + "  ".join("-" * width for width in col_widths))
        section_lines.append(_format_row(total_row))
        for display_label, *_ in rows:
            add_process_notes(display_label)
        section_lines.append("")
    else:
        section_lines.append("  (no bucket data)")
        section_lines.append("")

    return section_lines, total_cost, total_minutes, rows


__all__ = [
    "ColumnSpec",
    "DEFAULT_WIDTH",
    "ascii_table",
    "draw_boxed_table",
    "draw_kv_table",
    "render_process_sections",
]
