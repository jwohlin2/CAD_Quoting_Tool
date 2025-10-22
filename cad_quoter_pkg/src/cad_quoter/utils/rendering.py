"""Rendering and formatting utilities used across the CAD Quoter project."""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none

ELLIPSIS = "…"
MISSING_VALUE = "—"
DEFAULT_WIDTH = 114


# ---------------------------------------------------------------------------
# Basic formatting helpers
# ---------------------------------------------------------------------------


def ellipsize(text: object, width: int) -> str:
    """Clamp ``text`` to ``width`` characters using a single ellipsis if needed."""

    if width <= 0:
        return ""
    clean = text if isinstance(text, str) else str(text)
    if len(clean) <= width:
        return clean
    if width == 1:
        return ELLIPSIS
    return f"{clean[: width - 1]}{ELLIPSIS}"


def fmt_money(value: Any, currency: str) -> str:
    """Return *value* formatted as a currency string."""

    try:
        amount = float(value or 0.0)
    except Exception:
        amount = 0.0
    return f"{currency}{amount:,.2f}"


def money(value: float | int | None, currency: str = "$", *, missing: str = MISSING_VALUE) -> str:
    """Format ``value`` as a money string with thousands separators."""

    if value is None:
        return missing
    try:
        number = float(value)
    except Exception:
        return missing
    formatted = fmt_money(abs(number), currency)
    return f"-{formatted}" if number < 0 else formatted


def fmt_percent(value: Any, *, decimals: int = 1) -> str:
    """Return ``value`` as a percentage with a configurable precision."""

    try:
        pct_value = float(value or 0.0)
    except Exception:
        pct_value = 0.0
    return f"{pct_value * 100:.{decimals}f}%"


def pct(value: float | int | None, *, decimals: int = 1, missing: str = MISSING_VALUE) -> str:
    """Format a ratio (0.0–1.0) or percentage (0–100) with the desired precision."""

    if value is None:
        return missing
    try:
        number = float(value)
    except Exception:
        return missing
    ratio = number / 100.0 if abs(number) > 1.0 else number
    formatted = fmt_percent(ratio, decimals=decimals)
    return formatted if formatted else missing


def fmt_hours(
    value: Any,
    *,
    unit: str = "hr",
    include_unit: bool = True,
    decimals: int = 2,
) -> str:
    """Format an hour value with an optional unit suffix."""

    try:
        hours = float(value or 0.0)
    except Exception:
        hours = 0.0
    hours_text = f"{max(hours, 0.0):.{decimals}f}"
    if include_unit and unit:
        return f"{hours_text} {unit}"
    return hours_text


def fmt_range(
    lower: Any,
    upper: Any,
    *,
    formatter: Callable[[Any], str] | None = None,
    separator: str = "–",
    unit: str | None = None,
) -> str:
    """Render a range where both ends share the same formatting."""

    format_value = formatter or (lambda value: str(value))
    lower_text = format_value(lower)
    upper_text = format_value(upper)
    if unit:
        return f"{lower_text}{separator}{upper_text} {unit}".rstrip()
    return f"{lower_text}{separator}{upper_text}"


def format_currency(value: Any, currency: str) -> str:
    """Return *value* formatted as a currency string."""

    return fmt_money(value, currency)


def format_hours(value: Any) -> str:
    """Format an hour value with a ``hr`` suffix."""

    return fmt_hours(value)


def format_hours_with_rate(hours: Any, rate: Any, currency: str) -> str:
    """Return a human readable ``hours × rate`` string."""

    try:
        hours_val = float(hours or 0.0)
    except Exception:
        hours_val = 0.0
    try:
        rate_val = float(rate or 0.0)
    except Exception:
        rate_val = 0.0
    hours_text = fmt_hours(hours_val)
    if rate_val <= 0:
        return f"{hours_text} @ —/hr"
    rate_text = fmt_money(rate_val, currency)
    return f"{hours_text} @ {rate_text}/hr"


def format_percent(value: Any) -> str:
    """Return ``value`` as a percentage with a single decimal place."""

    return fmt_percent(value)


def format_dimension(value: Any) -> str:
    """Render numeric dimensions while keeping existing text untouched."""

    if isinstance(value, (int, float)):
        text = f"{float(value):.3f}".rstrip("0").rstrip(".")
        return text or "0"
    if value is None:
        return "—"
    text = str(value).strip()
    return text if text else "—"


def format_weight_lb_decimal(mass_g: float | None) -> str:
    """Convert grams into decimal pounds."""

    if mass_g is None:
        grams = 0.0
    else:
        try:
            grams = float(mass_g or 0.0)
        except Exception:
            grams = 0.0
    grams = max(0.0, grams)
    pounds = grams / 1000.0 * 2.2046226218487757
    if pounds <= 0:
        return "0.00 lb"
    text = f"{pounds:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{text} lb"


def format_weight_lb_oz(mass_g: float | None) -> str:
    """Convert grams into a pounds/ounces breakdown."""

    if mass_g is None:
        grams = 0.0
    else:
        try:
            grams = float(mass_g or 0.0)
        except Exception:
            grams = 0.0
    grams = max(0.0, grams)
    if grams <= 0:
        return "0 oz"
    pounds_total = grams / 1000.0 * 2.2046226218487757
    total_ounces = pounds_total * 16.0
    pounds = int(total_ounces // 16)
    ounces = total_ounces - pounds * 16
    precision = 1 if pounds > 0 or ounces >= 1.0 else 2
    ounces = round(ounces, precision)
    if ounces >= 16.0:
        pounds += 1
        ounces = 0.0
    parts: list[str] = []
    if pounds > 0:
        parts.append(f"{pounds} lb" if pounds != 1 else "1 lb")
    if ounces > 0 or pounds == 0:
        ounce_text = f"{ounces:.{precision}f}".rstrip("0").rstrip(".")
        if not ounce_text:
            ounce_text = "0"
        parts.append(f"{ounce_text} oz")
    return " ".join(parts) if parts else "0 oz"


# ---------------------------------------------------------------------------
# ASCII table rendering
# ---------------------------------------------------------------------------


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

    def _pad_body(text: str, width: int, align: str) -> str:
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
                pieces.append(_pad_body(segment, col_widths[col_idx], aligns[col_idx]))
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


# ---------------------------------------------------------------------------
# Quote document helpers
# ---------------------------------------------------------------------------


@dataclass
class QuoteRow:
    """Single rendered line of quote output."""

    index: int
    text: str


@dataclass
class QuoteSection:
    """Group of related rows under an optional title."""

    title: str | None
    rows: list[QuoteRow] = field(default_factory=list)


@dataclass
class QuoteDoc:
    """Structured representation of a rendered quote."""

    title: str
    sections: list[QuoteSection] = field(default_factory=list)


class QuoteDocRecorder:
    """Track lines emitted by the legacy renderer and build a schema."""

    def __init__(self, divider: str) -> None:
        self._divider = divider
        self._title: str | None = None
        self._sections: list[QuoteSection] = []
        self._current_section: QuoteSection | None = None
        self._line_map: dict[int, QuoteRow] = {}

    def observe_line(self, index: int, line: str, previous: str | None) -> None:
        """Record a raw line emitted by the renderer."""

        if self._title is None:
            self._title = line
            return
        if line == self._divider and previous is not None:
            if self._current_section and self._current_section.rows:
                last_row = self._current_section.rows[-1]
                if last_row.text == previous:
                    self._line_map.pop(last_row.index, None)
                    self._current_section.rows.pop()
            section = QuoteSection(title=previous, rows=[])
            self._sections.append(section)
            self._current_section = section
            return
        if self._current_section is None:
            self._current_section = QuoteSection(title=None, rows=[])
            self._sections.append(self._current_section)
        row = QuoteRow(index=index, text=line)
        self._current_section.rows.append(row)
        self._line_map[index] = row

    def replace_line(self, index: int, text: str) -> None:
        if index == 0:
            self._title = text
            return
        row = self._line_map.get(index)
        if row is not None:
            row.text = text

    def build_doc(self) -> QuoteDoc:
        title = self._title or ""
        sections = [section for section in self._sections if section.rows or section.title]
        return QuoteDoc(title=title, sections=sections)


def render_quote_doc_lines(doc: QuoteDoc, *, divider: str) -> list[str]:
    """Return the rendered quote output as a list of text lines."""

    lines: list[str] = []
    if doc.title:
        lines.append(doc.title)
    for section in doc.sections:
        if section.title:
            if not lines or lines[-1] != section.title:
                lines.append(section.title)
            lines.append(divider)
        lines.extend(row.text for row in section.rows)
    return lines


def render_quote_doc(doc: QuoteDoc, *, divider: str) -> str:
    """Render a :class:`QuoteDoc` back into legacy text output."""

    return "\n".join(render_quote_doc_lines(doc, divider=divider))


def render_quote_doc_to_path(doc: QuoteDoc, *, divider: str, out_path: str | Path) -> None:
    """Persist rendered quote text to ``out_path`` using UTF-8 encoding."""

    lines = render_quote_doc_lines(doc, divider=divider)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Debug table helpers
# ---------------------------------------------------------------------------


def _jsonify_debug_value(value: Any, depth: int = 0, max_depth: int = 6) -> Any:
    """Convert debug structures to JSON-friendly primitives."""

    if depth >= max_depth:
        return None
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key): _jsonify_debug_value(val, depth + 1, max_depth)
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _jsonify_debug_value(item, depth + 1, max_depth)
            for item in value
        ]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "ignore")
        except Exception:
            return repr(value)
    if callable(value):
        try:
            return repr(value)
        except Exception:
            return "<callable>"
    try:
        coerced = float(value)
    except Exception:
        try:
            return str(value)
        except Exception:
            return None
    return coerced if math.isfinite(coerced) else None


def _jsonify_debug_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Safely serialize debugging metadata for JSON storage."""

    return {
        str(key): _jsonify_debug_value(value)
        for key, value in summary.items()
    }


def _accumulate_drill_debug(dest: list[str], *sources: Any) -> None:
    """Accumulate normalized drill debug entries from heterogeneous sources."""

    for source in sources:
        if source is None:
            continue
        if isinstance(source, Mapping):
            _accumulate_drill_debug(dest, source.get("drill_debug"))
            continue
        if isinstance(source, str):
            text = source.strip()
            if text and text not in dest:
                dest.append(text)
            continue
        if isinstance(source, (list, tuple, set)):
            for entry in source:
                _accumulate_drill_debug(dest, entry)
            continue
        try:
            text = str(source).strip()
        except Exception:
            text = ""
        if text and text not in dest:
            dest.append(text)


def _format_range(vals: Sequence[float | None]) -> str:
    values = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not values:
        return "?"
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-6:
        return f"{vmin:.0f}"
    return fmt_range(
        vmin,
        vmax,
        formatter=lambda value: f"{float(value):.0f}",
    )


def _format_range_f(vals: Sequence[float | None], prec: int = 2) -> str:
    values = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not values:
        return "?"
    vmin = min(values)
    vmax = max(values)
    tol = 10 ** (-prec)
    if abs(vmax - vmin) < tol:
        return f"{vmin:.{prec}f}"
    return fmt_range(
        vmin,
        vmax,
        formatter=lambda value: f"{float(value):.{prec}f}",
    )


def build_removal_debug_table(
    *,
    op_name: str,
    mat_canon: str,
    mat_group: str,
    row_group: str,
    sfm: float,
    ipr_bins: Sequence[float | None],
    rpm_bins: Sequence[float | None],
    ipm_bins: Sequence[float | None],
    dia_bins: Sequence[float | None],
    depth_bins: Sequence[float | None],
    holes: int,
    index_sec_per_hole: float | None,
    peck_min_per_hole: float | None,
    toolchange_min: float | None,
    total_minutes: float,
) -> str:
    lines: list[str] = []
    lines.append("Material Removal Debug")
    lines.append("-" * 74)
    mat_group_display = mat_group or "?"
    row_group_display = row_group or "?"
    lines.append(
        f"{op_name}  |  material: {mat_canon or '?'} (group {mat_group_display}, row {row_group_display})"
    )
    lines.append("")
    lines.append(f"  SFM: {sfm:.0f}   IPR: {_format_range_f(ipr_bins, 3)}")
    lines.append(f"  RPM: {_format_range(rpm_bins)}   IPM: {_format_range_f(ipm_bins, 1)}")
    lines.append(
        "  A~:   "
        f"{_format_range_f(dia_bins, 3)} in   depth/hole: {_format_range_f(depth_bins, 2)} in   holes: {int(holes)}"
    )
    ix_text = "?"
    if index_sec_per_hole is not None and math.isfinite(index_sec_per_hole):
        ix_text = f"{float(index_sec_per_hole):.1f} s/hole"
    peck_text = "?"
    if peck_min_per_hole is not None and math.isfinite(peck_min_per_hole):
        peck_text = f"{float(peck_min_per_hole):.2f} min/hole"
    toolchange_text = "?"
    if toolchange_min is not None and math.isfinite(toolchange_min):
        toolchange_text = f"{float(toolchange_min):.2f} min"
    lines.append(
        f"  overhead + index: {ix_text}   peck: {peck_text}   toolchange: {toolchange_text}"
    )
    lines.append("")
    lines.append(
        f"  subtotal time: {float(total_minutes):.1f} min  ({fmt_hours(float(total_minutes) / 60.0)})"
    )
    return "\n".join(lines)


def _render_drilling_debug_table(summary: dict[str, dict], label: str, out_lines: list[str]) -> None:
    if not isinstance(summary, dict) or not summary:
        return

    def _as_float(v: Any) -> float | None:
        try:
            f = float(v)
        except Exception:
            return None
        return f if math.isfinite(f) else None

    out_lines.append(f"\n{label}")
    out_lines.append("  " + ("-" * 110))
    for op_key, s in summary.items():
        if not isinstance(s, dict):
            continue

        mat = str(s.get("material") or s.get("material_display") or "").strip()
        group = str(s.get("material_group") or s.get("row_group") or "").strip()

        dia_w = _as_float(s.get("diameter_weight_sum"))
        dia_q = _as_float(s.get("diameter_qty_sum"))
        dia_avg = (dia_w / dia_q) if (dia_w is not None and dia_q and dia_q > 0) else None

        sfm_avg = None
        if _as_float(s.get("sfm_sum")) is not None and _as_float(s.get("sfm_count")):
            try:
                sfm_avg = float(s["sfm_sum"]) / float(s["sfm_count"])
            except Exception:
                sfm_avg = None
        if sfm_avg is None:
            sfm_avg = _as_float(s.get("sfm"))

        rpm_avg = None
        if _as_float(s.get("rpm_sum")) is not None and _as_float(s.get("rpm_count")):
            try:
                rpm_avg = float(s["rpm_sum"]) / float(s["rpm_count"])
            except Exception:
                rpm_avg = None
        if rpm_avg is None:
            rpm_avg = _as_float(s.get("rpm"))

        ipm_avg = None
        if _as_float(s.get("ipm_sum")) is not None and _as_float(s.get("ipm_count")):
            try:
                ipm_avg = float(s["ipm_sum"]) / float(s["ipm_count"])
            except Exception:
                ipm_avg = None
        if ipm_avg is None:
            ipm_avg = _as_float(s.get("ipm"))

        ipr_avg = None
        if _as_float(s.get("ipr_sum")) is not None and _as_float(s.get("ipr_count")):
            try:
                ipr_avg = float(s["ipr_sum"]) / float(s["ipr_count"])
            except Exception:
                ipr_avg = None
        if ipr_avg is None and ipm_avg is not None and rpm_avg and rpm_avg > 0:
            ipr_avg = ipm_avg / rpm_avg

        depth_avg = None
        if _as_float(s.get("depth_weight_sum")) is not None and _as_float(s.get("depth_qty_sum")):
            try:
                depth_avg = float(s["depth_weight_sum"]) / float(s["depth_qty_sum"])
            except Exception:
                depth_avg = None
        if depth_avg is None:
            dmin = _as_float(s.get("depth_min"))
            dmax = _as_float(s.get("depth_max"))
            if dmin is not None and dmax is not None:
                depth_avg = 0.5 * (dmin + dmax)

        mrr = None
        if ipm_avg is not None and dia_avg is not None and dia_avg > 0:
            try:
                area = math.pi * (0.5 * float(dia_avg)) ** 2
                mrr = float(ipm_avg) * area
            except Exception:
                mrr = None

        qty = int(_as_float(s.get("qty")) or 0)
        totm = float(_as_float(s.get("total_minutes")) or 0.0)

        sfm_txt = f"{sfm_avg:.0f}" if sfm_avg is not None else "?"
        ipm_txt = f"{ipm_avg:.1f}" if ipm_avg is not None else "?"
        mrr_txt = f"{mrr:.2f}" if mrr is not None else "?"
        group_txt = f" [{group}]" if group else ""
        mat_txt = (mat or "?") + group_txt
        out_lines.append(
            f"  {op_key}  | qty {qty} | {mat_txt} | SFM {sfm_txt} | F {ipm_txt} | MRR {mrr_txt} | {totm:.2f} min"
        )

        bins = s.get("bins", {})
        if isinstance(bins, dict):
            def _dia_of(b: dict) -> float:
                d = _as_float(b.get("diameter_in"))
                return d if d is not None else 0.0

            for bkey, b in sorted(bins.items(), key=lambda kv: _dia_of(kv[1]) if isinstance(kv[1], dict) else 0.0):
                if not isinstance(b, dict):
                    continue
                d = _as_float(b.get("diameter_in"))
                q = int(_as_float(b.get("qty")) or 0)
                m = _as_float(b.get("minutes"))

                _sfm = sfm_avg if sfm_avg is not None else _as_float(s.get("sfm"))
                _ipr = None
                if "ipr_sum" in s and _as_float(s.get("ipr_count")):
                    try:
                        _ipr = float(s["ipr_sum"]) / float(s["ipr_count"])
                    except Exception:
                        _ipr = None
                if (_ipr is None) and (ipm_avg is not None) and (rpm_avg and rpm_avg > 0):
                    _ipr = ipm_avg / rpm_avg
                _mat = (mat or "").strip()
                _group = (group or "").strip()
                if _group:
                    _mat = f"{_mat}"
                if (d is not None) and (depth_avg is not None) and (_sfm is not None) and (_ipr is not None):
                    out_lines.append(
                        f"    OK {op_key} dia={d:.3f} depth={depth_avg:.3f} qty={q} material={_mat} sfm={_sfm:.0f} ipr={_ipr:.4f}"
                    )
                d_txt = f"{d:.3f}" if d is not None else "?"
                m_txt = f"{m:.2f}" if m is not None else "?"
                out_lines.append(f"    - A~{d_txt} in A- {q} + {m_txt} min")
    out_lines.append("  " + ("-" * 110) + "\n")


def append_removal_debug_if_enabled(
    lines: list[str],
    summary: Mapping[str, Any] | None,
) -> None:
    """Append a compact, non-wrapping material-removal summary table when enabled."""

    if not isinstance(summary, Mapping):
        return

    def _as_float(value: Any) -> float | None:
        val = _coerce_float_or_none(value)
        if val is None:
            return None
        try:
            f = float(val)
        except Exception:
            return None
        return f if math.isfinite(f) else None

    def _avg(sum_key: str, count_key: str) -> float | None:
        total = _as_float(summary.get(sum_key))
        count = _as_float(summary.get(count_key))
        if total is None or count is None or count <= 0:
            return None
        return total / count

    def _weighted_avg(weight_key: str, qty_key: str) -> float | None:
        weight = _as_float(summary.get(weight_key))
        qty = _as_float(summary.get(qty_key))
        if weight is None or qty is None or qty <= 0:
            return None
        return weight / qty

    material_text = str(
        summary.get("material")
        or summary.get("material_display")
        or summary.get("mat_canon")
        or ""
    ).strip() or "?"

    dia_avg = _weighted_avg("diameter_weight_sum", "diameter_qty_sum")
    if dia_avg is None:
        dia_avg = _as_float(summary.get("diam_min")) or _as_float(summary.get("diam_max"))

    depth_avg = _weighted_avg("depth_weight_sum", "depth_qty_sum")
    if depth_avg is None:
        depth_avg = _as_float(summary.get("depth_min")) or _as_float(summary.get("depth_max"))

    sfm_avg = _avg("sfm_sum", "sfm_count") or _as_float(summary.get("sfm"))
    rpm_avg = _avg("rpm_sum", "rpm_count") or _as_float(summary.get("rpm"))
    if rpm_avg is None and sfm_avg and dia_avg and dia_avg > 0:
        rpm_avg = (sfm_avg * 12.0) / (math.pi * dia_avg)

    ipr_avg = _avg("ipr_sum", "ipr_count") or _as_float(summary.get("ipr"))
    ipm_avg = _avg("ipm_sum", "ipm_count") or _as_float(summary.get("ipm"))
    if ipm_avg is None and ipr_avg and rpm_avg:
        ipm_avg = ipr_avg * rpm_avg

    holes = int(_as_float(summary.get("qty")) or 0)
    base_minutes = _as_float(summary.get("total_minutes")) or 0.0
    toolchange_minutes = _as_float(summary.get("toolchange_total")) or 0.0
    total_minutes = base_minutes + toolchange_minutes
    per_hole_minutes = (total_minutes / holes) if holes > 0 else None

    def _fmt(val: float | None, fmt: str) -> str:
        return fmt.format(val) if (val is not None and math.isfinite(val)) else "?"

    lines.append("Material Removal Debug")
    lines.append("-" * 74)
    lines.append(f"  Material: {material_text}")
    lines.append(
        "  Tool A~: "
        f"{_fmt(dia_avg, '{:.3f} in')}   Avg depth: {_fmt(depth_avg, '{:.2f} in')}   Holes: {holes}"
    )
    lines.append(
        "  SFM+RPM: "
        f"{_fmt(sfm_avg, '{:.0f}')} + {_fmt(rpm_avg, '{:.0f}')}   IPR: {_fmt(ipr_avg, '{:.4f}')}   IPM: {_fmt(ipm_avg, '{:.1f}')}"
    )
    lines.append(
        "  Time: "
        f"{_fmt(per_hole_minutes, '{:.2f}')} min/hole   Total: {_fmt(total_minutes, '{:.1f}')} min"
    )
    lines.append("")


__all__ = [
    "ELLIPSIS",
    "MISSING_VALUE",
    "DEFAULT_WIDTH",
    "ColumnSpec",
    "QuoteDoc",
    "QuoteDocRecorder",
    "QuoteRow",
    "QuoteSection",
    "append_removal_debug_if_enabled",
    "ascii_table",
    "build_removal_debug_table",
    "draw_boxed_table",
    "draw_kv_table",
    "ellipsize",
    "fmt_hours",
    "fmt_money",
    "fmt_percent",
    "fmt_range",
    "format_currency",
    "format_dimension",
    "format_hours",
    "format_hours_with_rate",
    "format_percent",
    "format_weight_lb_decimal",
    "format_weight_lb_oz",
    "money",
    "pct",
    "render_quote_doc",
    "render_quote_doc_lines",
    "render_quote_doc_to_path",
    "_accumulate_drill_debug",
    "_jsonify_debug_summary",
    "_jsonify_debug_value",
]
