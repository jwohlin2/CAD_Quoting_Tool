from __future__ import annotations

import math
from typing import Any, Mapping as _MappingABC, Sequence

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none


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
    if isinstance(value, _MappingABC):
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


def _jsonify_debug_summary(summary: _MappingABC[str, Any]) -> dict[str, Any]:
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
        if isinstance(source, _MappingABC):
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
    return f"{vmin:.0f}–{vmax:.0f}"


def _format_range_f(vals: Sequence[float | None], prec: int = 2) -> str:
    values = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not values:
        return "?"
    vmin = min(values)
    vmax = max(values)
    tol = 10 ** (-prec)
    if abs(vmax - vmin) < tol:
        return f"{vmin:.{prec}f}"
    return f"{vmin:.{prec}f}–{vmax:.{prec}f}"


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
        f"  subtotal time: {float(total_minutes):.1f} min  ({float(total_minutes) / 60.0:.2f} hr)"
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

        # Material and grouping context
        mat = str(s.get("material") or s.get("material_display") or "").strip()
        group = str(s.get("material_group") or s.get("row_group") or "").strip()

        # Average dia from weights
        dia_w = _as_float(s.get("diameter_weight_sum"))
        dia_q = _as_float(s.get("diameter_qty_sum"))
        dia_avg = (dia_w / dia_q) if (dia_w is not None and dia_q and dia_q > 0) else None

        # Feeds/Speeds — favor aggregated averages if present
        sfm_avg = None
        if _as_float(s.get("sfm_sum")) is not None and _as_float(s.get("sfm_count")):
            try:
                sfm_avg = float(s["sfm_sum"]) / float(s["sfm_count"])  # type: ignore[index]
            except Exception:
                sfm_avg = None
        if sfm_avg is None:
            sfm_avg = _as_float(s.get("sfm"))

        rpm_avg = None
        if _as_float(s.get("rpm_sum")) is not None and _as_float(s.get("rpm_count")):
            try:
                rpm_avg = float(s["rpm_sum"]) / float(s["rpm_count"])  # type: ignore[index]
            except Exception:
                rpm_avg = None
        if rpm_avg is None:
            rpm_avg = _as_float(s.get("rpm"))

        ipm_avg = None
        if _as_float(s.get("ipm_sum")) is not None and _as_float(s.get("ipm_count")):
            try:
                ipm_avg = float(s["ipm_sum"]) / float(s["ipm_count"])  # type: ignore[index]
            except Exception:
                ipm_avg = None
        if ipm_avg is None:
            ipm_avg = _as_float(s.get("ipm"))

        ipr_avg = None
        if _as_float(s.get("ipr_sum")) is not None and _as_float(s.get("ipr_count")):
            try:
                ipr_avg = float(s["ipr_sum"]) / float(s["ipr_count"])  # type: ignore[index]
            except Exception:
                ipr_avg = None
        if ipr_avg is None and ipm_avg is not None and rpm_avg and rpm_avg > 0:
            ipr_avg = ipm_avg / rpm_avg

        depth_avg = None
        if _as_float(s.get("depth_weight_sum")) is not None and _as_float(s.get("depth_qty_sum")):
            try:
                depth_avg = float(s["depth_weight_sum"]) / float(s["depth_qty_sum"])  # type: ignore[index]
            except Exception:
                depth_avg = None
        if depth_avg is None:
            dmin = _as_float(s.get("depth_min"))
            dmax = _as_float(s.get("depth_max"))
            if dmin is not None and dmax is not None:
                depth_avg = 0.5 * (dmin + dmax)

        # MRR estimate (cu in/min) for drilling: ipm * area
        mrr = None
        if ipm_avg is not None and dia_avg is not None and dia_avg > 0:
            try:
                area = math.pi * (0.5 * float(dia_avg)) ** 2
                mrr = float(ipm_avg) * area
            except Exception:
                mrr = None

        qty = int(_as_float(s.get("qty")) or 0)
        totm = float(_as_float(s.get("total_minutes")) or 0.0)

        # Header line for this op
        sfm_txt = f"{sfm_avg:.0f}" if sfm_avg is not None else "?"
        ipm_txt = f"{ipm_avg:.1f}" if ipm_avg is not None else "?"
        mrr_txt = f"{mrr:.2f}" if mrr is not None else "?"
        group_txt = f" [{group}]" if group else ""
        mat_txt = (mat or "?") + group_txt
        out_lines.append(
            f"  {op_key}  | qty {qty} | {mat_txt} | SFM {sfm_txt} | F {ipm_txt} | MRR {mrr_txt} | {totm:.2f} min"
        )

        # Per-bin minutes
        bins = s.get("bins", {})
        if isinstance(bins, dict):
            # Sort bins by diameter
            def _dia_of(b: dict) -> float:
                d = _as_float(b.get("diameter_in"))
                return d if d is not None else 0.0

            for bkey, b in sorted(bins.items(), key=lambda kv: _dia_of(kv[1]) if isinstance(kv[1], dict) else 0.0):
                if not isinstance(b, dict):
                    continue
                d = _as_float(b.get("diameter_in"))
                q = int(_as_float(b.get("qty")) or 0)
                m = _as_float(b.get("minutes"))

                # Legacy OK line when enough context is present
                _sfm = sfm_avg if sfm_avg is not None else _as_float(s.get("sfm"))
                _ipr = None
                if "ipr_sum" in s and _as_float(s.get("ipr_count")):
                    try:
                        _ipr = float(s["ipr_sum"]) / float(s["ipr_count"])  # type: ignore[index]
                    except Exception:
                        _ipr = None
                if (_ipr is None) and (ipm_avg is not None) and (rpm_avg and rpm_avg > 0):
                    _ipr = ipm_avg / rpm_avg
                _mat = (mat or "").strip()
                _group = (group or "").strip()
                if _group:
                    _mat = f"{_mat}"
                if (d is not None) and (depth_avg is not None) and (_sfm is not None) and (_ipr is not None):
                    # Format: OK <op> dia=<in> depth=<in> qty=<n> material=<name> sfm=<n> ipr=<n>
                    out_lines.append(
                        f"    OK {op_key} dia={d:.3f} depth={depth_avg:.3f} qty={q} material={_mat} sfm={_sfm:.0f} ipr={_ipr:.4f}"
                    )
                d_txt = f"{d:.3f}" if d is not None else "?"
                m_txt = f"{m:.2f}" if m is not None else "?"
                out_lines.append(f"    - A~{d_txt} in A- {q} + {m_txt} min")
    out_lines.append("  " + ("-" * 110) + "\n")


def append_removal_debug_if_enabled(
    lines: list[str],
    summary: _MappingABC[str, Any] | None,
) -> None:
    """Append a compact, non-wrapping material-removal summary table when enabled."""

    if not isinstance(summary, _MappingABC):
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
    "_jsonify_debug_value",
    "_jsonify_debug_summary",
    "_accumulate_drill_debug",
    "_format_range",
    "build_removal_debug_table",
    "_render_drilling_debug_table",
    "append_removal_debug_if_enabled",
]
