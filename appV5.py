def append_removal_debug_if_enabled(
    lines: list[str],
    summary: _MappingABC[str, Any] | None,
) -> None:
    """Append a compact material-removal table when debug mode is active."""

    if not APP_ENV.llm_debug_enabled:
        return
    if not isinstance(summary, _MappingABC):
        return

    def _as_float(value: Any) -> float | None:
        coerced = _coerce_float_or_none(value)
        if coerced is None:
            return None
        try:
            numeric = float(coerced)
        except Exception:
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

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
    ).strip()
    if not material_text:
        material_text = "—"

    dia_avg = _weighted_avg("diameter_weight_sum", "diameter_qty_sum")
    if dia_avg is None:
        dia_avg = _as_float(summary.get("diam_min"))
        if dia_avg is None:
            dia_avg = _as_float(summary.get("diam_max"))

    depth_avg = _weighted_avg("depth_weight_sum", "depth_qty_sum")
    if depth_avg is None:
        depth_avg = _as_float(summary.get("depth_min"))
        if depth_avg is None:
            depth_avg = _as_float(summary.get("depth_max"))

    sfm_avg = _avg("sfm_sum", "sfm_count")
    if sfm_avg is None:
        sfm_avg = _as_float(summary.get("sfm"))

    rpm_avg = _avg("rpm_sum", "rpm_count")
    if rpm_avg is None:
        rpm_avg = _as_float(summary.get("rpm"))
    if rpm_avg is None and sfm_avg and dia_avg and dia_avg > 0:
        rpm_avg = (sfm_avg * 12.0) / (math.pi * dia_avg)

    ipr_avg = _avg("ipr_sum", "ipr_count")
    if ipr_avg is None:
        ipr_avg = _as_float(summary.get("ipr"))

    ipm_avg = _avg("ipm_sum", "ipm_count")
    if ipm_avg is None:
        ipm_avg = _as_float(summary.get("ipm"))
    if ipm_avg is None and ipr_avg and rpm_avg:
        ipm_avg = ipr_avg * rpm_avg

    holes = int(_as_float(summary.get("qty")) or 0)
    base_minutes = _as_float(summary.get("total_minutes")) or 0.0
    toolchange_minutes = _as_float(summary.get("toolchange_total")) or 0.0
    total_minutes = base_minutes + toolchange_minutes
    per_hole_minutes = total_minutes / holes if holes > 0 else None

    def _fmt(value: float | None, fmt: str) -> str:
        if value is None:
            return "—"
        return fmt.format(value)

    tool_dia_text = _fmt(dia_avg, "{:.3f} in")
    depth_text = _fmt(depth_avg, "{:.2f} in")
    sfm_text = "—"
    rpm_text = "—"
    if sfm_avg is not None:
        sfm_text = f"{sfm_avg:.0f}"
    if rpm_avg is not None:
        rpm_text = f"{rpm_avg:.0f}"
    ipr_text = "—"
    if ipr_avg is not None:
        ipr_text = f"{ipr_avg:.3f}"
    ipm_text = "—"
    if ipm_avg is not None:
        ipm_text = f"{ipm_avg:.1f}"
    per_hole_text = _fmt(per_hole_minutes, "{:.2f}")
    total_text = _fmt(total_minutes, "{:.1f}")
    holes_text = str(holes) if holes > 0 else "—"

    if lines and lines[-1] != "":
        lines.append("")
    lines.append("Material Removal Debug")
    lines.append(
        "Material              Tool Ø    Holes  Avg Depth  SFM→RPM     IPR/IPM      Min/Hole  Total Min"
    )
    lines.append(
        f"{material_text:<20} {tool_dia_text:>8} {holes_text:>7} {depth_text:>10} "
        f"{sfm_text:>6}→{rpm_text:<6} {ipr_text:>7}/{ipm_text:<7} {per_hole_text:>8} {total_text:>10}"
    )
    lines.append("")


    append_removal_debug_if_enabled(lines, removal_summary_for_display)

    removal_summary_for_display: Mapping[str, Any] | None = None
        selected_summary_candidate = drill_debug_summary.get(selected_op_name)
        if isinstance(selected_summary_candidate, _MappingABC):
            removal_summary_for_display = typing.cast(
                Mapping[str, Any], selected_summary_candidate
            )
        if removal_summary_for_display is None:
            for summary in drill_debug_summary.values():
                if isinstance(summary, _MappingABC):
                    removal_summary_for_display = typing.cast(Mapping[str, Any], summary)
                    break
            removal_summary_for_display = typing.cast(
                Mapping[str, Any], removal_summary
            )
    parser.add_argument(
        "--debug-removal",
        action="store_true",
        help="Force-enable material removal debug output.",
    )
    if getattr(args, "debug_removal", False):
        global APP_ENV
        APP_ENV = replace(APP_ENV, llm_debug_enabled=True)

