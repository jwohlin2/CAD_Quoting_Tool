            lines,
            bins=bins,
            index_min=index_min,
            peck_min_deep=peck_min_deep,
            peck_min_std=peck_min_std,
            f"Subtotal (per-hole × qty) ............... {subtotal_min:.2f} min  ({subtotal_min/60:.2f} hr)"
            f"TOTAL DRILLING (with toolchange) ........ {subtotal_min + tool_add:.2f} min  ({(subtotal_min + tool_add)/60:.2f} hr)"
                            "bins_list": [],
                    sfm_candidate = _coerce_float_or_none(
                        bin_speed_snapshot.get("sfm") if bin_speed_snapshot else None
                    )
                    if sfm_candidate is None and sfm_float is not None and math.isfinite(sfm_float):
                        sfm_candidate = float(sfm_float)
                    if sfm_candidate is None:
                        sfm_candidate = _coerce_float_or_none(speeds_for_bin.get("sfm"))
                    if sfm_candidate is not None and math.isfinite(float(sfm_candidate)):
                        bin_summary["sfm"] = float(sfm_candidate)
                        bin_entry["sfm"] = float(sfm_candidate)
                    ipr_candidate = _coerce_float_or_none(
                        bin_speed_snapshot.get("ipr") if bin_speed_snapshot else None
                    )
                    if ipr_candidate is None and ipr_effective_float is not None and math.isfinite(ipr_effective_float):
                        ipr_candidate = float(ipr_effective_float)
                    if ipr_candidate is None and ipr_float is not None and math.isfinite(ipr_float):
                        ipr_candidate = float(ipr_float)
                    if ipr_candidate is None:
                        ipr_candidate = _coerce_float_or_none(speeds_for_bin.get("ipr"))
                    if ipr_candidate is None:
                        rpm_candidate = _coerce_float_or_none(
                            (bin_speed_snapshot.get("rpm") if bin_speed_snapshot else None)
                            or rpm_float
                        )
                        ipm_candidate = _coerce_float_or_none(
                            (bin_speed_snapshot.get("ipm") if bin_speed_snapshot else None)
                            or ipm_float
                        )
                        if (
                            rpm_candidate is not None
                            and ipm_candidate is not None
                            and math.isfinite(rpm_candidate)
                            and abs(float(rpm_candidate)) > 1e-9
                        ):
                            ratio = float(ipm_candidate) / float(rpm_candidate)
                            if math.isfinite(ratio):
                                ipr_candidate = float(ratio)
                    if ipr_candidate is not None and math.isfinite(float(ipr_candidate)):
                        bin_summary["ipr"] = float(ipr_candidate)
                        bin_entry["ipr"] = float(ipr_candidate)
                            "op": operation_name,
                            "op_name": operation_name,
                            "depth_weight_sum": 0.0,
                            "depth_qty_sum": 0,
                            "depth_in": None,
                            "sfm": None,
                            "ipr": None,
                    if not bin_summary.get("op"):
                        bin_summary["op"] = operation_name
                    bin_summary["op_name"] = operation_name

                    summary_bins_list = summary.setdefault("bins_list", [])
                    bin_entry: dict[str, Any] | None = None
                    for existing_entry in summary_bins_list:
                        if not isinstance(existing_entry, dict):
                            continue
                        if existing_entry.get("_bin_key") == bin_key:
                            bin_entry = existing_entry
                            break
                    if bin_entry is None:
                        bin_entry = {
                            "_bin_key": bin_key,
                            "op": operation_name,
                            "op_name": operation_name,
                            "diameter_in": float(tool_dia_in),
                            "qty": 0,
                            "depth_in": None,
                            "sfm": None,
                            "ipr": None,
                            "_depth_weight_sum": 0.0,
                            "_depth_qty_sum": 0,
                        }
                        summary_bins_list.append(bin_entry)

                    bin_entry_qty = int(bin_entry.get("qty", 0) or 0)
                    bin_entry["qty"] = bin_entry_qty + qty_for_debug
                        bin_summary["depth_weight_sum"] = (
                            float(bin_summary.get("depth_weight_sum", 0.0) or 0.0)
                            + float(depth_float) * qty_for_debug
                        )
                        bin_summary["depth_qty_sum"] = int(
                            (bin_summary.get("depth_qty_sum") or 0)
                        ) + qty_for_debug
                        bin_summary["depth_in"] = float(depth_float)
                        bin_entry["_depth_weight_sum"] = (
                            float(bin_entry.get("_depth_weight_sum", 0.0) or 0.0)
                            + float(depth_float) * qty_for_debug
                        )
                        bin_entry["_depth_qty_sum"] = int(
                            (bin_entry.get("_depth_qty_sum") or 0)
                        ) + qty_for_debug
                        try:
                            depth_qty_total = bin_entry["_depth_qty_sum"]
                            depth_weight_total = bin_entry["_depth_weight_sum"]
                            if depth_qty_total:
                                bin_entry["depth_in"] = float(depth_weight_total) / float(depth_qty_total)
                        except Exception:
                            pass
                raw_bins_list = summary.get("bins_list")
                cleaned_bins_list: list[dict[str, Any]] = []
                if isinstance(raw_bins_list, list):
                    for entry in raw_bins_list:
                        if not isinstance(entry, dict):
                            continue
                        bin_key = entry.get("_bin_key")
                        bin_source = None
                        if isinstance(bins_map, _MappingABC) and bin_key in bins_map:
                            candidate = bins_map.get(bin_key)
                            if isinstance(candidate, _MappingABC):
                                bin_source = candidate
                        qty_val = to_int(entry.get("qty")) or 0
                        dia_val = _coerce_float_or_none(entry.get("diameter_in"))
                        if dia_val is None or not math.isfinite(dia_val) or qty_val <= 0:
                            continue
                        depth_val = None
                        depth_weight = _coerce_float_or_none(entry.get("_depth_weight_sum"))
                        depth_qty = _coerce_float_or_none(entry.get("_depth_qty_sum"))
                        if depth_weight is not None and depth_qty and depth_qty > 0:
                            try:
                                depth_val = float(depth_weight) / float(depth_qty)
                            except Exception:
                                depth_val = None
                        if depth_val is None:
                            depth_val = _coerce_float_or_none(entry.get("depth_in"))
                        if depth_val is None and isinstance(bin_source, _MappingABC):
                            for key in ("depth_weight_sum", "depth_in", "depth_min", "depth_max"):
                                candidate_depth = _coerce_float_or_none(bin_source.get(key))
                                if candidate_depth is not None and math.isfinite(candidate_depth):
                                    depth_val = float(candidate_depth)
                                    break
                        if depth_val is None or not math.isfinite(depth_val):
                            continue
                        sfm_val = _coerce_float_or_none(entry.get("sfm"))
                        if sfm_val is None and isinstance(bin_source, _MappingABC):
                            speeds_map = bin_source.get("speeds")
                            if not isinstance(speeds_map, _MappingABC):
                                speeds_map = {}
                            sfm_val = _coerce_float_or_none(
                                bin_source.get("sfm")
                                or speeds_map.get("sfm")
                            )
                            if sfm_val is None:
                                rpm_candidate = _coerce_float_or_none(speeds_map.get("rpm"))
                                if rpm_candidate is not None and math.isfinite(rpm_candidate):
                                    try:
                                        sfm_candidate = (float(rpm_candidate) * math.pi * float(dia_val)) / 12.0
                                    except (TypeError, ValueError, ZeroDivisionError):
                                        sfm_candidate = None
                                    if sfm_candidate is not None and math.isfinite(sfm_candidate):
                                        sfm_val = float(sfm_candidate)
                        if sfm_val is None or not math.isfinite(sfm_val):
                            continue
                        ipr_val = _coerce_float_or_none(entry.get("ipr"))
                        if ipr_val is None and isinstance(bin_source, _MappingABC):
                            speeds_map = bin_source.get("speeds")
                            if not isinstance(speeds_map, _MappingABC):
                                speeds_map = {}
                            ipr_val = _coerce_float_or_none(
                                bin_source.get("ipr")
                                or speeds_map.get("ipr")
                            )
                            if ipr_val is None:
                                rpm_candidate = _coerce_float_or_none(
                                    speeds_map.get("rpm")
                                    or bin_source.get("rpm_min")
                                    or bin_source.get("rpm_max")
                                )
                                ipm_candidate = _coerce_float_or_none(
                                    speeds_map.get("ipm")
                                    or bin_source.get("ipm_min")
                                    or bin_source.get("ipm_max")
                                )
                                if (
                                    rpm_candidate is not None
                                    and ipm_candidate is not None
                                    and math.isfinite(rpm_candidate)
                                    and abs(float(rpm_candidate)) > 1e-9
                                ):
                                    ratio = float(ipm_candidate) / float(rpm_candidate)
                                    if math.isfinite(ratio):
                                        ipr_val = float(ratio)
                        if ipr_val is None or not math.isfinite(ipr_val):
                            continue
                        op_label = str(entry.get("op") or summary.get("operation") or op_key or "").strip()
                        cleaned_bins_list.append(
                            {
                                "op": op_label,
                                "diameter_in": float(dia_val),
                                "depth_in": float(depth_val),
                                "qty": int(qty_val),
                                "sfm": float(sfm_val),
                                "ipr": float(ipr_val),
                            }
                        )
                if cleaned_bins_list:
                    cleaned_bins_list.sort(key=lambda item: (float(item.get("diameter_in", 0.0)), str(item.get("op", ""))))
                summary["bins_list"] = cleaned_bins_list
    drill_bins_list: list[dict[str, Any]] = []
    if isinstance(drill_debug_summary, _MappingABC):
        for op_key, summary in drill_debug_summary.items():
            if not isinstance(summary, _MappingABC):
                continue
            op_default = str(summary.get("operation") or op_key or "").strip()
            bins_entries = summary.get("bins_list")
            if not isinstance(bins_entries, list):
                continue
            for entry in bins_entries:
                if not isinstance(entry, _MappingABC):
                    continue
                op_val = str(entry.get("op") or op_default or op_key or "").strip()
                dia_val = _coerce_float_or_none(entry.get("diameter_in"))
                depth_val = _coerce_float_or_none(entry.get("depth_in"))
                qty_val = to_int(entry.get("qty"))
                sfm_val = _coerce_float_or_none(entry.get("sfm"))
                ipr_val = _coerce_float_or_none(entry.get("ipr"))
                if (
                    op_val is None
                    or dia_val is None
                    or depth_val is None
                    or qty_val is None
                    or sfm_val is None
                    or ipr_val is None
                ):
                    continue
                if not math.isfinite(dia_val) or not math.isfinite(depth_val):
                    continue
                if not math.isfinite(sfm_val) or not math.isfinite(ipr_val):
                    continue
                try:
                    qty_int = int(qty_val)
                except Exception:
                    continue
                if qty_int <= 0:
                    continue
                drill_bins_list.append(
                    {
                        "op": op_val,
                        "diameter_in": float(dia_val),
                        "depth_in": float(depth_val),
                        "qty": qty_int,
                        "sfm": float(sfm_val),
                        "ipr": float(ipr_val),
                    }
                )
    if drill_bins_list:
        drill_bins_list.sort(
            key=lambda item: (
                str(item.get("op") or ""),
                float(item.get("diameter_in", 0.0)),
            )
        )

    if drill_bins_list:
        drilling_meta["bins_list"] = drill_bins_list
    drilling_meta = locals().get("drilling_meta") or {}
    bins_list = drilling_meta.get("bins_list") or []
    if bins_list:
        try:
            _render_removal_card(
                lines,
                mat_canon=drilling_meta["material_canonical"],
                mat_group=drilling_meta.get("material_group"),
                row_group=drilling_meta.get("row_material_group"),
                holes_deep=drilling_meta.get("holes_deep", 0),
                holes_std=drilling_meta.get("holes_std", 0),
                dia_vals_in=drilling_meta.get("dia_in_vals", []),
                depth_vals_in=drilling_meta.get("depth_in_vals", []),
                sfm_deep=drilling_meta.get("sfm_deep", 0),
                sfm_std=drilling_meta.get("sfm_std", 0),
                ipr_deep_vals=drilling_meta.get("ipr_deep_vals", []),
                ipr_std_val=drilling_meta.get("ipr_std_val", 0.0),
                rpm_deep_vals=drilling_meta.get("rpm_deep_vals", []),
                rpm_std_vals=drilling_meta.get("rpm_std_vals", []),
                ipm_deep_vals=drilling_meta.get("ipm_deep_vals", []),
                ipm_std_vals=drilling_meta.get("ipm_std_vals", []),
                index_min_per_hole=drilling_meta.get("index_min_per_hole", 0.13),
                peck_min_rng=drilling_meta.get("peck_min_per_hole_vals", [0.07, 0.08]),
                toolchange_min_deep=drilling_meta.get("toolchange_min_deep", 8.0),
                toolchange_min_std=drilling_meta.get("toolchange_min_std", 2.5),
            )
            subtotal_min, seen_deep, seen_std = _render_time_per_hole(
                lines,
                bins=bins_list,
                index_min=drilling_meta.get("index_min_per_hole", 0.13),
                peck_min_deep=min(drilling_meta.get("peck_min_per_hole_vals", [0.07, 0.08])),
                peck_min_std=max(drilling_meta.get("peck_min_per_hole_vals", [0.07, 0.08])),
            )
            tool_add = (
                (drilling_meta.get("toolchange_min_deep", 8.0) if seen_deep else 0.0)
                + (drilling_meta.get("toolchange_min_std", 2.5) if seen_std else 0.0)
            )
            lines.append(
                f"Toolchange adders: Deep-Drill {drilling_meta.get('toolchange_min_deep', 8.0):.2f} min + "
                f"Drill {drilling_meta.get('toolchange_min_std', 2.5):.2f} min = {tool_add:.2f} min"
            )
            lines.append("-" * 66)
            lines.append(
                f"Subtotal (per-hole × qty) ............... {subtotal_min:.2f} min  ({subtotal_min/60:.2f} hr)"
            )
            lines.append(
                f"TOTAL DRILLING (with toolchange) ........ {subtotal_min + tool_add:.2f} min  ({(subtotal_min + tool_add)/60:.2f} hr)"
            )
            lines.append("")
        except Exception:
            # If anything goes sideways here, do not break the quote – just skip this block.
            pass
