        # Force estimator path regardless of planner signals
        used_planner = False
def _ensure_drilling_override(
    drill_hours: Any,
    *,
    planner_bucket_view: Mapping[str, Any] | None,
    canonical_bucket_rollup: _MutableMappingABC[str, Any] | None,
    process_meta: _MutableMappingABC[str, Any] | None,
    hour_summary_entries: _MutableMappingABC[str, tuple[float, bool]] | None,
    label_overrides: Mapping[str, str] | None = None,
) -> None:
    """Propagate estimator drilling hours into all planner-derived views."""

    try:
        hours = float(drill_hours or 0.0)
    except Exception:
        hours = 0.0
    if not math.isfinite(hours):
        hours = 0.0
    hours = max(0.0, hours)

    minutes_value = round(hours * 60.0, 2)
    minutes_for_meta = round(hours * 60.0, 1)
    hr_for_meta = round(hours, 2)
    hr_precise = round(hours, 3)

    if isinstance(canonical_bucket_rollup, _MutableMappingABC):
        existing_rollup = canonical_bucket_rollup.get("drilling")
        if isinstance(existing_rollup, _MutableMappingABC):
            try:
                existing_rollup["minutes"] = minutes_value
            except Exception:
                pass
            try:
                existing_rollup["hours"] = hr_precise
            except Exception:
                pass
        canonical_bucket_rollup["drilling"] = hours

    if isinstance(process_meta, _MutableMappingABC):
        drilling_meta = process_meta.get("drilling")
        if isinstance(drilling_meta, _MutableMappingABC):
            drilling_meta["hr"] = hr_for_meta
            drilling_meta["minutes"] = minutes_for_meta
        elif isinstance(drilling_meta, _MappingABC):
            new_meta = dict(drilling_meta)
            new_meta["hr"] = hr_for_meta
            new_meta["minutes"] = minutes_for_meta
            process_meta["drilling"] = new_meta
        else:
            process_meta["drilling"] = {
                "hr": hr_for_meta,
                "minutes": minutes_for_meta,
            }

    if isinstance(hour_summary_entries, _MutableMappingABC):
        label = _display_bucket_label("drilling", label_overrides) or "Drilling"
        include_flag = True
        existing = hour_summary_entries.get(label)
        if isinstance(existing, (list, tuple)) and existing:
            try:
                include_flag = bool(existing[1])
            except Exception:
                include_flag = True
        hour_summary_entries[label] = (hr_for_meta, include_flag)

    def _update_bucket_map(bucket_map: Mapping[str, Any] | None) -> None:
        if not isinstance(bucket_map, _MutableMappingABC):
            return
        target_key: str | None = None
        for key in list(bucket_map.keys()):
            if _canonical_bucket_key(key) == "drilling":
                target_key = key
                break
        if target_key is None:
            target_key = "Drilling"
        entry_raw = bucket_map.get(target_key)
        if isinstance(entry_raw, _MutableMappingABC):
            entry = entry_raw
        elif isinstance(entry_raw, _MappingABC):
            entry = dict(entry_raw)
        else:
            entry = {}
        entry["minutes"] = minutes_value
        entry["hr"] = hr_precise
        bucket_map[target_key] = entry

    if isinstance(planner_bucket_view, _MutableMappingABC):
        buckets_map = planner_bucket_view.get("buckets")
        if isinstance(buckets_map, _MutableMappingABC):
            _update_bucket_map(buckets_map)
        else:
            _update_bucket_map(planner_bucket_view)

        order = planner_bucket_view.get("order")
        if isinstance(order, list):
            canonical_order = {
                _canonical_bucket_key(item)
                for item in order
                if isinstance(item, str)
            }
            if "drilling" not in canonical_order:
                candidate_key: str | None = None
                if isinstance(buckets_map, _MutableMappingABC):
                    for key in buckets_map.keys():
                        if _canonical_bucket_key(key) == "drilling":
                            candidate_key = key
                            break
                order.append(candidate_key or "Drilling")

    _ensure_drilling_override(
        drill_hr_total_final,
        planner_bucket_view=planner_bucket_view,
        canonical_bucket_rollup=canonical_bucket_rollup,
        process_meta=process_meta,
        hour_summary_entries=locals().get("hour_summary_entries"),
        label_overrides=label_overrides,
    )

