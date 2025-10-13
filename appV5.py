SF_CSV_DEFAULT = r"D:\CAD_Quoting_Tool\speeds_feeds_merged.csv"


def _find_speeds_feeds_csv() -> str | None:
    """Return the first existing speeds/feeds CSV path using the configured order."""

    cand = os.environ.get("CADQ_SF_CSV")
    if cand and os.path.isfile(cand):
        return cand

    for path_text in (
        os.path.join(os.getcwd(), "speeds_feeds_merged.csv"),
        os.path.join(os.path.dirname(__file__), "speeds_feeds_merged.csv"),
        SF_CSV_DEFAULT,
    ):
        try:
            if os.path.isfile(path_text):
                return path_text
        except Exception:
            continue

    return None

class SpeedsFeedsUnavailableError(RuntimeError):
    """Raised when a speeds/feeds CSV is required but unavailable."""


    if speeds_feeds_table is None:
        raise SpeedsFeedsUnavailableError("Speeds/feeds CSV missing.")

    speeds_feeds_error: dict[str, Any] | None = None
    speeds_feeds_error_detail: str | None = None

    def _set_speeds_feeds_error(detail: str) -> None:
        nonlocal speeds_feeds_error, speeds_feeds_error_detail
        speeds_feeds_error_detail = detail
        speeds_feeds_error = {
            "code": "speeds_feeds_csv_missing",
            "message": "Speeds/feeds CSV missing.",
            "detail": detail,
        }

                "Invalid speeds/feeds CSV path %r: %s; drilling calculation skipped.",
            _set_speeds_feeds_error(
                f"Speeds/feeds CSV path invalid ({raw_path_text})."
                        "Failed to load speeds/feeds CSV at %s; drilling calculation skipped.",
                    _set_speeds_feeds_error(
                        f"Failed to load speeds/feeds CSV at {resolved_candidate}."
                    "Speeds/feeds CSV not found at %s; drilling calculation skipped.",
                _set_speeds_feeds_error(
                    f"Speeds/feeds CSV not found at {resolved_candidate}."

    if speeds_feeds_table is None:
        fallback_csv = _find_speeds_feeds_csv()
        if fallback_csv:
            try:
                resolved_fallback = Path(fallback_csv).expanduser()
            except Exception:
                resolved_fallback = Path(fallback_csv)
            speeds_feeds_path = _stringify_resolved_path(resolved_fallback)
            speeds_feeds_table = _load_speeds_feeds_table(str(resolved_fallback))
            if speeds_feeds_table is None:
                logger.warning(
                    "Failed to load speeds/feeds CSV at %s; drilling calculation skipped.",
                    resolved_fallback,
                )
                _set_speeds_feeds_error(
                    f"Failed to load speeds/feeds CSV at {resolved_fallback}."
                )
            else:
                speeds_feeds_error = None
                speeds_feeds_error_detail = None
        else:
            if speeds_feeds_error is None:
                logger.warning(
                    "Speeds/feeds CSV path not configured; drilling calculation skipped."
                )
                _set_speeds_feeds_error("Speeds/feeds CSV not configured.")
    if speeds_feeds_error is not None and speeds_feeds_table is None:
        _record_red_flag("Speeds/feeds CSV missing — drilling calculation skipped.")
    if speeds_feeds_table is None:
        drill_hr = 0.0
        error_line = "ERROR: Speeds/feeds CSV missing — drilling calculation skipped."
        if speeds_feeds_error_detail:
            error_line = f"{error_line[:-1]} ({speeds_feeds_error_detail})"
        drill_debug_lines.append(error_line)
        if speeds_feeds_error_detail:
            speeds_feeds_warnings.append(speeds_feeds_error_detail)
    else:
        drill_hr = estimate_drilling_hours(
            hole_diams_mm=hole_diams_list,
            thickness_in=thickness_in,
            mat_key=material_key_for_drill or "",
            material_group=material_group_for_speeds or None,
            hole_groups=hole_groups_geo,
            speeds_feeds_table=speeds_feeds_table,
            machine_params=machine_params_default,
            overhead_params=drill_overhead_default,
            warnings=speeds_feeds_warnings,
            debug_lines=drill_debug_lines,
            debug_summary=drill_debug_summary,
        )
    if speeds_feeds_error is not None:
        drilling_meta["error"] = dict(speeds_feeds_error)
            if speeds_feeds_error is not None:
                guard_ctx["speeds_feeds_error"] = dict(speeds_feeds_error)
    if speeds_feeds_error is not None:
        breakdown["speeds_feeds_error"] = dict(speeds_feeds_error)
        result_payload["speeds_feeds_error"] = dict(speeds_feeds_error)

