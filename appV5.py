# ──────────────────────────────────────────────────────────────────────────────
# Helpers: formatting + removal card + per-hole lines (no material per line)
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_rng(vals, prec=2, unit: str | None = None):
    vs = []
    for v in (vals or []):
        try:
            f = float(v)
            if math.isfinite(f):
                vs.append(f)
        except Exception:
            pass
    if not vs:
        return "—"
    lo, hi = min(vs), max(vs)
    s = (
        f"{lo:.{prec}f}"
        if abs(hi - lo) < 10 ** (-prec)
        else f"{lo:.{prec}f}–{hi:.{prec}f}"
    )
    return f"{s}{unit}" if unit else s


def _rpm_from_sfm(sfm: float, d_in: float) -> float:
    try:
        d = max(float(d_in), 1e-6)
        return (float(sfm) * 12.0) / (math.pi * d)
    except Exception:
        return 0.0


def _render_removal_card(
    lines: List[str],
    *,
    mat_canon: str,
    mat_group: str | None,
    row_group: str | None,
    holes_deep: int,
    holes_std: int,
    dia_vals_in: List[float],
    depth_vals_in: List[float],
    sfm_deep: float,
    sfm_std: float,
    ipr_deep_vals: List[float],
    ipr_std_val: float,
    rpm_deep_vals: List[float],
    rpm_std_vals: List[float],
    ipm_deep_vals: List[float],
    ipm_std_vals: List[float],
    index_min_per_hole: float,
    peck_min_rng: List[float],
    toolchange_min_deep: float,
    toolchange_min_std: float,
):
    lines.append("MATERIAL REMOVAL — DRILLING")
    lines.append("=" * 64)
    # Inputs
    lines.append("Inputs")
    lines.append(f"  Material .......... {mat_canon}  [group {mat_group or '—'}]")
    mismatch = False
    if row_group:
        rg = str(row_group).upper()
        mg = str(mat_group or "").upper()
        mismatch = rg != mg and (rg and mg)
        note = "   (!) mismatch — used row from different group" if mismatch else ""
        lines.append(f"  CSV row group ..... {row_group}{note}")
    lines.append("  Operations ........ Deep-Drill (L/D ≥ 3), Drill")
    lines.append(
        f"  Holes ............. {int(holes_deep)} deep + {int(holes_std)} std  = {int(holes_deep + holes_std)}"
    )
    lines.append(f"  Diameter range .... {_fmt_rng(dia_vals_in, 3)}\"")
    lines.append(f"  Depth per hole .... {_fmt_rng(depth_vals_in, 2)} in")
    lines.append("")
    # Feeds & Speeds
    lines.append("Feeds & Speeds (used)")
    lines.append(
        f"  SFM ............... {int(round(sfm_deep))} (deep)   | {int(round(sfm_std))} (std)"
    )
    lines.append(
        f"  IPR ............... {_fmt_rng(ipr_deep_vals, 4)} (deep) | {float(ipr_std_val):.4f} (std)"
    )
    lines.append(
        f"  RPM ............... {_fmt_rng(rpm_deep_vals, 0)} (deep)      | {_fmt_rng(rpm_std_vals, 0)} (std)"
    )
    lines.append(
        f"  IPM ............... {_fmt_rng(ipm_deep_vals, 1)} (deep)       | {_fmt_rng(ipm_std_vals, 1)} (std)"
    )
    lines.append("")
    # Overheads
    lines.append("Overheads")
    lines.append(f"  Index per hole .... {float(index_min_per_hole):.2f} min")
    lines.append(f"  Peck per hole ..... {_fmt_rng(peck_min_rng, 2)} min")
    lines.append(
        f"  Toolchange ........ {float(toolchange_min_deep):.2f} min (deep) | {float(toolchange_min_std):.2f} min (std)"
    )
    lines.append("")


def _render_time_per_hole(
    lines: List[str], *, bins: List[Dict[str, Any]], index_min: float, peck_min_deep: float, peck_min_std: float
):
    lines.append("TIME PER HOLE — DRILL GROUPS")
    lines.append("-" * 66)
    subtotal_min = 0.0
    seen_deep = False
    seen_std = False
    for b in bins:
        try:
            op = (b.get("op") or b.get("op_name") or "").strip().lower()
            deep = op.startswith("deep")
            if deep:
                seen_deep = True
            else:
                seen_std = True
            d_in = float(b.get("diameter_in"))
            depth = float(b.get("depth_in"))
            qty = int(b.get("qty") or 0)
            sfm = float(b.get("sfm"))
            ipr = float(b.get("ipr"))
            rpm = _rpm_from_sfm(sfm, d_in)
            ipm = rpm * ipr
            peck = float(peck_min_deep if deep else peck_min_std)
            t_hole = (depth / max(ipm, 1e-6)) + float(index_min) + peck
            group_min = t_hole * qty
            subtotal_min += group_min
            # single-line, no material
            lines.append(
                f'Dia {d_in:.3f}" ×{qty}  | depth {depth:.3f}" | {int(round(sfm))} sfm | {ipr:.4f} ipr | '
                f't/hole {t_hole:.2f} min | group {qty}×{t_hole:.2f} = {group_min:.2f} min'
            )
        except Exception:
            continue
    lines.append("")
    return subtotal_min, seen_deep, seen_std

        mat_canon = str(drilling_meta.get("material_canonical") or drilling_meta.get("material") or "—")
        mat_group = drilling_meta.get("material_group") or drilling_meta.get("group") or "—"
            else "Toolchange adders: —"
