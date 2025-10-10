"""Helpers for applying 2D geometry feature hints to variable tables."""

from __future__ import annotations

from typing import Any

from cad_quoter.coerce import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.domain_models import DEFAULT_MATERIAL_DISPLAY
from cad_quoter.geometry import upsert_var_row

__all__ = [
    "apply_2d_features_to_variables",
    "to_noncapturing",
    "_to_noncapturing",
]


def to_noncapturing(expr: str) -> str:
    """
    Convert every capturing '(' to non-capturing '(?:', preserving escaped parens and
    existing '(?...)' constructs.
    """
    out: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        prev = expr[i - 1] if i > 0 else ""
        nxt = expr[i + 1] if i + 1 < len(expr) else ""
        if ch == "(" and prev != "\\" and nxt != "?":
            out.append("(?:")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# Backwards-compatible alias for older imports.
_to_noncapturing = to_noncapturing


def apply_2d_features_to_variables(df, g2d: dict, *, params: dict, rates: dict):
    """Write a few cycle-time rows based on 2D perimeter/holes so compute_quote_from_df() can price it."""

    geo = g2d.get("geo") if isinstance(g2d, dict) else None
    thickness_mm = _coerce_float_or_none(g2d.get("thickness_mm")) if isinstance(g2d, dict) else None
    thickness_in = (float(thickness_mm) / 25.4) if thickness_mm else None
    thickness_from_deepest = False
    if (not thickness_in) and isinstance(geo, dict):
        guess = _coerce_float_or_none(geo.get("thickness_in_guess"))
        if guess:
            try:
                bounded = min(3.0, max(0.125, float(guess)))
                thickness_in = bounded
                thickness_from_deepest = True
            except Exception:
                thickness_in = None

    material_note = ""
    if isinstance(geo, dict) and geo.get("material_note"):
        material_note = str(geo.get("material_note") or "").strip()
    material_value = (
        material_note
        or (g2d.get("material") if isinstance(g2d, dict) else "")
        or DEFAULT_MATERIAL_DISPLAY
    )
    df = upsert_var_row(df, "Material", material_value, dtype="text")
    if thickness_in:
        df = upsert_var_row(
            df,
            "Thickness (in)",
            round(float(thickness_in), 4),
            dtype="number",
        )
    elif thickness_from_deepest:
        df = upsert_var_row(df, "Thickness (in)", 0.125, dtype="number")

    plate_len = None
    plate_wid = None
    if isinstance(geo, dict):
        plate_len = geo.get("plate_len_in")
        plate_wid = geo.get("plate_wid_in")
    df = upsert_var_row(
        df,
        "Plate Length (in)",
        round(float(plate_len), 3) if plate_len else 12.0,
        dtype="number",
    )
    df = upsert_var_row(
        df,
        "Plate Width (in)",
        round(float(plate_wid), 3) if plate_wid else 14.0,
        dtype="number",
    )
    df = upsert_var_row(df, "Scrap Percent (%)", 15.0, dtype="number")

    def _update_if_blank(label: str, value: Any, dtype: str = "number", zero_is_blank: bool = True) -> None:
        if value is None:
            return
        mask = df["Item"].astype(str).str.fullmatch(label, case=False, na=False)
        if mask.any():
            existing_raw = str(df.loc[mask, "Example Values / Options"].iloc[0]).strip()
            existing_val = _coerce_float_or_none(existing_raw)
            is_blank = (existing_raw == "") or (zero_is_blank and (existing_val is None or abs(existing_val) < 1e-9))
            if not is_blank:
                return
            df.loc[mask, "Example Values / Options"] = value
            df.loc[mask, "Data Type / Input Method"] = dtype
        else:
            df.loc[len(df)] = [label, value, dtype]

    tap_qty_geo = None
    from_back_geo = False
    if isinstance(geo, dict):
        tap_qty_geo = _coerce_float_or_none(geo.get("tap_qty"))
        from_back_geo = bool(geo.get("needs_back_face") or geo.get("from_back"))
    if tap_qty_geo and tap_qty_geo > 0:
        _update_if_blank("Tap Qty (LLM/GEO)", int(round(tap_qty_geo)))
    if from_back_geo:
        mask_setups = df["Item"].astype(str).str.fullmatch("Number of Milling Setups", case=False, na=False)
        if mask_setups.any():
            current = _coerce_float_or_none(df.loc[mask_setups, "Example Values / Options"].iloc[0])
            if current is None or current < 2:
                df.loc[mask_setups, "Example Values / Options"] = 2
                df.loc[mask_setups, "Data Type / Input Method"] = "number"
        else:
            df.loc[len(df)] = ["Number of Milling Setups", 2, "number"]

    def set_row(pattern: str, value: float):
        regex = to_noncapturing(pattern)
        mask = df["Item"].astype(str).str.contains(regex, case=False, regex=True, na=False)
        if mask.any():
            df.loc[mask, "Example Values / Options"] = value
        else:
            df.loc[len(df)] = [pattern, value, "number"]

    L = float(g2d.get("profile_length_mm", 0.0))
    t = float(g2d.get("thickness_mm") or 6.0)
    holes = g2d.get("hole_diams_mm", [])
    # crude process pick
    use_jet = (t <= 12.0 and (L >= 300.0 or len(holes) >= 2))

    if use_jet:
        cut_min = L / (300.0 if t <= 10 else 120.0)   # mm/min ? minutes
        deburr_min = (L / 1000.0) * 2.0
        set_row(r"(Sawing|Waterjet|Blank\s*Prep)", round(cut_min / 60.0, 3))
        set_row(r"(Deburr|Edge\s*Break)", round(deburr_min / 60.0, 3))
        set_row(r"(Programming|2D\s*CAM)", 0.5)
        set_row(r"(Setup\s*Time\s*per\s*Setup)", 0.25)
        set_row(r"(Milling\s*Setups)", 1)
    else:
        mill_min = L / 800.0
        drill_min = max(0.2, (t / 50.0)) * max(1, len(holes))
        deburr_min = (L / 1000.0) * 3.0
        set_row(r"(Finishing\s*Cycle\s*Time)", round(mill_min / 60.0, 3))
        set_row(r"(ID\s*Boring|Drilling|Reaming)", round(drill_min / 60.0, 3))
        set_row(r"(Deburr|Edge\s*Break)", round(deburr_min / 60.0, 3))
        set_row(r"(Programming|2D\s*CAM)", 0.75)
        set_row(r"(Setup\s*Time\s*per\s*Setup)", 0.5)
        set_row(r"(Milling\s*Setups)", 1)

    set_row(r"(Final\s*Inspection|Manual\s*Inspection)", 0.2)
    set_row(r"(Packaging|Boxing|Crating\s*Labor)", 0.1)
    return df.reset_index(drop=True)
