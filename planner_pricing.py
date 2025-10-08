from __future__ import annotations

from math import log1p, sqrt
from typing import Any, Dict, Tuple

from process_planner import plan_job


# -------- helpers

def _get_rate(rates2: dict, bucket: str, key: str, default: float = 90.0) -> float:
    try:
        return float(rates2.get(bucket, {}).get(key, default) or default)
    except Exception:
        return default


def _as_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _geom(geom: dict) -> dict:
    """Normalize geometry payload into scalar values used by the models."""

    d = dict(geom or {})
    derived = d.get("derived") if isinstance(d.get("derived"), dict) else {}
    out: Dict[str, Any] = {}

    def g(*keys, default=None):
        for k in keys:
            v = d.get(k, None)
            if v is None:
                k_derived = k.split(".", 1)[-1]
                if k_derived in derived:
                    v = derived.get(k_derived)
            if v is not None:
                return v
        return default

    out["hole_count"] = int(_as_float(g("hole_count", "derived.hole_count"), 0) or 0)
    out["tap_qty"] = int(_as_float(g("tap_qty"), 0) or 0)
    out["cbore_qty"] = int(_as_float(g("cbore_qty"), 0) or 0)
    out["slot_count"] = int(_as_float(g("slot_count"), 0) or 0)
    out["edge_len_in"] = _as_float(g("edge_len_in"), 0.0) or 0.0
    out["pocket_area_in2"] = _as_float(g("pocket_area_total_in2"), 0.0) or 0.0
    out["plate_area_in2"] = _as_float(g("plate_area_in2"), 0.0) or 0.0
    out["thickness_in"] = _as_float(g("thickness_in"), None)
    if out["thickness_in"] is None:
        mm = _as_float(g("thickness_mm"), 0.0) or 0.0
        out["thickness_in"] = mm / 25.4 if mm else 0.0
    return out


def _material_factor(material: str | None) -> Tuple[float, float]:
    """Return (mrr_in3_per_min, grind_area_rate_in2_per_min) rough factors by material."""

    m = (material or "").lower()
    if "al" in m or "aluminum" in m:
        return (3.0, 220.0)
    if "cast" in m:
        return (2.0, 180.0)
    if "ss" in m or "stainless" in m:
        return (1.0, 120.0)
    return (1.5, 160.0)  # default steels


# -------- curved models (no hard caps)

def _inspection_minutes(geom: dict, tol: dict) -> float:
    n_holes = max(0, int(geom["hole_count"]))
    edge_len = max(0.0, float(geom["edge_len_in"]))
    slots = max(0, int(geom["slot_count"]))

    def _tight_score(x, ref):
        if x is None or x <= 0:
            return 0.0
        r = ref / max(x, 1e-6)
        return log1p(r)

    score = 0.0
    score += _tight_score(tol.get("profile_tol"), 0.001)
    score += _tight_score(tol.get("flatness_spec"), 0.001)
    score += _tight_score(tol.get("parallelism_spec"), 0.001)

    base = 18.0
    features = 1.8 * sqrt(n_holes)
    geometry = 0.04 * edge_len + 0.5 * slots
    tight = 22.0 * score
    return base + features + geometry + tight


def _fixture_build_minutes(setups: int, geom: dict, tol: dict) -> float:
    n_holes = max(0, int(geom["hole_count"]))
    flat = tol.get("flatness_spec")
    term_setup = 18.0 * max(1, setups)
    term_holes = 0.06 * sqrt(n_holes)
    term_precision = 6.0 if (flat and flat <= 0.001) else 0.0
    return term_setup + term_holes + term_precision


def _drilling_minutes(geom: dict, thickness_in: float, taps: int, cbrores: int, holes: int) -> float:
    approach = 10.0
    peck = 6.0
    pecks = max(1, int(1 + thickness_in / 0.5))
    drill_penetration = max(0.6, 1.2 - 0.3 * thickness_in)
    cut = 60.0 * (thickness_in / drill_penetration)
    per_hole = approach + pecks * peck + cut
    taps_term = 12.0 * max(0, taps)
    cbore_term = 20.0 * max(0, cbrores)
    return (holes * per_hole + taps_term + cbore_term) / 60.0


def _milling_minutes(geom: dict, thickness_in: float, mrr_in3_min: float) -> float:
    pocket_vol = max(0.0, float(geom["pocket_area_in2"])) * max(0.0, thickness_in)
    edge_len = max(0.0, float(geom["edge_len_in"]))
    rough = (pocket_vol / max(0.2, mrr_in3_min)) * 60.0
    finish = (edge_len / max(10.0, 50.0)) * 60.0
    return rough + finish


def _grind_minutes(plate_area_in2: float, passes: int, grind_rate_in2_per_min: float) -> float:
    setup = 8.0
    sweep = max(1, passes) * (plate_area_in2 / max(60.0, grind_rate_in2_per_min))
    return setup + sweep


def _wedm_minutes(cut_len_in: float, thickness_in: float, skims: int) -> float:
    base_ipm = 0.23 / max(0.4, thickness_in)
    rough = (cut_len_in / max(0.05, base_ipm))
    skim = 1.8 * skims * (cut_len_in / max(0.3, 3.0 * base_ipm))
    return rough + skim


def price_with_planner(
    family: str,
    planner_inputs: Dict[str, Any],
    geom_payload: Dict[str, Any],
    two_bucket_rates: Dict[str, Dict[str, float]],
    *,
    oee: float = 0.85,
) -> Dict[str, Any]:
    """
    Return planner-driven pricing including minute models and two-bucket costs.
    """

    plan = plan_job(family, planner_inputs)
    ops = [op.get("op") for op in plan.get("ops", [])]

    g = _geom(geom_payload)
    t = {
        "profile_tol": _as_float(planner_inputs.get("profile_tol")),
        "flatness_spec": _as_float(planner_inputs.get("flatness_spec")),
        "parallelism_spec": _as_float(planner_inputs.get("parallelism_spec")),
    }
    material = str(planner_inputs.get("material") or "").strip()
    mrr, grind_rate = _material_factor(material)
    thickness = float(g["thickness_in"] or 0.0)
    setups = int(_as_float(geom_payload.get("setups"), 0) or 2)

    minutes: Dict[str, float] = {}

    minutes["Inspection"] = _inspection_minutes(g, t)
    minutes["Fixture Build (amortized)"] = _fixture_build_minutes(setups, g, t)

    total_cut_guess_hr = (
        _milling_minutes(g, thickness, mrr)
        + _drilling_minutes(g, thickness, g["tap_qty"], g["cbore_qty"], g["hole_count"])
    ) / 60.0
    minutes["Programming (amortized)"] = max(21.0, 12.0 + 9.0 * log1p(total_cut_guess_hr))

    minutes["Milling"] = _milling_minutes(g, thickness, mrr)
    minutes["Drilling"] = _drilling_minutes(g, thickness, g["tap_qty"], g["cbore_qty"], g["hole_count"])

    if any(k in ops for k in ("wire_edm_windows", "wire_edm_cam_slot_or_profile", "wire_edm_outline")):
        skims = 0
        for op in plan.get("ops", []):
            if op.get("op") == "wire_edm_windows":
                txt = str(op.get("passes") or "")
                if "R+" in txt and "S" in txt:
                    try:
                        skims = int(txt.split("R+")[1].split("S")[0])
                    except Exception:
                        pass
        cut_len = max(0.0, float(g["edge_len_in"]))
        minutes["Wire EDM"] = _wedm_minutes(cut_len, thickness, skims)

    grind_passes = 0
    if (
        "blanchard_grind_pre" in ops
        or "surface_grind_faces/pads_to_final" in ops
        or t["flatness_spec"]
        or t["parallelism_spec"]
    ):
        grind_passes = 1
        if t["flatness_spec"] and t["flatness_spec"] <= 0.001:
            grind_passes += 1
        if t["parallelism_spec"] and t["parallelism_spec"] <= 0.001:
            grind_passes += 1
    if grind_passes:
        minutes["Grinding"] = _grind_minutes(g.get("plate_area_in2", 0.0), grind_passes, grind_rate)

    minutes["Deburr"] = 4.0 + 0.6 * sqrt(max(0, g["hole_count"])) + 0.02 * max(0.0, g["edge_len_in"])

    line_items = []
    labor_min = 0.0
    machine_min = 0.0
    labor_cost = 0.0
    machine_cost = 0.0

    def _add(name: str, bucket: str, rate_key: str) -> None:
        nonlocal labor_min, machine_min, labor_cost, machine_cost
        mins = float(minutes.get(name, 0.0) or 0.0)
        if mins <= 0:
            return
        rate = _get_rate(two_bucket_rates, "labor" if bucket == "labor" else "machine", rate_key, 90.0)
        if bucket == "machine" and oee > 0:
            effective_minutes = mins / max(oee, 0.01)
        else:
            effective_minutes = mins
        cost = (effective_minutes / 60.0) * rate
        entry: Dict[str, Any] = {"name": name, "minutes": round(mins, 2)}
        entry[f"{bucket}_cost"] = round(cost, 2)
        line_items.append(entry)
        if bucket == "labor":
            labor_min += mins
            labor_cost += cost
        else:
            machine_min += mins
            machine_cost += cost

    _add("Inspection", "labor", "InspectionRate")
    _add("Fixture Build (amortized)", "labor", "FixtureBuildRate")
    _add("Programming (amortized)", "labor", "ProgrammingRate")
    _add("Deburr", "labor", "DeburrRate")
    _add("Drilling", "machine", "DrillingRate")
    _add("Milling", "machine", "MillingRate")
    _add("Wire EDM", "machine", "WireEDMRate")
    _add("Grinding", "machine", "SurfaceGrindRate")

    total_minutes = labor_min + machine_min
    total_cost = labor_cost + machine_cost

    return {
        "plan": plan,
        "line_items": line_items,
        "totals": {
            "labor_minutes": round(labor_min, 2),
            "machine_minutes": round(machine_min, 2),
            "total_minutes": round(total_minutes, 2),
            "minutes": round(total_minutes, 2),
            "labor_cost": round(labor_cost, 2),
            "machine_cost": round(machine_cost, 2),
            "total_cost": round(total_cost, 2),
        },
        "assumptions": {
            "family": family,
            "setups": setups,
            "material": material,
            "thickness_in": thickness,
            "tolerances": t,
            "oee": oee,
        },
    }
