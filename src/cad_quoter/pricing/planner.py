from __future__ import annotations
from math import sqrt, log1p
from typing import Any, Dict, Tuple, List, Iterable, Mapping

from cad_quoter.geo_extractor import ops_manifest
from cad_quoter.planning.process_planner import plan_job
from cad_quoter.rates import ensure_two_bucket_defaults
from cad_quoter.utils import _dict
from cad_quoter.pricing.rate_buckets import bucket_cost_breakdown
from cad_quoter.utils.chart_buckets import classify_chart_rows

def _as_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _geom(geom: dict) -> dict:
    d = dict(geom or {})
    derived = _dict(d.get("derived"))
    def g(*keys, default=None):
        for k in keys:
            # dotted lookup into derived support
            if "." in k:
                k1, k2 = k.split(".", 1)
                d_inner = _dict(d.get(k1))
                if k2 in d_inner:
                    return d_inner[k2]
                derived_inner = _dict(derived.get(k1))
                if k2 in derived_inner:
                    return derived_inner[k2]
            v = d.get(k)
            if v is None and k in derived:
                v = derived.get(k)
            if v is not None:
                return v
        return default

    out = {}

    def _first_int(*keys: str) -> int:
        for key in keys:
            val = _as_float(g(key), None)
            if val is not None and val > 0:
                return int(val)
        return 0

    # prefer table/derived before raw geometry fallbacks
    out["hole_count"] = _first_int(
        "hole_count",  # top-level (table or dedup now flows here)
        "geo.hole_count",  # explicit table count if nested
        "derived.hole_count",
        "hole_count_geom",  # our dedup writes here
        "derived.hole_count_geom",
    )
    if out["hole_count"] <= 0:
        holes = g("hole_diams_mm", "derived.hole_diams_mm", default=())
        if isinstance(holes, (list, tuple)):
            out["hole_count"] = sum(1 for h in holes if _as_float(h) is not None)

    feature_counts = _dict(g("feature_counts", "derived.feature_counts"))

    def _feature_count(key: str, *extra_keys: str) -> int:
        candidates = [key, f"derived.{key}"]
        candidates.extend(extra_keys)
        count = _first_int(*candidates)
        if count <= 0 and feature_counts:
            count = int(_as_float(feature_counts.get(key), 0) or 0)
        return count

    out["tap_qty"] = _feature_count("tap_qty")
    out["cbore_qty"] = _feature_count("cbore_qty", "cbore_pairs_geom", "derived.cbore_pairs_geom")
    out["slot_count"] = _feature_count("slot_count", "slot_qty", "derived.slot_qty")

    edge_len = _as_float(
        g(
            "edge_len_in",
            "derived.edge_len_in",
            "edge_length_in",
            "derived.edge_length_in",
        ),
        0.0,
    ) or 0.0
    if edge_len <= 0:
        edge_mm = _as_float(
            g(
                "edge_len_mm",
                "derived.edge_len_mm",
                "edge_length_mm",
                "profile_length_mm",
                "derived.profile_length_mm",
            ),
            0.0,
        ) or 0.0
        if edge_mm > 0:
            edge_len = edge_mm / 25.4
    if edge_len <= 0:
        wedm = _dict(d.get("wedm"))
        if wedm:
            edge_len = _as_float(wedm.get("perimeter_in"), edge_len) or edge_len
    out["edge_len_in"] = edge_len

    out["pocket_area_in2"] = _as_float(
        g("pocket_area_in2", "pocket_area_total_in2", "derived.pocket_area_in2"), 0.0
    ) or 0.0
    if out["pocket_area_in2"] <= 0:
        milling = _dict(d.get("milling"))
        area = _as_float(milling.get("area_in2"))
        if area is not None:
            out["pocket_area_in2"] = area

    plate_area = _as_float(
        g(
            "plate_area_in2",
            "derived.plate_area_in2",
            "outline_area_in2",
            "derived.outline_area_in2",
        ),
        0.0,
    ) or 0.0
    if plate_area <= 0:
        plate_area_mm2 = _as_float(
            g(
                "plate_bbox_area_mm2",
                "derived.plate_bbox_area_mm2",
            ),
            0.0,
        ) or 0.0
        if plate_area_mm2 > 0:
            plate_area = plate_area_mm2 / (25.4 ** 2)
    if plate_area <= 0:
        sg = _dict(d.get("sg"))
        if sg:
            plate_area = _as_float(sg.get("area_sq_in"), plate_area) or plate_area
    out["plate_area_in2"] = plate_area

    thk_in = _as_float(
        g(
            "thickness_in",
            "derived.thickness_in",
            "plate_thickness_in",
            "stock_thickness_in",
        ),
        None,
    )
    thk_mm = _as_float(
        g(
            "thickness_mm",
            "derived.thickness_mm",
            "plate_thickness_mm",
            "stock_thickness_mm",
        ),
        None,
    )
    if thk_in is None and thk_mm is not None:
        thk_in = thk_mm / 25.4
    if thk_in is None:
        thk_in = 0.0
    out["thickness_in"] = thk_in

    ops_summary = _dict(d.get("ops_summary"))
    if not ops_summary:
        ops_summary = _dict(_dict(d.get("geo")).get("ops_summary"))

    chart_rows_iter: Iterable[Mapping[str, Any]] | None = None
    rows_candidate = ops_summary.get("rows")
    if isinstance(rows_candidate, list):
        chart_rows_iter = [row for row in rows_candidate if isinstance(row, Mapping)]

    chart_buckets, chart_row_count, chart_qty_sum = classify_chart_rows(chart_rows_iter)

    def _hole_sets_from_geo(source: Mapping[str, Any] | None) -> Any:
        if not isinstance(source, Mapping):
            return None
        hole_sets_val = source.get("hole_sets")
        if hole_sets_val:
            return hole_sets_val
        nested_geo = source.get("geo")
        if isinstance(nested_geo, Mapping):
            return _hole_sets_from_geo(nested_geo)
        return None

    hole_sets_payload = _hole_sets_from_geo(d)
    ops_manifest_payload = ops_manifest(chart_rows_iter, hole_sets=hole_sets_payload)
    manifest_totals = (
        ops_manifest_payload.get("total", {})
        if isinstance(ops_manifest_payload, Mapping)
        else {}
    )

    ops_totals = _dict(ops_summary.get("totals"))
    out["ops"] = {
        "drill": int(
            _as_float(manifest_totals.get("drill"), _as_float(ops_totals.get("drill"), 0))
            or 0
        ),
        "tap_front": int(_as_float(ops_totals.get("tap_front"), 0) or 0),
        "tap_back": int(_as_float(ops_totals.get("tap_back"), 0) or 0),
        "cbore_front": int(_as_float(ops_totals.get("cbore_front"), 0) or 0),
        "cbore_back": int(_as_float(ops_totals.get("cbore_back"), 0) or 0),
        "csk_front": int(_as_float(ops_totals.get("csk_front"), 0) or 0),
        "csk_back": int(_as_float(ops_totals.get("csk_back"), 0) or 0),
        "spot_front": int(_as_float(ops_totals.get("spot_front"), 0) or 0),
        "spot_back": int(_as_float(ops_totals.get("spot_back"), 0) or 0),
        "jig_grind": int(_as_float(ops_totals.get("jig_grind"), 0) or 0),
    }
    out["ops"]["manifest"] = ops_manifest_payload
    out["flip_required"] = bool(ops_summary.get("flip_required"))

    if chart_buckets:
        out["ops"]["chart_rows"] = {
            "buckets": chart_buckets,
            "row_count": chart_row_count,
            "qty_sum": chart_qty_sum,
        }

    if chart_buckets.get("tap", 0) > out.get("tap_qty", 0):
        out["tap_qty"] = int(chart_buckets.get("tap", 0))
    if chart_buckets.get("cbore", 0) > out.get("cbore_qty", 0):
        out["cbore_qty"] = int(chart_buckets.get("cbore", 0))

    return out

def _material_factor(material: str | None) -> Tuple[float, float]:
    """(mrr_in3_per_min, grind_area_rate_in2_per_min) rough factors by material."""
    m = (material or "").lower()
    if "al" in m:
        return (3.0, 220.0)
    if "cast" in m:
        return (2.0, 180.0)
    if "ss" in m:
        return (1.0, 120.0)
    return (1.5, 160.0)  # default steels

def _ops_by_name(plan: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for op in (plan.get("ops") or []):
        name = str(op.get("op") or "")
        out.setdefault(name, []).append(op)
    return out

# -------- shared curved models (no hard caps)
def _inspection_minutes(geom: dict, tol: dict) -> float:
    n_holes = max(0, int(geom["hole_count"]))
    edge_len = max(0.0, float(geom["edge_len_in"]))
    slots = max(0, int(geom["slot_count"]))
    def _tight_score(x, ref):
        if x is None or x <= 0:
            return 0.0
        r = ref / max(x, 1e-6)
        return log1p(r)  # smooth growth as tolerances tighten
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
    term_setup = 18.0 * max(1, setups)           # ~0.3 h per setup
    term_holes = 0.06 * sqrt(n_holes)
    term_precision = 6.0 if (flat and flat <= 0.001) else 0.0
    return term_setup + term_holes + term_precision  # e.g. 3 setups ≈ ~54 min

def _drilling_minutes(
    geom: dict,
    thickness_in: float,
    taps: int,
    cbrores: int,
    holes: int,
) -> dict[str, float]:
    approach = 10.0
    peck = 6.0
    pecks = max(1, int(1 + thickness_in / 0.5))
    drill_penetration = max(0.6, 1.2 - 0.3 * thickness_in)  # in/min (thin plates faster)
    cut = 60.0 * (thickness_in / drill_penetration)
    per_hole = approach + pecks * peck + cut
    taps_term = 12.0 * max(0, taps)
    cbore_term = 20.0 * max(0, cbrores)

    drill_cycle_seconds = holes * per_hole
    taps_minutes = taps_term / 60.0
    cbore_minutes = cbore_term / 60.0
    spot_minutes = (holes * (approach + pecks * peck)) / 60.0
    drill_total_minutes = (drill_cycle_seconds / 60.0) + taps_minutes + cbore_minutes

    return {
        "drill_total_min": drill_total_minutes,
        "tapping_min": taps_minutes,
        "counterbore_min": cbore_minutes,
        "spot_min": spot_minutes,
    }

def _milling_minutes(geom: dict, thickness_in: float, mrr_in3_min: float) -> float:
    pocket_vol = max(0.0, float(geom["pocket_area_in2"])) * max(0.0, thickness_in)
    edge_len = max(0.0, float(geom["edge_len_in"]))
    rough = (pocket_vol / max(0.2, mrr_in3_min)) * 60.0
    finish = (edge_len / max(10.0, 50.0)) * 60.0
    return rough + finish

def _grind_minutes(plate_area_in2: float, passes: int, grind_rate_in2_min: float) -> float:
    setup = 8.0
    sweep = max(1, passes) * (plate_area_in2 / max(60.0, grind_rate_in2_min))
    return setup + sweep

def _profile_grind_minutes(edge_len_in: float, precision_bonus: float = 0.0) -> float:
    # linear inches with a small precision adder
    return 6.0 + (edge_len_in / 15.0) * 60.0 + precision_bonus  # ~15 IPM baseline

def _wedm_minutes(cut_len_in: float, thickness_in: float, skims: int) -> float:
    base_ipm = 0.23 / max(0.4, thickness_in)
    rough = (cut_len_in / max(0.05, base_ipm))
    skim = 1.8 * skims * (cut_len_in / max(0.3, 3.0 * base_ipm))
    return rough + skim

def _sinker_minutes(tight: bool) -> float:
    # very rough “finish burn” model; you can tune this or key off EDM depth if you have it
    return 35.0 + (15.0 if tight else 0.0)

def _lap_minutes(target_ra: float | None, length_in: float = 0.0) -> float:
    # gentle model: tighter Ra and more length → more minutes
    base = 8.0
    ra_term = 10.0 if (target_ra is not None and target_ra <= 8) else 4.0
    len_term = 0.4 * length_in
    return base + ra_term + len_term

# -------- family-specific aggregators (all curved; no hard caps)

def _minutes_die_plate(plan, ops, g, t, material):
    mrr, grind_rate = _material_factor(material)
    thk = float(g["thickness_in"] or 0.0)
    minutes = {}
    # Labor-ish buckets
    minutes["Inspection"] = _inspection_minutes(g, t)
    # Fixture build (toned down)
    setups = 2
    minutes["Fixture Build (amortized)"] = _fixture_build_minutes(setups, g, t)
    # Programming as mild curve on cut time
    drill_minutes = _drilling_minutes(
        g,
        thk,
        g["tap_qty"],
        g["cbore_qty"],
        g["hole_count"],
    )
    total_cut_guess_hr = (
        _milling_minutes(g, thk, mrr) + drill_minutes["drill_total_min"]
    ) / 60.0
    minutes["Programming (per part)"] = max(21.0, 12.0 + 9.0 * log1p(total_cut_guess_hr))
    # Machine buckets
    minutes["Milling"] = _milling_minutes(g, thk, mrr)
    minutes["Drilling"] = drill_minutes["drill_total_min"]
    minutes["Tapping"] = drill_minutes["tapping_min"]
    minutes["Counterbore"] = drill_minutes["counterbore_min"]
    minutes["Spot-Drill"] = drill_minutes["spot_min"]
    if "wire_edm_windows" in ops:
        # planner provides passes like "R+2S" on the op
        skims = 0
        for op in ops["wire_edm_windows"]:
            txt = str(op.get("passes") or "")
            if "R+" in txt and "S" in txt:
                try:
                    skims = max(skims, int(txt.split("R+")[1].split("S")[0]))
                except Exception:
                    pass
        minutes["Wire EDM"] = _wedm_minutes(max(0.0, g["edge_len_in"]), thk, skims)
    grind_passes = 0
    if "blanchard_grind_pre" in ops or "face_mill_pre" in ops or t["flatness_spec"] or t["parallelism_spec"]:
        grind_passes = 1 + (1 if (t["flatness_spec"] and t["flatness_spec"] <= 0.001) else 0) + (1 if (t["parallelism_spec"] and t["parallelism_spec"] <= 0.001) else 0)
    if grind_passes:
        minutes["Grinding"] = _grind_minutes(g.get("plate_area_in2", 0.0), grind_passes, grind_rate)
    minutes["Deburr"] = 4.0 + 0.6 * sqrt(max(0, g["hole_count"])) + 0.02 * max(0.0, g["edge_len_in"])
    return minutes

def _minutes_punch_or_pilot(plan, ops, g, t, material):
    # Punch plan emits: rough path (steel vs carbide), carrier tabs, wire outline (+skims), optional sinker, grind/lap bearing, pilot TIR grind.
    mrr, _ = _material_factor(material)
    thk = float(g["thickness_in"] or 0.0)
    minutes = {}
    minutes["Inspection"] = _inspection_minutes(g, t)
    # Programming mildly follows cut size
    minutes["Programming (per part)"] = 18.0 + 6.0 * log1p(max(0.0, g["edge_len_in"]) / 30.0)
    # Machine: WEDM outline (skims from op), small milling rough/time if present
    skims = 0
    for op in ops.get("wire_edm_outline", []):
        txt = str(op.get("passes") or "")
        if "R+" in txt and "S" in txt:
            try:
                skims = max(skims, int(txt.split("R+")[1].split("S")[0]))
            except Exception:
                pass
    minutes["Wire EDM"] = _wedm_minutes(max(0.0, g["edge_len_in"]), thk, skims)
    if "cnc_mill_rough" in ops:
        minutes["Milling"] = 0.35 * _milling_minutes(g, thk, mrr)  # light roughing only
    # Sinker if requested
    if "sinker_edm_finish_burn" in ops:
        tight = bool(t.get("profile_tol") and t["profile_tol"] <= 0.0003)
        minutes["Sinker EDM"] = _sinker_minutes(tight)
    # Grinding/lapping for bearing lands or cleanup
    if "surface_or_profile_grind_bearing" in ops:
        minutes["Grinding"] = _profile_grind_minutes(max(0.0, g["edge_len_in"]), precision_bonus=10.0)
    elif "light_grind_cleanup" in ops:
        minutes["Grinding"] = 10.0 + 0.2 * max(0.0, g["edge_len_in"])
    # Tiny deburr
    minutes["Deburr"] = 6.0 + 0.02 * max(0.0, g["edge_len_in"])
    return minutes

def _minutes_bushing(plan, ops, g, t, material):
    # Bushing plan: OD prepped or purchased, open ID (wire or drill/trepan), jig grind ID, lap ID (optional).
    _, grind_rate = _material_factor(material)
    thk = float(g["thickness_in"] or 0.0)
    minutes = {}
    minutes["Inspection"] = 14.0 + 10.0 * log1p(1.0 / max(t.get("flatness_spec") or 0.001, 1e-6))
    # OD work if not purchased ground
    if "turn_or_mill_OD" in ops:
        minutes["Milling"] = minutes.get("Milling", 0.0) + 18.0
        minutes["Grinding"] = minutes.get("Grinding", 0.0) + 12.0  # cleanup OD
    # ID opening
    if "wire_edm_open_ID" in ops:
        minutes["Wire EDM"] = _wedm_minutes(max(0.0, g["edge_len_in"]), thk, 0)
    else:
        minutes["Drilling"] = 12.0 + 8.0 * thk  # trepan/drill setup + depth
    # Jig grind ID to size & roundness, then optional lap
    minutes["Grinding"] = minutes.get("Grinding", 0.0) + (20.0 + 30.0 * log1p(1e-4 / max(t.get("profile_tol") or 2e-4, 1e-6)))
    if "lap_ID" in ops:
        target_ra = None
        for op in ops["lap_ID"]:
            target_ra = op.get("target_Ra")
        minutes["Lapping/Honing"] = _lap_minutes(target_ra, length_in=0.0)
    minutes["Deburr"] = 5.0
    return minutes

def _minutes_cam_or_hemmer(plan, ops, g, t, material):
    # Cam plan: rough blocks, HT (if wear part), WEDM cam path or finish-mill, grind wear faces, jig-bore pivots.
    mrr, grind_rate = _material_factor(material)
    thk = float(g["thickness_in"] or 0.0)
    minutes = {}
    minutes["Inspection"] = 18.0 + 12.0 * log1p(1.0 / max(t.get("profile_tol") or 0.001, 1e-6))
    # Cam slot/profile
    if "wire_edm_cam_slot_or_profile" in ops:
        skims = 0
        for op in ops["wire_edm_cam_slot_or_profile"]:
            txt = str(op.get("passes") or "")
            if "R+" in txt and "S" in txt:
                try:
                    skims = max(skims, int(txt.split("R+")[1].split("S")[0]))
                except Exception:
                    pass
        minutes["Wire EDM"] = _wedm_minutes(max(0.0, g["edge_len_in"]), thk, skims)
    else:
        minutes["Milling"] = _milling_minutes(g, thk, mrr) * 0.6  # finishing passes
    # Wear faces grind + pivot bores
    minutes["Grinding"] = (minutes.get("Grinding", 0.0)
                           + _grind_minutes(g.get("plate_area_in2", 60.0), 1, grind_rate)
                           + 24.0)  # jig-bore or grind pivots
    minutes["Deburr"] = 6.0 + 0.02 * max(0.0, g["edge_len_in"])
    return minutes

def _minutes_flat_die_chaser(plan, ops, g, t, material):
    # Chasers: rough by mill or wire, HT, profile-grind flanks/reliefs, lap edges.
    thk = float(g["thickness_in"] or 0.0)
    minutes = {}
    minutes["Inspection"] = 12.0 + 8.0 * log1p(max(0.0, g["edge_len_in"]) / 30.0)
    # If rough via wire, add some WEDM time; else light milling
    if "mill_or_wire_rough_form" in ops:
        # assume some wire usage when very tight profiles show up
        minutes["Milling"] = 18.0
        minutes["Wire EDM"] = 18.0 if (t.get("profile_tol") and t["profile_tol"] <= 0.001) else 0.0
    minutes["Grinding"] = _profile_grind_minutes(max(0.0, g["edge_len_in"]), precision_bonus=12.0)
    minutes["Lapping/Honing"] = 10.0
    minutes["Deburr"] = 4.0
    return minutes

def _minutes_pm_compaction_die(plan, ops, g, t, material):
    # Carbide ring: wire ID leave, **tight jig grind** to tenths, lap land.
    thk = float(g["thickness_in"] or 0.0)
    minutes = {}
    minutes["Inspection"] = 16.0 + 14.0 * log1p(1e-4 / 1e-4)  # modest bump
    minutes["Wire EDM"] = 12.0 + 1.4 * max(0.0, g["edge_len_in"])  # short ID cut
    minutes["Grinding"] = 40.0  # jig grind to 0.0001" + straightness
    minutes["Lapping/Honing"] = 12.0
    return minutes

def _minutes_shear_blade(plan, ops, g, t, material):
    # Shear blades: waterjet/saw blanks, HT, profile grind edges/angles, match grind set, hone.
    minutes = {}
    minutes["Inspection"] = 12.0
    minutes["Saw/Waterjet"] = 10.0
    minutes["Grinding"] = _profile_grind_minutes(max(0.0, g["edge_len_in"]), precision_bonus=8.0) + 20.0  # match-grind set
    minutes["Deburr"] = 6.0
    return minutes

def _minutes_extrude_hone(plan, ops, g, t, material):
    # AFM: verify/mask, abrasive flow polish to target Ra, clean/flush.
    minutes = {}
    minutes["Inspection"] = 10.0
    minutes["Abrasive Flow"] = 25.0 + (8.0 if (t.get("profile_tol") and t["profile_tol"] <= 0.001) else 0.0)
    minutes["Deburr"] = 2.0
    return minutes

# -------- main entry

def price_with_planner(
    family: str,
    planner_inputs: Dict[str, Any],
    geom_payload: Dict[str, Any],
    two_bucket_rates: Dict[str, Dict[str, float]],
    *,
    oee: float = 0.85,
) -> Dict[str, Any]:
    rates = ensure_two_bucket_defaults(two_bucket_rates)
    plan = plan_job(family, planner_inputs)
    ops = _ops_by_name(plan)

    g = _geom(geom_payload)
    t = {
        "profile_tol": _as_float(planner_inputs.get("profile_tol")),
        "flatness_spec": _as_float(planner_inputs.get("flatness_spec")),
        "parallelism_spec": _as_float(planner_inputs.get("parallelism_spec")),
    }
    material = str(planner_inputs.get("material") or "").strip()

    # family dispatch
    if family in {"die_plate"}:
        minutes = _minutes_die_plate(plan, ops, g, t, material)
    elif family in {"punch", "pilot_punch"}:
        minutes = _minutes_punch_or_pilot(plan, ops, g, t, material)
    elif family == "bushing_id_critical":
        minutes = _minutes_bushing(plan, ops, g, t, material)
    elif family == "cam_or_hemmer":
        minutes = _minutes_cam_or_hemmer(plan, ops, g, t, material)
    elif family == "flat_die_chaser":
        minutes = _minutes_flat_die_chaser(plan, ops, g, t, material)
    elif family == "pm_compaction_die":
        minutes = _minutes_pm_compaction_die(plan, ops, g, t, material)
    elif family == "shear_blade":
        minutes = _minutes_shear_blade(plan, ops, g, t, material)
    elif family == "extrude_hone":
        minutes = _minutes_extrude_hone(plan, ops, g, t, material)
    else:
        # fallback: light generic mapping
        minutes = {"Inspection": _inspection_minutes(g, t), "Deburr": 6.0}

    # ---- convert to costs (two-bucket rates)
    line_items, totals = bucket_cost_breakdown(minutes, rates)
    for entry in line_items:
        # Older callers expect both cost keys to be present on every item.
        entry.setdefault("labor_cost", 0.0)
        entry.setdefault("machine_cost", 0.0)

    ops_seen = sorted(ops.keys())
    if "wire_edm_windows" in ops and "wire_edm_outline" not in ops:
        ops_seen = sorted(set(ops_seen) | {"wire_edm_outline"})

    return {
        "plan": plan,
        "line_items": line_items,
        "totals": totals,
        "assumptions": {
            "family": family,
            "material": material,
            "thickness_in": g["thickness_in"],
            "tolerances": t,
            "ops_seen": ops_seen,
        },
    }
