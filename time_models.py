# time_models.py
# Turn planner ops into minutes (not dollars).
from typing import Dict, Any, Iterable
from math import ceil

# ---- Tunables (defaults; put in a JSON or env later) ----
IPM_WEDM = {  # cutting speed (in/min) -> conservative shop numbers
    0.010: {"rough": 0.60, "skim": 0.90},
    0.008: {"rough": 0.45, "skim": 0.70},
    0.006: {"rough": 0.30, "skim": 0.55},
}
WEDM_START_STOP_MIN = 0.6              # start/land thread per cut (min)
WEDM_TAB_BREAK_MIN   = 0.3              # per slug/tab removal (min)

SG_PASS_REMOVAL_IN   = 0.0005           # stock removed per pass (in)
SG_TRAVERSE_IPM      = 60.0             # table feed (in/min)
SG_EFF_WIDTH_IN      = 2.0              # effective wheel width swept (in)
SG_SETUP_MIN         = 6.0

BLANCHARD_SFPM_MIN   = 0.0025           # min per sq.in per 0.001" stock (rough rule)
MILL_REMOVAL_CUIN_PER_MIN = 1.8         # alu >> larger; tool steel <<; tune by material
DRILL_IN_PER_MIN     = 8.0              # penetration rate; adjust by dia/material
TAP_MIN_PER_HOLE     = 0.25
THREAD_MILL_MIN_PER_HOLE = 0.6

JIG_BORE_MIN_PER_BORE = 3.0             # light cut, indicate, measure
JIG_GRIND_MIN_PER_BORE = 6.0
SINKER_ROUGH_MIN_PER_CUIN = 15.0        # very geometry-dependent; start conservative
SINKER_FINISH_MIN     = 8.0             # per feature finish burn

LAP_MIN_PER_SQIN      = 2.0
EDGE_BREAK_MIN_PER_FT = 1.2
MARK_MIN              = 1.0

# ---- Geometry keys we expect on the quote/job (add as you wire CAD/worksheet) ----
# geom = {
#   "wedm": {"perimeter_in": float, "starts": int, "tabs": int, "passes": int, "wire_in": 0.010},
#   "sg":   {"area_sq_in": float, "stock_in": float},
#   "blanchard": {"area_sq_in": float, "stock_in": float},
#   "milling": {"volume_cuin": float},
#   "windows_count": int,
#   "drill": [{"dia_in": d, "depth_in": L}] ,
#   "tapped": [{"dia_in": d, "depth_in": L}],
#   "thread_mill": [{"dia_in": d, "depth_in": L}],
#   "bores": [{"tol": 0.0002, "method":"jig_grind"|"jig_bore"|"ream"}],
#   "sinker": [{"vol_cuin": v, "finish": True|False}],
#   "length_ft_edges": float,
#   "lap_area_sq_in": float
# }

def minutes_wedm(d: Dict[str, Any]) -> float:
    perim = d.get("perimeter_in", 0.0)
    starts = d.get("starts", 1)
    tabs   = d.get("tabs", 0)
    passes = max(1, d.get("passes", 1))
    wire   = d.get("wire_in", 0.010)

    ipm_r = IPM_WEDM[wire]["rough"]
    ipm_s = IPM_WEDM[wire]["skim"]
    # Speeds are in inches/minute, so perimeter divided by ipm_* yields minutes directly.
    cut_min = (perim / ipm_r) + (passes - 1) * (perim / ipm_s)
    anc_min = starts*WEDM_START_STOP_MIN + tabs*WEDM_TAB_BREAK_MIN
    return cut_min + anc_min

def minutes_surface_grind(d: Dict[str, Any]) -> float:
    area = d.get("area_sq_in", 0.0)
    stock = d.get("stock_in", 0.001)
    passes = ceil(max(0.0, stock) / SG_PASS_REMOVAL_IN)
    # simple raster time: (area / eff_width) / feed * passes
    # Traverse rate is in inches/minute, so each raster stroke calculation is already minutes.
    strokes = (area / SG_EFF_WIDTH_IN) / SG_TRAVERSE_IPM
    return SG_SETUP_MIN + passes * strokes

def minutes_blanchard(d: Dict[str, Any]) -> float:
    area = d.get("area_sq_in", 0.0)
    stock = d.get("stock_in", 0.002)
    return area * (stock/0.001) * BLANCHARD_SFPM_MIN

def minutes_mill(d: Dict[str, Any]) -> float:
    vol = d.get("volume_cuin", 0.0)
    return (vol / max(0.01, MILL_REMOVAL_CUIN_PER_MIN))

def minutes_drill(holes: Iterable[Dict[str, float]]) -> float:
    total = 0.0
    for h in holes or []:
        total += (h["depth_in"] / max(0.01, DRILL_IN_PER_MIN)) + 0.1  # spot/peck overhead
    return total

def minutes_tap(n: int) -> float:
    return n * TAP_MIN_PER_HOLE

def minutes_thread_mill(n: int) -> float:
    return n * THREAD_MILL_MIN_PER_HOLE

def minutes_bores(items: Iterable[Dict[str, Any]]) -> float:
    total = 0.0
    for b in items or []:
        m = b.get("method","jig_bore")
        if m == "jig_grind":
            total += JIG_GRIND_MIN_PER_BORE
        elif m == "jig_bore":
            total += JIG_BORE_MIN_PER_BORE
        else:  # ream
            total += 0.8
    return total

def minutes_sinker(items: Iterable[Dict[str, Any]]) -> float:
    total = 0.0
    for f in items or []:
        total += f.get("vol_cuin", 0.0) * SINKER_ROUGH_MIN_PER_CUIN
        if f.get("finish", False):
            total += SINKER_FINISH_MIN
    return total

def minutes_edgebreak(ft: float) -> float:
    return ft * EDGE_BREAK_MIN_PER_FT

def minutes_lap(area_sq_in: float) -> float:
    return area_sq_in * LAP_MIN_PER_SQIN
