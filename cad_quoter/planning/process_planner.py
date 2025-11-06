"""
process_planner_v2.py — compact, rule‑based process planner with clean inputs
-----------------------------------------------------------------------------
Goals
- Keep the public API simple: plan_job(family: str, params: dict) -> dict
- Make decisions from small, testable helpers (wire size, skims, slot strategy)
- Accept hole/dimension adapters (e.g., DummyHoleHelper, DummyDimsHelper)
- Emit a stable plan schema: {ops: [...], fixturing: [...], qa: [...], warnings: [...], directs: {...}}

Families supported (extensible):
- die_plate  (main one used in CAD quoting flow)
- punch, pilot_punch, bushing_id_critical, cam_or_hemmer, flat_die_chaser,
  pm_compaction_die, shear_blade, extrude_hone   (lightweight placeholders)

Notes
- This file stands alone; no external deps beyond stdlib.
- The die_plate logic is practical and opinionated; tweak thresholds to your shop.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterable
from pathlib import Path

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_job(family: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Route to a planner by family and return a normalized plan dict.

    Required (by family):
      die_plate: expects at least plate_LxW=(L,W) OR L,W in params; optional hole_sets

    Returns a dict with keys: ops, fixturing, qa, warnings, directs
    """
    family = (family or "").strip().lower()
    if family not in PLANNERS:
        raise ValueError(f"Unsupported family '{family}'. Known: {sorted(PLANNERS)}")
    plan = PLANNERS[family](params)
    return normalize_plan(plan)

# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    ops: List[Dict[str, Any]] = field(default_factory=list)
    fixturing: List[str] = field(default_factory=list)
    qa: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    directs: Dict[str, bool] = field(default_factory=lambda: {
        "hardware": False,
        "outsourced": False,
        "utilities": False,
        "consumables_flat": False,
        "packaging_flat": True,
    })

    def add(self, op: str, **kwargs: Any) -> None:
        self.ops.append({"op": op, **compact_dict(kwargs)})


def base_plan() -> Plan:
    return Plan()


def normalize_plan(plan: Plan | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(plan, Plan):
        d = {
            "ops": plan.ops,
            "fixturing": plan.fixturing,
            "qa": plan.qa,
            "warnings": plan.warnings,
            "directs": dict(plan.directs),
        }
    else:
        d = {
            "ops": list(plan.get("ops", [])),
            "fixturing": list(plan.get("fixturing", [])),
            "qa": list(plan.get("qa", [])),
            "warnings": list(plan.get("warnings", [])),
            "directs": dict(plan.get("directs", {})),
        }
    derive_directs(d)
    return d


def compact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None and v != ""}


# ---------------------------------------------------------------------------
# Decision helpers (tunable thresholds)
# ---------------------------------------------------------------------------

def choose_wire_size(min_inside_radius: Optional[float], min_feature_width: Optional[float]) -> float:
    """Return wire diameter in inches based on tightest geometry."""
    mir = min_inside_radius or 0.0
    mfw = min_feature_width or 0.0
    # conservative ladder
    if mir <= 0.003 or mfw <= 0.006:
        return 0.006
    if mir <= 0.004 or mfw <= 0.010:
        return 0.008
    return 0.010


def choose_skims(profile_tol: Optional[float]) -> int:
    t = (profile_tol or 0.0)
    if t <= 0.0002:
        return 3
    if t <= 0.0003:
        return 2
    if t <= 0.0005:
        return 1
    return 0


def needs_wedm_for_windows(windows_need_sharp: bool, window_corner_radius_req: Optional[float], profile_tol: Optional[float]) -> bool:
    if windows_need_sharp:
        return True
    if (window_corner_radius_req or 99) <= 0.030:
        return True
    if (profile_tol or 1.0) <= 0.001:
        return True
    return False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def derive_directs(plan_dict: Dict[str, Any]) -> None:
    ops = plan_dict.get("ops", [])
    if ops:
        plan_dict["directs"]["utilities"] = True
        plan_dict["directs"]["consumables_flat"] = True
    # outsourced if any heat treat or coating listed
    for op in ops:
        name = (op.get("op") or "").lower()
        if name.startswith("heat_treat") or name.startswith("coat_"):
            plan_dict["directs"]["outsourced"] = True
    # hardware flag is left to caller to set explicitly or via hole semantics


# ---------------------------------------------------------------------------
# Family planners
# ---------------------------------------------------------------------------

# ---- die_plate -------------------------------------------------------------

def planner_die_plate(params: Dict[str, Any]) -> Plan:
    p = base_plan()

    # Dimensions
    L, W = _read_plate_LW(params)
    profile_tol = params.get("profile_tol")  # float in inches
    flatness_spec = params.get("flatness_spec")
    parallelism_spec = params.get("parallelism_spec")
    windows_need_sharp = bool(params.get("windows_need_sharp", False))
    window_corner_radius_req = params.get("window_corner_radius_req")
    # 1) Face strategy (Blanchard vs mill) — big or tight spec → Blanchard first
    if max(L, W) > 10.0 or (flatness_spec is not None and flatness_spec <= 0.001):
        p.add("blanchard_pre")
    else:
        p.add("face_mill_pre")

    # 2) Base drilling ops (spot + drill patterns handled generically)
    p.add("spot_drill_all")
    p.add("drill_patterns")

    # 3) Window/Profile strategy (WEDM vs finish mill)
    if needs_wedm_for_windows(windows_need_sharp, window_corner_radius_req, profile_tol):
        wire = choose_wire_size(params.get("min_inside_radius"), params.get("min_feature_width"))
        skims = choose_skims(profile_tol)
        p.add("wedm_windows", wire=wire, skims=skims)
    else:
        p.add("finish_mill_windows", profile_tol=profile_tol)

    # 4) Special holes from hole_sets
    hole_sets = list(params.get("hole_sets", []) or [])
    _apply_hole_sets(p, hole_sets)

    # 5) Final faces if specs call for it
    if (flatness_spec is not None) or (parallelism_spec is not None):
        p.add("surface_grind_faces", flatness=flatness_spec, parallelism=parallelism_spec)
        p.fixturing.append("Mag‑chuck with parallels; qualify datums before grind.")
    return p


def _read_plate_LW(params: Dict[str, Any]) -> Tuple[float, float]:
    # accept either plate_LxW=(L,W) or separate L,W fields
    if "plate_LxW" in params:
        L, W = params["plate_LxW"]
        return float(L), float(W)
    # fallbacks
    L = float(params.get("L") or params.get("length") or 0.0)
    W = float(params.get("W") or params.get("width") or 0.0)
    return L, W


def _apply_hole_sets(p: Plan, hole_sets: List[Dict[str, Any]]) -> None:
    for h in hole_sets:
        htype = (h.get("type") or "").lower()
        if htype == "tapped":
            dia = float(h.get("dia", 0.0) or 0.0)
            depth = float(h.get("depth", 0.0) or 0.0)
            # thread-mill if dia ≥ 0.5 or depth > 1.5×dia; else rigid tap
            if dia >= 0.5 or (depth > 1.5 * max(dia, 1e-6)):
                p.add("thread_mill", dia=dia, depth=depth)
            else:
                p.add("rigid_tap", dia=dia, depth=depth)
        elif htype in {"post_bore", "bushing_seat"}:
            tol = float(h.get("tol", 0.0) or 0.0)
            if tol <= 0.0005 or h.get("coax_pair_id"):
                p.add("assemble_and_jig_bore", tol=tol)
                # ultra-tight → jig grind cleanup
                if tol <= 0.0002:
                    p.add("jig_grind_bore", tol=tol)
            else:
                p.add("drill_and_ream_bore", tol=tol)
        elif htype == "dowel_press":
            p.add("ream_press_fit_dowel")
            p.qa.append("Verify press-fit orientation & support.")
        elif htype == "dowel_slip":
            p.add("ream_slip_fit_dowel")
            p.qa.append("Ream slip in assembly for alignment.")
        elif htype == "counterbore":
            p.add("counterbore", dia=h.get("dia"), depth=h.get("depth"), side=h.get("side"))
        elif htype in {"c_drill", "counterdrill"}:
            p.add("counterdrill", dia=h.get("dia"), depth=h.get("depth"), side=h.get("side"))
        else:
            # Unrecognized types are ignored (safe default)
            p.warnings.append(f"Ignored hole type '{htype}' for ref={h.get('ref')}")


# ---- Placeholders for other families (min viable stubs you can flesh out) --

def _stub_plan(name: str, note: str) -> Plan:
    p = base_plan()
    p.add("placeholder", family=name, note=note)
    return p


def planner_punch(params: Dict[str, Any]) -> Plan:
    return _stub_plan("punch", "Add WEDM outline, HT route, grind/lap bearing as needed.")


def planner_pilot_punch(params: Dict[str, Any]) -> Plan:
    return _stub_plan("pilot_punch", "Pilot runout defaults tight; grind to TIR if specified.")


def planner_bushing(params: Dict[str, Any]) -> Plan:
    return _stub_plan("bushing_id_critical", "Wire/drill open ID, jig grind to tol, lap for low Ra.")


def planner_cam_or_hemmer(params: Dict[str, Any]) -> Plan:
    return _stub_plan("cam_or_hemmer", "Decide WEDM vs finish-mill for slots; grind wear faces.")


def planner_flat_die_chaser(params: Dict[str, Any]) -> Plan:
    return _stub_plan("flat_die_chaser", "Standard route: mill/wire → HT → profile grind → lap.")


def planner_pm_compaction_die(params: Dict[str, Any]) -> Plan:
    return _stub_plan("pm_compaction_die", "Carbide ring route: wire, jig grind, lap to tenths.")


def planner_shear_blade(params: Dict[str, Any]) -> Plan:
    return _stub_plan("shear_blade", "HT → profile grind → match grind → hone.")


def planner_extrude_hone(params: Dict[str, Any]) -> Plan:
    return _stub_plan("extrude_hone", "Hone to target Ra; verify flow & geometry.")


# Registry
PLANNERS = {
    "die_plate": planner_die_plate,
    "punch": planner_punch,
    "pilot_punch": planner_pilot_punch,
    "bushing_id_critical": planner_bushing,
    "cam_or_hemmer": planner_cam_or_hemmer,
    "flat_die_chaser": planner_flat_die_chaser,
    "pm_compaction_die": planner_pm_compaction_die,
    "shear_blade": planner_shear_blade,
    "extrude_hone": planner_extrude_hone,
}


# ---------------------------------------------------------------------------
# Optional: tiny orchestrator to plug Dummy* helpers directly
# ---------------------------------------------------------------------------

def plan_from_helpers(dummy_dims_helper, dummy_hole_helper) -> Dict[str, Any]:
    """Convenience wrapper to call into the planner using two helper objects.

    The dims helper should expose .get_dims() -> {"L": float, "W": float, "T": float}
    The hole helper should expose .get_rows() -> list of dicts
      with keys: ref, ref_diam_text, qty, desc (your existing shape)

    Use your existing row→hole_set translator before calling plan_job
    if you want counterbores/counterdrills explicit. Otherwise, pass only
    post_bore / bushing_seat / tapped / dowel_* entries.
    """
    dims = dummy_dims_helper.get_dims()
    L, W = float(dims["L"]), float(dims["W"])
    T = float(dims.get("T", 0.0))  # may be useful to set tap depths (=T) upstream

    rows = dummy_hole_helper.get_rows()
    hole_sets = translate_rows_to_holesets(rows, plate_T=T)

    params = {
        "plate_LxW": (L, W),
        "profile_tol": 0.001,
        "windows_need_sharp": False,
        "hole_sets": hole_sets,
    }
    return plan_job("die_plate", params)


# Basic row→holeset translator (mirrors earlier adapter rules)
import re
from fractions import Fraction

FRACTIONAL_THREAD_MAJORS = {
    "5/8": 0.6250, "3/8": 0.3750, "5/16": 0.3125, "1/2": 0.5000, "1/4": 0.2500,
}
NUMBER_THREAD_MAJORS = {"#10": 0.1900, "#8": 0.1640, "#6": 0.1380, "#4": 0.1120}


def translate_rows_to_holesets(rows: List[Dict[str, Any]], plate_T: float) -> List[Dict[str, Any]]:
    hs: List[Dict[str, Any]] = []
    for r in rows:
        ref = r.get("ref"); qty = int(r.get("qty", 0) or 0)
        desc = str(r.get("desc", ""))
        if _is_jig_grind(desc):
            hs.append({"ref": ref, "qty": qty, "type": "post_bore", "tol": 0.0002})
            continue
        if "TAP" in desc.upper():
            major = _parse_thread_major(desc)
            depth = _parse_depth(desc, plate_T)
            if major is None:
                # fallback if we only saw a drill size in the diameter column
                major = 0.1900
            hs.append({"ref": ref, "qty": qty, "type": "tapped", "dia": round(float(major), 4), "depth": round(float(depth), 4)})
            continue
        if "DOWEL" in desc.upper() and "PRESS" in desc.upper():
            hs.append({"ref": ref, "qty": qty, "type": "dowel_press"}); continue
        if "DOWEL" in desc.upper() and ("SLIP" in desc.upper() or "SLP" in desc.upper()):
            hs.append({"ref": ref, "qty": qty, "type": "dowel_slip"}); continue
        # counterbore / counterdrill (optional)
        if "C'BORE" in desc.upper() or "CBORE" in desc.upper() or "COUNTERBORE" in desc.upper():
            m = re.search(r"X\s*([0-9.]+)\s*DEEP", desc, flags=re.I)
            depth = float(m.group(1)) if m else 0.0
            side = "back" if "BACK" in desc.upper() else ("front" if "FRONT" in desc.upper() else None)
            # expose explicitly if you added support in planner
            hs.append({"ref": ref, "qty": qty, "type": "counterbore", "depth": depth, "side": side})
            continue
        if "C'DRILL" in desc.upper() or "COUNTERDRILL" in desc.upper():
            m = re.search(r"X\s*([0-9.]+)\s*DEEP", desc, flags=re.I)
            depth = float(m.group(1)) if m else 0.0
            side = "back" if "BACK" in desc.upper() else ("front" if "FRONT" in desc.upper() else None)
            hs.append({"ref": ref, "qty": qty, "type": "c_drill", "depth": depth, "side": side})
            continue
        # Otherwise, THRU holes are covered by base drill ops.
    return hs


def _is_jig_grind(desc: str) -> bool:
    d = desc.upper()
    return ("JIG GRIND" in d) or ("±.0001" in d or "+/-.0001" in d or "± 0.0001" in d)


def _parse_depth(desc: str, plate_T: float) -> float:
    if "THRU" in desc.upper():
        return float(plate_T)
    m = re.search(r"[Xx]\s*([0-9.]+)\s*DEEP", desc, flags=re.I)
    return float(m.group(1)) if m else 0.0


def _parse_thread_major(desc: str) -> Optional[float]:
    m = re.search(r"(\d+/\d+|#\d+)\s*-\s*\d+", desc)
    if not m:
        return None
    nom = m.group(1)
    if nom.startswith("#"):
        return NUMBER_THREAD_MAJORS.get(nom, None)
    return FRACTIONAL_THREAD_MAJORS.get(nom, float(Fraction(nom)))


# ---------------------------------------------------------------------------
# Quick self-test (remove or keep for dev)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: tiny plate, dims from DummyDimsHelper-like output
    params = {
        "plate_LxW": (8.72, 3.247),
        "profile_tol": 0.001,
        "flatness_spec": None,
        "parallelism_spec": None,
        "windows_need_sharp": False,
        "hole_sets": [
            {"ref": "A", "qty": 4, "type": "post_bore", "tol": 0.0002},
            {"ref": "C", "qty": 6, "type": "tapped", "dia": 0.6250, "depth": 0.25},
        ],
    }
    out = plan_job("die_plate", params)
    import json
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# Family picker (keyword-driven) — optional helper
# ---------------------------------------------------------------------------

from typing import Iterable

# Minimal keyword sets; expand per your shop’s vocabulary.
_FAM_KEYWORDS = {
    "die_plate": {
        "any": {
            # common plate words
            "plate", "shoe", "punch shoe", "die set", "punch holder", "stripper",
            # hole table signals (often present on plates)
            "hole table", "c'boRE", "counterbore", "c'drill", "counterdrill",
        },
        "none_of": set(),
    },
    "punch": {
        "any": {"punch detail", "pilot punch", "bearing land", "edge hone"},
        "none_of": {"shoe", "die set"},
    },
    "bushing_id_critical": {
        "any": {"bushing", "id grind", "jig grind id", "retainer"},
        "none_of": set(),
    },
    "cam_or_hemmer": {
        "any": {"cam", "hemmer", "slot cam", "cam slot"},
        "none_of": set(),
    },
    "shear_blade": {
        "any": {"shear blade", "knife", "edge hone", "match grind"},
        "none_of": set(),
    },
}

_EXTRA_OP_KEYWORDS = [
    ("edge_break_all", {"break all outside sharp corners", "break all edges"}),
    ("etch_marking", {"etch", "etch on detail", "laser etch"}),
    ("pipe_tap", {"n.p.t", "npt"}),
    ("callout_coords", {"list of coordinates", "see sheet 2 for hole chart"}),
]


def _normalize_lines(raw_text: str | Iterable[str]) -> List[str]:
    if isinstance(raw_text, str):
        lines = raw_text.splitlines()
    else:
        lines = list(raw_text)
    return [ln.strip().lower() for ln in lines if ln and ln.strip()]


def pick_family_and_hints(all_text: str | Iterable[str]) -> Dict[str, Any]:
    """Heuristic family picker + extra-op hints from CAD text dump.

    Returns: {
      "family": str | None,
      "extra_ops": list[{op: str, ...}],
      "notes": list[str]
    }
    """
    lines = _normalize_lines(all_text)
    blob = "\n".join(lines)

    # 1) Family scores
    scores: Dict[str, int] = {}
    for fam, rules in _FAM_KEYWORDS.items():
        s = 0
        for kw in rules["any"]:
            if kw in blob:
                s += 1
        for kw in rules.get("none_of", set()):
            if kw in blob:
                s -= 2
        scores[fam] = s

    # winner if non-zero and unique top
    fam_sorted = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    chosen_family: Optional[str] = None
    if fam_sorted and fam_sorted[0][1] > 0:
        if len(fam_sorted) == 1 or fam_sorted[0][1] > fam_sorted[1][1]:
            chosen_family = fam_sorted[0][0]

    # 2) Extra ops
    extra_ops: List[Dict[str, Any]] = []
    notes: List[str] = []
    for op_name, kws in _EXTRA_OP_KEYWORDS:
        if any(kw in blob for kw in kws):
            extra_ops.append({"op": op_name})
            notes.append(f"Detected '{op_name}' from text cues")

    return {"family": chosen_family, "extra_ops": extra_ops, "notes": notes}


def plan_with_text(fallback_family: str, params: Dict[str, Any], all_text: str | Iterable[str]) -> Dict[str, Any]:
    """Pick family from text (if possible), then generate plan and append extra ops.

    - If no family is confidently detected, uses fallback_family.
    - Any detected extra ops are appended at the end.
    """
    hints = pick_family_and_hints(all_text)
    fam = hints.get("family") or fallback_family
    plan = plan_job(fam, params)
    for x in hints.get("extra_ops", []):
        plan["ops"].append(x)
    if hints.get("notes"):
        plan.setdefault("warnings", []).extend(hints["notes"])  # surface detection notes
    return plan


# ---------------------------------------------------------------------------
# CAD File Integration (geo_dump + PaddleOCR)
# ---------------------------------------------------------------------------

def extract_dimensions_from_cad(file_path: str | Path) -> Optional[Tuple[float, float, float]]:
    """Extract L, W, T dimensions from CAD file using PaddleOCR.

    Returns (length, width, thickness) in inches, or None if extraction fails.
    """
    try:
        import sys
        from pathlib import Path

        # Add tools directory to path if needed
        tools_dir = Path(__file__).resolve().parent.parent.parent / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))

        from paddle_dims_extractor import PaddleOCRDimensionExtractor, DrawingRenderer
        from PIL import Image
        import tempfile

        file_path = Path(file_path)

        # Render CAD to PNG
        with tempfile.TemporaryDirectory(prefix="cad_dims_") as tmpdir:
            png_path = Path(tmpdir) / f"{file_path.stem}_render.png"
            renderer = DrawingRenderer(verbose=False)
            renderer.render(str(file_path), str(png_path))

            # Extract dimensions with OCR
            extractor = PaddleOCRDimensionExtractor(verbose=False)
            with Image.open(png_path) as img:
                img = img.convert("RGB")
                dims = extractor.extract(img)

            if dims:
                return (dims.length, dims.width, dims.thickness)
            return None

    except Exception as e:
        print(f"[WARN] PaddleOCR dimension extraction failed: {e}")
        return None


def extract_hole_table_from_cad(file_path: str | Path) -> List[Dict[str, Any]]:
    """Extract hole table from CAD file using geo_dump.

    Returns list of hole dicts with keys: HOLE, REF_DIAM, QTY, DESCRIPTION
    """
    try:
        import sys
        from pathlib import Path

        # Add cad_quoter to path if needed
        cad_quoter_dir = Path(__file__).resolve().parent.parent
        if str(cad_quoter_dir) not in sys.path:
            sys.path.insert(0, str(cad_quoter_dir.parent))

        from cad_quoter.geo_dump import extract_hole_table_from_file

        holes = extract_hole_table_from_file(file_path)
        return holes

    except Exception as e:
        print(f"[WARN] Hole table extraction failed: {e}")
        return []


def extract_hole_operations_from_cad(file_path: str | Path) -> List[Dict[str, Any]]:
    """Extract hole operations (expanded) from CAD file using geo_dump.

    Returns list of operation dicts with keys: HOLE, REF_DIAM, QTY, OPERATION
    This expands multi-operation holes into separate entries (e.g., drill then tap).
    """
    try:
        import sys
        from pathlib import Path

        # Add cad_quoter to path if needed
        cad_quoter_dir = Path(__file__).resolve().parent.parent
        if str(cad_quoter_dir) not in sys.path:
            sys.path.insert(0, str(cad_quoter_dir.parent))

        from cad_quoter.geo_dump import extract_hole_operations_from_file

        ops = extract_hole_operations_from_file(file_path)
        return ops

    except Exception as e:
        print(f"[WARN] Hole operations extraction failed: {e}")
        return []


def extract_all_text_from_cad(file_path: str | Path) -> List[str]:
    """Extract all text from CAD file using geo_dump.

    Returns list of text strings.
    """
    try:
        import sys
        from pathlib import Path

        # Add cad_quoter to path if needed
        cad_quoter_dir = Path(__file__).resolve().parent.parent
        if str(cad_quoter_dir) not in sys.path:
            sys.path.insert(0, str(cad_quoter_dir.parent))

        from cad_quoter.geo_dump import extract_all_text_from_file

        text_records = extract_all_text_from_file(file_path)
        return [r["text"] for r in text_records]

    except Exception as e:
        print(f"[WARN] Text extraction failed: {e}")
        return []


def plan_from_cad_file(
    file_path: str | Path,
    fallback_family: str = "die_plate",
    use_paddle_ocr: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """High-level API: Generate process plan directly from a CAD file.

    This function:
    1. Extracts dimensions (L, W, T) using PaddleOCR (if use_paddle_ocr=True)
    2. Extracts hole table using geo_dump
    3. Extracts all text for family detection
    4. Converts hole table to hole_sets format
    5. Auto-detects family from text (or uses fallback)
    6. Generates complete process plan

    Args:
        file_path: Path to DXF or DWG file
        fallback_family: Family to use if auto-detection fails (default: "die_plate")
        use_paddle_ocr: Whether to use PaddleOCR for dimensions (default: True)
        verbose: Print extraction progress (default: False)

    Returns:
        Process plan dict with keys: ops, fixturing, qa, warnings, directs

    Example:
        >>> plan = plan_from_cad_file("301.dxf")
        >>> for op in plan["ops"]:
        ...     print(f"{op['op']}: {op}")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CAD file not found: {file_path}")

    if verbose:
        print(f"[PLANNER] Processing: {file_path.name}")

    # 1. Extract dimensions (L, W, T)
    dims = None
    if use_paddle_ocr:
        if verbose:
            print("[PLANNER] Extracting dimensions with PaddleOCR...")
        dims = extract_dimensions_from_cad(file_path)
        if dims:
            L, W, T = dims
            if verbose:
                print(f"[PLANNER] Dimensions: L={L:.3f}\", W={W:.3f}\", T={T:.3f}\"")
        else:
            if verbose:
                print("[PLANNER] Could not extract dimensions with PaddleOCR")

    # 2. Extract hole table and operations
    if verbose:
        print("[PLANNER] Extracting hole table...")
    hole_table = extract_hole_table_from_cad(file_path)
    hole_operations = extract_hole_operations_from_cad(file_path)
    hole_operation_rows = len(hole_operations)
    hole_operation_qty = _sum_hole_qty(hole_operations)
    if verbose:
        print(
            "[PLANNER] Found "
            f"{len(hole_table)} unique holes -> {hole_operation_rows} operations "
            f"totaling {hole_operation_qty} holes"
        )

    # 3. Extract all text for family detection
    if verbose:
        print("[PLANNER] Extracting text for family detection...")
    all_text = extract_all_text_from_cad(file_path)
    if verbose:
        print(f"[PLANNER] Extracted {len(all_text)} text records")

    # 4. Convert hole table to hole_sets format
    hole_sets = _convert_hole_table_to_hole_sets(hole_table)

    # 5. Build params dict
    params: Dict[str, Any] = {
        "hole_sets": hole_sets,
    }

    if dims:
        L, W, T = dims
        params["plate_LxW"] = (L, W)
        params["T"] = T
    else:
        # Provide defaults if dimensions not extracted
        params["plate_LxW"] = (0.0, 0.0)
        params["T"] = 0.0

    # Optional: Set reasonable defaults for other params
    params.setdefault("profile_tol", 0.001)
    params.setdefault("windows_need_sharp", False)

    # 6. Generate plan with auto family detection
    if verbose:
        print("[PLANNER] Generating process plan...")
    plan = plan_with_text(fallback_family, params, all_text)

    # Add source info to plan
    plan["source_file"] = str(file_path)
    if dims:
        plan["extracted_dims"] = {"L": L, "W": W, "T": T}
    plan["extracted_holes"] = len(hole_table)
    plan["extracted_hole_operations"] = hole_operation_qty
    plan["extracted_hole_operation_rows"] = hole_operation_rows

    if verbose:
        print(f"[PLANNER] Plan complete: {len(plan['ops'])} operations")

    return plan


def _convert_hole_table_to_hole_sets(hole_table: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert geo_dump hole table format to process_planner hole_sets format.

    Input format (from geo_dump):
        {"HOLE": "A", "REF_DIAM": "Ø0.7500", "QTY": 4, "DESCRIPTION": "THRU"}

    Output format (for process_planner):
        {"ref": "A", "dia": 0.7500, "qty": 4, "type": "thru", ...}
    """
    hole_sets = []

    for hole in hole_table:
        ref = hole.get("HOLE", "")
        ref_diam = hole.get("REF_DIAM", "")
        qty = hole.get("QTY", 0)
        desc = (hole.get("DESCRIPTION", "") or "").upper()

        # Parse diameter from REF_DIAM (e.g., "Ø0.7500" -> 0.7500)
        dia = _parse_diameter(ref_diam)

        # Determine hole type from description
        hole_entry: Dict[str, Any] = {
            "ref": ref,
            "dia": dia,
            "qty": qty,
        }

        # Classify hole type based on description keywords
        if "JIG GRIND" in desc or "±.0001" in desc or "±0.0001" in desc:
            hole_entry["type"] = "post_bore"
            hole_entry["tol"] = 0.0002
        elif "TAP" in desc:
            hole_entry["type"] = "tapped"
            # Extract tap depth if present (e.g., "X .25 DEEP")
            depth = _parse_depth(desc)
            if depth:
                hole_entry["depth"] = depth
        elif "DOWEL" in desc and "PRESS" in desc:
            hole_entry["type"] = "dowel_press"
        elif "DOWEL" in desc and ("SLIP" in desc or "SLP" in desc):
            hole_entry["type"] = "dowel_slip"
        elif "C'BORE" in desc or "CBORE" in desc or "COUNTERBORE" in desc:
            hole_entry["type"] = "counterbore"
            depth = _parse_depth(desc)
            if depth:
                hole_entry["depth"] = depth
            # Determine side (FRONT or BACK)
            if "BACK" in desc:
                hole_entry["side"] = "back"
            elif "FRONT" in desc:
                hole_entry["side"] = "front"
        elif "C'DRILL" in desc or "COUNTERDRILL" in desc:
            hole_entry["type"] = "c_drill"
            depth = _parse_depth(desc)
            if depth:
                hole_entry["depth"] = depth
            if "BACK" in desc:
                hole_entry["side"] = "back"
            elif "FRONT" in desc:
                hole_entry["side"] = "front"
        elif "THRU" in desc:
            hole_entry["type"] = "thru"
        else:
            # Default to thru if no specific operation identified
            hole_entry["type"] = "thru"

        hole_sets.append(hole_entry)

    return hole_sets


def _sum_hole_qty(rows: Iterable[Dict[str, Any]]) -> int:
    """Return the total quantity from a collection of hole operation rows."""

    total = 0
    for row in rows:
        qty = row.get("QTY", 0)
        if isinstance(qty, str):
            qty = qty.strip()
            if not qty:
                continue
            try:
                qty_val = float(qty)
            except ValueError:
                continue
        elif isinstance(qty, (int, float)):
            qty_val = qty
        else:
            continue

        if qty_val < 0:
            continue

        total += int(round(qty_val))

    return total


def _parse_diameter(ref_diam: str) -> float:
    """Parse diameter from string like 'Ø0.7500' or '1/2'."""
    import re
    from fractions import Fraction

    # Remove Ø symbol
    s = ref_diam.replace("Ø", "").replace("∅", "").strip()

    # Try fraction first (e.g., "1/2")
    if "/" in s:
        try:
            return float(Fraction(s))
        except Exception:
            pass

    # Try decimal
    try:
        return float(s)
    except Exception:
        return 0.0


def _parse_depth(desc: str) -> Optional[float]:
    """Parse depth from description like 'X .25 DEEP' or 'X 0.125 DEEP'."""
    import re

    m = re.search(r"[Xx]\s*([0-9.]+)\s*DEEP", desc)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Labor Estimation (merged from LaborOpsHelper.py)
# ---------------------------------------------------------------------------

@dataclass
class LaborInputs:
    """
    Input parameters for labor minute calculations.

    This dataclass captures all the metrics needed to estimate human labor time
    across five buckets: Setup, Programming, Machining, Inspection, and Finishing.
    """
    # Core tallies
    ops_total: int = 0
    holes_total: int = 0

    # Setup drivers (counts)
    tool_changes: int = 0
    fixturing_complexity: int = 0  # 0=none, 1=light, 2=moderate, 3=complex

    # Features / special processes
    edm_window_count: int = 0
    edm_skim_passes: int = 0
    thread_mill: int = 0
    jig_grind_bore_qty: int = 0
    grind_face_pairs: int = 0
    deep_holes: int = 0
    counterbore_qty: int = 0
    counterdrill_qty: int = 0
    ream_press_dowel: int = 0
    ream_slip_dowel: int = 0
    tap_rigid: int = 0
    tap_npt: int = 0

    # Logistics / flow
    outsource_touches: int = 0  # heat treat, coating, etc.

    # Sampling — e.g., every 5th part => 0.2
    inspection_frequency: float = 0.0

    # Handling
    part_flips: int = 0


def setup_minutes(i: LaborInputs) -> float:
    """
    Calculate Setup / Prep labor minutes (base = 10).

    NOTE: part_flips intentionally EXCLUDED from setup and counted in Machining.

    Setup minutes =
        10
      + 2·tool_changes
      + 5·fixturing_complexity
      + 2·edm_window_count
      + 1·edm_skim_passes
      + 3·grind_face_pairs
      + 2·jig_grind_bore_qty
      + 1·(tap_rigid + thread_mill)
      + 2·tap_npt
      + 1·(ream_press_dowel + ream_slip_dowel)
      + 1·(counterbore_qty + counterdrill_qty)
      + 4·outsource_touches
    """
    return (
        10
        + 2 * i.tool_changes
        + 5 * i.fixturing_complexity
        + 2 * i.edm_window_count
        + 1 * i.edm_skim_passes
        + 3 * i.grind_face_pairs
        + 2 * i.jig_grind_bore_qty
        + 1 * (i.tap_rigid + i.thread_mill)
        + 2 * i.tap_npt
        + 1 * (i.ream_press_dowel + i.ream_slip_dowel)
        + 1 * (i.counterbore_qty + i.counterdrill_qty)
        + 4 * i.outsource_touches
    )


def programming_minutes(i: LaborInputs) -> float:
    """
    Calculate Programming / Prove-out labor minutes.

    Programming minutes =
        1·holes_total
      + 2·edm_window_count
      + 2·thread_mill
      + 2·jig_grind_bore_qty
      + 1·deep_holes
      + 1·grind_face_pairs

    Note: Holes appear here and in Inspection by design. If you consider that
    "double counting" across buckets, set the `holes_total` term to 0 here.
    """
    return (
        1 * i.holes_total
        + 2 * i.edm_window_count
        + 2 * i.thread_mill
        + 2 * i.jig_grind_bore_qty
        + 1 * i.deep_holes
        + 1 * i.grind_face_pairs
    )


def machining_minutes(i: LaborInputs) -> float:
    """
    Calculate Machining Steps labor minutes.

    Human time while the machine runs: load, chip clear, coolant checks, tool swaps oversight.

    Machining minutes =
        0.5·ops_total
      + 0.2·holes_total
      + 0.5·tool_changes
      + 0.5·part_flips      # moved here per user request
      + 1·deep_holes
      + 1·edm_window_count
      + 0.5·edm_skim_passes
      + 1·grind_face_pairs
    """
    return (
        0.5 * i.ops_total
        + 0.2 * i.holes_total
        + 0.5 * i.tool_changes
        + 0.5 * i.part_flips
        + 1 * i.deep_holes
        + 1 * i.edm_window_count
        + 0.5 * i.edm_skim_passes
        + 1 * i.grind_face_pairs
    )


def inspection_minutes(i: LaborInputs) -> float:
    """
    Calculate Inspection labor minutes (base = 6 minutes).

    Includes hole-driven and feature-driven checks.

    Inspection minutes =
        6
      + 1·holes_total
      + 2·jig_grind_bore_qty
      + 1·ream_press_dowel
      + 1·ream_slip_dowel
      + 0.5·(counterbore_qty + counterdrill_qty)
      + 1·deep_holes
      + 2·grind_face_pairs
      + 1·edm_window_count
      + inspection_frequency·ops_total
    """
    return (
        6
        + 1 * i.holes_total
        + 2 * i.jig_grind_bore_qty
        + 1 * i.ream_press_dowel
        + 1 * i.ream_slip_dowel
        + 0.5 * (i.counterbore_qty + i.counterdrill_qty)
        + 1 * i.deep_holes
        + 2 * i.grind_face_pairs
        + 1 * i.edm_window_count
        + i.inspection_frequency * i.ops_total
    )


def finishing_minutes(i: LaborInputs) -> float:
    """
    Calculate Finishing labor minutes.

    Deburr/edge break, cosmetic touch-ups, clean, bag/label.

    Finishing minutes =
        0.5·ops_total
      + 0.2·holes_total
      + 0.5·(counterbore_qty + counterdrill_qty)
      + 0.5·(ream_press_dowel + ream_slip_dowel)
      + 1·grind_face_pairs
      + 1·outsource_touches
    """
    return (
        0.5 * i.ops_total
        + 0.2 * i.holes_total
        + 0.5 * (i.counterbore_qty + i.counterdrill_qty)
        + 0.5 * (i.ream_press_dowel + i.ream_slip_dowel)
        + 1 * i.grind_face_pairs
        + 1 * i.outsource_touches
    )


def compute_labor_minutes(i: LaborInputs) -> Dict[str, Any]:
    """
    Calculate labor minutes across all buckets.

    Returns a dict with per-bucket minutes in this order:
      1) Setup
      2) Programming
      3) Machining_Steps
      4) Inspection
      5) Finishing

    Plus a Labor_Total field.

    Args:
        i: LaborInputs dataclass with all the operation counts

    Returns:
        Dict with 'inputs' (as dict) and 'minutes' (breakdown by bucket)

    Example:
        >>> from dataclasses import asdict
        >>> inputs = LaborInputs(ops_total=15, holes_total=22, tool_changes=8)
        >>> result = compute_labor_minutes(inputs)
        >>> print(result['minutes']['Labor_Total'])
    """
    from dataclasses import asdict

    setup = setup_minutes(i)
    programming = programming_minutes(i)
    machining = machining_minutes(i)
    inspection = inspection_minutes(i)
    finishing = finishing_minutes(i)

    buckets = {
        "Setup": setup,
        "Programming": programming,
        "Machining_Steps": machining,
        "Inspection": inspection,
        "Finishing": finishing,
    }
    buckets["Labor_Total"] = sum(buckets.values())

    return {
        "inputs": asdict(i),
        "minutes": buckets,
    }


# ---------------------------------------------------------------------------
# Machine Time Estimation (using speeds_feeds_merged.csv)
# ---------------------------------------------------------------------------

# Cache for speeds/feeds data
_SPEEDS_FEEDS_CACHE: Optional[List[Dict[str, Any]]] = None


def load_speeds_feeds_data() -> List[Dict[str, Any]]:
    """
    Load speeds and feeds data from CSV file.

    Returns cached data if already loaded.
    """
    global _SPEEDS_FEEDS_CACHE

    if _SPEEDS_FEEDS_CACHE is not None:
        return _SPEEDS_FEEDS_CACHE

    import csv
    from pathlib import Path

    # Find the CSV file
    csv_path = Path(__file__).resolve().parent.parent / "pricing" / "resources" / "speeds_feeds_merged.csv"

    if not csv_path.exists():
        print(f"[WARN] Speeds/feeds CSV not found: {csv_path}")
        _SPEEDS_FEEDS_CACHE = []
        return _SPEEDS_FEEDS_CACHE

    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ['sfm_start', 'fz_ipr_0_125in', 'fz_ipr_0_25in', 'fz_ipr_0_5in',
                       'doc_axial_in', 'woc_radial_pct', 'linear_cut_rate_ipm']:
                if row.get(key) and row[key].strip():
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
                else:
                    row[key] = None
            data.append(row)

    _SPEEDS_FEEDS_CACHE = data
    return data


def get_speeds_feeds(material: str, operation: str) -> Optional[Dict[str, Any]]:
    """
    Look up speeds and feeds for a material and operation.

    Falls back to GENERIC if specific material not found.

    Args:
        material: Material name (e.g., "Aluminum 6061-T6", "P20 Tool Steel")
        operation: Operation type (e.g., "Drill", "Endmill_Profile", "Wire_EDM_Rough")

    Returns:
        Dict with speeds/feeds data, or None if not found
    """
    data = load_speeds_feeds_data()

    # Try exact match first
    for row in data:
        if row['material'] == material and row['operation'] == operation:
            return row

    # Fall back to GENERIC
    for row in data:
        if row['material_group'] == 'GENERIC' and row['operation'] == operation:
            return row

    return None


def calculate_drill_time(
    diameter: float,
    depth: float,
    qty: int,
    material: str = "GENERIC",
    is_deep_hole: bool = False
) -> float:
    """
    Calculate drilling time in minutes.

    Args:
        diameter: Hole diameter in inches
        depth: Hole depth in inches
        qty: Number of holes
        material: Material name
        is_deep_hole: Whether this is a deep hole (depth > 3*diameter)

    Returns:
        Time in minutes
    """
    operation = "Deep_Drill" if is_deep_hole or (depth > 3 * diameter) else "Drill"
    sf = get_speeds_feeds(material, operation)

    if not sf:
        # Fallback estimate: 0.5 min per inch depth per hole
        return depth * qty * 0.5

    # Select feed based on diameter
    if diameter <= 0.1875:  # <= 3/16"
        feed = sf.get('fz_ipr_0_125in') or 0.001
    elif diameter <= 0.375:  # <= 3/8"
        feed = sf.get('fz_ipr_0_25in') or 0.0015
    else:
        feed = sf.get('fz_ipr_0_5in') or 0.002

    sfm = sf.get('sfm_start') or 100

    # Calculate RPM
    rpm = (sfm * 12) / (3.14159 * diameter) if diameter > 0 else 1000
    rpm = min(rpm, 3000)  # Cap at typical machine limit

    # Calculate feed rate (IPM)
    ipm = feed * rpm

    # Time per hole (minutes) = depth / feed_rate + approach/retract
    time_per_hole = (depth / ipm) + 0.1  # 0.1 min for approach/retract

    return time_per_hole * qty


def calculate_milling_time(
    length: float,
    width: float,
    depth: float,
    material: str = "GENERIC",
    operation: str = "Endmill_Profile"
) -> float:
    """
    Calculate milling time in minutes.

    Args:
        length: Cut length in inches
        width: Cut width in inches
        depth: Total depth in inches
        material: Material name
        operation: "Endmill_Profile" or "Endmill_Slot"

    Returns:
        Time in minutes
    """
    sf = get_speeds_feeds(material, operation)

    if not sf:
        # Fallback: 2 minutes per square inch of material removal
        volume = length * width * depth
        return volume * 2

    # Use 0.25" tool as default
    fz = sf.get('fz_ipr_0_25in') or 0.0015
    sfm = sf.get('sfm_start') or 200
    doc = sf.get('doc_axial_in') or 0.2
    woc_pct = sf.get('woc_radial_pct') or 50

    tool_dia = 0.25
    rpm = (sfm * 12) / (3.14159 * tool_dia)
    rpm = min(rpm, 8000)

    # Feed rate
    num_flutes = 4
    ipm = fz * num_flutes * rpm

    # Number of passes needed
    axial_passes = max(1, int(depth / doc) + 1)
    radial_passes = max(1, int(100 / woc_pct))

    # Cut length per pass
    if operation == "Endmill_Slot":
        cut_length_per_pass = length
    else:
        cut_length_per_pass = 2 * (length + width)  # Perimeter

    # Total time
    total_length = cut_length_per_pass * axial_passes * radial_passes
    time = (total_length / ipm) + (axial_passes * 0.5)  # Add time for plunges

    return time


def calculate_edm_time(
    perimeter: float,
    thickness: float,
    num_windows: int,
    num_skims: int = 0,
    material: str = "GENERIC"
) -> float:
    """
    Calculate Wire EDM time in minutes.

    Args:
        perimeter: Total perimeter to cut in inches
        thickness: Part thickness in inches
        num_windows: Number of windows to cut
        num_skims: Number of skim passes
        material: Material name

    Returns:
        Time in minutes
    """
    # Rough cut
    sf_rough = get_speeds_feeds(material, "Wire_EDM_Rough")
    rough_ipm = sf_rough.get('linear_cut_rate_ipm') or 3.0 if sf_rough else 3.0

    # Skim cut
    sf_skim = get_speeds_feeds(material, "Wire_EDM_Skim")
    skim_ipm = sf_skim.get('linear_cut_rate_ipm') or 2.0 if sf_skim else 2.0

    # Time = (perimeter * thickness / rate)
    # For rough cut
    rough_time = (perimeter * thickness * num_windows) / rough_ipm

    # For skim passes
    skim_time = (perimeter * thickness * num_windows * num_skims) / skim_ipm

    # Add setup time per window (threading wire, etc.)
    setup_time = num_windows * 5  # 5 minutes per window

    return rough_time + skim_time + setup_time


def calculate_tap_time(
    diameter: float,
    depth: float,
    qty: int,
    is_rigid_tap: bool = True
) -> float:
    """
    Calculate tapping time in minutes.

    Args:
        diameter: Tap diameter in inches
        depth: Thread depth in inches
        qty: Number of holes to tap
        is_rigid_tap: Whether using rigid tapping (faster)

    Returns:
        Time in minutes
    """
    # Typical tapping speed: 20-40 SFM
    sfm = 30
    rpm = (sfm * 12) / (3.14159 * diameter) if diameter > 0 else 500
    rpm = min(rpm, 1000)  # Cap at reasonable limit

    # Feed = pitch (assume standard coarse thread)
    # For simplicity, use approximate TPI
    if diameter <= 0.25:
        tpi = 20
    elif diameter <= 0.5:
        tpi = 13
    else:
        tpi = 10

    pitch = 1.0 / tpi
    ipm = rpm * pitch

    # Time = (depth / ipm) * 2 (in and out) + dwell
    time_per_hole = ((depth / ipm) * 2) + 0.15 if ipm > 0 else 0.5

    # Rigid tap is faster
    if is_rigid_tap:
        time_per_hole *= 0.7

    return time_per_hole * qty


def estimate_machine_hours_from_plan(
    plan: Dict[str, Any],
    material: str = "GENERIC",
    plate_LxW: Tuple[float, float] = (0, 0),
    thickness: float = 0
) -> Dict[str, Any]:
    """
    Estimate machine hours from a process plan.

    Args:
        plan: Process plan dict (from plan_job or plan_from_cad_file)
        material: Material name for speeds/feeds lookup
        plate_LxW: Plate length and width in inches
        thickness: Plate thickness in inches

    Returns:
        Dict with machine time breakdown by operation type

    Example:
        >>> plan = plan_from_cad_file("part.dxf")
        >>> machine_time = estimate_machine_hours_from_plan(plan, "P20 Tool Steel", (8, 4), 0.5)
        >>> print(f"Total: {machine_time['total_hours']:.2f} hours")
    """
    from dataclasses import asdict

    ops = plan.get('ops', [])
    L, W = plate_LxW
    T = thickness

    time_breakdown = {
        'drilling': 0,
        'milling': 0,
        'edm': 0,
        'tapping': 0,
        'grinding': 0,
        'other': 0,
    }

    for op in ops:
        op_type = op.get('op', '').lower()

        # Drilling operations
        if 'drill' in op_type and 'pattern' in op_type:
            # drill_patterns - estimate based on typical plate
            num_holes = 20  # Rough estimate
            avg_depth = T or 0.5
            time_breakdown['drilling'] += calculate_drill_time(0.25, avg_depth, num_holes, material)

        elif 'spot_drill' in op_type:
            # Spot drilling - quick operation
            time_breakdown['drilling'] += 2  # 2 minutes estimate

        # Milling operations
        elif 'face_mill' in op_type or 'mill_face' in op_type:
            # Face milling
            area = L * W if (L and W) else 10  # Default 10 sq in
            time_breakdown['milling'] += calculate_milling_time(L or 3, W or 3, 0.05, material)

        elif 'endmill' in op_type or 'profile' in op_type:
            perimeter = 2 * ((L or 4) + (W or 3))
            time_breakdown['milling'] += calculate_milling_time(perimeter, 0.1, T or 0.5, material, "Endmill_Profile")

        # EDM operations
        elif 'wedm' in op_type or 'wire_edm' in op_type:
            num_windows = op.get('windows', 1)
            skims = op.get('skims', 0)
            # Estimate perimeter
            perimeter = 4  # inches, typical window
            time_breakdown['edm'] += calculate_edm_time(perimeter, T or 0.5, num_windows, skims, material)

        # Tapping operations
        elif 'tap' in op_type:
            dia = op.get('dia', 0.25)
            depth = op.get('depth', 0.5)
            qty = op.get('qty', 1)
            is_rigid = 'rigid' in op_type
            time_breakdown['tapping'] += calculate_tap_time(dia, depth, qty, is_rigid)

        elif 'thread_mill' in op_type:
            dia = op.get('dia', 0.5)
            depth = op.get('depth', 0.5)
            # Thread milling is slower than tapping
            time_breakdown['tapping'] += calculate_tap_time(dia, depth, 1, False) * 1.5

        # Counterbore operations
        elif 'counterbore' in op_type or 'c_bore' in op_type or 'cbore' in op_type:
            dia = op.get('dia', 0.5)
            depth = op.get('depth', 0.25)
            qty = op.get('qty', 1)
            time_breakdown['drilling'] += calculate_drill_time(dia, depth, qty, material)

        # Grinding operations
        elif 'grind' in op_type or 'jig_grind' in op_type:
            # Grinding is slow and precise
            if 'bore' in op_type:
                time_breakdown['grinding'] += 15  # 15 min per bore
            elif 'face' in op_type:
                area = L * W if (L and W) else 10
                time_breakdown['grinding'] += area * 2  # 2 min per sq in
            else:
                time_breakdown['grinding'] += 10  # Generic estimate

        # Assembly operations don't count as machine time
        elif 'assemble' in op_type:
            pass  # No machine time

        # Other operations
        else:
            # Generic estimate
            time_breakdown['other'] += 5  # 5 minutes

    # Calculate total
    total_minutes = sum(time_breakdown.values())

    return {
        'breakdown_minutes': time_breakdown,
        'total_minutes': total_minutes,
        'total_hours': total_minutes / 60,
        'material': material,
        'dimensions': {'L': L, 'W': W, 'T': T},
    }


def estimate_hole_table_times(
    hole_table: List[Dict[str, Any]],
    material: str = "GENERIC",
    thickness: float = 0
) -> Dict[str, Any]:
    """
    Calculate detailed time estimates for each hole table entry.

    Args:
        hole_table: List of hole table entries from extract_hole_table_from_cad()
        material: Material name for speeds/feeds lookup
        thickness: Plate thickness in inches (for THRU holes)

    Returns:
        Dict with detailed time breakdown by hole and operation type

    Example:
        >>> hole_table = extract_hole_table_from_cad("part.dxf")
        >>> times = estimate_hole_table_times(hole_table, "17-4 PH Stainless", 2.0)
        >>> print(times['drill_groups'])
    """
    import re

    # Storage for different operation types
    drill_groups = []
    jig_grind_groups = []
    tap_groups = []
    cbore_groups = []
    cdrill_groups = []

    for entry in hole_table:
        hole_id = entry.get('HOLE', '?')
        ref_diam_str = entry.get('REF_DIAM', '')
        qty_raw = entry.get('QTY', 1)
        qty = int(qty_raw) if isinstance(qty_raw, str) else qty_raw

        # Check if we have expanded operations (with OPERATION field) or compressed table (with DESCRIPTION)
        operation = entry.get('OPERATION', '').upper()
        description = entry.get('DESCRIPTION', '').upper()

        # Use OPERATION if available, otherwise use DESCRIPTION
        op_text = operation if operation else description

        # Parse diameter - match decimal after ∅ symbol or standalone
        dia_match = re.search(r'[∅Ø]\s*(\d*\.\d+)', ref_diam_str)
        if not dia_match:
            # Try fractional format like 11/32
            frac_match = re.search(r'(\d+)/(\d+)', ref_diam_str)
            if frac_match:
                ref_dia = float(frac_match.group(1)) / float(frac_match.group(2))
            else:
                dia_match = re.search(r'(\d+\.\d+)', ref_diam_str)
                ref_dia = float(dia_match.group(1)) if dia_match else 0.5
        else:
            ref_dia = float(dia_match.group(1))

        # Determine operation type
        is_jig_grind = 'JIG GRIND' in op_text
        is_thru = 'THRU' in op_text
        is_tap = 'TAP' in op_text
        is_cbore = "C'BORE" in op_text or 'CBORE' in op_text or 'COUNTERBORE' in op_text
        is_cdrill = "C'DRILL" in op_text or 'CDRILL' in op_text or 'CENTER DRILL' in op_text

        # Determine depth for drilling operation
        if is_thru and not is_jig_grind:
            depth = thickness if thickness > 0 else 2.0
        else:
            depth = 0.5  # Default for non-THRU holes

        # Get speeds/feeds for drilling
        sf_drill = get_speeds_feeds(material, "Drill")
        if not sf_drill:
            sf_drill = {'sfm_start': 100, 'fz_ipr_0_125in': 0.002, 'fz_ipr_0_25in': 0.004, 'fz_ipr_0_5in': 0.008}

        sfm = sf_drill.get('sfm_start', 100)

        # Select feed based on diameter
        if ref_dia <= 0.1875:  # <= 3/16"
            feed_per_tooth = sf_drill.get('fz_ipr_0_125in', 0.002)
        elif ref_dia <= 0.375:  # <= 3/8"
            feed_per_tooth = sf_drill.get('fz_ipr_0_25in', 0.004)
        else:
            feed_per_tooth = sf_drill.get('fz_ipr_0_5in', 0.008)

        # Calculate RPM and feed rate
        rpm = (sfm * 12) / (3.14159 * ref_dia) if ref_dia > 0 else 1000
        rpm = min(rpm, 3500)  # Max spindle RPM

        # For drilling, assume 2 flutes
        feed_rate = rpm * 2 * feed_per_tooth  # IPM

        # DRILL operations (main hole) - skip if it's a jig grind operation
        if not is_jig_grind:
            # Time per hole: depth / feed_rate gives minutes (since feed_rate is IPM)
            time_per_hole = (depth / feed_rate) if feed_rate > 0 else 1.0
            time_per_hole += 0.1  # Add approach/retract time

            total_time = time_per_hole * qty

            drill_groups.append({
                'hole_id': hole_id,
                'diameter': ref_dia,
                'depth': depth,
                'qty': qty,
                'sfm': sfm,
                'ipr': feed_per_tooth,
                'rpm': rpm,
                'feed_rate': feed_rate,
                'time_per_hole': time_per_hole,
                'total_time': total_time,
                'description': entry.get('DESCRIPTION', '')
            })

        # JIG GRIND operations (using is_jig_grind check from above)
        if is_jig_grind:
            # Jig grinding time calculation
            # Constants (can be made configurable later)
            setup_min = 0  # Setup time per bore
            mpsi = 7  # Minutes per square inch ground
            stock_diam = 0.003  # Diametral stock to remove (inches)
            stock_rate_diam = 0.003  # Diametral removal rate (inches)

            # Calculate grinding surface area: π × D × depth
            grind_area = 3.14159 * ref_dia * depth

            # Spark out time: 0.7 + 0.2 if depth ≥ 3×D
            spark_out_min = 0.7
            if depth >= 3 * ref_dia:
                spark_out_min += 0.2

            # Total time per hole
            time_per_hole = (
                setup_min +
                (grind_area * mpsi) +
                (stock_diam / stock_rate_diam) +
                spark_out_min
            )

            total_time = time_per_hole * qty

            jig_grind_groups.append({
                'hole_id': hole_id,
                'diameter': ref_dia,
                'depth': depth,
                'qty': qty,
                'time_per_hole': time_per_hole,
                'total_time': total_time,
                'description': entry.get('DESCRIPTION', '')
            })

        # TAP operations
        if is_tap:
            # Extract tap size
            tap_match = re.search(r'(\d+/\d+)-(\d+)', op_text)
            if tap_match:
                # Fractional tap (e.g., 5/8-11)
                frac_parts = tap_match.group(1).split('/')
                tap_dia = float(frac_parts[0]) / float(frac_parts[1])
                tpi = int(tap_match.group(2))
            else:
                # Try #10-32 format
                num_tap_match = re.search(r'#(\d+)-(\d+)', op_text)
                if num_tap_match:
                    # #10 screw ≈ 0.190"
                    screw_num = int(num_tap_match.group(1))
                    tap_dia = 0.060 + (screw_num * 0.013)
                    tpi = int(num_tap_match.group(2))
                else:
                    tap_dia = ref_dia * 0.8  # Estimate tap drill size
                    tpi = int(20 / tap_dia) if tap_dia > 0 else 20

            # Extract TAP depth - look for "TAP X {number} DEEP" or "X {number} DEEP"
            tap_depth_match = re.search(r'[TAP\s+]*X\s+(\d*\.\d+|\d+)\s+DEEP', op_text)
            if tap_depth_match:
                tap_depth = float(tap_depth_match.group(1))
            elif 'TAP THRU' in op_text or is_thru:
                tap_depth = thickness if thickness > 0 else 2.0
            else:
                tap_depth = 0.5  # Default

            is_rigid = 'RIGID' in op_text

            # Tapping speed (much slower than drilling)
            tap_rpm = min(rpm * 0.3, 500)
            tap_feed_rate = tap_rpm / tpi  # IPM

            time_per_hole = (tap_depth / tap_feed_rate) if tap_feed_rate > 0 else 2.0
            time_per_hole += 0.5  # Add approach/retract/dwell

            if is_rigid:
                time_per_hole *= 0.7  # Rigid tap is faster

            total_time = time_per_hole * qty

            tap_groups.append({
                'hole_id': hole_id,
                'diameter': tap_dia,
                'depth': tap_depth,
                'qty': qty,
                'tpi': tpi,
                'rpm': tap_rpm,
                'feed_rate': tap_feed_rate,
                'time_per_hole': time_per_hole,
                'total_time': total_time,
                'is_rigid': is_rigid,
                'description': entry.get('DESCRIPTION', '')
            })

        # COUNTERBORE operations
        if is_cbore:
            # For expanded operations, the counterbore diameter is in REF_DIAM
            # For compressed table, need to extract from description
            cbore_dia = ref_dia  # Start with REF_DIAM

            # If description has a different diameter specified, use that
            if description:
                cbore_dia_match = re.search(r'(\d*\.\d+)[∅Ø]\s*C[\'"]?BORE', description)
                if cbore_dia_match:
                    cbore_dia = float(cbore_dia_match.group(1))
                else:
                    # Try fractional format like "∅13/32"
                    cbore_dia_match = re.search(r'[∅Ø]\s*(\d+)/(\d+)\s*C[\'"]?BORE', description)
                    if cbore_dia_match:
                        cbore_dia = float(cbore_dia_match.group(1)) / float(cbore_dia_match.group(2))

            # Extract counterbore depth - look for "X {number} DEEP"
            cbore_depth_match = re.search(r'X\s+(\d*\.\d+|\d+)\s+DEEP', op_text)
            if cbore_depth_match:
                cbore_depth = float(cbore_depth_match.group(1))
            else:
                cbore_depth = 0.25  # Default

            # Counterboring is like drilling but larger diameter
            cbore_rpm = (sfm * 12) / (3.14159 * cbore_dia) if cbore_dia > 0 else 1000
            cbore_rpm = min(cbore_rpm, 2000)

            cbore_feed = cbore_rpm * 2 * 0.006  # IPM

            time_per_hole = (cbore_depth / cbore_feed) if cbore_feed > 0 else 0.5
            time_per_hole += 0.1

            total_time = time_per_hole * qty

            cbore_groups.append({
                'hole_id': hole_id,
                'diameter': cbore_dia,
                'depth': cbore_depth,
                'qty': qty,
                'sfm': sfm,
                'rpm': cbore_rpm,
                'feed_rate': cbore_feed,
                'time_per_hole': time_per_hole,
                'total_time': total_time,
                'description': entry.get('DESCRIPTION', '')
            })

        # CENTER DRILL operations
        if is_cdrill:
            # Extract center drill depth if specified
            cdrill_depth_match = re.search(r'[Xx]\s+(\d*\.\d+|\d+)\s+DEEP', op_text)
            if cdrill_depth_match:
                cdrill_depth = float(cdrill_depth_match.group(1))
            else:
                cdrill_depth = 0.1  # Default shallow depth

            # Center drilling is quick - estimate based on depth
            time_per_hole = max(0.05, cdrill_depth * 0.5)  # Min 3 seconds, or 30 sec per inch
            total_time = time_per_hole * qty

            cdrill_groups.append({
                'hole_id': hole_id,
                'diameter': ref_dia,
                'depth': cdrill_depth,
                'qty': qty,
                'time_per_hole': time_per_hole,
                'total_time': total_time,
                'description': entry.get('DESCRIPTION', '')
            })

    # Calculate totals
    total_drill = sum(g['total_time'] for g in drill_groups)
    total_jig_grind = sum(g['total_time'] for g in jig_grind_groups)
    total_tap = sum(g['total_time'] for g in tap_groups)
    total_cbore = sum(g['total_time'] for g in cbore_groups)
    total_cdrill = sum(g['total_time'] for g in cdrill_groups)

    total_minutes = total_drill + total_jig_grind + total_tap + total_cbore + total_cdrill

    return {
        'drill_groups': drill_groups,
        'jig_grind_groups': jig_grind_groups,
        'tap_groups': tap_groups,
        'cbore_groups': cbore_groups,
        'cdrill_groups': cdrill_groups,
        'total_drill_minutes': total_drill,
        'total_jig_grind_minutes': total_jig_grind,
        'total_tap_minutes': total_tap,
        'total_cbore_minutes': total_cbore,
        'total_cdrill_minutes': total_cdrill,
        'total_minutes': total_minutes,
        'total_hours': total_minutes / 60,
        'material': material,
        'thickness': thickness,
    }
