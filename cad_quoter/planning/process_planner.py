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
    if verbose:
        print(f"[PLANNER] Found {len(hole_table)} unique holes -> {len(hole_operations)} operations")

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
    plan["extracted_hole_operations"] = len(hole_operations)

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
