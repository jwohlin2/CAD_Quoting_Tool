"""
process_planner_v2.py — compact, rule‑based process planner with clean inputs
-----------------------------------------------------------------------------
Goals
- Keep the public API simple: plan_job(family: str, params: dict) -> dict
- Make decisions from small, testable helpers (wire size, skims, slot strategy)
- Accept hole/dimension adapters (e.g., DummyHoleHelper, DummyDimsHelper)
- Emit a stable plan schema: {ops: [...], fixturing: [...], qa: [...], warnings: [...], directs: {...}}

Families supported (extensible):
- Plates  (main one used in CAD quoting flow - die plates, shoes, flats)
- Punches  (punch, pilot punch, spring punch, guide posts)
- bushing_id_critical  (guide bushings, ring gauges)
- Sections_blocks  (cam, hemmer, die chasers, sensor blocks, die sections)
- Special_processes  (PM compaction dies, shear blades, extrude hone)

Notes
- This file stands alone; no external deps beyond stdlib.
- The Plates logic is practical and opinionated; tweak thresholds to your shop.
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
      Plates: expects at least plate_LxW=(L,W) OR L,W in params; optional hole_sets

    Returns a dict with keys: ops, fixturing, qa, warnings, directs
    """
    # Backward compatibility: map old family names to new ones
    FAMILY_ALIASES = {
        "die_plate": "Plates",
        "punch": "Punches",
        "pilot_punch": "Punches",
        "cam_or_hemmer": "Sections_blocks",
        "flat_die_chaser": "Sections_blocks",
        "pm_compaction_die": "Special_processes",
        "shear_blade": "Special_processes",
        "extrude_hone": "Special_processes",
    }

    family = (family or "").strip()

    # First check if it's an old name that needs mapping
    if family in FAMILY_ALIASES:
        family = FAMILY_ALIASES[family]
    # Then try exact match
    elif family not in PLANNERS:
        # Try case-insensitive match
        family_lower = family.lower()
        matched = None
        for key in PLANNERS:
            if key.lower() == family_lower:
                matched = key
                break
        if matched:
            family = matched
        else:
            raise ValueError(f"Unsupported family '{family}'. Known: {sorted(PLANNERS)}")

    # Validate family classification and potentially correct it
    corrected_family, classification_warning = _validate_family_classification(family, params)
    if corrected_family != family:
        family = corrected_family

    plan = PLANNERS[family](params)

    # Add classification warning if present
    if classification_warning:
        plan_dict = normalize_plan(plan)
        plan_dict.setdefault("warnings", []).append(classification_warning)
        return plan_dict

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
        # Preserve meta key for die sections and other families with metadata
        if "meta" in plan:
            d["meta"] = dict(plan["meta"])
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


def _validate_thickness_removal(
    stock_thk: float,
    finished_thk: float,
    rough_mill_thk: float,
    grind_thk_total: float,
    context: str = ""
) -> Optional[str]:
    """Validate thickness removal consistency.

    Args:
        stock_thk: Starting stock thickness (inches)
        finished_thk: Final finished thickness (inches)
        rough_mill_thk: Thickness removed by rough milling (inches)
        grind_thk_total: Thickness removed by grinding (inches)
        context: Context string for the warning message

    Returns:
        Warning message if mismatch > 0.050", None otherwise
    """
    modeled_removed = rough_mill_thk + grind_thk_total
    actual_removed = stock_thk - finished_thk
    difference = abs(actual_removed - modeled_removed)

    if difference > 0.050:
        warning = (
            f"THICKNESS REMOVAL MISMATCH{' (' + context + ')' if context else ''}: "
            f"|actual_removed - modeled_removed| = {difference:.4f}\" > 0.050\"\n"
            f"  stock_thk={stock_thk:.4f}\", finished_thk={finished_thk:.4f}\"\n"
            f"  rough_mill_thk={rough_mill_thk:.4f}\", grind_thk_total={grind_thk_total:.4f}\"\n"
            f"  actual_removed={actual_removed:.4f}\", modeled_removed={modeled_removed:.4f}\""
        )
        print(f"DEBUG: {warning}")
        return warning

    # Debug output even when no warning
    print(f"DEBUG: Thickness removal validation{' (' + context + ')' if context else ''}:")
    print(f"  stock_thk={stock_thk:.4f}\", finished_thk={finished_thk:.4f}\"")
    print(f"  rough_mill_thk={rough_mill_thk:.4f}\", grind_thk_total={grind_thk_total:.4f}\"")
    print(f"  actual_removed={actual_removed:.4f}\", modeled_removed={modeled_removed:.4f}\"")
    print(f"  difference={difference:.4f}\" (OK)")

    return None


def _has_cylindrical_feature(params: Dict[str, Any], min_length_threshold: float = 0.25) -> bool:
    """Check if part has cylindrical features long enough to warrant turning operations.

    Args:
        params: Part parameters dictionary
        min_length_threshold: Minimum cylindrical section length (inches) to qualify

    Returns:
        True if part has at least one cylindrical section with length > threshold
    """
    # Check for round shape type (punch family)
    if params.get("shape_type") == "round":
        # Check if there are ground diameters with sufficient length
        num_diams = params.get("num_ground_diams", 0)
        total_ground_length = params.get("total_ground_length_in", 0.0)
        if num_diams > 0 and total_ground_length > min_length_threshold:
            return True

    # Check for explicit cylindrical sections in feature map
    feature_map = params.get("feature_map", {})
    if feature_map:
        cylindrical_sections = feature_map.get("cylindrical_sections", [])
        for section in cylindrical_sections:
            section_length = section.get("length", 0.0)
            if section_length > min_length_threshold:
                return True

    # Check for OD dimensions that suggest cylindrical geometry
    max_od = params.get("max_od_or_width_in", 0.0)
    overall_length = params.get("overall_length_in", 0.0)
    if max_od > 0 and overall_length > min_length_threshold:
        # If we have a diameter and sufficient length, might be cylindrical
        # But only if not explicitly classified as rectangular
        has_width_thickness = (
            params.get("body_width_in") is not None or
            params.get("width_in") is not None
        )
        if not has_width_thickness:
            return True

    return False


def _validate_family_classification(family: str, params: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Validate and potentially correct family classification based on geometry and features.

    Family classification guards:
    - If thickness < min(length, width) AND no long cylindrical shaft AND
      majority ops are drilling/EDM/milling on a rectangular profile
      → classify as "die_section" / "block_plate", not "form_punch" or "round_punch"

    Args:
        family: Proposed family classification
        params: Part parameters dictionary

    Returns:
        Tuple of (corrected_family, warning_message)
    """
    # Extract dimensions based on family type
    if family in ["Punches", "punch", "round_punch", "form_punch"]:
        # For punches, check if this should actually be a block/plate
        length = params.get("overall_length_in", 0.0)
        width = params.get("max_od_or_width_in", 0.0) or params.get("body_width_in", 0.0)
        thickness = params.get("body_thickness_in", 0.0)

        # If no thickness specified, might be round - use diameter as "thickness"
        if thickness == 0.0 and params.get("shape_type") == "round":
            thickness = width  # For round parts, OD is effectively the "thickness"

    elif family in ["Sections_blocks", "die_section", "cam_or_hemmer"]:
        length = params.get("length_in", 0.0) or params.get("L", 0.0)
        width = params.get("width_in", 0.0) or params.get("W", 0.0)
        thickness = params.get("thickness_in", 0.0) or params.get("T", 0.0)

    elif family in ["Plates", "die_plate"]:
        length = params.get("L", 0.0) or params.get("length", 0.0)
        width = params.get("W", 0.0) or params.get("width", 0.0)
        thickness = params.get("T", 0.0) or params.get("thickness", 0.0)

    else:
        # Unknown family, can't validate
        return family, None

    # Skip validation if dimensions are missing
    if length == 0.0 or width == 0.0 or thickness == 0.0:
        return family, None

    # Check if this looks like a block/plate (thickness < min(length, width))
    min_planar_dim = min(length, width)
    is_plate_like = thickness < min_planar_dim

    # Check for cylindrical features
    has_cylindrical = _has_cylindrical_feature(params, min_length_threshold=0.25)

    # Check for rectangular/plate operations
    has_drilling = params.get("hole_count", 0) > 0 or len(params.get("hole_sets", [])) > 0
    has_edm = params.get("requires_wire_edm", False) or params.get("has_wire_profile", False)
    has_milling = params.get("has_internal_form", False) or params.get("has_edge_form", False)
    majority_ops_rectangular = has_drilling or has_edm or has_milling

    # Classification logic
    if is_plate_like and not has_cylindrical and majority_ops_rectangular:
        # This should be classified as a block/plate, not a punch
        if family in ["Punches", "punch", "round_punch", "form_punch"]:
            corrected_family = "Sections_blocks"
            warning = (
                f"FAMILY CLASSIFICATION CORRECTED: {family} → {corrected_family}\n"
                f"  Reason: thickness ({thickness:.3f}\") < min(length, width) ({min_planar_dim:.3f}\"), "
                f"no cylindrical features > 0.25\", and majority ops are drilling/EDM/milling\n"
                f"  Dimensions: L={length:.3f}\", W={width:.3f}\", T={thickness:.3f}\""
            )
            print(f"DEBUG: {warning}")
            return corrected_family, warning

    # Family is appropriate
    return family, None


# ---------------------------------------------------------------------------
# Family planners
# ---------------------------------------------------------------------------

# ---- die_plate -------------------------------------------------------------

def planner_die_plate(params: Dict[str, Any]) -> Plan:
    p = base_plan()

    # Dimensions
    L, W = _read_plate_LW(params)
    T = float(params.get("T") or params.get("thickness") or 0.0)
    profile_tol = params.get("profile_tol")  # float in inches
    flatness_spec = params.get("flatness_spec")
    parallelism_spec = params.get("parallelism_spec")
    windows_need_sharp = bool(params.get("windows_need_sharp", False))
    window_corner_radius_req = params.get("window_corner_radius_req")
    material = params.get("material", "GENERIC")

    # Squaring/Finishing strategy with volume-based time calculation
    if L > 5.0 and W > 5.0 and T > 1.0:
        # Get starting stock dimensions (default: +0.50" L/W, +0.25" T)
        stock_L = params.get('stock_length', L + 0.50)
        stock_W = params.get('stock_width', W + 0.50)
        stock_T = params.get('stock_thickness', T + 0.25)

        # Calculate volume removed
        volume_start = stock_L * stock_W * stock_T
        volume_finish = L * W * T
        volume_removed = volume_start - volume_finish

        # Split: 90% milling (rough), 10% grinding (finish)
        volume_milling = volume_removed * 0.90
        volume_grinding = volume_removed * 0.10

        # Get material-specific removal rates
        rates = get_material_removal_rates(material)
        milling_rate = rates['milling_min_per_cuin']
        grinding_rate = rates['grinding_min_per_cuin']
        setup_time = rates['setup_overhead_min']
        flip_time = rates['flip_deburr_min']

        # Calculate raw cutting times
        milling_time = volume_milling * milling_rate
        grinding_time = volume_grinding * grinding_rate

        # Total time with overhead
        total_square_up_time = milling_time + grinding_time + setup_time + flip_time

        # Thickness removal calculations for validation
        stock_thk = stock_T
        finished_thk = T
        total_thickness_to_remove = stock_thk - finished_thk

        # Calculate thickness removed by each operation
        # For square-up: thickness removal comes from top/bottom faces
        rough_mill_thk = total_thickness_to_remove * 0.90  # 90% by rough milling
        grind_thk_total = total_thickness_to_remove * 0.10  # 10% by grinding

        # Calculate modeled thickness removed for square-up specifically
        # radial_stock affects sides, face operations affect top/bottom
        modeled_thickness_removed_squareup = rough_mill_thk + grind_thk_total

        # DEBUG output for square-up
        print(f"DEBUG: Square-up operation (die_plate):")
        print(f"  final_thickness={finished_thk:.4f}\", stock_thickness={stock_thk:.4f}\"")
        print(f"  total_thickness_to_remove={total_thickness_to_remove:.4f}\"")
        print(f"  modeled_thickness_removed_squareup={modeled_thickness_removed_squareup:.4f}\"")
        print(f"  rough_mill_thk={rough_mill_thk:.4f}\" (90% of removal)")
        print(f"  grind_thk_total={grind_thk_total:.4f}\" (10% of removal)")

        # Check for mismatch
        thickness_mismatch = abs(total_thickness_to_remove - modeled_thickness_removed_squareup)
        if thickness_mismatch > 0.050:
            warning = f"Square-up thickness mismatch > 0.050\": {thickness_mismatch:.4f}\""
            print(f"DEBUG: WARNING - {warning}")
            p.warnings.append(warning)

        # Validate thickness removal consistency
        validation_warning = _validate_thickness_removal(
            stock_thk=stock_thk,
            finished_thk=finished_thk,
            rough_mill_thk=rough_mill_thk,
            grind_thk_total=grind_thk_total,
            context="die_plate square-up"
        )
        if validation_warning:
            p.warnings.append(validation_warning)

        # Add operations with calculated times
        D = W / 3.0
        axial_step = min(0.75, T / 2.0)

        # Pass override times to suppress physics-based calc
        p.add("square_up_rough_sides",
              radial_stock=0.250,
              axial_step=axial_step,
              tool_diameter_in=D,
              override_time_minutes=milling_time + (setup_time / 2))

        p.add("square_up_rough_faces",
              finish_doc=0.025,
              tool_diameter_in=D,
              target_pass_count=3,
              override_time_minutes=grinding_time + (flip_time + setup_time / 2))
    else:
        # Wet grind for smaller parts
        p.add("wet_grind_square_all", stock_removed_total=0.050, faces=2)

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


def planner_punches(params: Dict[str, Any]) -> Plan:
    """
    Enhanced planner for Punches (punch + pilot_punch + form_punch).

    Uses DWG punch feature extraction when available.
    Falls back to manual params if no DXF path provided.
    """
    try:
        # Get plan dict from enhanced planner (defined at end of this file)
        plan_dict = planner_punches_enhanced(params)

        # Convert to Plan dataclass
        p = base_plan()
        p.ops = plan_dict.get("ops", [])
        p.fixturing = plan_dict.get("fixturing", [])
        p.qa = plan_dict.get("qa", [])
        p.warnings = plan_dict.get("warnings", [])
        p.directs = plan_dict.get("directs", p.directs)

        return p

    except Exception as e:
        # Fall back to stub if enhanced planner fails
        p = _stub_plan("Punches", "Add WEDM outline, HT route, grind/lap bearing as needed. For pilot: tight runout/TIR.")
        p.warnings.append(f"Enhanced punch planner failed: {str(e)}")
        return p


def planner_bushing(params: Dict[str, Any]) -> Plan:
    return _stub_plan("bushing_id_critical", "Wire/drill open ID, jig grind to tol, lap for low Ra.")


def planner_sections_blocks(params: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidated planner for Sections_blocks (cam_or_hemmer + flat_die_chaser + die_section).

    Handles carbide die sections with proper operation stacks including:
    - Stock procurement from oversize blank
    - Square up / rough grind all faces
    - Finish grind thickness and working faces
    - Form cutting (wire EDM and/or grinding)
    - Chamfers and reliefs

    Ensures proper time estimation with carbide material factors.

    Returns a dict (not Plan) to preserve the meta key with timing/cost data.
    """
    try:
        plan_dict = create_die_section_plan(params)
        # Return as dict to preserve meta key
        return plan_dict
    except Exception as e:
        p = _stub_plan("Sections_blocks", "WEDM/mill slots/profiles → HT → profile grind → lap wear faces.")
        p.warnings.append(f"Die section planner failed: {str(e)}")
        # Convert Plan to dict and add error info to meta
        return {
            "ops": p.ops,
            "fixturing": p.fixturing,
            "qa": p.qa,
            "warnings": p.warnings,
            "directs": dict(p.directs),
            "meta": {"error": str(e)},
        }


# ---------------------------------------------------------------------------
# Die Section / Carbide Block Planner
# ---------------------------------------------------------------------------

@dataclass
class DieSectionParams:
    """Parameters for die section planning."""
    # Basic geometry
    length_in: float = 0.0
    width_in: float = 0.0
    thickness_in: float = 0.0

    # Material
    material: str = "A2"
    material_group: str = ""

    # Form characteristics
    has_internal_form: bool = False
    has_edge_form: bool = False
    form_perimeter_in: float = 0.0
    form_depth_in: float = 0.0
    form_complexity: int = 1  # 1=simple, 2=moderate, 3=complex

    # Tolerances
    tight_tolerances: List[Dict[str, Any]] = field(default_factory=list)
    min_tolerance_in: float = 0.001  # Tightest tolerance on part
    has_over_r_callout: bool = False  # "OVER R" dimension callouts

    # Features
    num_chamfers: int = 0
    num_reliefs: int = 0
    num_working_faces: int = 2
    has_land_height: bool = False

    # Hole info (usually none for form die sections)
    hole_count: int = 0
    hole_sets: List[Dict[str, Any]] = field(default_factory=list)

    # Processing hints
    requires_wire_edm: bool = True
    requires_form_grind: bool = False
    requires_polish: bool = False

    # Family sub-type
    sub_type: str = "die_section"  # die_section, cam, hemmer, sensor_block

    def __post_init__(self):
        if self.tight_tolerances is None:
            self.tight_tolerances = []
        if self.hole_sets is None:
            self.hole_sets = []


def _extract_die_section_params(params: Dict[str, Any]) -> DieSectionParams:
    """Extract and normalize die section parameters from input dict."""
    # Get dimensions
    L, W, T = 0.0, 0.0, 0.0
    if "plate_LxW" in params:
        L, W = params["plate_LxW"]
    else:
        L = float(params.get("L") or params.get("length") or params.get("length_in") or 0.0)
        W = float(params.get("W") or params.get("width") or params.get("width_in") or 0.0)
    T = float(params.get("T") or params.get("thickness") or params.get("thickness_in") or 0.0)

    # Material detection
    material = params.get("material", "A2")
    material_group = params.get("material_group", "")

    # Form detection
    has_internal = bool(params.get("has_internal_form", False))
    has_edge = bool(params.get("has_edge_form", False))

    # If no explicit form flags, infer from part type keywords
    sub_type = params.get("sub_type", "die_section")
    part_name = str(params.get("part_name", "")).lower()
    if not has_internal and not has_edge:
        if any(kw in part_name for kw in ["die section", "form die", "carbide insert"]):
            has_internal = True
        elif any(kw in part_name for kw in ["cam", "hemmer", "sensor"]):
            has_edge = True

    # Tolerance extraction
    tight_tols = params.get("tight_tolerances", [])
    min_tol = float(params.get("min_tolerance_in", 0.001))
    has_over_r = bool(params.get("has_over_r_callout", False))

    # Check for tight tolerances from dimensions
    if not tight_tols and "dimensions" in params:
        for dim in params.get("dimensions", []):
            tol = dim.get("tolerance", 0.001)
            if tol <= 0.0005:
                tight_tols.append(dim)
                min_tol = min(min_tol, tol)

    return DieSectionParams(
        length_in=float(L),
        width_in=float(W),
        thickness_in=float(T),
        material=material,
        material_group=material_group,
        has_internal_form=has_internal,
        has_edge_form=has_edge,
        form_perimeter_in=float(params.get("form_perimeter_in", 0.0)),
        form_depth_in=float(params.get("form_depth_in", T * 0.5 if T > 0 else 0.25)),
        form_complexity=int(params.get("form_complexity", 2)),
        tight_tolerances=tight_tols,
        min_tolerance_in=min_tol,
        has_over_r_callout=has_over_r,
        num_chamfers=int(params.get("num_chamfers", 4)),
        num_reliefs=int(params.get("num_reliefs", 0)),
        num_working_faces=int(params.get("num_working_faces", 2)),
        has_land_height=bool(params.get("has_land_height", False)),
        hole_count=int(params.get("hole_count", 0)),
        hole_sets=params.get("hole_sets", []),
        requires_wire_edm=bool(params.get("requires_wire_edm", True)),
        requires_form_grind=bool(params.get("requires_form_grind", False)),
        requires_polish=bool(params.get("requires_polish", False)),
        sub_type=sub_type,
    )


def _is_carbide_die_section(params: DieSectionParams) -> bool:
    """Determine if this is a carbide die section (not a plain block).

    A carbide die section is characterized by:
    - Material is carbide
    - Small block geometry (typically < 4" in any dimension)
    - Has internal or edge form
    - No drilled holes (form-only)
    """
    is_carbide = _is_carbide(params.material, params.material_group)
    is_small = max(params.length_in, params.width_in, params.thickness_in) < 4.0
    has_form = params.has_internal_form or params.has_edge_form
    no_holes = params.hole_count == 0

    return is_carbide and is_small and has_form and no_holes


def _calculate_die_section_complexity(params: DieSectionParams) -> Dict[str, Any]:
    """Calculate complexity score for a die section.

    Returns complexity metrics used to scale:
    - Programming time
    - Machining time
    - Inspection time
    - Finishing time
    """
    score = 0
    factors = []

    # Form complexity base
    score += params.form_complexity * 2
    factors.append(f"Form complexity: {params.form_complexity}")

    # Tight tolerances
    num_tight = len(params.tight_tolerances)
    if params.min_tolerance_in <= 0.0002:
        score += 3
        factors.append(f"Very tight tolerance: ±{params.min_tolerance_in:.4f}\"")
    elif params.min_tolerance_in <= 0.0005:
        score += 2
        factors.append(f"Tight tolerance: ±{params.min_tolerance_in:.4f}\"")

    if num_tight > 3:
        score += num_tight - 3
        factors.append(f"Multiple tight tolerances: {num_tight}")

    # Special callouts
    if params.has_over_r_callout:
        score += 2
        factors.append("'OVER R' dimension callout")

    if params.has_land_height:
        score += 1
        factors.append("Land height specification")

    # Feature counts
    if params.num_chamfers > 4:
        score += 1
        factors.append(f"Multiple chamfers: {params.num_chamfers}")

    if params.num_reliefs > 0:
        score += params.num_reliefs
        factors.append(f"Reliefs: {params.num_reliefs}")

    # Wire EDM + form grind combo
    if params.requires_wire_edm and params.requires_form_grind:
        score += 2
        factors.append("Wire EDM + form grind combination")

    # Polish requirement
    if params.requires_polish:
        score += 2
        factors.append("Polish/lap finish required")

    # Carbide material factor
    if _is_carbide(params.material, params.material_group):
        score += 3
        factors.append("Carbide material (slower machining)")

    return {
        "score": score,
        "level": "high" if score >= 10 else "medium" if score >= 5 else "low",
        "factors": factors,
    }


def create_die_section_plan(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a detailed manufacturing plan for a die section or similar block.

    Generates proper operation stack for carbide form die sections:
    1. Stock from oversize blank
    2. Square up / rough grind all faces
    3. Finish grind thickness and working faces
    4. Form cutting (wire EDM and/or form grinding)
    5. Chamfers and reliefs
    6. Polish/finishing

    Ensures:
    - Programming time is never 0
    - Machining time has minimum floors for carbide
    - Inspection scales with complexity
    - Machine costs are populated
    """
    p = _extract_die_section_params(params)
    is_carbide = _is_carbide(p.material, p.material_group)
    is_die_section = _is_carbide_die_section(p)
    complexity = _calculate_die_section_complexity(p)

    plan = {
        "ops": [],
        "fixturing": [],
        "qa": [],
        "warnings": [],
        "directs": {
            "hardware": False,
            "outsourced": not is_carbide,  # Carbide skips HT
            "utilities": False,
            "consumables_flat": False,
            "packaging_flat": True,
        },
        "meta": {
            "family": "Sections_blocks",
            "sub_type": p.sub_type,
            "is_carbide_die_section": is_die_section,
            "complexity": complexity,
        }
    }

    # 1. Stock procurement
    _add_die_section_stock_ops(plan, p)

    # 2. Square up / rough grinding
    _add_die_section_squaring_ops(plan, p, is_carbide)

    # 3. Finish grinding
    _add_die_section_finish_grind_ops(plan, p, is_carbide)

    # 4. Form operations (Wire EDM and/or form grinding)
    _add_die_section_form_ops(plan, p, is_carbide)

    # 5. Heat treatment (if not carbide)
    if not is_carbide:
        _add_die_section_heat_treat_ops(plan, p)

    # 6. Chamfers and reliefs
    _add_die_section_edge_ops(plan, p)

    # 7. Polish and finishing
    _add_die_section_finishing_ops(plan, p, is_carbide, complexity)

    # 8. QA checks
    _add_die_section_qa_checks(plan, p, complexity)

    # 9. Fixturing notes
    _add_die_section_fixturing_notes(plan, p)

    # 10. Apply guardrails and sanity checks
    _apply_die_section_guardrails(plan, p, is_carbide, complexity)

    return plan


def _add_die_section_stock_ops(plan: Dict[str, Any], p: DieSectionParams) -> None:
    """Add stock procurement operation with oversize allowance."""
    # Calculate stock size with grinding allowance
    stock_L = p.length_in + 0.125 if p.length_in > 0 else 1.0
    stock_W = p.width_in + 0.125 if p.width_in > 0 else 1.0
    stock_T = p.thickness_in + 0.100 if p.thickness_in > 0 else 0.5

    is_carbide = _is_carbide(p.material, p.material_group)
    stock_size = f"{stock_L:.3f}\" x {stock_W:.3f}\" x {stock_T:.3f}\""

    plan["ops"].append({
        "op": "stock_procurement",
        "material": p.material,
        "stock_size": stock_size,
        "stock_L": stock_L,
        "stock_W": stock_W,
        "stock_T": stock_T,
        "note": f"Order {p.material} stock: {stock_size}",
        "is_carbide": is_carbide,
    })


def _add_die_section_squaring_ops(plan: Dict[str, Any], p: DieSectionParams, is_carbide: bool) -> None:
    """Add square-up and rough grinding operations."""
    # Calculate volume for time estimation
    volume_cuin = p.length_in * p.width_in * p.thickness_in

    # Stock removal (0.125" L/W, 0.100" T total)
    stock_removed = (0.125 * p.width_in * p.thickness_in * 2 +  # sides
                     0.125 * p.length_in * p.thickness_in * 2 +  # ends
                     0.100 * p.length_in * p.width_in)  # faces

    # Thickness-specific calculations
    stock_thk = p.thickness_in + 0.100  # Total stock on thickness (0.050" per face)
    finished_thk = p.thickness_in
    total_thickness_to_remove = stock_thk - finished_thk

    # Calculate thickness removed by rough grinding vs finish grinding
    # Rough grind removes most of the stock, finish grind removes the last bit
    rough_mill_thk = total_thickness_to_remove * 0.75  # 75% in rough grind
    grind_thk_total = total_thickness_to_remove * 0.25  # 25% in finish grind
    modeled_thickness_removed_squareup = rough_mill_thk + grind_thk_total

    # DEBUG output for square-up
    print(f"DEBUG: Square-up operation (die_section):")
    print(f"  final_thickness={finished_thk:.4f}\", stock_thickness={stock_thk:.4f}\"")
    print(f"  total_thickness_to_remove={total_thickness_to_remove:.4f}\"")
    print(f"  modeled_thickness_removed_squareup={modeled_thickness_removed_squareup:.4f}\"")
    print(f"  rough_mill_thk (rough grind)={rough_mill_thk:.4f}\" (75% of removal)")
    print(f"  grind_thk_total (finish grind)={grind_thk_total:.4f}\" (25% of removal)")

    # Check for mismatch
    thickness_mismatch = abs(total_thickness_to_remove - modeled_thickness_removed_squareup)
    if thickness_mismatch > 0.050:
        warning = f"Die section square-up thickness mismatch > 0.050\": {thickness_mismatch:.4f}\""
        print(f"DEBUG: WARNING - {warning}")
        plan["warnings"].append(warning)

    # Material factor for carbide
    grind_factor = 2.5 if is_carbide else 1.0

    # Rough grind time: volume × base_rate × material_factor
    # Base rate: 4.0 min/in³ for grinding
    rough_grind_time = stock_removed * 4.0 * grind_factor

    # Minimum time for die sections (at least 15 minutes for squaring)
    rough_grind_time = max(rough_grind_time, 15.0 if is_carbide else 8.0)

    plan["ops"].append({
        "op": "rough_grind_all_faces",
        "faces": 6,
        "stock_removed_cuin": stock_removed,
        "material_factor": grind_factor,
        "time_minutes": rough_grind_time,
        "note": f"Rough grind all 6 faces to establish datums ({stock_removed:.3f} in³ removed)",
    })


def _add_die_section_finish_grind_ops(plan: Dict[str, Any], p: DieSectionParams, is_carbide: bool) -> None:
    """Add finish grinding for thickness and working faces."""
    grind_factor = 2.5 if is_carbide else 1.0

    # Finish grind thickness (top and bottom faces)
    face_area = p.length_in * p.width_in
    finish_stock = 0.010  # 0.005" per face
    finish_volume = face_area * finish_stock

    # Time calculation: precision grinding is slower
    # 6.0 min/in³ for finish grinding × material factor
    finish_time = finish_volume * 6.0 * grind_factor
    finish_time = max(finish_time, 10.0 if is_carbide else 5.0)

    plan["ops"].append({
        "op": "finish_grind_thickness",
        "faces": 2,
        "stock_removed_cuin": finish_volume,
        "material_factor": grind_factor,
        "time_minutes": finish_time,
        "flatness_target": 0.0002 if p.min_tolerance_in <= 0.0005 else 0.0005,
        "note": f"Finish grind top/bottom to thickness ±{p.min_tolerance_in:.4f}\"",
    })

    # Finish grind working faces if more than 2
    if p.num_working_faces > 2:
        extra_faces = p.num_working_faces - 2
        extra_time = extra_faces * 5.0 * grind_factor
        plan["ops"].append({
            "op": "finish_grind_working_faces",
            "faces": extra_faces,
            "material_factor": grind_factor,
            "time_minutes": extra_time,
            "note": f"Finish grind {extra_faces} additional working faces",
        })


def _add_die_section_form_ops(plan: Dict[str, Any], p: DieSectionParams, is_carbide: bool) -> None:
    """Add form cutting operations (Wire EDM and/or form grinding)."""
    if not p.has_internal_form and not p.has_edge_form:
        return

    # Calculate form perimeter if not provided
    perimeter = p.form_perimeter_in
    if perimeter <= 0:
        # Estimate from geometry: assume form is ~60% of part perimeter
        perimeter = 2 * (p.length_in + p.width_in) * 0.6

    depth = p.form_depth_in if p.form_depth_in > 0 else p.thickness_in * 0.5

    # Wire EDM for internal/edge forms
    if p.requires_wire_edm or p.has_internal_form:
        # Wire EDM time: perimeter × minutes_per_inch × material_factor
        # Carbide: ~0.36 min/in (2.8 ipm), Tool steel: ~0.25 min/in (4 ipm)
        wire_mpi = 0.36 if is_carbide else 0.25
        edm_mat_factor = 1.3 if is_carbide else 1.0

        # Base time: path_length × min_per_in × material_factor
        wire_time = perimeter * wire_mpi * edm_mat_factor

        # Add skim passes for tight tolerances
        skims = 2 if p.min_tolerance_in <= 0.0005 else 1
        skim_mpi = 0.2  # Skims are faster
        wire_time += perimeter * skim_mpi * edm_mat_factor * skims

        # Minimum wire EDM time for carbide die sections
        wire_time = max(wire_time, 20.0 if is_carbide else 10.0)

        # t_window = edm_path_length_in * edm_min_per_in * edm_material_factor
        t_window = perimeter * wire_mpi * edm_mat_factor

        plan["ops"].append({
            "op": "wire_edm_form",
            "wire_profile_perimeter_in": perimeter,
            "thickness_in": depth,
            "skims": skims,
            "material_factor": edm_mat_factor,
            "time_minutes": wire_time,
            "note": f"Wire EDM form profile: {perimeter:.2f}\" perimeter × {depth:.3f}\" deep",
            # EDM time model transparency fields
            "edm_path_length_in": perimeter,
            "edm_min_per_in": wire_mpi,
            "edm_material_factor": edm_mat_factor,
            "part_thickness": depth,
            "t_window": t_window,
        })

    # Form grinding for precision forms
    if p.requires_form_grind:
        # Form grind time: perimeter × depth × rate × material factor
        grind_factor = 2.5 if is_carbide else 1.0
        form_grind_time = (perimeter * depth * 0.5) * grind_factor
        form_grind_time = max(form_grind_time, 15.0 if is_carbide else 8.0)

        plan["ops"].append({
            "op": "form_grind",
            "perimeter_in": perimeter,
            "depth_in": depth,
            "material_factor": grind_factor,
            "time_minutes": form_grind_time,
            "note": f"Form grind to final profile and tolerance",
        })


def _add_die_section_heat_treat_ops(plan: Dict[str, Any], p: DieSectionParams) -> None:
    """Add heat treatment operation (skipped for carbide)."""
    hardness_map = {
        "A2": "60-62 RC", "D2": "58-60 RC", "M2": "62-64 RC",
        "O1": "60-62 RC", "S7": "54-56 RC",
    }
    target_hardness = hardness_map.get(p.material, "58-62 RC")

    plan["ops"].append({
        "op": "heat_treat",
        "material": p.material,
        "target_hardness": target_hardness,
        "note": f"Heat treat to {target_hardness}",
    })


def _add_die_section_edge_ops(plan: Dict[str, Any], p: DieSectionParams) -> None:
    """Add chamfer and relief operations."""
    if p.num_chamfers > 0:
        chamfer_time = p.num_chamfers * 1.5  # 1.5 min per chamfer
        plan["ops"].append({
            "op": "chamfer_edges",
            "qty": p.num_chamfers,
            "time_minutes": chamfer_time,
            "note": f"Chamfer {p.num_chamfers} edges on entry/exit",
        })

    if p.num_reliefs > 0:
        relief_time = p.num_reliefs * 3.0  # 3 min per relief
        plan["ops"].append({
            "op": "machine_reliefs",
            "qty": p.num_reliefs,
            "time_minutes": relief_time,
            "note": f"Machine {p.num_reliefs} relief cuts",
        })


def _add_die_section_finishing_ops(plan: Dict[str, Any], p: DieSectionParams,
                                    is_carbide: bool, complexity: Dict[str, Any]) -> None:
    """Add polish and finishing operations based on complexity.

    Ensures minimum finishing time for carbide die sections.
    """
    # Base finishing time
    base_time = 10.0 if is_carbide else 5.0

    # Add time for cavity/working surface polish
    if p.requires_polish or p.has_internal_form:
        polish_time = 8.0 if is_carbide else 4.0
        plan["ops"].append({
            "op": "polish_cavity",
            "time_minutes": polish_time,
            "note": "Polish cavity/working surfaces to spec Ra",
        })
        base_time += polish_time

    # Edge break time
    edge_break_time = max(3.0, p.num_chamfers * 0.5 + 2.0)
    plan["ops"].append({
        "op": "edge_break",
        "time_minutes": edge_break_time,
        "note": "Edge break all entry/exit edges and corners",
    })

    # Deburr based on complexity
    deburr_time = 3.0 + (complexity["score"] * 0.3)
    plan["ops"].append({
        "op": "deburr_and_clean",
        "time_minutes": deburr_time,
        "note": "Deburr, clean, and inspect surfaces",
    })

    # Total finishing time floor
    total_finishing = base_time + edge_break_time + deburr_time
    min_finishing = 10.0 if is_carbide else 5.0

    if total_finishing < min_finishing:
        plan["warnings"].append(
            f"Finishing time ({total_finishing:.1f} min) below minimum; adjusted to {min_finishing} min"
        )


def _add_die_section_qa_checks(plan: Dict[str, Any], p: DieSectionParams,
                                complexity: Dict[str, Any]) -> None:
    """Add QA checks scaled by complexity and tolerances."""
    qa = plan["qa"]

    # Base dimensional inspection
    qa.append(f"Verify overall dimensions L×W×T to ±{p.min_tolerance_in:.4f}\"")

    # Form verification
    if p.has_internal_form or p.has_edge_form:
        qa.append("Verify form profile to optical comparator or CMM")

    # Tight tolerance checks
    if p.min_tolerance_in <= 0.0002:
        qa.append(f"Critical dimension check: tolerance ±{p.min_tolerance_in:.4f}\"")
        for tol in p.tight_tolerances[:5]:  # Show up to 5
            if isinstance(tol, dict):
                qa.append(f"  - {tol.get('name', 'Dim')}: {tol.get('value', '?')} ±{tol.get('tolerance', '?')}")

    # Special callouts
    if p.has_over_r_callout:
        qa.append("Verify 'OVER R' dimensions with radius gauge")

    if p.has_land_height:
        qa.append("Verify land height to specification")

    # Surface finish checks
    if p.requires_polish:
        qa.append("Verify surface finish Ra on working surfaces")

    # Flatness/parallelism for ground faces
    qa.append("Verify flatness and parallelism on ground faces")


def _add_die_section_fixturing_notes(plan: Dict[str, Any], p: DieSectionParams) -> None:
    """Add fixturing recommendations."""
    fix = plan["fixturing"]

    is_carbide = _is_carbide(p.material, p.material_group)

    if is_carbide:
        fix.append("Use diamond wheel for carbide grinding; maintain coolant flow")
        fix.append("Support part fully during grinding to prevent chipping")

    fix.append("Use precision vise or fixture block for squaring operations")

    if p.has_internal_form:
        fix.append("Wire EDM: Use stable workholding; verify perpendicularity before cutting")

    if p.min_tolerance_in <= 0.0005:
        fix.append("Temperature-stabilize part before final inspection")


def _apply_die_section_guardrails(plan: Dict[str, Any], p: DieSectionParams,
                                   is_carbide: bool, complexity: Dict[str, Any]) -> None:
    """Apply sanity checks and guardrails for die sections.

    Implements:
    - Minimum programming time (never 0)
    - Minimum machining time floors
    - Machine cost verification
    - High material cost warnings
    """
    warnings = plan["warnings"]
    ops = plan["ops"]

    # Calculate totals from operations
    total_machine_time = sum(op.get("time_minutes", 0) for op in ops)
    num_unique_ops = len(set(op.get("op", "") for op in ops))

    # --- Task 2: Programming time must not be 0 ---
    base_prog_time = 15.0  # Minimum for die sections

    # Complexity adders
    prog_time = base_prog_time
    prog_time += num_unique_ops * 3.0  # +3 min per unique operation type

    # Tight tolerance adder
    if p.min_tolerance_in <= 0.0002:
        prog_time += 10.0  # +10 min for very tight tolerances
    elif p.min_tolerance_in <= 0.0005:
        prog_time += 5.0  # +5 min for tight tolerances

    # Over R callout adder
    if p.has_over_r_callout:
        prog_time += 5.0

    # Carbide adder (slower prove-out)
    if is_carbide:
        prog_time *= 1.25

    # Store in plan metadata
    if "meta" not in plan:
        plan["meta"] = {}
    plan["meta"]["programming_time_min"] = prog_time
    plan["meta"]["num_unique_operations"] = num_unique_ops

    # --- Task 3: Minimum machining time floors ---
    min_machine_time = 30.0 if is_carbide else 15.0
    if total_machine_time < min_machine_time:
        warnings.append(
            f"Machining time ({total_machine_time:.1f} min) below minimum for "
            f"{'carbide ' if is_carbide else ''}die section; floor is {min_machine_time} min"
        )
        plan["meta"]["machining_time_adjusted"] = True
        plan["meta"]["machining_time_floor"] = min_machine_time

    plan["meta"]["total_machine_time_min"] = max(total_machine_time, min_machine_time)

    # --- Task 4: Inspection time scaling ---
    base_insp_time = 10.0  # Higher base for die sections
    insp_time = base_insp_time

    # Per-tight-dimension adder
    insp_time += len(p.tight_tolerances) * 2.0

    # Complexity scaling
    if complexity["level"] == "high":
        insp_time *= 1.5
    elif complexity["level"] == "medium":
        insp_time *= 1.2

    # Over R and land height checks
    if p.has_over_r_callout:
        insp_time += 5.0
    if p.has_land_height:
        insp_time += 3.0

    plan["meta"]["inspection_time_min"] = insp_time

    # --- Task 5: Machine cost tracking ---
    # Calculate expected machine costs (will be used by cost calculator)
    machine_rates = {
        "wire_edm": 130.0,  # $/hr
        "surface_grind": 95.0,
        "form_grind": 110.0,
        "default": 90.0,
    }

    estimated_machine_cost = 0.0
    for op in ops:
        op_name = op.get("op", "").lower()
        time_min = op.get("time_minutes", 0)

        if "wire_edm" in op_name:
            rate = machine_rates["wire_edm"]
        elif "form_grind" in op_name:
            rate = machine_rates["form_grind"]
        elif "grind" in op_name:
            rate = machine_rates["surface_grind"]
        else:
            rate = machine_rates["default"]

        op_cost = (time_min / 60.0) * rate
        estimated_machine_cost += op_cost
        op["estimated_cost"] = op_cost

    plan["meta"]["estimated_machine_cost"] = estimated_machine_cost

    # Guard: Flag if machine time > 0 but cost might be missing
    if total_machine_time > 0 and estimated_machine_cost <= 0:
        warnings.append(
            "WARNING: Machining time present but estimated machine cost is $0. "
            "Flag for review."
        )

    # --- Task 7: High material cost sanity check ---
    # Estimate material cost for carbide
    if is_carbide:
        # Carbide density ~15.6 g/cc, price ~$0.30-0.50/g
        volume_cuin = p.length_in * p.width_in * p.thickness_in
        volume_cc = volume_cuin * 16.387  # 1 in³ = 16.387 cm³
        weight_g = volume_cc * 15.6
        est_material_cost = weight_g * 0.40  # $0.40/g average

        plan["meta"]["estimated_material_cost"] = est_material_cost

        # Flag if material cost exceeds threshold or is disproportionate
        if est_material_cost > 1000:
            warnings.append(
                f"Manual Review Recommended: Carbide stock estimate (${est_material_cost:.2f}) "
                f"exceeds $1,000 threshold. Verify catalog and SMB pricing."
            )
        elif est_material_cost > estimated_machine_cost * 2:
            warnings.append(
                f"Note: Carbide material cost (${est_material_cost:.2f}) is high relative "
                f"to machining cost (${estimated_machine_cost:.2f}). Verify stock pricing."
            )

    # --- Task 8: Bounds checking ---
    # Setup time bounds for die sections
    setup_min = 20.0  # Minimum setup for die sections
    setup_max = 120.0  # Maximum reasonable setup

    calculated_setup = 15 + num_unique_ops * 3 + (5 if is_carbide else 0)
    bounded_setup = max(setup_min, min(setup_max, calculated_setup))

    plan["meta"]["setup_time_min"] = bounded_setup

    # Store complexity score for downstream use
    plan["meta"]["complexity_score"] = complexity["score"]
    plan["meta"]["complexity_level"] = complexity["level"]
    plan["meta"]["complexity_factors"] = complexity["factors"]


def planner_special_processes(params: Dict[str, Any]) -> Plan:
    """Consolidated planner for Special_processes (pm_compaction_die + shear_blade + extrude_hone)."""
    return _stub_plan("Special_processes", "Carbide/hardened parts: wire/grind/lap → HT → match grind → hone to Ra.")


# Registry
PLANNERS = {
    "Plates": planner_die_plate,
    "Punches": planner_punches,
    "bushing_id_critical": planner_bushing,
    "Sections_blocks": planner_sections_blocks,
    "Special_processes": planner_special_processes,
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
    return plan_job("Plates", params)


# Basic row→holeset translator (mirrors earlier adapter rules)
import re
from fractions import Fraction

# Import thread validation functions from hole_table_parser
from cad_quoter.geometry.hole_table_parser import (
    validate_and_correct_thread,
    is_valid_thread_spec,
    STANDARD_THREADS,
    THREAD_MAJOR_DIAMETERS,
)

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

# Minimal keyword sets; expand per your shop's vocabulary.
_FAM_KEYWORDS = {
    "Plates": {
        "any": {
            # common plate words
            "plate", "shoe", "punch shoe", "die set", "punch holder", "stripper",
            # hole table signals (often present on plates)
            "hole table", "c'boRE", "counterbore", "c'drill", "counterdrill",
            # new part name keywords
            "retainer plate", "stripper", "stripper plate",
        },
        "none_of": set(),
    },
    "Punches": {
        "any": {
            "punch detail", "pilot punch", "bearing land", "edge hone",
            # new part name keywords
            "spring punch", "guide post", "spring pin", "punch",
        },
        "none_of": {"shoe", "die set"},
    },
    "bushing_id_critical": {
        "any": {"bushing", "id grind", "jig grind id", "retainer"},
        "none_of": set(),
    },
    "Sections_blocks": {
        "any": {
            "cam", "hemmer", "slot cam", "cam slot",
            # new part name keywords
            "sensor block", "die section", "stock guide", "die chase",
            "punch block", "stripper insert", "pressure pad",
            # carbide die section keywords
            "form die", "carbide insert", "carbide section", "carbide block",
            "form insert", "die insert", "cutting insert",
        },
        "none_of": set(),
    },
    "Special_processes": {
        "any": {
            "shear blade", "knife", "edge hone", "match grind",
            "pm compaction", "extrude hone", "carbide",
        },
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
        else:
            # Tie-breaker: prefer Sections_blocks for carbide die sections
            # If carbide is mentioned along with die section keywords, prefer Sections_blocks
            top_score = fam_sorted[0][1]
            tied = [fam for fam, s in fam_sorted if s == top_score]
            if "Sections_blocks" in tied and "Special_processes" in tied:
                # Check for die section keywords that indicate form work
                die_section_keywords = {"die section", "form die", "carbide insert",
                                        "form insert", "die insert", "carbide section"}
                if any(kw in blob for kw in die_section_keywords):
                    chosen_family = "Sections_blocks"
                else:
                    chosen_family = fam_sorted[0][0]
            else:
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
# CAD File Integration (geo_dump + DimensionFinder)
# ---------------------------------------------------------------------------

def extract_dimensions_from_cad(file_path: str | Path) -> Optional[Tuple[float, float, float]]:
    """Extract L, W, T dimensions from CAD file using DimensionFinder.

    Uses the DimensionFinder to analyze DIMENSION entities in DXF files
    and infer bounding box dimensions. For DWG files, looks for pre-extracted
    mtext_results.json or converts to DXF first.

    Returns (length, width, thickness) in inches, or None if extraction fails.
    """
    try:
        from pathlib import Path
        from cad_quoter.geometry.dimension_finder import DimensionFinder

        file_path = Path(file_path)
        finder = DimensionFinder()

        # Handle different file types
        ext = file_path.suffix.lower()

        if ext == '.dxf':
            # Load directly from DXF
            finder.load_dxf(file_path)

        elif ext == '.dwg':
            # Convert DWG to DXF using ODA for fresh extraction
            import tempfile
            import subprocess
            import shutil
            import os

            # Find ODA converter
            oda_exe = os.getenv("ODA_FILE_CONVERTER")
            if not oda_exe:
                common_paths = [
                    r"D:\ODA\ODAFileConverter 26.8.0\ODAFileConverter.exe",
                    r"C:\Program Files\ODA\OdaFileConverter.exe",
                    r"C:\Program Files (x86)\ODA\OdaFileConverter.exe",
                ]
                for path in common_paths:
                    if Path(path).exists():
                        oda_exe = path
                        break

            if not oda_exe or not Path(oda_exe).exists():
                print(f"[WARN] ODA converter not available for DWG: {file_path}")
                return None

            # Convert DWG to DXF
            with tempfile.TemporaryDirectory(prefix="dwg_convert_") as tmpdir:
                input_dir = Path(tmpdir) / "_input"
                output_dir = Path(tmpdir) / "_output"
                input_dir.mkdir()
                output_dir.mkdir()

                # Copy DWG to input dir
                shutil.copy2(file_path, input_dir / file_path.name)

                # Run ODA converter
                cmd = [
                    oda_exe,
                    str(input_dir),
                    str(output_dir),
                    "ACAD2018",
                    "DXF",
                    "0",
                    "1",
                    file_path.name
                ]
                subprocess.run(cmd, capture_output=True, text=True)

                # Find output DXF
                dxf_files = list(output_dir.glob(f"{file_path.stem}*.dxf"))
                if not dxf_files:
                    print(f"[WARN] ODA conversion failed for: {file_path}")
                    return None

                finder.load_dxf(dxf_files[0])

        elif ext == '.json' and 'mtext_results' in file_path.name:
            # Direct JSON file
            finder.load_results(file_path)

        else:
            print(f"[WARN] Unsupported file type: {ext}")
            return None

        # Get inferred bounding box dimensions
        bbox_candidates = finder.find_bounding_box()

        if len(bbox_candidates) < 3:
            print(f"[WARN] Not enough dimension candidates found: {len(bbox_candidates)}")
            return None

        # Strategy for bbox extraction:
        # - Linear dimensions (dimtype 0/1) are almost always THICKNESS
        # - Ordinate dimensions (dimtype 6) give L and W from max extents in each direction

        # Separate dimensions by type
        linear_dims = []
        ordinate_x = []  # X-direction ordinates
        ordinate_y = []  # Y-direction ordinates
        ordinate_unknown = []  # Direction unknown

        for dim in finder.dimensions:
            dimtype = dim.get("dimtype", 0)
            val = dim.get("measurement_in", 0)
            if val < 0.05:
                continue
            if dim.get("is_diameter", False):
                continue

            if dimtype in (0, 1):  # Linear or Aligned
                linear_dims.append(val)
            elif dimtype == 6:  # Ordinate
                direction = dim.get("ordinate_direction")
                if direction == "X":
                    ordinate_x.append(val)
                elif direction == "Y":
                    ordinate_y.append(val)
                else:
                    ordinate_unknown.append(val)

        # Get thickness from largest linear dimension
        T = max(linear_dims) if linear_dims else None

        # Get L and W from max ordinates in each direction
        L = None
        W = None

        if ordinate_x and ordinate_y:
            # We have direction information - use max in each direction
            L = max(ordinate_x)
            W = max(ordinate_y)
            # Ensure L >= W
            if W > L:
                L, W = W, L
        elif ordinate_unknown:
            # No direction info - use two largest unique ordinates
            unique_ordinates = sorted(set(ordinate_unknown), reverse=True)
            if len(unique_ordinates) >= 2:
                L = unique_ordinates[0]
                W = unique_ordinates[1]
            elif len(unique_ordinates) >= 1:
                L = unique_ordinates[0]
                W = unique_ordinates[0]

        if L and W and T:
            # Sort as L >= W >= T
            dims = sorted([L, W, T], reverse=True)
            return (dims[0], dims[1], dims[2])
        else:
            # Fallback: use top 3 scored dimensions
            seen = set()
            top_3 = []
            for val, score in bbox_candidates:
                rounded = round(val, 4)
                if rounded not in seen:
                    seen.add(rounded)
                    top_3.append(val)
                    if len(top_3) >= 3:
                        break

            if len(top_3) < 3:
                return None

            dims = sorted(top_3, reverse=True)
            return (dims[0], dims[1], dims[2])

    except Exception as e:
        print(f"[WARN] Dimension extraction failed: {e}")
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
    fallback_family: str = "Plates",
    use_paddle_ocr: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """High-level API: Generate process plan directly from a CAD file.

    This function:
    1. Extracts dimensions (L, W, T) using DimensionFinder (if use_paddle_ocr=True)
    2. Extracts hole table using geo_dump
    3. Extracts all text for family detection
    4. Converts hole table to hole_sets format
    5. Auto-detects family from text (or uses fallback)
    6. Generates complete process plan

    Args:
        file_path: Path to DXF or DWG file
        fallback_family: Family to use if auto-detection fails (default: "Plates")
        use_paddle_ocr: Whether to extract dimensions (default: True)
            Note: Now uses DimensionFinder instead of PaddleOCR for better accuracy
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

    # Pre-convert DWG to DXF once to avoid multiple ODA converter invocations
    # Each extraction function would otherwise convert independently
    cad_path_for_extraction = file_path
    if file_path.suffix.lower() == '.dwg':
        if verbose:
            print("[PLANNER] Converting DWG to DXF (one-time conversion)...")
        try:
            from cad_quoter.geometry import convert_dwg_to_dxf
            dxf_path = convert_dwg_to_dxf(str(file_path))
            if dxf_path:
                cad_path_for_extraction = Path(dxf_path)
                if verbose:
                    print(f"[PLANNER] Using cached DXF: {cad_path_for_extraction.name}")
        except Exception as e:
            if verbose:
                print(f"[PLANNER] DWG conversion failed, functions will convert individually: {e}")
            # Fall back to original path - each function will try its own conversion

    # 1. Extract dimensions (L, W, T)
    dims = None
    if use_paddle_ocr:
        if verbose:
            print("[PLANNER] Extracting dimensions with DimensionFinder...")
        dims = extract_dimensions_from_cad(cad_path_for_extraction)
        if dims:
            L, W, T = dims
            if verbose:
                print(f"[PLANNER] Dimensions: L={L:.3f}\", W={W:.3f}\", T={T:.3f}\"")
        else:
            if verbose:
                print("[PLANNER] Could not extract dimensions")

    # 2. Extract hole table and operations
    if verbose:
        print("[PLANNER] Extracting hole table...")
    hole_table = extract_hole_table_from_cad(cad_path_for_extraction)
    hole_operations = extract_hole_operations_from_cad(cad_path_for_extraction)
    if verbose:
        print(f"[PLANNER] Found {len(hole_table)} unique holes -> {len(hole_operations)} operations")

    # 3. Extract all text for family detection
    if verbose:
        print("[PLANNER] Extracting text for family detection...")
    all_text = extract_all_text_from_cad(cad_path_for_extraction)
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
    # Store cached DXF path if DWG was converted (avoids redundant ODA conversions)
    if cad_path_for_extraction != file_path:
        plan["cached_dxf_path"] = str(cad_path_for_extraction)
    if dims:
        plan["extracted_dims"] = {"L": L, "W": W, "T": T}
    plan["extracted_holes"] = len(hole_table)
    plan["extracted_hole_operations"] = len(hole_operations)
    # Store actual hole operations data for reuse (avoids redundant ODA conversions)
    plan["hole_operations_data"] = hole_operations
    # Add text dump for punch detection
    plan["text_dump"] = "\n".join(all_text) if all_text else ""

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

    # CMM inspection setup (labor)
    cmm_setup_min: float = 0.0

    # Handling
    part_flips: int = 0

    # Machine time for setup guardrail comparison
    machine_time_minutes: float = 0.0


def setup_minutes(i: LaborInputs) -> float:
    """
    Calculate Setup / Prep labor minutes (base = 15, multipliers increased 1.5x).

    NOTE: part_flips intentionally EXCLUDED from setup and counted in Machining.

    Setup minutes =
        15                              (was 10)
      + 3·tool_changes                  (was 2)
      + 8·fixturing_complexity          (was 5)
      + 3·edm_window_count              (was 2)
      + 2·edm_skim_passes               (was 1)
      + 5·grind_face_pairs              (was 3)
      + 3·jig_grind_bore_qty            (was 2)
      + 2·(tap_rigid + thread_mill)     (was 1)
      + 3·tap_npt                       (was 2)
      + 2·(ream_press_dowel + ream_slip_dowel)  (was 1)
      + 2·(counterbore_qty + counterdrill_qty)  (was 1)
      + 6·outsource_touches             (was 4)

    Simple-part guardrail:
      If machine_time is provided and the part is simple (few ops, few holes,
      no complex operations), setup is capped to avoid excessive ratios.
      - Simple part: ops <= 8, holes <= 12, no EDM/jig-grind/thread-mill/NPT
      - If setup > 4× machine_time on simple parts, reduce to 3× machine_time
        with a floor of 10 minutes
    """
    # Calculate raw setup time
    raw_setup = (
        15  # Was 10
        + 3 * i.tool_changes  # Was 2
        + 8 * i.fixturing_complexity  # Was 5
        + 3 * i.edm_window_count  # Was 2
        + 2 * i.edm_skim_passes  # Was 1
        + 5 * i.grind_face_pairs  # Was 3
        + 3 * i.jig_grind_bore_qty  # Was 2
        + 2 * (i.tap_rigid + i.thread_mill)  # Was 1
        + 3 * i.tap_npt  # Was 2
        + 2 * (i.ream_press_dowel + i.ream_slip_dowel)  # Was 1
        + 2 * (i.counterbore_qty + i.counterdrill_qty)  # Was 1
        + 6 * i.outsource_touches  # Was 4
    )

    # Apply simple-part guardrail if machine time is provided
    if i.machine_time_minutes > 0:
        # Check if this is a simple part
        is_simple = (
            i.ops_total <= 8
            and i.holes_total <= 12
            and i.edm_window_count == 0
            and i.edm_skim_passes == 0
            and i.jig_grind_bore_qty == 0
            and i.thread_mill == 0
            and i.tap_npt == 0
            and i.outsource_touches == 0
        )

        if is_simple:
            # For simple parts, cap setup at 4× machine time
            # If exceeded, reduce to 3× machine time with floor of 10 min
            max_ratio = 4.0
            target_ratio = 3.0
            min_setup = 10.0

            if raw_setup > max_ratio * i.machine_time_minutes:
                # Calculate reduced setup based on machine time
                reduced_setup = max(min_setup, target_ratio * i.machine_time_minutes)
                return reduced_setup

    return raw_setup


def programming_minutes(i: LaborInputs) -> float:
    """
    Calculate Programming / Prove-out labor minutes.

    Programming minutes =
        base (10 min if any machine ops, else 0)
      + 1·holes_total
      + 2·edm_window_count
      + 2·thread_mill
      + 2·jig_grind_bore_qty
      + 1·deep_holes
      + 1·grind_face_pairs

    Note: Holes appear here and in Inspection by design. If you consider that
    "double counting" across buckets, set the `holes_total` term to 0 here.

    The base minimum (10 min) ensures that any part with machining operations
    gets at least basic programming/prove-out time for tool offsets, program
    verification, and first article setup - even if there are no holes.
    """
    # Base programming time: 10 minutes if there are any machine operations
    # This covers basic program setup, tool offsets, and prove-out for turned/ground parts
    has_machine_ops = (
        i.ops_total > 0 or
        i.grind_face_pairs > 0 or
        i.edm_window_count > 0 or
        i.tool_changes > 0
    )
    base_programming = 10 if has_machine_ops else 0

    return (
        base_programming
        + 1 * i.holes_total
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

    Includes hole-driven and feature-driven checks, plus CMM setup.

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
      + cmm_setup_min (load, clamp, datum setup)
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
        + i.cmm_setup_min
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


def cmm_inspection_minutes(holes_total: int) -> Dict[str, float]:
    """
    Calculate CMM inspection time split into setup (labor) and checking (machine).

    Formula: CMM_time_min = base_block_min + holes_total × minutes_per_hole

    Base time (~30 min) - LABOR:
    - Load, clamp, warm up: 5-10 min
    - Pick up 3 datums/planes, coordinate system: 5-10 min
    - Quick size/flatness/squareness checks: 10-15 min

    Per hole time (~1.0 min) - MACHINE:
    - First article (building/debugging program)
    - Move to position, take 4-6 circle touches, retract, process

    Args:
        holes_total: Total number of holes to inspect

    Returns:
        Dict with 'setup_labor_min', 'checking_machine_min', 'total_min', 'holes_checked'

    Example:
        >>> cmm_inspection_minutes(88)
        {'setup_labor_min': 30, 'checking_machine_min': 88, 'total_min': 118, 'holes_checked': 88}
    """
    base_block_min = 30  # Base setup and datum time (LABOR)
    minutes_per_hole = 1.0  # First article inspection time per hole (MACHINE)

    # Only do CMM if there are holes to inspect (threshold: >20 holes)
    if holes_total > 20:
        checking_min = holes_total * minutes_per_hole
        total_min = base_block_min + checking_min
        return {
            'setup_labor_min': base_block_min,
            'checking_machine_min': checking_min,
            'total_min': total_min,
            'holes_checked': holes_total
        }
    else:
        return {
            'setup_labor_min': 0.0,
            'checking_machine_min': 0.0,
            'total_min': 0.0,
            'holes_checked': 0
        }


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
                       'doc_axial_in', 'woc_radial_pct', 'linear_cut_rate_ipm',
                       'tap_sfm_start', 'tap_overhead_sec_per_hole', 'grinding_time_factor']:
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


def get_grind_factor(material: str) -> float:
    """
    Look up grind material factor for wet grinding time calculations.

    Searches for grind_material_factor, grind_factor, or material_factor columns.
    Falls back to 1.0 if not found.

    Args:
        material: Material name (e.g., "P20 Tool Steel", "17-4 PH Stainless")

    Returns:
        Material factor for grinding (default 1.0)
    """
    data = load_speeds_feeds_data()

    # Try to find factor for this material
    for row in data:
        if row.get('material') == material:
            # Try different column names
            for col_name in ['grind_material_factor', 'grind_factor', 'material_factor']:
                if col_name in row and row[col_name]:
                    try:
                        return float(row[col_name])
                    except (ValueError, TypeError):
                        pass

    # Fall back to GENERIC material
    for row in data:
        if row.get('material_group') == 'GENERIC':
            for col_name in ['grind_material_factor', 'grind_factor', 'material_factor']:
                if col_name in row and row[col_name]:
                    try:
                        return float(row[col_name])
                    except (ValueError, TypeError):
                        pass

    # Default fallback
    return 1.0


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


def get_material_removal_rates(material: str) -> Dict[str, float]:
    """
    Get material-specific removal rates for square-up operations.

    Returns:
        Dict with 'milling_min_per_cuin' and 'grinding_min_per_cuin'

    Rates based on typical die-shop practice:
    - Milling: Rough removal to near size (~90% of total volume)
    - Grinding: Finish last few thou for flatness/parallelism (~10% of volume)
    """
    # Default rates (conservative)
    default_rates = {
        'milling_min_per_cuin': 0.25,    # ~4 in³/min
        'grinding_min_per_cuin': 4.0,    # ~0.25 in³/min (slow, precise)
        'setup_overhead_min': 20,         # Setup, indicating, clamping
        'flip_deburr_min': 15            # Flips, deburr, edge break
    }

    # Material-specific adjustments
    # Use speeds_feeds lookup for grinding_time_factor
    sf = get_speeds_feeds(material, "Endmill_Profile")

    if sf:
        grinding_factor = sf.get('grinding_time_factor', 1.0)
        default_rates['grinding_min_per_cuin'] = 4.0 * grinding_factor

        # Adjust milling rate based on SFM (harder materials = slower)
        sfm = sf.get('sfm_start', 250)
        if sfm < 150:  # Hard material (tool steel, stainless)
            default_rates['milling_min_per_cuin'] = 0.35
        elif sfm > 400:  # Soft material (aluminum)
            default_rates['milling_min_per_cuin'] = 0.15

    return default_rates


def estimate_machine_hours_from_plan(
    plan: Dict[str, Any],
    material: str = "GENERIC",
    plate_LxW: Tuple[float, float] = (0, 0),
    thickness: float = 0,
    stock_thickness: float = 0
) -> Dict[str, Any]:
    """
    Estimate machine hours from a process plan.

    Args:
        plan: Process plan dict (from plan_job or plan_from_cad_file)
        material: Material name for speeds/feeds lookup
        plate_LxW: Plate length and width in inches
        thickness: Plate thickness in inches (final part thickness)
        stock_thickness: Starting stock thickness in inches (McMaster stock)

    Returns:
        Dict with machine time breakdown by operation type

    Example:
        >>> plan = plan_from_cad_file("part.dxf")
        >>> machine_time = estimate_machine_hours_from_plan(plan, "P20 Tool Steel", (8, 4), 0.5, 0.75)
        >>> print(f"Total: {machine_time['total_hours']:.2f} hours")
    """
    from dataclasses import asdict

    ops = plan.get('ops', [])
    L, W = plate_LxW
    T = thickness
    stock_T = stock_thickness if stock_thickness > 0 else T  # Fall back to part thickness if not provided

    time_breakdown = {
        'drilling': 0,
        'milling': 0,
        'edm': 0,
        'tapping': 0,
        'grinding': 0,
        'other': 0,
    }

    # Lists to collect detailed operation breakdowns
    milling_operations_detailed = []
    grinding_operations_detailed = []

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

        # ---------- Punch Planner Operations (MUST be before general handlers) ----------
        # Wire EDM profile for punch outline (before 'profile' check at line 1477)
        elif op_type == 'wire_edm_profile':
            from cad_quoter.pricing.time_estimator import estimate_wire_edm_minutes, _edm_material_factor
            material_group = op.get('material_group', '')
            minutes = estimate_wire_edm_minutes(op, material, material_group)
            time_breakdown['edm'] += minutes

            # EDM debug output
            perim = op.get("wire_profile_perimeter_in", 0.0)
            thk = op.get("thickness_in", 0.0)
            from cad_quoter.pricing.time_estimator import _wire_mins_per_in
            mpi = _wire_mins_per_in(material, material_group, thk)
            mat_factor = _edm_material_factor(material, material_group)
            t_window = perim * mpi * mat_factor
            print(f"  DEBUG [EDM wire_edm_profile]: path_length={perim:.3f}\", min_per_in={mpi:.3f}, "
                  f"material_factor={mat_factor:.2f}, part_thickness={thk:.3f}\", t_window={t_window:.2f} min")

        # Face grinding (top/bottom surfaces)
        elif op_type == 'grind_faces':
            from cad_quoter.pricing.time_estimator import estimate_face_grind_minutes
            material_group = op.get('material_group', '')
            stock_total = op.get('stock_removed_total', 0.006)  # Conservative default
            op_L = op.get('width_in', L)
            op_W = op.get('thickness_in', W)
            minutes = estimate_face_grind_minutes(op_L, op_W, stock_total, material, material_group, faces=2)
            time_breakdown['grinding'] += minutes

            # Add detailed operation for renderer
            volume_cuin = op_L * op_W * stock_total
            from cad_quoter.pricing.time_estimator import _grind_factor
            factor = _grind_factor(material, material_group)
            grinding_operations_detailed.append({
                'op_name': 'grind_faces',
                'op_description': 'Face Grind - Final Kiss',
                'length': op_L,
                'width': op_W,
                'stock_removed_total': stock_total,
                'faces': 2,
                'volume_removed': volume_cuin,
                'grind_material_factor': factor,
                'time_minutes': minutes,
            })

        # Grind_reference_faces - Establish datums before profile (punch datum heuristics)
        elif op_type == 'grind_reference_faces':
            from cad_quoter.pricing.time_estimator import estimate_face_grind_minutes, _grind_factor
            material_group = op.get('material_group', '')
            stock_total = op.get('stock_removed_total', 0.006)
            faces = op.get('faces', 2)
            op_L = op.get('length_in', L)
            op_W = op.get('width_in', W)
            minutes = estimate_face_grind_minutes(op_L, op_W, stock_total, material, material_group, faces=faces)
            time_breakdown['grinding'] += minutes

            # Add detailed operation for renderer (DATUM line)
            volume_cuin = op_L * op_W * stock_total
            factor = _grind_factor(material, material_group)
            grinding_operations_detailed.append({
                'op_name': 'grind_reference_faces',
                'op_description': 'Grind Reference Faces (Datum)',
                'length': op_L,
                'width': op_W,
                'stock_removed_total': stock_total,
                'faces': faces,
                'volume_removed': volume_cuin,
                'grind_material_factor': factor,
                'time_minutes': minutes,
                'is_datum': True,  # Flag for renderer
            })

        # Grind_length - End face grinding for both round and non-round punches
        elif op_type == 'grind_length':
            from cad_quoter.pricing.time_estimator import estimate_face_grind_minutes, _grind_factor
            material_group = op.get('material_group', '')
            stock_total = op.get('stock_removed_total', 0.006)
            faces = op.get('faces', 2)
            # Use diameter for round or width/thickness for rectangular
            if op.get('diameter'):
                # Round punch - use diameter as both L and W for face area
                d = op.get('diameter', 0)
                op_L = op_W = d
            else:
                op_L = op.get('width_in', L)
                op_W = op.get('thickness_in', W)
            minutes = estimate_face_grind_minutes(op_L, op_W, stock_total, material, material_group, faces=faces)
            time_breakdown['grinding'] += minutes

            # Add detailed operation
            volume_cuin = op_L * op_W * stock_total
            factor = _grind_factor(material, material_group)
            grinding_operations_detailed.append({
                'op_name': 'grind_length',
                'op_description': 'Grind to Length (End Faces)',
                'length': op_L,
                'width': op_W,
                'stock_removed_total': stock_total,
                'faces': faces,
                'volume_removed': volume_cuin,
                'grind_material_factor': factor,
                'time_minutes': minutes,
            })

        # OD grinding for round pierce punches (rough and finish)
        elif op_type in ('grind_od', 'od_grind', 'od_grind_rough', 'od_grind_finish'):
            from cad_quoter.pricing.time_estimator import estimate_od_grind_minutes, _grind_factor
            material_group = op.get('material_group', '')
            # Build meta dict from op parameters
            meta = {
                'od_grind_volume_removed_cuin': op.get('od_grind_volume_removed_cuin', 0),
                'diameter': op.get('diameter', op.get('total_length_in', 0)),
                'thickness': op.get('total_length_in', L),
                'stock_allow_radial': 0.003,
            }
            # Also check plan meta
            plan_meta = plan.get('meta', {})
            for k, v in plan_meta.items():
                if k not in meta or not meta[k]:
                    meta[k] = v

            minutes = estimate_od_grind_minutes(meta, material, material_group)
            time_breakdown['grinding'] += minutes

            # Add detailed operation
            factor = _grind_factor(material, material_group)
            op_desc = 'OD Grind - Rough' if 'rough' in op_type else 'OD Grind - Finish'
            grinding_operations_detailed.append({
                'op_name': op_type,
                'op_description': op_desc,
                'num_diams': op.get('num_diams', 1),
                'total_length_in': op.get('total_length_in', L),
                'grind_material_factor': factor,
                'time_minutes': minutes,
            })

        # Optional: rough milling/turning operations (placeholder)
        elif op_type in ('mill_rough_profile', 'mill_turn_rough', 'turn_rough'):
            # Placeholder - can be expanded later with actual milling time estimates
            # For now, use a simple conservative estimate
            time_breakdown['milling'] += 5  # 5 minutes placeholder

        # Milling operations (general handlers - must come after punch-specific ones)
        elif 'face_mill' in op_type or 'mill_face' in op_type:
            # Face milling
            area = L * W if (L and W) else 10  # Default 10 sq in
            time_breakdown['milling'] += calculate_milling_time(L or 3, W or 3, 0.05, material)

        elif 'endmill' in op_type or 'profile' in op_type:
            perimeter = 2 * ((L or 4) + (W or 3))
            time_breakdown['milling'] += calculate_milling_time(perimeter, 0.1, T or 0.5, material, "Endmill_Profile")

        # EDM operations (general handler - must come after specific punch handlers)
        elif 'wedm' in op_type or 'wire_edm' in op_type:
            num_windows = op.get('windows', 1)
            skims = op.get('skims', 0)
            # Estimate perimeter
            perimeter = op.get('wire_profile_perimeter_in', 4.0)  # inches, typical window
            thickness = op.get('thickness_in', T or 0.5)
            edm_time = calculate_edm_time(perimeter, thickness, num_windows, skims, material)
            time_breakdown['edm'] += edm_time

            # EDM debug output
            from cad_quoter.pricing.time_estimator import _edm_material_factor
            mat_factor = op.get('edm_material_factor') or op.get('material_factor') or _edm_material_factor(material, "")
            mpi = op.get('edm_min_per_in', 0.33)  # Default estimate
            t_window = perimeter * mpi * mat_factor
            print(f"  DEBUG [EDM {op_type}]: path_length={perimeter:.3f}\", min_per_in={mpi:.3f}, "
                  f"material_factor={mat_factor:.2f}, part_thickness={thickness:.3f}\", "
                  f"edm_windows_qty={num_windows}, t_window={t_window:.2f} min, total_time={edm_time:.2f} min")

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

        # Squaring operations (wet grind)
        elif op_type == 'wet_grind_square_all':
            # Wet grind squaring: time based on volume and material factor
            from cad_quoter.pricing.time_estimator import estimate_wet_grind_minutes

            # Calculate actual thickness removal needed
            # Total to remove = stock_thickness - final_part_thickness
            total_thickness_to_remove = stock_T - T if (stock_T > T and T > 0) else 0.0

            # Finish grind is always 0.050" (0.025" per face)
            finish_grind_stock = 0.050

            # Rough removal is everything else (milling)
            rough_removal = max(0.0, total_thickness_to_remove - finish_grind_stock)

            # Add rough milling operation if there's significant material to remove
            if rough_removal > 0.010:  # More than 10 thou to remove
                # Calculate rough milling time for face milling the excess thickness
                # Use face milling rate: ~0.25 min/in³ for aluminum, adjusted for material
                sf_face = get_speeds_feeds(material, "Endmill_Face")
                if sf_face:
                    sfm = sf_face.get('sfm_start', 200)
                    # Slower for harder materials
                    milling_min_per_cuin = 0.35 if sfm < 150 else 0.25
                else:
                    milling_min_per_cuin = 0.25

                rough_volume = L * W * rough_removal
                rough_milling_time = rough_volume * milling_min_per_cuin

                # Add to milling time
                time_breakdown['milling'] += rough_milling_time

                # Create detailed operation for rough stock removal
                milling_operations_detailed.append({
                    'op_name': 'face_mill_rough_stock',
                    'op_description': f'Face Mill - Rough Stock Removal ({rough_removal:.3f}")',
                    'length': L,
                    'width': W,
                    'perimeter': 0,
                    'tool_diameter': W / 3.0 if W > 0 else 0.75,
                    'passes': 3,
                    'stepover': 0,
                    'radial_stock': None,
                    'axial_step': None,
                    'axial_passes': None,
                    'radial_passes': None,
                    'path_length': L * 3,  # 3 passes
                    'feed_rate': 0,
                    'time_minutes': rough_milling_time,
                    '_used_override': False,
                    'override_time_minutes': None
                })

            # Now handle the finish grind (0.050" total, 0.025" per face)
            stock_removed = finish_grind_stock
            faces = op.get('faces', 2)
            grind_time, material_factor = estimate_wet_grind_minutes(
                L, W, stock_removed, material, material_group='', faces=faces
            )
            volume_cuin = L * W * stock_removed
            min_per_cuin = 3.0
            time_breakdown['grinding'] += grind_time

            # Create detailed operation object for finish grind
            grinding_operations_detailed.append({
                'op_name': 'wet_grind_square_all',
                'op_description': 'Wet Grind - Top/Bottom Pair (Finish)',
                'length': L,
                'width': W,
                'area': L * W,
                'stock_removed_total': stock_removed,
                'faces': faces,
                'volume_removed': volume_cuin,
                'min_per_cuin': min_per_cuin,
                'material_factor': material_factor,
                'grind_material_factor': material_factor,  # For renderer display
                'time_minutes': grind_time,
                '_used_override': False  # Wet grind doesn't use overrides
            })

        # Squaring operations (mill - rough faces)
        elif op_type == 'square_up_rough_faces':
            # Get operation parameters (needed for detailed object regardless of override)
            D = op.get('tool_diameter_in') or (W / 3.0 if W > 0 else 0.75)
            target_passes = op.get('target_pass_count', 3)
            path_in = target_passes * L if L > 0 else 0
            stepover = (W / 3.0) * 0.95 if W > 0 else 0  # ~5% overlap for 3 stripes

            # Check if time is overridden
            override_time = op.get('override_time_minutes')
            used_override = override_time is not None
            if used_override:
                minutes = override_time
                ipm = 0  # Not calculated when overridden
            else:
                # Look up face milling IPM
                sf = get_speeds_feeds(material, "Endmill_Face")
                if sf:
                    fz = sf.get('fz_ipr_0_25in', 0.004)
                    sfm = sf.get('sfm_start', 200)
                    rpm = (sfm * 12) / (3.14159 * D) if D > 0 else 1000
                    rpm = min(rpm, 8000)
                    ipm = fz * 4 * rpm  # 4 flutes typical for face mill
                else:
                    ipm = 50  # Fallback IPM

                minutes = (path_in / max(1e-6, ipm)) * 1.05

            time_breakdown['milling'] += minutes

            # Create detailed operation object
            milling_operations_detailed.append({
                'op_name': 'square_up_rough_faces',
                'op_description': 'Face Mill - Top & Bottom',
                'length': L,
                'width': W,
                'perimeter': 0,  # Not applicable for face ops
                'tool_diameter': D,
                'passes': target_passes,
                'stepover': stepover,
                'radial_stock': None,
                'axial_step': None,
                'axial_passes': None,
                'radial_passes': None,
                'path_length': path_in,
                'feed_rate': ipm,
                'time_minutes': minutes,
                '_used_override': used_override,
                'override_time_minutes': override_time
            })

        # Squaring operations (mill - rough sides)
        elif op_type == 'square_up_rough_sides':
            # Get operation parameters (needed for detailed object regardless of override)
            import math
            D = op.get('tool_diameter_in', 0.75)
            radial_stock = op.get('radial_stock', 0.250)
            axial_step = op.get('axial_step', 0.75)

            # Calculate number of passes
            woc = min(radial_stock, 0.5 * D)
            axial_passes = max(1, math.ceil(T / axial_step) if axial_step > 0 else 1)
            radial_passes = max(1, math.ceil(radial_stock / woc) if woc > 0 else 1)

            # Path length: perimeter × passes
            perimeter = 2 * (L + W)
            path_in = perimeter * axial_passes * radial_passes

            # Check if time is overridden
            override_time = op.get('override_time_minutes')
            used_override = override_time is not None
            if used_override:
                minutes = override_time
                ipm = 0  # Not calculated when overridden
            else:
                # Look up side milling IPM
                sf = get_speeds_feeds(material, "Endmill_Profile")
                if sf:
                    fz = sf.get('fz_ipr_0_25in', 0.003)
                    sfm = sf.get('sfm_start', 200)
                    rpm = (sfm * 12) / (3.14159 * D) if D > 0 else 1000
                    rpm = min(rpm, 8000)
                    ipm = fz * 4 * rpm
                else:
                    ipm = 40  # Fallback IPM

                minutes = (path_in / max(1e-6, ipm)) * 1.05

            time_breakdown['milling'] += minutes

            # Create detailed operation object
            milling_operations_detailed.append({
                'op_name': 'square_up_rough_sides',
                'op_description': 'Side Mill - Square Up (Rough)',
                'length': L,
                'width': W,
                'perimeter': perimeter,
                'tool_diameter': D,
                'passes': axial_passes * radial_passes,
                'stepover': None,  # Not applicable for side ops
                'radial_stock': radial_stock,
                'axial_step': axial_step,
                'axial_passes': axial_passes,
                'radial_passes': radial_passes,
                'path_length': path_in,
                'feed_rate': ipm,
                'time_minutes': minutes,
                '_used_override': used_override,
                'override_time_minutes': override_time
            })

        # Grinding operations (general handler - punch-specific handlers are earlier)
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
        'milling_operations': milling_operations_detailed,
        'grinding_operations': grinding_operations_detailed,
    }


def render_square_up_block(
    plan: Dict[str, Any],
    milling_ops: List[Dict[str, Any]],
    grinding_ops: List[Dict[str, Any]],
    setup_time_min: float = 0.0,
    flip_time_min: float = 0.0
) -> List[str]:
    """Return pre-wrapped lines for the Square-Up section.

    Produces ≤106 char lines matching hole table style for consistency.

    Args:
        plan: Process plan dict with 'ops' list
        milling_ops: Detailed milling operations from estimate_machine_hours_from_plan
        grinding_ops: Detailed grinding operations from estimate_machine_hours_from_plan
        setup_time_min: Setup overhead in minutes
        flip_time_min: Flip/deburr time in minutes

    Returns:
        List of formatted strings for display
    """
    lines: List[str] = []
    total_time = 0.0

    # Find square-up operations
    side_op = None
    face_op = None
    grind_op = None

    for op in milling_ops:
        if op.get('op_name') == 'square_up_rough_sides':
            side_op = op
        elif op.get('op_name') == 'square_up_rough_faces':
            face_op = op

    for op in grinding_ops:
        if op.get('op_name') == 'wet_grind_square_all':
            grind_op = op

    # Determine method
    is_mill_route = side_op is not None or face_op is not None
    is_grind_route = grind_op is not None

    if not is_mill_route and not is_grind_route:
        return []  # No square-up operations

    # Header line
    if is_mill_route:
        lines.append("SQUARE-UP — MILLING")
    else:
        lines.append("SQUARE-UP — WET GRIND")

    lines.append("-" * 106)

    # Context lines
    if is_mill_route:
        # Get tool diameter from side_op or face_op
        D = side_op.get('tool_diameter', 0) if side_op else (face_op.get('tool_diameter', 0) if face_op else 0)
        ctx1 = f"Method: Mill | ToolØ = W/3 ({D:.3f}\") | Side stock 0.250\" | Top/Bottom stock 0.025\""
        ctx2 = "Strategy: 3-pass face | Setup+Flip included in times"
        lines.append(ctx1)
        lines.append(ctx2)
    else:
        # Grind context line
        gf = grind_op.get('grind_material_factor', 1.0) if grind_op else 1.0
        ctx = f"Method: Wet Grind | Faces: Top & Bottom | Stock total 0.050\" | min/in³ = 3.0 | Factor {gf:.2f}"
        lines.append(ctx)

    # Subheader
    lines.append("")
    lines.append("TIME PER OP - SQUARE/FINISH")
    lines.append("-" * 106)

    # Mill route op lines
    if is_mill_route:
        # Side mill line
        if side_op:
            perim = side_op.get('perimeter', 0)
            ax_passes = side_op.get('axial_passes', 1)
            rad_passes = side_op.get('radial_passes', 1)
            tool_d = side_op.get('tool_diameter', 0)
            ipm = side_op.get('feed_rate', 0)
            path = side_op.get('path_length', 0)
            time_min = side_op.get('time_minutes', 0)
            used_ovr = side_op.get('_used_override', False)

            ovr_badge = " (ovr)" if used_ovr else ""

            # Build line with optional IPM - use compact format to stay ≤106 chars
            if ipm > 0:
                # Compact format: use "P" instead of "Perim", compact passes format
                line = (f"Side Mill – SQ UP (Rough) | P {perim:.1f}\" | {ax_passes}×{rad_passes} "
                        f"| Ø {tool_d:.3f}\" |{ipm:.0f}ipm| Path {path:.1f}\" | t/op {time_min:.2f} min{ovr_badge}")
            else:
                line = (f"Side Mill – SQ UP (Rough) | Perim {perim:.1f}\" | Ax {ax_passes}× Rad {rad_passes}× "
                        f"| Ø {tool_d:.3f}\" | Path {path:.1f}\" | t/op {time_min:.2f} min{ovr_badge}")

            lines.append(line[:106])
            total_time += time_min

        # Face mill line
        if face_op:
            tool_d = face_op.get('tool_diameter', 0)
            stepover = face_op.get('stepover', 0)
            ipm = face_op.get('feed_rate', 0)
            path = face_op.get('path_length', 0)
            time_min = face_op.get('time_minutes', 0)
            used_ovr = face_op.get('_used_override', False)

            ovr_badge = " (ovr)" if used_ovr else ""

            # Build line with optional IPM - use compact format to stay ≤106 chars
            if ipm > 0:
                # Compact format: shorter step format when IPM is present
                line = (f"Face Mill – Top & Bottom | Passes 3×L | Ø {tool_d:.3f}\" | S {stepover:.3f}\" "
                        f"|{ipm:.0f}ipm| Path {path:.1f}\" | t/op {time_min:.2f} min{ovr_badge}")
            else:
                line = (f"Face Mill – Top & Bottom  | Passes 3×L  | Ø {tool_d:.3f}\" | Step {stepover:.3f}\" "
                        f"| Path {path:.1f}\" | t/op {time_min:.2f} min{ovr_badge}")

            lines.append(line[:106])
            total_time += time_min

    # Grind route op line
    if is_grind_route:
        L = grind_op.get('length', 0)
        W = grind_op.get('width', 0)
        stock = grind_op.get('stock_removed_total', 0.050)
        vol = L * W * stock
        time_min = grind_op.get('time_minutes', 0)

        line = f"Face Grind – Pair         | Vol L×W×0.050 = {vol:.3f} in³ | t/op {time_min:.2f} min"
        lines.append(line[:106])
        total_time += time_min

    # Total line
    lines.append("")
    lines.append(f"Total Square/Finish Time: {total_time:.2f} min")

    return lines


def render_punch_datum_block(
    grinding_ops: List[Dict[str, Any]],
) -> List[str]:
    """Render DATUM line for punch reference face grinding.

    When punch plan includes Grind_reference_faces, prints a one-liner:
    DATUM: Grind 2 faces | L×W×stock → min | factor {grind_factor}

    Args:
        grinding_ops: Detailed grinding operations from estimate_machine_hours_from_plan

    Returns:
        List of formatted strings for display (single line for DATUM)
    """
    lines = []

    # Find datum operation (Grind_reference_faces)
    datum_op = None
    for op in grinding_ops:
        if op.get('op_name') == 'grind_reference_faces' or op.get('is_datum'):
            datum_op = op
            break

    if not datum_op:
        return []

    L = datum_op.get('length', 0)
    W = datum_op.get('width', 0)
    stock = datum_op.get('stock_removed_total', 0.006)
    factor = datum_op.get('grind_material_factor', 1.0)
    time_min = datum_op.get('time_minutes', 0)
    faces = datum_op.get('faces', 2)

    # Format: DATUM: Grind 2 faces | L×W×stock → min | factor {grind_factor}
    line = f"DATUM: Grind {faces} faces | {L:.3f}×{W:.3f}×{stock:.3f} → {time_min:.2f} min | factor {factor:.2f}"
    lines.append(line[:106])

    return lines


def render_punch_grind_block(
    grinding_ops: List[Dict[str, Any]],
) -> List[str]:
    """Render grinding operations for punches (OD grind, length grind, etc.).

    Shows OD/length grinds for round punches without plate-style square-up block.

    Args:
        grinding_ops: Detailed grinding operations from estimate_machine_hours_from_plan

    Returns:
        List of formatted strings for display
    """
    lines = []
    total_time = 0.0

    # Group operations
    od_ops = []
    length_ops = []
    face_ops = []

    for op in grinding_ops:
        op_name = op.get('op_name', '')
        if 'od_grind' in op_name:
            od_ops.append(op)
        elif op_name == 'grind_length':
            length_ops.append(op)
        elif op_name == 'grind_faces':
            face_ops.append(op)

    # OD Grind operations
    for op in od_ops:
        op_desc = op.get('op_description', 'OD Grind')
        num_diams = op.get('num_diams', 1)
        length = op.get('total_length_in', 0)
        factor = op.get('grind_material_factor', 1.0)
        time_min = op.get('time_minutes', 0)

        line = f"  {op_desc}: {num_diams} diam(s) × {length:.2f}\" | factor {factor:.2f} | {time_min:.2f} min"
        lines.append(line[:106])
        total_time += time_min

    # Length grind operations
    for op in length_ops:
        L = op.get('length', 0)
        W = op.get('width', 0)
        stock = op.get('stock_removed_total', 0.006)
        factor = op.get('grind_material_factor', 1.0)
        time_min = op.get('time_minutes', 0)

        line = f"  Grind Length: {L:.3f}×{W:.3f}×{stock:.3f} | factor {factor:.2f} | {time_min:.2f} min"
        lines.append(line[:106])
        total_time += time_min

    # Face grind operations (kiss pass)
    for op in face_ops:
        L = op.get('length', 0)
        W = op.get('width', 0)
        stock = op.get('stock_removed_total', 0.004)
        factor = op.get('grind_material_factor', 1.0)
        time_min = op.get('time_minutes', 0)

        line = f"  Face Grind: {L:.3f}×{W:.3f}×{stock:.3f} | factor {factor:.2f} | {time_min:.2f} min"
        lines.append(line[:106])
        total_time += time_min

    if lines:
        lines.append(f"  TOTAL Punch Grind Time: {total_time:.2f} minutes")

    return lines


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
    edm_groups = []  # Wire EDM operations from "FOR WIRE EDM" notes
    slot_groups = []  # Slot/obround features requiring milling

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

        # For TAP detection, check both fields since description may contain TAP info
        # even when OPERATION doesn't (e.g., compressed hole tables)
        combined_text = f"{operation} {description}".upper()

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
        # Use combined_text for TAP detection to catch TAP info in description field
        is_jig_grind = 'JIG GRIND' in op_text
        is_thru = 'THRU' in op_text
        is_tap = 'TAP' in combined_text  # Check both OPERATION and DESCRIPTION for TAP
        # Handle both straight apostrophe (') and curly apostrophe (') for C'BORE/C'DRILL
        is_cbore = "C'BORE" in op_text or "C\u2019BORE" in op_text or 'CBORE' in op_text or 'COUNTERBORE' in op_text
        is_cdrill = "C'DRILL" in op_text or "C\u2019DRILL" in op_text or 'CDRILL' in op_text or 'CENTER DRILL' in op_text
        # Check combined_text for EDM to catch "FOR WIRE EDM" in either OPERATION or DESCRIPTION field
        is_for_edm = 'FOR WIRE EDM' in combined_text or 'FOR EDM' in combined_text or 'WIRE EDM' in combined_text
        # Check for slot/obround features
        is_slot = ('SLOT' in combined_text or 'OBROUND' in combined_text or 'ELONGATED' in combined_text or
                   re.search(r'\bR[\.\d]+\s*(?:X\s*[\d\.]+|OVER\s*R)', combined_text) is not None)

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

        # DRILL operations (main hole) - skip if it's a jig grind or standalone cbore
        # C'BORE entries should not be added to drill_groups (they go in cbore_groups)
        # C'DRILL entries should not be added to drill_groups (they go in cdrill_groups)
        # TAP entries need a drill operation first (tap drill hole), then tapping
        if not is_jig_grind and not is_cbore and not is_tap and not is_cdrill:
            # Time per hole: depth / feed_rate gives minutes (since feed_rate is IPM)
            time_per_hole = (depth / feed_rate) if feed_rate > 0 else 1.0
            time_per_hole += 0.1  # Add approach/retract time

            # Round times to 2 decimals for display consistency
            time_per_hole = round(time_per_hole, 2)
            total_time = round(time_per_hole * qty, 2)

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

        # TAP holes also need a drill operation for the tap drill hole
        # Calculate tap drill diameter from tap size (major_dia - 1/TPI)
        if is_tap:
            # Extract tap size to calculate tap drill diameter
            # Use combined_text to find tap spec in either OPERATION or DESCRIPTION
            tap_match = re.search(r'(\d+/\d+)-(\d+)', combined_text)
            if tap_match:
                # Fractional tap (e.g., 5/16-18)
                major_str = tap_match.group(1)
                tpi = int(tap_match.group(2))

                # Validate and correct thread specification
                corrected_major, corrected_tpi, was_corrected = validate_and_correct_thread(
                    major_str, tpi
                )
                if was_corrected:
                    import logging
                    logging.debug(f"Thread corrected (drill calc): {major_str}-{tpi} -> {corrected_major}-{corrected_tpi}")
                    tpi = corrected_tpi
                    major_str = corrected_major

                # Get major diameter from standard table or calculate from fraction
                if corrected_major in THREAD_MAJOR_DIAMETERS:
                    tap_major_dia = THREAD_MAJOR_DIAMETERS[corrected_major]
                else:
                    frac_parts = major_str.split('/')
                    tap_major_dia = float(frac_parts[0]) / float(frac_parts[1])
            else:
                # Try #10-32 format
                num_tap_match = re.search(r'#(\d+)-(\d+)', combined_text)
                if num_tap_match:
                    screw_num = int(num_tap_match.group(1))
                    tpi = int(num_tap_match.group(2))
                    major_str = f"#{screw_num}"

                    # Validate and correct thread specification
                    corrected_major, corrected_tpi, was_corrected = validate_and_correct_thread(
                        major_str, tpi
                    )
                    if was_corrected:
                        import logging
                        logging.debug(f"Thread corrected (drill calc): {major_str}-{tpi} -> {corrected_major}-{corrected_tpi}")
                        tpi = corrected_tpi

                    # Get major diameter from standard table
                    if corrected_major in THREAD_MAJOR_DIAMETERS:
                        tap_major_dia = THREAD_MAJOR_DIAMETERS[corrected_major]
                    else:
                        tap_major_dia = 0.060 + (screw_num * 0.013)
                else:
                    # Try bare number format like 4-40 (without # prefix)
                    bare_num_match = re.search(r'\b(\d+)-(\d+)\b', combined_text)
                    if bare_num_match:
                        screw_num = int(bare_num_match.group(1))
                        tpi = int(bare_num_match.group(2))
                        # Only treat as number thread if screw_num <= 12 (valid number sizes)
                        if screw_num <= 12:
                            major_str = f"#{screw_num}"

                            # Validate and correct thread specification
                            corrected_major, corrected_tpi, was_corrected = validate_and_correct_thread(
                                major_str, tpi
                            )
                            if was_corrected:
                                import logging
                                logging.debug(f"Thread corrected (drill calc): {major_str}-{tpi} -> {corrected_major}-{corrected_tpi}")
                                tpi = corrected_tpi

                            # Get major diameter from standard table
                            if corrected_major in THREAD_MAJOR_DIAMETERS:
                                tap_major_dia = THREAD_MAJOR_DIAMETERS[corrected_major]
                            else:
                                tap_major_dia = 0.060 + (screw_num * 0.013)
                        else:
                            tap_major_dia = ref_dia
                            tpi = 20  # Default TPI
                    else:
                        tap_major_dia = ref_dia
                        tpi = 20  # Default TPI

            # Sanity check: guard against bogus high TPI on larger taps
            # If TPI > 40 and tap_major_dia > 0.19", mark "CHECK TPI" and fall back to default
            if tpi > 40 and tap_major_dia > 0.19:
                import logging
                logging.debug(
                    f"CHECK TPI (drill calc): Suspicious TPI {tpi} for tap diameter {tap_major_dia:.3f}\". "
                    f"Falling back to default coarse thread."
                )
                # Fall back to coarse (first) TPI for this nominal size
                # Need to determine the nominal size for lookup
                # Check if we have major_str from earlier parsing
                if 'major_str' in locals() and major_str in STANDARD_THREADS:
                    tpi = STANDARD_THREADS[major_str][0]
                    logging.debug(f"Using default TPI {tpi} for {major_str}")
                else:
                    # Conservative fallback based on diameter
                    if tap_major_dia <= 0.25:
                        tpi = 20
                    elif tap_major_dia <= 0.5:
                        tpi = 13
                    else:
                        tpi = 10
                    logging.debug(f"Using conservative TPI {tpi} based on diameter {tap_major_dia:.3f}\"")

            # Calculate tap drill diameter: major_dia - (1/TPI)
            # This gives approximately 75% thread engagement
            tap_drill_dia = tap_major_dia - (1.0 / tpi) if tpi > 0 else tap_major_dia * 0.8
            tap_drill_dia = max(tap_drill_dia, 0.05)  # Minimum drill size

            # Determine drill depth for tap drill
            if is_thru or 'TAP THRU' in combined_text:
                tap_drill_depth = thickness if thickness > 0 else 2.0
            else:
                # Extract depth from "TAP X {number} DEEP"
                tap_depth_match = re.search(r'[TAP\s+]*X\s+(\d*\.\d+|\d+)\s+DEEP', combined_text)
                if tap_depth_match:
                    tap_drill_depth = float(tap_depth_match.group(1)) + 0.1  # Drill slightly deeper than tap
                else:
                    tap_drill_depth = 0.6  # Default

            # Sanity check: tap drill depth cannot exceed material thickness
            if thickness > 0 and tap_drill_depth > thickness + 0.1:
                import logging
                logging.warning(
                    f"Tap drill depth {tap_drill_depth:.3f}\" exceeds thickness {thickness:.3f}\" for hole {hole_id}, "
                    f"clamping to thickness"
                )
                tap_drill_depth = thickness

            # Calculate drill time using tap drill diameter
            tap_drill_rpm = (sfm * 12) / (3.14159 * tap_drill_dia) if tap_drill_dia > 0 else 1000
            tap_drill_rpm = min(tap_drill_rpm, 3500)

            # Select feed based on tap drill diameter
            if tap_drill_dia <= 0.1875:
                tap_drill_fpt = sf_drill.get('fz_ipr_0_125in', 0.002)
            elif tap_drill_dia <= 0.375:
                tap_drill_fpt = sf_drill.get('fz_ipr_0_25in', 0.004)
            else:
                tap_drill_fpt = sf_drill.get('fz_ipr_0_5in', 0.008)

            tap_drill_feed = tap_drill_rpm * 2 * tap_drill_fpt

            time_per_hole = (tap_drill_depth / tap_drill_feed) if tap_drill_feed > 0 else 1.0
            time_per_hole += 0.1  # Approach/retract

            total_time = time_per_hole * qty

            drill_groups.append({
                'hole_id': hole_id,
                'diameter': tap_drill_dia,
                'depth': tap_drill_depth,
                'qty': qty,
                'sfm': sfm,
                'ipr': tap_drill_fpt,
                'rpm': tap_drill_rpm,
                'feed_rate': tap_drill_feed,
                'time_per_hole': time_per_hole,
                'total_time': total_time,
                'description': f"Tap drill for {entry.get('DESCRIPTION', op_text)}"
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

            # Base time per hole (geometry-based)
            time_per_hole_base = (
                setup_min +
                (grind_area * mpsi) +
                (stock_diam / stock_rate_diam) +
                spark_out_min
            )

            # Apply material grinding factor (aluminum < 1.0, tool steel = 1.0, carbide = 2.5, etc.)
            # Try Grinding operation first, then Endmill_Profile for backward compatibility
            sf_grind = get_speeds_feeds(material, "Grinding")
            if not sf_grind:
                sf_grind = get_speeds_feeds(material, "Endmill_Profile")

            grinding_time_factor = 1.0
            if sf_grind:
                grinding_time_factor = sf_grind.get('grinding_time_factor', 1.0)

            # Apply small-diameter factor: if dia < 0.080", multiply by 1.2-1.4
            small_dia_factor = 1.0
            if ref_dia < 0.080:
                # Use 1.3 as the middle of the range (1.2-1.4)
                # Scale based on how small: at 0.060" use 1.2, at 0.040" or less use 1.4
                if ref_dia <= 0.040:
                    small_dia_factor = 1.4
                elif ref_dia <= 0.060:
                    # Linear interpolation between 0.040 (1.4) and 0.060 (1.2)
                    small_dia_factor = 1.4 - (ref_dia - 0.040) / (0.060 - 0.040) * (1.4 - 1.2)
                else:
                    # Linear interpolation between 0.060 (1.2) and 0.080 (1.0)
                    small_dia_factor = 1.2 - (ref_dia - 0.060) / (0.080 - 0.060) * (1.2 - 1.0)

            # Calculate time per hole with material and diameter factors
            time_per_hole = time_per_hole_base * grinding_time_factor * small_dia_factor

            # Detect die sections/inserts from description
            desc_upper = entry.get('DESCRIPTION', '').upper()
            is_die_section = any(kw in desc_upper for kw in [
                'DIE SECTION', 'DIE INSERT', 'FORM DIE', 'CARBIDE INSERT',
                'DIE CHASER', 'INSERT', 'SECTION'
            ])

            # Enforce minimum grinding time for die sections/inserts (20.0 minutes)
            time_before_min = time_per_hole
            if is_die_section and time_per_hole < 20.0:
                time_per_hole = 20.0

            # Round times to 2 decimals for display consistency
            time_per_hole = round(time_per_hole, 2)
            total_time = round(time_per_hole * qty, 2)

            # Log jig-grind calculation details
            import logging
            logging.debug(
                f"Jig grind {hole_id}: dia={ref_dia:.4f}\", depth={depth:.3f}\", "
                f"t_base={time_per_hole_base:.2f}min, material_factor={grinding_time_factor:.2f}, "
                f"small_dia_factor={small_dia_factor:.2f}, t_hole={time_per_hole:.2f}min"
                + (f", die_section_min_applied (was {time_before_min:.2f}min)" if is_die_section and time_before_min < 20.0 else "")
            )

            jig_grind_groups.append({
                'hole_id': hole_id,
                'diameter': ref_dia,
                'depth': depth,
                'qty': qty,
                'time_per_hole': time_per_hole,
                'total_time': total_time,
                'description': entry.get('DESCRIPTION', ''),
                'grinding_time_factor': grinding_time_factor,
                # JIG GRIND TIME MODEL TRANSPARENCY FIELDS
                'jig_grind_dia': ref_dia,
                'jig_grind_depth': depth,
                't_hole': time_per_hole,
                'material_factor': grinding_time_factor,
                'small_dia_factor': small_dia_factor,
                'is_die_section': is_die_section,
                'grind_area_sq_in': grind_area,
                't_base': round(time_per_hole_base, 2),
                'spark_out_min': spark_out_min,
            })

        # TAP operations
        if is_tap:
            # Extract tap size - use combined_text to find tap spec
            tap_match = re.search(r'(\d+/\d+)-(\d+)', combined_text)
            if tap_match:
                # Fractional tap (e.g., 5/8-11)
                major_str = tap_match.group(1)
                tpi = int(tap_match.group(2))

                # Validate and correct thread specification
                corrected_major, corrected_tpi, was_corrected = validate_and_correct_thread(
                    major_str, tpi
                )
                if was_corrected:
                    import logging
                    logging.debug(f"Thread corrected (tap time): {major_str}-{tpi} -> {corrected_major}-{corrected_tpi}")
                    tpi = corrected_tpi
                    major_str = corrected_major

                # Get tap diameter from standard table or calculate from fraction
                if corrected_major in THREAD_MAJOR_DIAMETERS:
                    tap_dia = THREAD_MAJOR_DIAMETERS[corrected_major]
                else:
                    frac_parts = major_str.split('/')
                    tap_dia = float(frac_parts[0]) / float(frac_parts[1])
            else:
                # Try #10-32 format
                num_tap_match = re.search(r'#(\d+)-(\d+)', combined_text)
                if num_tap_match:
                    screw_num = int(num_tap_match.group(1))
                    tpi = int(num_tap_match.group(2))
                    major_str = f"#{screw_num}"

                    # Validate and correct thread specification
                    corrected_major, corrected_tpi, was_corrected = validate_and_correct_thread(
                        major_str, tpi
                    )
                    if was_corrected:
                        import logging
                        logging.debug(f"Thread corrected (tap time): {major_str}-{tpi} -> {corrected_major}-{corrected_tpi}")
                        tpi = corrected_tpi

                    # Get tap diameter from standard table
                    if corrected_major in THREAD_MAJOR_DIAMETERS:
                        tap_dia = THREAD_MAJOR_DIAMETERS[corrected_major]
                    else:
                        tap_dia = 0.060 + (screw_num * 0.013)
                else:
                    # Try bare number format like 4-40 (without # prefix)
                    bare_num_match = re.search(r'\b(\d+)-(\d+)\b', combined_text)
                    if bare_num_match:
                        screw_num = int(bare_num_match.group(1))
                        tpi = int(bare_num_match.group(2))
                        # Only treat as number thread if screw_num <= 12 (valid number sizes)
                        if screw_num <= 12:
                            major_str = f"#{screw_num}"

                            # Validate and correct thread specification
                            corrected_major, corrected_tpi, was_corrected = validate_and_correct_thread(
                                major_str, tpi
                            )
                            if was_corrected:
                                import logging
                                logging.debug(f"Thread corrected (tap time): {major_str}-{tpi} -> {corrected_major}-{corrected_tpi}")
                                tpi = corrected_tpi

                            # Get tap diameter from standard table
                            if corrected_major in THREAD_MAJOR_DIAMETERS:
                                tap_dia = THREAD_MAJOR_DIAMETERS[corrected_major]
                            else:
                                tap_dia = 0.060 + (screw_num * 0.013)
                        else:
                            tap_dia = ref_dia * 0.8  # Estimate tap drill size
                            tpi = int(20 / tap_dia) if tap_dia > 0 else 20
                    else:
                        tap_dia = ref_dia * 0.8  # Estimate tap drill size
                        tpi = int(20 / tap_dia) if tap_dia > 0 else 20

            # Extract TAP depth - look for "TAP X {number} DEEP" or "X {number} DEEP"
            tap_depth_match = re.search(r'[TAP\s+]*X\s+(\d*\.\d+|\d+)\s+DEEP', combined_text)
            if tap_depth_match:
                tap_depth = float(tap_depth_match.group(1))
            elif 'TAP THRU' in combined_text or is_thru:
                tap_depth = thickness if thickness > 0 else 2.0
            else:
                tap_depth = 0.5  # Default

            # Sanity check: tap depth cannot exceed material thickness
            if thickness > 0 and tap_depth > thickness + 0.1:
                import logging
                logging.warning(
                    f"Tap depth {tap_depth:.3f}\" exceeds thickness {thickness:.3f}\" for hole {hole_id}, "
                    f"clamping to thickness"
                )
                tap_depth = thickness

            is_rigid = 'RIGID' in combined_text

            # Sanity check: guard against bogus high TPI on larger taps
            # If TPI > 40 and tap_dia > 0.19", mark "CHECK TPI" and fall back to default
            if tpi > 40 and tap_dia > 0.19:
                import logging
                logging.debug(
                    f"CHECK TPI: {major_str}-{tpi} has suspicious TPI {tpi} for diameter {tap_dia:.3f}\". "
                    f"Falling back to default coarse thread."
                )
                # Fall back to coarse (first) TPI for this nominal size
                if major_str in STANDARD_THREADS:
                    tpi = STANDARD_THREADS[major_str][0]
                    logging.debug(f"Using default TPI {tpi} for {major_str}")
                else:
                    # If not in standard threads, use conservative estimate
                    tpi = 20
                    logging.debug(f"Using conservative TPI {tpi} for non-standard size")

            # Look up material-specific tapping speeds/feeds
            sf_tap = get_speeds_feeds(material, "Tapping")
            if sf_tap:
                tap_sfm = sf_tap.get('sfm_start', 40)  # Use sfm_start for tapping
                overhead_sec = sf_tap.get('tap_overhead_sec_per_hole', 3.0)
            else:
                # Fallback values if no speeds/feeds found
                tap_sfm = 40
                overhead_sec = 3.0

            # Calculate tapping RPM from material-specific SFM
            tap_rpm = (tap_sfm * 12) / (3.14159 * tap_dia) if tap_dia > 0 else 500
            tap_rpm = min(tap_rpm, 1000)  # Cap at reasonable limit

            # Feed rate = RPM × pitch (pitch = 1/TPI)
            pitch = 1.0 / tpi if tpi > 0 else 0.05
            tap_feed_rate = tap_rpm * pitch  # IPM

            # Calculate cutting time
            time_per_hole = (tap_depth / tap_feed_rate) if tap_feed_rate > 0 else 2.0

            # Add material-specific overhead (approach/retract/dwell)
            time_per_hole += overhead_sec / 60.0  # Convert seconds to minutes

            # Rigid tap is faster (can reverse at full speed)
            if is_rigid:
                time_per_hole *= 0.7

            # Round times to 2 decimals for display consistency
            time_per_hole = round(time_per_hole, 2)
            total_time = round(time_per_hole * qty, 2)

            tap_groups.append({
                'hole_id': hole_id,
                'diameter': tap_dia,
                'depth': tap_depth,
                'qty': qty,
                'tpi': tpi,
                'sfm': tap_sfm,
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
            # Handle both x/X and × (multiplication sign U+00D7)
            cbore_depth_match = re.search(r'[Xx×]\s+(\d*\.\d+|\d+)\s+DEEP', op_text)
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

            # Round times to 2 decimals for display consistency
            time_per_hole = round(time_per_hole, 2)
            total_time = round(time_per_hole * qty, 2)

            # Extract side info (FRONT or BACK) from description or op_text
            cbore_side = None
            if "BACK" in op_text.upper():
                cbore_side = "back"
            elif "FRONT" in op_text.upper():
                cbore_side = "front"

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
                'description': entry.get('DESCRIPTION', ''),
                'side': cbore_side
            })

        # CENTER DRILL operations
        if is_cdrill:
            # Extract center drill depth if specified
            # Handle both x/X and × (multiplication sign U+00D7)
            cdrill_depth_match = re.search(r'[Xx×]\s+(\d*\.\d+|\d+)\s+DEEP', op_text)
            if cdrill_depth_match:
                cdrill_depth = float(cdrill_depth_match.group(1))
            else:
                cdrill_depth = 0.1  # Default shallow depth

            # Center drilling is quick - estimate based on depth
            time_per_hole = max(0.05, cdrill_depth * 0.5)  # Min 3 seconds, or 30 sec per inch
            # Round times to 2 decimals for display consistency
            time_per_hole = round(time_per_hole, 2)
            total_time = round(time_per_hole * qty, 2)

            cdrill_groups.append({
                'hole_id': hole_id,
                'diameter': ref_dia,
                'depth': cdrill_depth,
                'qty': qty,
                'time_per_hole': time_per_hole,
                'total_time': total_time,
                'description': entry.get('DESCRIPTION', '')
            })

        # WIRE EDM operations - holes marked "FOR WIRE EDM" are starter holes
        # These indicate wire EDM will be used to cut out shapes from these threading points
        if is_for_edm:
            # Wire EDM time calculation for starter holes
            # Each starter hole represents a window to be cut
            # Estimate perimeter per window based on typical die section geometry
            # Default 4" perimeter per window if no other info available
            estimated_perimeter_per_window = 4.0

            # Use part thickness or default (actual part thickness, not stock)
            edm_thickness = thickness if thickness > 0 else 0.5

            # Calculate time using existing EDM formula:
            # rough_time = (perimeter * thickness * num_windows) / rough_ipm
            # skim_time = similar
            # setup_time = num_windows * 5 minutes
            sf_rough = get_speeds_feeds(material, "Wire_EDM_Rough")
            rough_ipm = sf_rough.get('linear_cut_rate_ipm', 3.0) if sf_rough else 3.0

            sf_skim = get_speeds_feeds(material, "Wire_EDM_Skim")
            skim_ipm = sf_skim.get('linear_cut_rate_ipm', 2.0) if sf_skim else 2.0

            # Get EDM material factor
            from cad_quoter.pricing.time_estimator import _edm_material_factor
            material_group = ""  # Would need to be passed in
            edm_mat_factor = _edm_material_factor(material, material_group)

            # Calculate time per window with material factor
            # Convert IPM to min/in for rough and skim passes
            rough_min_per_in = 1.0 / rough_ipm if rough_ipm > 0 else 0.33
            skim_min_per_in = 1.0 / skim_ipm if skim_ipm > 0 else 0.50

            # rough_time = path_length * min_per_in * material_factor (no thickness in this model)
            # But for wire EDM, we need to account for thickness affecting cut time
            # So: time = perimeter * thickness * min_per_in * material_factor
            rough_time_per = estimated_perimeter_per_window * edm_thickness * rough_min_per_in * edm_mat_factor
            # Add one skim pass for precision
            skim_time_per = estimated_perimeter_per_window * edm_thickness * skim_min_per_in * edm_mat_factor
            setup_time_per = 5.0  # 5 minutes setup per window (threading wire, etc.)

            # t_window = edm_path_length_in * edm_min_per_in * edm_material_factor
            # (for rough pass, combining perimeter and thickness as path length × thickness)
            t_window_rough = estimated_perimeter_per_window * rough_min_per_in * edm_mat_factor
            t_window_skim = estimated_perimeter_per_window * skim_min_per_in * edm_mat_factor

            time_per_hole = round(rough_time_per + skim_time_per + setup_time_per, 2)
            total_time = round(time_per_hole * qty, 2)

            edm_groups.append({
                'hole_id': hole_id,
                'diameter': ref_dia,
                'depth': edm_thickness,
                'qty': qty,
                'perimeter_per_window': estimated_perimeter_per_window,
                'rough_ipm': rough_ipm,
                'skim_ipm': skim_ipm,
                'time_per_hole': time_per_hole,
                'total_time': total_time,
                'description': entry.get('DESCRIPTION', ''),
                # EDM time model transparency fields
                'edm_path_length_in': estimated_perimeter_per_window,
                'edm_min_per_in_rough': rough_min_per_in,
                'edm_min_per_in_skim': skim_min_per_in,
                'edm_material_factor': edm_mat_factor,
                'part_thickness': edm_thickness,
                't_window_rough': t_window_rough,
                't_window_skim': t_window_skim,
            })

        # SLOT operations (milled slots/obrounds)
        if is_slot:
            # Extract slot geometry from description
            # Pattern: "2X R.094 x 0.697 OVER R" or similar
            slot_desc = combined_text

            # Try to extract radius from pattern like "R.094" or "R0.094"
            # Capture the full decimal number including leading period
            radius_match = re.search(r'\bR(\.?\d*\.?\d+)', slot_desc)
            slot_radius = float(radius_match.group(1)) if radius_match else ref_dia / 2

            # Try to extract length from patterns like "0.697 OVER R" or "X 0.697"
            length_match = re.search(r'(?:X\s*|^)([\d\.]+)\s*(?:OVER\s*R|LONG)', slot_desc)
            if length_match:
                slot_length = float(length_match.group(1))
            else:
                # Try alternate pattern like "0.697 OVER R"
                over_r_match = re.search(r'([\d\.]+)\s*OVER\s*R', slot_desc)
                if over_r_match:
                    slot_length = float(over_r_match.group(1))
                else:
                    # Default to diameter if no length found
                    slot_length = ref_dia

            # Slot depth is plate thickness for THRU or extracted value
            slot_depth = thickness if (is_thru or 'THRU' in slot_desc) and thickness > 0 else 0.5

            # Calculate slot milling time
            # Slot path length = straight section + two semicircles
            # For obround: perimeter = 2 * straight_length + 2 * pi * radius
            straight_section = max(0, slot_length - 2 * slot_radius)
            slot_perimeter = 2 * straight_section + 2 * 3.14159 * slot_radius

            # Get endmill speeds/feeds
            sf_endmill = get_speeds_feeds(material, "Endmill_Profile")
            if sf_endmill:
                sfm = sf_endmill.get('sfm_start', 200)
                fz = sf_endmill.get('fz_ipt', 0.003)
                doc = sf_endmill.get('doc_ax_in', 0.1)
            else:
                sfm = 200
                fz = 0.003
                doc = 0.1

            # Use slot radius * 2 as tool diameter (match end radius)
            tool_dia = slot_radius * 2
            rpm = (sfm * 12) / (3.14159 * tool_dia) if tool_dia > 0 else 1000
            rpm = min(rpm, 5000)

            # IPM = RPM * teeth * feed_per_tooth (assume 4 flute endmill)
            ipm = rpm * 4 * fz

            # Number of passes (depth of cut)
            num_passes = max(1, int(slot_depth / doc) + 1)

            # Time = (perimeter * passes) / feed_rate
            cut_time = (slot_perimeter * num_passes) / ipm if ipm > 0 else 1.0
            time_per_slot = cut_time + 0.2  # Add approach/retract

            time_per_hole = round(time_per_slot, 2)
            total_time = round(time_per_hole * qty, 2)

            slot_groups.append({
                'hole_id': hole_id,
                'radius': slot_radius,
                'length': slot_length,
                'depth': slot_depth,
                'qty': qty,
                'sfm': sfm,
                'rpm': rpm,
                'ipm': ipm,
                'time_per_hole': time_per_hole,
                'total_time': total_time,
                'description': entry.get('DESCRIPTION', '')
            })

    # Calculate totals - round after summing to ensure display consistency
    total_drill = round(sum(g['total_time'] for g in drill_groups), 2)
    total_jig_grind = round(sum(g['total_time'] for g in jig_grind_groups), 2)
    total_tap = round(sum(g['total_time'] for g in tap_groups), 2)
    total_cbore = round(sum(g['total_time'] for g in cbore_groups), 2)
    total_cdrill = round(sum(g['total_time'] for g in cdrill_groups), 2)
    total_edm = round(sum(g['total_time'] for g in edm_groups), 2)
    total_slot = round(sum(g['total_time'] for g in slot_groups), 2)

    total_minutes = round(total_drill + total_jig_grind + total_tap + total_cbore + total_cdrill + total_edm + total_slot, 2)

    # Print EDM debug information for each group
    if edm_groups:
        print("  DEBUG [EDM hole table groups]:")
        for g in edm_groups:
            print(f"    Hole {g['hole_id']} (qty={g['qty']}): "
                  f"path_length={g.get('edm_path_length_in', 0.0):.3f}\", "
                  f"min_per_in_rough={g.get('edm_min_per_in_rough', 0.0):.3f}, "
                  f"min_per_in_skim={g.get('edm_min_per_in_skim', 0.0):.3f}, "
                  f"material_factor={g.get('edm_material_factor', 1.0):.2f}, "
                  f"part_thickness={g.get('part_thickness', 0.0):.3f}\", "
                  f"t_window_rough={g.get('t_window_rough', 0.0):.2f} min, "
                  f"t_window_skim={g.get('t_window_skim', 0.0):.2f} min, "
                  f"total_time={g['total_time']:.2f} min")

    return {
        'drill_groups': drill_groups,
        'jig_grind_groups': jig_grind_groups,
        'tap_groups': tap_groups,
        'cbore_groups': cbore_groups,
        'cdrill_groups': cdrill_groups,
        'edm_groups': edm_groups,
        'slot_groups': slot_groups,
        'total_drill_minutes': total_drill,
        'total_jig_grind_minutes': total_jig_grind,
        'total_tap_minutes': total_tap,
        'total_cbore_minutes': total_cbore,
        'total_cdrill_minutes': total_cdrill,
        'total_edm_minutes': total_edm,
        'total_slot_minutes': total_slot,
        'total_minutes': total_minutes,
        'total_hours': total_minutes / 60,
        'material': material,
        'thickness': thickness,
    }


# ============================================================================
# PUNCH PLANNER - Detailed planning for punch manufacturing
# ============================================================================


@dataclass
class PunchPlannerParams:
    """Parameters for punch planning."""
    family: str = "round_punch"
    shape_type: str = "round"
    overall_length_in: float = 0.0
    max_od_or_width_in: float = 0.0
    body_width_in: Optional[float] = None
    body_thickness_in: Optional[float] = None
    num_ground_diams: int = 0
    total_ground_length_in: float = 0.0
    tap_count: int = 0
    tap_summary: list = None
    num_chamfers: int = 0
    has_perp_face_grind: bool = False
    has_3d_surface: bool = False
    has_polish_contour: bool = False
    has_no_step_permitted: bool = False
    min_dia_tol_in: Optional[float] = None
    min_len_tol_in: Optional[float] = None
    material: str = "A2"
    material_group: str = ""
    # Punch datum heuristics fields
    has_flats: bool = False  # Round punches with machined flats
    has_wire_profile: bool = False  # Requires wire EDM profiling
    wire_profile_perimeter_in: float = 0.0  # Perimeter for wire EDM

    def __post_init__(self):
        if self.tap_summary is None:
            self.tap_summary = []


def _is_carbide(material: str, material_group: str) -> bool:
    """Check if material is carbide (disallow milling for datum faces)."""
    mat_upper = (material or "").upper()
    group_upper = (material_group or "").upper()
    return "CARBIDE" in mat_upper or "CARBIDE" in group_upper


def create_punch_plan(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a detailed manufacturing plan for a punch."""
    p = _extract_punch_params(params)

    plan = {
        "ops": [],
        "fixturing": [],
        "qa": [],
        "warnings": [],
        "directs": {
            "hardware": False,
            "outsourced": False,
            "utilities": False,
            "consumables_flat": False,
            "packaging_flat": True,
        }
    }

    _add_punch_stock_ops(plan, p)
    _add_punch_roughing_ops(plan, p)
    _add_punch_heat_treat_ops(plan, p)
    _add_punch_grinding_ops(plan, p)
    _add_punch_hole_ops(plan, p)
    _add_punch_edge_ops(plan, p)
    _add_punch_form_ops(plan, p)
    _add_punch_qa_checks(plan, p)
    _add_punch_fixturing_notes(plan, p)

    return plan


def _extract_punch_params(params: Dict[str, Any]) -> PunchPlannerParams:
    """Extract and normalize parameters from input dict."""
    return PunchPlannerParams(
        family=params.get("family", "round_punch"),
        shape_type=params.get("shape_type", "round"),
        overall_length_in=float(params.get("overall_length_in", 0.0)),
        max_od_or_width_in=float(params.get("max_od_or_width_in", 0.0)),
        body_width_in=params.get("body_width_in"),
        body_thickness_in=params.get("body_thickness_in"),
        num_ground_diams=int(params.get("num_ground_diams", 0)),
        total_ground_length_in=float(params.get("total_ground_length_in", 0.0)),
        tap_count=int(params.get("tap_count", 0)),
        tap_summary=params.get("tap_summary", []),
        num_chamfers=int(params.get("num_chamfers", 0)),
        has_perp_face_grind=bool(params.get("has_perp_face_grind", False)),
        has_3d_surface=bool(params.get("has_3d_surface", False)),
        has_polish_contour=bool(params.get("has_polish_contour", False)),
        has_no_step_permitted=bool(params.get("has_no_step_permitted", False)),
        min_dia_tol_in=params.get("min_dia_tol_in"),
        min_len_tol_in=params.get("min_len_tol_in"),
        material=params.get("material", "A2"),
        material_group=params.get("material_group", ""),
        has_flats=bool(params.get("has_flats", False)),
        has_wire_profile=bool(params.get("has_wire_profile", False)),
        wire_profile_perimeter_in=float(params.get("wire_profile_perimeter_in", 0.0)),
    )


def _add_punch_stock_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add stock procurement operation."""
    if p.shape_type == "round":
        stock_size = f"{p.max_od_or_width_in:.3f}\" DIA x {p.overall_length_in:.2f}\" L"
    else:
        stock_size = f"{p.body_width_in:.3f}\" x {p.body_thickness_in:.3f}\" x {p.overall_length_in:.2f}\" L"

    plan["ops"].append({
        "op": "stock_procurement",
        "material": p.material,
        "stock_size": stock_size,
        "note": f"Order {p.material} stock: {stock_size}"
    })


def _add_punch_roughing_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add roughing operations."""
    plan["ops"].append({
        "op": "saw_to_length",
        "length_in": p.overall_length_in,
        "note": "Saw to length with HT allowance"
    })

    if p.shape_type == "round":
        if p.num_ground_diams > 0:
            # Check for cylindrical features before allowing turning
            params_dict = {
                "shape_type": p.shape_type,
                "num_ground_diams": p.num_ground_diams,
                "total_ground_length_in": p.total_ground_length_in,
                "max_od_or_width_in": p.max_od_or_width_in,
                "overall_length_in": p.overall_length_in,
            }
            has_cylindrical = _has_cylindrical_feature(params_dict, min_length_threshold=0.25)

            if has_cylindrical:
                plan["ops"].append({
                    "op": "rough_turning",
                    "num_diams": p.num_ground_diams,
                    "note": f"Rough turn {p.num_ground_diams} diameter sections"
                })
            else:
                # No cylindrical features long enough - use milling instead
                warning = (
                    f"Turning operation disabled: No cylindrical sections > 0.25\" found. "
                    f"Using plate-style milling instead."
                )
                print(f"DEBUG: {warning}")
                plan.setdefault("warnings", []).append(warning)
                plan["ops"].append({
                    "op": "rough_milling",
                    "note": "Rough mill (no cylindrical features > 0.25\" for turning)"
                })
    else:
        plan["ops"].append({
            "op": "rough_milling",
            "note": "Rough mill rectangular sections"
        })


def _add_punch_heat_treat_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add heat treatment operation."""
    hardness_map = {
        "A2": "60-62 RC", "D2": "58-60 RC", "M2": "62-64 RC",
        "O1": "60-62 RC", "S7": "54-56 RC", "CARBIDE": "N/A (as-sintered)",
    }
    target_hardness = hardness_map.get(p.material, "58-62 RC")

    if p.material != "CARBIDE":
        plan["ops"].append({
            "op": "heat_treat",
            "material": p.material,
            "target_hardness": target_hardness,
            "note": f"Heat treat to {target_hardness}"
        })
        plan["directs"]["outsourced"] = True


def _add_punch_wire_edm_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add wire EDM profile operations for punches with flats or form profiles."""
    if not p.has_wire_profile and not p.has_flats:
        return

    # Calculate perimeter if not provided
    perimeter = p.wire_profile_perimeter_in
    if perimeter <= 0:
        if p.shape_type == "round":
            # Round with flats: approximate perimeter
            perimeter = 3.14159 * p.max_od_or_width_in + (p.max_od_or_width_in * 0.5)
        else:
            # Rectangular: 2(L + W)
            w = p.body_width_in or p.max_od_or_width_in
            t = p.body_thickness_in or w
            perimeter = 2 * (p.overall_length_in + max(w, t))

    plan["ops"].append({
        "op": "Wire_EDM_profile",
        "wire_profile_perimeter_in": perimeter,
        "thickness_in": p.body_thickness_in or p.max_od_or_width_in,
        "note": f"Wire EDM profile cut, perimeter {perimeter:.2f}\""
    })


def _add_punch_grinding_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add grinding operations with punch datum heuristics.

    Implements:
    - Round punches: OD + end faces only (no side square-up)
    - Non-round/form punches: Grind_reference_faces before profiling
    - Carbide punches: only grind + WEDM (no milling)
    - Very small parts: prefer Grind_reference_faces over mill
    """
    is_carbide = _is_carbide(p.material, p.material_group)

    if p.shape_type == "round":
        # ROUND PUNCHES: OD grind + face grind (Grind_length)
        # If has_flats, Wire_EDM_profile comes BEFORE OD grind

        if p.has_flats:
            _add_punch_wire_edm_ops(plan, p)

        if p.num_ground_diams > 0:
            # Check for cylindrical features before allowing OD grinding (turning-style operation)
            params_dict = {
                "shape_type": p.shape_type,
                "num_ground_diams": p.num_ground_diams,
                "total_ground_length_in": p.total_ground_length_in,
                "max_od_or_width_in": p.max_od_or_width_in,
                "overall_length_in": p.overall_length_in,
            }
            has_cylindrical = _has_cylindrical_feature(params_dict, min_length_threshold=0.25)

            if has_cylindrical:
                tol_factor = 1.0
                if p.min_dia_tol_in and p.min_dia_tol_in < 0.0002:
                    tol_factor = 1.5
                if p.has_no_step_permitted:
                    tol_factor *= 1.3

                # Add rough and finish OD grind
                plan["ops"].append({
                    "op": "OD_grind_rough",
                    "num_diams": p.num_ground_diams,
                    "total_length_in": p.total_ground_length_in,
                    "tol_factor": tol_factor,
                    "note": f"Rough grind {p.num_ground_diams} diameters"
                })
                plan["ops"].append({
                    "op": "OD_grind_finish",
                    "num_diams": p.num_ground_diams,
                    "total_length_in": p.total_ground_length_in,
                    "tol_factor": tol_factor,
                    "note": f"Finish grind {p.num_ground_diams} diameters, {p.total_ground_length_in:.2f}\" total"
                })
            else:
                # No cylindrical features - use plate-style grinding instead
                warning = (
                    f"OD grinding disabled: No cylindrical sections > 0.25\" found. "
                    f"Using plate-style surface grinding instead."
                )
                print(f"DEBUG: {warning}")
                plan.setdefault("warnings", []).append(warning)
                plan["ops"].append({
                    "op": "surface_grind_faces",
                    "stock_removed_total": 0.010,
                    "faces": 4,  # All sides
                    "note": "Surface grind all faces (no cylindrical features > 0.25\" for OD grind)"
                })

        # Grind_length: faces both ends
        plan["ops"].append({
            "op": "Grind_length",
            "diameter": p.max_od_or_width_in,
            "length_in": p.overall_length_in,
            "stock_removed_total": 0.006,
            "faces": 2,
            "note": "Grind both end faces to length"
        })

        # Do NOT emit square_up_* for round punches

    else:
        # NON-ROUND / FORM PUNCHES
        # Determine dimensions for small-part check
        w = p.body_width_in or p.max_od_or_width_in
        t = p.body_thickness_in or w
        min_dim = min(w, t) if w and t else 1.0

        # Insert Grind_reference_faces BEFORE Wire_EDM_profile
        # For carbide or very small parts, always use grind (no mill)
        use_grind_datum = is_carbide or min_dim < 1.0

        if use_grind_datum or p.has_wire_profile:
            # Establish datum faces by grinding
            plan["ops"].append({
                "op": "Grind_reference_faces",
                "stock_removed_total": 0.006,
                "faces": 2,
                "length_in": p.overall_length_in,
                "width_in": w,
                "note": "Establish datums before profile"
            })

        # Wire EDM profile after datum establishment
        if p.has_wire_profile:
            _add_punch_wire_edm_ops(plan, p)

        # Final surface grind (Grind_length for non-round)
        plan["ops"].append({
            "op": "Grind_length",
            "width_in": w,
            "thickness_in": t,
            "length_in": p.overall_length_in,
            "stock_removed_total": 0.004,
            "faces": 2,
            "note": "Grind to final length"
        })

        # Optional final kiss grind on faces
        if p.has_perp_face_grind:
            plan["ops"].append({
                "op": "Grind_faces",
                "width_in": w,
                "thickness_in": t,
                "stock_removed_total": 0.004,
                "faces": 2,
                "note": "Final face grind (kiss pass)"
            })


def _add_punch_hole_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add drilling and tapping operations."""
    if p.tap_count > 0:
        for tap in p.tap_summary:
            size = tap.get("size", "Unknown")
            depth = tap.get("depth_in")
            depth_str = f"{depth:.2f}\"" if depth else "full depth"

            plan["ops"].append({
                "op": "tap",
                "size": size,
                "depth_in": depth,
                "note": f"Tap {size} x {depth_str}"
            })


def _add_punch_edge_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add chamfering and deburring operations."""
    if p.num_chamfers > 0:
        plan["ops"].append({
            "op": "chamfer",
            "count": p.num_chamfers,
            "note": f"Chamfer {p.num_chamfers} edges"
        })


def _add_punch_form_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add form/polish operations for contoured punches."""
    if p.has_3d_surface:
        plan["ops"].append({
            "op": "3d_mill_form",
            "note": "3D mill contoured nose section"
        })

    if p.has_polish_contour:
        plan["ops"].append({
            "op": "polish_contour",
            "note": "Polish contoured surface to spec"
        })


def _add_punch_qa_checks(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add quality assurance checks."""
    qa = plan["qa"]
    qa.append(f"Verify OAL: {p.overall_length_in:.3f}\"")

    if p.shape_type == "round":
        qa.append(f"Verify max OD: {p.max_od_or_width_in:.4f}\"")
    else:
        qa.append(f"Verify width: {p.body_width_in:.3f}\" x thickness: {p.body_thickness_in:.3f}\"")

    if p.min_dia_tol_in and p.min_dia_tol_in < 0.0005:
        qa.append(f"Critical diameter tolerance: ±{p.min_dia_tol_in:.4f}\"")

    if p.min_len_tol_in and p.min_len_tol_in < 0.005:
        qa.append(f"Critical length tolerance: ±{p.min_len_tol_in:.4f}\"")

    if p.has_perp_face_grind:
        qa.append("Verify face perpendicularity to centerline")

    if p.has_polish_contour:
        qa.append("Verify contour surface finish (8 µin or per spec)")

    qa.append(f"Verify hardness: {p.material} to spec")


def _add_punch_fixturing_notes(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add fixturing recommendations."""
    fix = plan["fixturing"]

    if p.shape_type == "round":
        fix.append("Use collet or 3-jaw chuck for turning")
        fix.append("Use centers for long/slender sections")

        if p.max_od_or_width_in > 0 and p.overall_length_in / p.max_od_or_width_in > 10:
            fix.append("Use steady rest for slender punch (L/D > 10)")
    else:
        fix.append("Use vise or fixture plate for milling")
        fix.append("Parallel supports for grinding")

    if p.has_perp_face_grind:
        fix.append("Use precision angle plate for perpendicular grinding")


def planner_punches_enhanced(params: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced punch planner for process_planner.py integration."""
    if not params.get("overall_length_in") and params.get("dxf_path"):
        try:
            from cad_quoter.geometry.dxf_enrich import extract_punch_features
            summary = extract_punch_features(params["dxf_path"])

            summary_dict = {
                "family": summary.family,
                "shape_type": summary.shape_type,
                "overall_length_in": summary.overall_length_in,
                "max_od_or_width_in": summary.max_od_or_width_in,
                "body_width_in": summary.body_width_in,
                "body_thickness_in": summary.body_thickness_in,
                "num_ground_diams": summary.num_ground_diams,
                "total_ground_length_in": summary.total_ground_length_in,
                "tap_count": summary.tap_count,
                "tap_summary": summary.tap_summary,
                "num_chamfers": summary.num_chamfers,
                "has_perp_face_grind": summary.has_perp_face_grind,
                "has_3d_surface": summary.has_3d_surface,
                "has_polish_contour": summary.has_polish_contour,
                "has_no_step_permitted": summary.has_no_step_permitted,
                "min_dia_tol_in": summary.min_dia_tol_in,
                "min_len_tol_in": summary.min_len_tol_in,
                "material": summary.material_callout or params.get("material", "A2"),
            }
            params.update(summary_dict)

            if summary.warnings:
                if "warnings" not in params:
                    params["warnings"] = []
                params["warnings"].extend(summary.warnings)

        except Exception as e:
            if "warnings" not in params:
                params["warnings"] = []
            params["warnings"].append(f"DXF extraction failed: {str(e)}")

    return create_punch_plan(params)