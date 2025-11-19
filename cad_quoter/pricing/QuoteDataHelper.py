"""
QuoteDataHelper - Unified data structure for all CAD extraction results.

This module provides a centralized data structure that holds all extracted information
from DirectCostHelper and ProcessPlanner, making it easy to cache, serialize, and
pass around quote data throughout the application.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

# Punch extraction imports
try:
    from cad_quoter.geometry.dxf_enrich import (
        extract_punch_features_from_dxf,
        PunchFeatureSummary,
    )
    from cad_quoter.planning.process_planner import create_punch_plan
    from cad_quoter.pricing.time_estimator import estimate_punch_times
    PUNCH_EXTRACTION_AVAILABLE = True
except ImportError:
    PUNCH_EXTRACTION_AVAILABLE = False


@dataclass
class PartDimensions:
    """Part dimensions extracted from CAD file."""
    length: float = 0.0  # inches
    width: float = 0.0   # inches
    thickness: float = 0.0  # inches
    volume: float = 0.0  # cubic inches
    area: float = 0.0    # square inches (L × W)

    def __post_init__(self):
        """Calculate derived properties if not set."""
        if self.volume == 0.0:
            self.volume = self.length * self.width * self.thickness
        if self.area == 0.0:
            self.area = self.length * self.width


@dataclass
class MaterialInfo:
    """Material information and properties."""
    material_name: str = "GENERIC"
    material_family: str = "aluminum"  # aluminum, steel, stainless, etc.
    density: float = 0.098  # lb/in³
    detected_from_cad: bool = False
    is_default: bool = False


@dataclass
class StockInfo:
    """McMaster stock information."""
    # Desired starting stock (part + allowances)
    desired_length: float = 0.0
    desired_width: float = 0.0
    desired_thickness: float = 0.0
    desired_volume: float = 0.0

    # McMaster catalog stock
    mcmaster_length: float = 0.0
    mcmaster_width: float = 0.0
    mcmaster_thickness: float = 0.0
    mcmaster_volume: float = 0.0
    mcmaster_part_number: Optional[str] = None
    mcmaster_price: Optional[float] = None  # Unit price
    price_is_estimated: bool = False  # True if price was estimated from volume

    # Weights
    mcmaster_weight: float = 0.0  # lbs
    final_part_weight: float = 0.0  # lbs


@dataclass
class ScrapInfo:
    """Scrap calculation results."""
    # Scrap breakdown by source
    stock_prep_scrap: float = 0.0  # McMaster → Desired (in³)
    face_milling_scrap: float = 0.0  # Desired → Part envelope (in³)
    hole_drilling_scrap: float = 0.0  # Holes removed from part (in³)
    total_scrap_volume: float = 0.0  # Total material removed (in³)
    total_scrap_weight: float = 0.0  # lbs

    # Scrap percentages
    scrap_percentage: float = 0.0  # Scrap as % of McMaster stock
    utilization_percentage: float = 0.0  # Final part as % of McMaster stock

    # Scrap value
    scrap_price_per_lb: Optional[float] = None  # $/lb
    scrap_value: float = 0.0  # Total scrap value ($)
    scrap_price_source: str = ""  # e.g., "Wieland aluminum scrap"


@dataclass
class DirectCostBreakdown:
    """Direct cost breakdown."""
    stock_cost: float = 0.0  # McMaster stock price
    tax: float = 0.0  # Tax on stock
    shipping: float = 0.0  # Shipping on stock
    scrap_credit: float = 0.0  # Credit for scrap value (negative cost)
    net_material_cost: float = 0.0  # Total direct cost


@dataclass
class HoleOperation:
    """Single hole operation details."""
    hole_id: str = ""
    diameter: float = 0.0  # inches
    depth: float = 0.0  # inches
    qty: int = 1
    operation_type: str = ""  # drill, tap, cbore, cdrill, jig_grind
    description: str = ""

    # Time estimates
    time_per_hole: float = 0.0  # minutes
    total_time: float = 0.0  # minutes (qty × time_per_hole)

    # Operation-specific parameters
    sfm: Optional[float] = None  # Surface feet per minute
    ipr: Optional[float] = None  # Inches per revolution
    tpi: Optional[int] = None  # Threads per inch (for taps)
    side: Optional[str] = None  # For C'BORE: "front", "back", or None


@dataclass
class MillingOperation:
    """Detailed milling operation breakdown."""
    op_name: str = ""  # e.g., "square_up_rough_faces", "square_up_rough_sides"
    op_description: str = ""  # Human-readable description

    # Geometry
    length: float = 0.0  # inches
    width: float = 0.0  # inches
    perimeter: float = 0.0  # inches (for side ops)

    # Tool parameters
    tool_diameter: float = 0.0  # inches

    # Cut parameters
    passes: int = 1
    stepover: Optional[float] = None  # inches
    radial_stock: Optional[float] = None  # inches
    axial_step: Optional[float] = None  # inches
    axial_passes: Optional[int] = None
    radial_passes: Optional[int] = None

    # Path and feed
    path_length: float = 0.0  # total inches
    feed_rate: float = 0.0  # IPM

    # Time
    time_minutes: float = 0.0

    # Override tracking
    _used_override: bool = False
    override_time_minutes: Optional[float] = None


@dataclass
class GrindingOperation:
    """Detailed grinding operation breakdown."""
    op_name: str = ""  # e.g., "wet_grind_square_all"
    op_description: str = ""  # Human-readable description

    # Geometry
    length: float = 0.0  # inches
    width: float = 0.0  # inches
    area: float = 0.0  # square inches

    # Grind parameters
    stock_removed_total: float = 0.0  # inches
    faces: int = 2  # number of faces
    volume_removed: float = 0.0  # cubic inches

    # Time calculation
    min_per_cuin: float = 3.0
    material_factor: float = 1.0
    grind_material_factor: float = 1.0  # Alias for renderer display
    time_minutes: float = 0.0

    # Override tracking
    _used_override: bool = False


@dataclass
class MachineHoursBreakdown:
    """Machine hours estimation breakdown."""
    # Operations by type (hole operations)
    drill_operations: Optional[List[HoleOperation]] = None
    tap_operations: Optional[List[HoleOperation]] = None
    cbore_operations: Optional[List[HoleOperation]] = None
    cdrill_operations: Optional[List[HoleOperation]] = None
    jig_grind_operations: Optional[List[HoleOperation]] = None

    # Operations by type (plan operations)
    milling_operations: Optional[List[MillingOperation]] = None
    grinding_operations: Optional[List[GrindingOperation]] = None

    # Time totals by operation type (from hole table)
    total_drill_minutes: float = 0.0
    total_tap_minutes: float = 0.0
    total_cbore_minutes: float = 0.0
    total_cdrill_minutes: float = 0.0
    total_jig_grind_minutes: float = 0.0

    # Time totals by operation category (from plan operations)
    total_milling_minutes: float = 0.0  # Includes squaring ops
    total_grinding_minutes: float = 0.0  # Includes wet grind squaring
    total_edm_minutes: float = 0.0
    total_other_minutes: float = 0.0
    total_cmm_minutes: float = 0.0  # CMM checking time (machine only, setup is in labor)
    cmm_holes_checked: int = 0  # Number of holes inspected by CMM

    # Overall totals
    total_minutes: float = 0.0
    total_hours: float = 0.0
    machine_cost: float = 0.0  # Total cost at machine rate

    def __post_init__(self):
        """Initialize empty lists."""
        if self.drill_operations is None:
            self.drill_operations = []
        if self.tap_operations is None:
            self.tap_operations = []
        if self.cbore_operations is None:
            self.cbore_operations = []
        if self.cdrill_operations is None:
            self.cdrill_operations = []
        if self.jig_grind_operations is None:
            self.jig_grind_operations = []
        if self.milling_operations is None:
            self.milling_operations = []
        if self.grinding_operations is None:
            self.grinding_operations = []


@dataclass
class LaborHoursBreakdown:
    """Labor hours estimation breakdown."""
    # Labor by category (minutes)
    setup_minutes: float = 0.0
    programming_minutes: float = 0.0
    machining_steps_minutes: float = 0.0
    inspection_minutes: float = 0.0
    finishing_minutes: float = 0.0

    # Totals
    total_minutes: float = 0.0
    total_hours: float = 0.0
    labor_cost: float = 0.0  # Total cost at labor rate

    # Input parameters (for reference)
    ops_total: int = 0
    holes_total: int = 0
    tool_changes: int = 0
    fixturing_complexity: int = 1  # 0=none, 1=light, 2=moderate, 3=complex


@dataclass
class CostSummary:
    """Overall cost summary."""
    # Per-unit costs
    direct_cost: float = 0.0
    machine_cost: float = 0.0
    labor_cost: float = 0.0
    total_cost: float = 0.0

    # Total costs (for quantity > 1)
    total_direct_cost: float = 0.0
    total_machine_cost: float = 0.0
    total_labor_cost: float = 0.0
    total_total_cost: float = 0.0

    # Margin and final price
    margin_rate: float = 0.15
    margin_amount: float = 0.0
    final_price: float = 0.0  # Per-unit price
    total_final_price: float = 0.0  # Total price for all units


@dataclass
class QuoteData:
    """
    Complete quote data structure holding all extraction results.

    This is the main data structure that aggregates all information from
    DirectCostHelper and ProcessPlanner.
    """
    # Metadata
    cad_file_path: str = ""
    cad_file_name: str = ""
    extraction_timestamp: str = ""
    quantity: int = 1  # Number of parts to quote

    # Core data
    part_dimensions: PartDimensions = None
    material_info: MaterialInfo = None
    stock_info: StockInfo = None
    scrap_info: ScrapInfo = None
    direct_cost_breakdown: DirectCostBreakdown = None
    machine_hours: MachineHoursBreakdown = None
    labor_hours: LaborHoursBreakdown = None
    cost_summary: CostSummary = None

    # Raw plan data (optional, for reference)
    raw_plan: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize nested dataclasses."""
        if self.part_dimensions is None:
            self.part_dimensions = PartDimensions()
        if self.material_info is None:
            self.material_info = MaterialInfo()
        if self.stock_info is None:
            self.stock_info = StockInfo()
        if self.scrap_info is None:
            self.scrap_info = ScrapInfo()
        if self.direct_cost_breakdown is None:
            self.direct_cost_breakdown = DirectCostBreakdown()
        if self.machine_hours is None:
            self.machine_hours = MachineHoursBreakdown()
        if self.labor_hours is None:
            self.labor_hours = LaborHoursBreakdown()
        if self.cost_summary is None:
            self.cost_summary = CostSummary()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        return asdict(self)

    def to_json(self, filepath: Optional[str | Path] = None, indent: int = 2) -> str:
        """
        Convert to JSON string.

        Args:
            filepath: Optional path to save JSON file
            indent: JSON indentation level

        Returns:
            JSON string
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)

        if filepath:
            Path(filepath).write_text(json_str)

        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuoteData:
        """Create QuoteData from dictionary."""
        # Convert nested dicts to dataclasses
        if 'part_dimensions' in data and isinstance(data['part_dimensions'], dict):
            data['part_dimensions'] = PartDimensions(**data['part_dimensions'])
        if 'material_info' in data and isinstance(data['material_info'], dict):
            data['material_info'] = MaterialInfo(**data['material_info'])
        if 'stock_info' in data and isinstance(data['stock_info'], dict):
            data['stock_info'] = StockInfo(**data['stock_info'])
        if 'scrap_info' in data and isinstance(data['scrap_info'], dict):
            data['scrap_info'] = ScrapInfo(**data['scrap_info'])
        if 'direct_cost_breakdown' in data and isinstance(data['direct_cost_breakdown'], dict):
            data['direct_cost_breakdown'] = DirectCostBreakdown(**data['direct_cost_breakdown'])
        if 'machine_hours' in data and isinstance(data['machine_hours'], dict):
            # Convert hole operations lists
            machine_data = data['machine_hours']
            for key in ['drill_operations', 'tap_operations', 'cbore_operations',
                       'cdrill_operations', 'jig_grind_operations']:
                if key in machine_data and isinstance(machine_data[key], list):
                    machine_data[key] = [HoleOperation(**op) if isinstance(op, dict) else op
                                        for op in machine_data[key]]
            data['machine_hours'] = MachineHoursBreakdown(**machine_data)
        if 'labor_hours' in data and isinstance(data['labor_hours'], dict):
            data['labor_hours'] = LaborHoursBreakdown(**data['labor_hours'])
        if 'cost_summary' in data and isinstance(data['cost_summary'], dict):
            data['cost_summary'] = CostSummary(**data['cost_summary'])

        return cls(**data)

    @classmethod
    def from_json(cls, json_str_or_path: str | Path) -> QuoteData:
        """
        Load QuoteData from JSON string or file.

        Args:
            json_str_or_path: JSON string or path to JSON file

        Returns:
            QuoteData instance
        """
        # Check if it's a file path
        try:
            path = Path(json_str_or_path)
            if path.exists():
                json_str = path.read_text()
            else:
                json_str = json_str_or_path
        except:
            json_str = json_str_or_path

        data = json.loads(json_str)
        return cls.from_dict(data)


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================


def detect_punch_drawing(cad_file_path: Path, text_dump: str = None) -> bool:
    """
    Detect if a CAD file is a punch drawing (individual punch component).

    Detection is based on:
    1. Filename containing 'punch', 'pilot', 'pin', etc.
    2. Text content containing PUNCH, PILOT PIN, etc.

    Excludes die shoes, holders, and other tooling that references punches.

    Args:
        cad_file_path: Path to CAD file
        text_dump: Optional text dump from drawing (if already extracted)

    Returns:
        True if file appears to be a punch drawing
    """
    # Check filename
    filename = cad_file_path.stem.upper()

    # Exclusion patterns - these are NOT punches even if they reference punches
    exclusion_patterns = ["SHOE", "HOLDER", "BASE", "PLATE", "DIE SET", "BLOCK", "INSERT"]
    if any(excl in filename for excl in exclusion_patterns):
        return False

    # Punch indicators in filename
    filename_indicators = ["PUNCH", "PILOT", "PIN", "FORM", "GUIDE POST"]
    if any(ind in filename for ind in filename_indicators):
        return True

    # Check text content if provided
    if text_dump:
        text_upper = text_dump.upper()

        # Check for exclusions first - die shoes, holders, etc.
        # These parts reference punches but are not themselves punches
        exclusion_indicators = [
            "DIE SHOE",
            "PUNCH SHOE",
            "PUNCH HOLDER",
            "DIE HOLDER",
            "DIE SET",
            "DIE BASE",
            "PUNCH PLATE",
            "DIE PLATE",
            "BACKING PLATE",
            "STRIPPER PLATE",
            "STRIPPER INSERT",
            "PUNCH BLOCK",
        ]
        if any(excl in text_upper for excl in exclusion_indicators):
            return False

        # Punch indicators in text
        text_indicators = [
            "FORM PUNCH",
            "DIE PUNCH",
            "PIERCING PUNCH",
            "PILOT PIN",
            "SPRING PIN",
            "PUNCH TIP",
            "GUIDE POST",
        ]
        # Only trigger on specific punch phrases, not just "PUNCH" alone
        # (since die shoes often reference punches in their title blocks)
        if any(ind in text_upper for ind in text_indicators):
            return True

        # Check for standalone "PUNCH" - but only if no exclusion words are nearby
        # This catches drawings titled just "PUNCH" without SHOE/HOLDER/etc.
        if "PUNCH" in text_upper:
            # Make sure PUNCH isn't followed by exclusion words
            # e.g., "PUNCH CLEARANCE" or "PUNCH LOCATION" in notes
            punch_exclusion_suffixes = ["SHOE", "HOLDER", "PLATE", "POCKET", "CLEARANCE", "LOCATION", "BLOCK"]
            # Find all occurrences of PUNCH
            import re
            punch_pattern = r'\bPUNCH\b'
            for match in re.finditer(punch_pattern, text_upper):
                # Get text after this PUNCH occurrence
                after_punch = text_upper[match.end():match.end()+15]  # Check next 15 chars
                # If PUNCH is not followed by an exclusion suffix, it's likely a punch drawing
                if not any(suffix in after_punch for suffix in punch_exclusion_suffixes):
                    return True

    return False


def extract_punch_quote_data(
    cad_file_path: Path,
    quote_data: 'QuoteData',
    verbose: bool = False,
    plan: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract punch-specific data and estimate times.

    Args:
        cad_file_path: Path to DXF/DWG file
        quote_data: QuoteData object to populate
        verbose: Print progress messages
        plan: Optional plan dict with cached text_dump and cached_dxf_path

    Returns:
        Dict with punch features, plan, and time estimates
    """
    if not PUNCH_EXTRACTION_AVAILABLE:
        return {"error": "Punch extraction modules not available"}

    try:
        # Use cached text from plan if available (avoids ODA conversion)
        if plan and plan.get('text_dump'):
            text_dump = plan['text_dump']
            text_lines = text_dump.split('\n')
            if verbose:
                print(f"  Using cached text ({len(text_lines)} lines) from plan")
        else:
            # Extract text from DXF
            from cad_quoter.geo_extractor import open_doc, collect_all_text

            doc = open_doc(cad_file_path)
            text_records = list(collect_all_text(doc))
            text_lines = [rec["text"] for rec in text_records if rec.get("text")]
            text_dump = "\n".join(text_lines)

            if verbose:
                print(f"  Extracted {len(text_lines)} text lines from drawing")

        # Extract punch features - use cached DXF path if available
        dxf_path_for_punch = cad_file_path
        if plan and plan.get('cached_dxf_path'):
            dxf_path_for_punch = Path(plan['cached_dxf_path'])
            if verbose:
                print(f"  Using cached DXF for punch extraction: {dxf_path_for_punch.name}")
        punch_features = extract_punch_features_from_dxf(dxf_path_for_punch, text_dump)

        if verbose:
            print(f"  Punch features extracted:")
            print(f"    - Family: {punch_features.family}")
            print(f"    - Shape: {punch_features.shape_type}")
            print(f"    - Length: {punch_features.overall_length_in:.3f}\"")
            print(f"    - Max OD: {punch_features.max_od_or_width_in:.3f}\"")
            print(f"    - Ground diameters: {punch_features.num_ground_diams}")
            print(f"    - Confidence: {punch_features.confidence_score:.2f}")

        # Convert PunchFeatureSummary to dict
        from dataclasses import asdict
        features_dict = asdict(punch_features)

        # Create punch manufacturing plan
        punch_plan = create_punch_plan(features_dict)

        if verbose:
            print(f"  Punch plan created with {len(punch_plan.get('ops', []))} operations")

        # Estimate times
        time_estimates = estimate_punch_times(punch_plan, features_dict)

        if verbose:
            mh = time_estimates.get("machine_hours", {})
            lh = time_estimates.get("labor_hours", {})
            print(f"  Time estimates:")
            print(f"    - Machine hours: {mh.get('total_hours', 0):.2f}")
            print(f"    - Labor hours: {lh.get('total_hours', 0):.2f}")

        return {
            "is_punch": True,
            "punch_features": features_dict,
            "punch_plan": punch_plan,
            "time_estimates": time_estimates,
            "text_dump": text_dump,
        }

    except Exception as e:
        if verbose:
            import traceback
            print(f"  [PUNCH ERROR] Failed to extract punch features: {str(e)}")
            print(f"  {traceback.format_exc()}")
        return {
            "is_punch": True,
            "error": str(e),
        }


def extract_quote_data_from_cad(
    cad_file_path: str | Path,
    machine_rate: float = 45.0,
    labor_rate: float = 45.0,
    margin_rate: float = 0.15,
    material_override: Optional[str] = None,
    catalog_csv_path: Optional[str] = None,
    dimension_override: Optional[tuple[float, float, float]] = None,
    mcmaster_price_override: Optional[float] = None,
    scrap_value_override: Optional[float] = None,
    quantity: int = 1,
    family_override: Optional[str] = None,
    verbose: bool = False
) -> QuoteData:
    """
    Extract complete quote data from a CAD file.

    This is the main extraction function that pulls all data from DirectCostHelper
    and ProcessPlanner and packages it into a unified QuoteData structure.

    Args:
        cad_file_path: Path to CAD file (DXF/DWG)
        machine_rate: Machine hourly rate ($/hr)
        labor_rate: Labor hourly rate ($/hr)
        margin_rate: Profit margin rate (decimal, e.g., 0.15 = 15%)
        material_override: Optional material name override
        catalog_csv_path: Optional path to McMaster catalog CSV
        dimension_override: Optional (length, width, thickness) tuple to override OCR extraction
        mcmaster_price_override: Optional manual stock price - skips McMaster API lookup
        scrap_value_override: Optional manual scrap value - skips automatic scrap value calculation
        quantity: Number of parts to quote (affects setup cost amortization and material pricing)
        family_override: Optional part family override (e.g., "Punches" to force punch pipeline)
        verbose: Print extraction progress

    Returns:
        QuoteData with all extraction results

    Example:
        >>> quote_data = extract_quote_data_from_cad("part.dxf", verbose=True)
        >>> print(f"Total cost: ${quote_data.cost_summary.total_cost:.2f}")

        >>> # With dimension override (when OCR fails)
        >>> quote_data = extract_quote_data_from_cad(
        ...     "part.dxf",
        ...     dimension_override=(10.0, 8.0, 0.5)
        ... )
        >>> quote_data.to_json("quote_results.json")
    """
    from cad_quoter.planning import (
        plan_from_cad_file,
        extract_hole_operations_from_cad,
        estimate_hole_table_times
    )
    from cad_quoter.planning.process_planner import (
        LaborInputs,
        compute_labor_minutes,
        estimate_machine_hours_from_plan
    )
    from cad_quoter.pricing.DirectCostHelper import (
        extract_part_info_from_plan,
        get_mcmaster_part_number,
        get_mcmaster_price,
        calculate_total_scrap,
        calculate_scrap_value,
        get_material_density,
        DEFAULT_MATERIAL
    )
    from cad_quoter.pricing.KeywordDetector import detect_material_in_cad
    from cad_quoter.pricing.MaterialMapper import material_mapper
    from cad_quoter.pricing.mcmaster_helpers import (
        pick_mcmaster_plate_sku,
        load_mcmaster_catalog_rows
    )
    from cad_quoter.resources import default_catalog_csv

    cad_file_path = Path(cad_file_path)

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXTRACTING QUOTE DATA: {cad_file_path.name}")
        print(f"{'='*70}\n")

    # Initialize QuoteData
    quote_data = QuoteData(
        cad_file_path=str(cad_file_path),
        cad_file_name=cad_file_path.name,
        extraction_timestamp=datetime.now().isoformat(),
        quantity=quantity
    )

    # ========================================================================
    # STEP 1: Extract process plan (ODA + OCR)
    # ========================================================================
    if verbose:
        print("[1/5] Extracting process plan (ODA + OCR)...")

    # Skip expensive OCR if manual dimensions provided (saves ~43 seconds)
    use_ocr = dimension_override is None
    if not use_ocr and verbose:
        print("  Skipping OCR dimension extraction (manual dimensions provided)")

    plan = plan_from_cad_file(cad_file_path, use_paddle_ocr=use_ocr, verbose=False)
    quote_data.raw_plan = plan if verbose else None  # Only store if verbose

    # ========================================================================
    # CHECK: Is this a punch drawing?
    # ========================================================================
    is_punch = False
    punch_data = None

    if PUNCH_EXTRACTION_AVAILABLE:
        # First check for family override - this takes priority
        if family_override and family_override.lower() in ("punches", "punch"):
            is_punch = True
            if verbose:
                print(f"[PUNCH] Using family override: {family_override}")
        else:
            # Auto-detect: First check by filename
            is_punch = detect_punch_drawing(cad_file_path)

            # If not detected by filename, check text content
            if not is_punch and plan.get('text_dump'):
                is_punch = detect_punch_drawing(cad_file_path, plan.get('text_dump'))

        if is_punch:
            if verbose:
                print("[PUNCH] Detected punch drawing - using punch extraction pipeline")

            # Extract punch-specific data
            punch_data = extract_punch_quote_data(cad_file_path, quote_data, verbose, plan)

            if "error" not in punch_data:
                # Use punch features for part dimensions
                features = punch_data.get("punch_features", {})

                # Set part dimensions from punch features
                quote_data.part_dimensions = PartDimensions(
                    length=features.get("overall_length_in", 0.0),
                    width=features.get("max_od_or_width_in", 0.0),
                    thickness=features.get("max_od_or_width_in", 0.0),  # OD for round
                )

                # If punch dimensions are zero, fall back to DimensionFinder
                if (quote_data.part_dimensions.length == 0.0 and
                    quote_data.part_dimensions.width == 0.0):
                    if verbose:
                        print("  [PUNCH] Punch dimensions are zero, falling back to DimensionFinder...")

                    # Use cached dimensions from plan if available (avoids ODA conversion)
                    if 'extracted_dims' in plan and plan['extracted_dims']:
                        dims_data = plan['extracted_dims']
                        L = dims_data.get('L', 0.0)
                        W = dims_data.get('W', 0.0)
                        T = dims_data.get('T', 0.0)
                        if L > 0 and W > 0:
                            quote_data.part_dimensions = PartDimensions(
                                length=L,
                                width=W,
                                thickness=T,
                            )
                            if verbose:
                                print(f"  [PUNCH] Using cached dimensions: {L:.3f} x {W:.3f} x {T:.3f}")
                        else:
                            if verbose:
                                print("  [PUNCH] Cached dimensions are zero, DimensionFinder also failed")
                    else:
                        # No cached dims, fall back to extraction (will trigger ODA conversion)
                        from cad_quoter.planning.process_planner import extract_dimensions_from_cad
                        dims = extract_dimensions_from_cad(cad_file_path)
                        if dims:
                            L, W, T = dims
                            quote_data.part_dimensions = PartDimensions(
                                length=L,
                                width=W,
                                thickness=T,
                            )
                            if verbose:
                                print(f"  [PUNCH] DimensionFinder found: {L:.3f} x {W:.3f} x {T:.3f}")
                        else:
                            if verbose:
                                print("  [PUNCH] DimensionFinder also failed to extract dimensions")

                # Apply dimension override for punch parts
                if dimension_override:
                    dim1, dim2, dim3 = dimension_override
                    if verbose:
                        print(f"  [PUNCH] Dimension override input: {dim1} x {dim2} x {dim3}")

                    # For punch parts, auto-reorder dimensions:
                    # Users often enter dimensions as smallest×middle×largest (e.g., .148x.445x2)
                    # but punch features need: length=OAL (largest), width=OD, thickness
                    # Sort to get: thickness (smallest), width (middle), length (largest)
                    sorted_dims = sorted([dim1, dim2, dim3])
                    thickness = sorted_dims[0]  # smallest
                    width = sorted_dims[1]      # middle
                    length = sorted_dims[2]     # largest (OAL)

                    if verbose:
                        print(f"  [PUNCH] Reordered to: length={length}, width={width}, thickness={thickness}")

                    quote_data.part_dimensions = PartDimensions(
                        length=length,
                        width=width,
                        thickness=thickness,
                    )

                    # Also update punch_features dict so punch plan uses correct values
                    features["overall_length_in"] = length
                    features["max_od_or_width_in"] = width
                    if width != thickness:
                        # Rectangular punch - set body dimensions
                        features["body_width_in"] = width
                        features["body_thickness_in"] = thickness

                    # Regenerate punch plan with new dimensions
                    from dataclasses import asdict
                    from cad_quoter.planning.process_planner import create_punch_plan

                    updated_features_dict = features  # Already a dict
                    punch_plan = create_punch_plan(updated_features_dict)
                    punch_data["punch_plan"] = punch_plan
                    punch_data["punch_features"] = features

                    if verbose:
                        print(f"  [PUNCH] Plan regenerated with dimension overrides")

                # Set material from punch features
                punch_material = features.get("material_callout") or DEFAULT_MATERIAL
                density = get_material_density(punch_material)
                quote_data.material_info = MaterialInfo(
                    material_name=punch_material,
                    material_family="tool_steel",
                    density=density,
                    detected_from_cad=features.get("material_callout") is not None,
                    is_default=features.get("material_callout") is None,
                )

                # Set machine hours from punch estimates
                # Round each component to 2 decimal places for consistent display
                time_estimates = punch_data.get("time_estimates", {})
                mh = time_estimates.get("machine_hours", {})
                punch_milling_min = round(mh.get("total_milling_minutes", 0.0), 2)
                punch_grinding_min = round(mh.get("total_grinding_minutes", 0.0), 2)
                punch_tap_min = round(mh.get("total_tap_minutes", 0.0), 2)
                punch_drill_min = round(mh.get("total_drill_minutes", 0.0), 2)
                punch_edm_min = round(mh.get("total_edm_minutes", 0.0), 2)
                punch_other_min = round(mh.get("total_other_minutes", 0.0), 2)
                punch_cmm_min = round(mh.get("total_cmm_minutes", 0.0), 2)
                punch_total_min = round(mh.get("total_minutes", 0.0), 2)
                punch_machine_hours = round(punch_total_min / 60.0, 2)
                quote_data.machine_hours = MachineHoursBreakdown(
                    total_milling_minutes=punch_milling_min,
                    total_grinding_minutes=punch_grinding_min,
                    total_tap_minutes=punch_tap_min,
                    total_drill_minutes=punch_drill_min,
                    total_edm_minutes=punch_edm_min,
                    total_other_minutes=punch_other_min,
                    total_cmm_minutes=punch_cmm_min,
                    total_minutes=punch_total_min,
                    total_hours=punch_machine_hours,
                    machine_cost=round(punch_machine_hours * machine_rate, 2),
                )

                # Set labor hours from punch estimates
                lh = time_estimates.get("labor_hours", {})
                punch_labor_hours = lh.get("total_hours", 0.0)
                quote_data.labor_hours = LaborHoursBreakdown(
                    setup_minutes=lh.get("total_setup_minutes", 0.0),
                    programming_minutes=lh.get("cam_programming_minutes", 0.0),
                    machining_steps_minutes=lh.get("handling_minutes", 0.0),
                    inspection_minutes=lh.get("inspection_minutes", 0.0),
                    finishing_minutes=lh.get("deburring_minutes", 0.0),
                    total_minutes=lh.get("total_minutes", 0.0),
                    total_hours=punch_labor_hours,
                    labor_cost=punch_labor_hours * labor_rate,
                )

                # Store punch-specific data in quote_data
                quote_data.raw_plan = {
                    "is_punch": True,
                    "punch_features": features,
                    "punch_plan": punch_data.get("punch_plan", {}),
                    "planner": "punch_planner",
                }

                if verbose:
                    print(f"[PUNCH] Extraction complete")
                    print(f"  Machine hours: {quote_data.machine_hours.total_hours:.2f}")
                    print(f"  Labor hours: {quote_data.labor_hours.total_hours:.2f}")

    # ========================================================================
    # STEP 2: Extract part dimensions and material
    # ========================================================================
    if verbose:
        print("[2/5] Extracting dimensions and material...")

    # Skip standard extraction if punch data was successfully extracted
    if is_punch and punch_data and "error" not in punch_data:
        if verbose:
            print("  [PUNCH] Using punch-extracted dimensions and material")
        # Jump to cost calculation section
        pass  # Continue with rest of function using punch data
    else:
        # Standard plate/die extraction path
        pass

    # Apply dimension override if provided (useful when OCR fails)
    if dimension_override and not (is_punch and punch_data and "error" not in punch_data):
        from cad_quoter.planning.process_planner import plan_job

        length, width, thickness = dimension_override
        if verbose:
            print(f"  Using dimension override: {length} x {width} x {thickness}")

        # Update extracted dimensions
        if 'extracted_dims' not in plan:
            plan['extracted_dims'] = {}
        plan['extracted_dims']['L'] = length
        plan['extracted_dims']['W'] = width
        plan['extracted_dims']['T'] = thickness

        # IMPORTANT: Regenerate plan with new dimensions
        # The original plan was generated with OCR dimensions, but squaring logic
        # depends on L/W/T, so we need to regenerate with the override dimensions
        params = {
            'plate_LxW': (length, width),
            'T': thickness,
            'profile_tol': plan.get('profile_tol'),
            'flatness_spec': plan.get('flatness_spec'),
            'parallelism_spec': plan.get('parallelism_spec'),
            'windows_need_sharp': plan.get('windows_need_sharp', False),
            'window_corner_radius_req': plan.get('window_corner_radius_req'),
            'hole_sets': plan.get('hole_sets', []),
        }

        # Get the planner type from original plan
        planner_type = plan.get('planner', 'die_plate')
        regenerated_plan = plan_job(planner_type, params)

        # Preserve extracted_dims and other metadata from original plan
        regenerated_plan['extracted_dims'] = plan['extracted_dims']
        if 'extracted_material' in plan:
            regenerated_plan['extracted_material'] = plan['extracted_material']

        # Replace plan with regenerated version
        plan = regenerated_plan

        if verbose:
            print(f"  Plan regenerated with dimension overrides")

    # Material detection and dimension extraction (skip for punch parts)
    if not (is_punch and punch_data and "error" not in punch_data):
        if material_override:
            material = material_override
            detected_from_cad = False
        else:
            # Use cached text from plan if available (avoids ODA conversion)
            cached_text = None
            if 'text_dump' in plan and plan['text_dump']:
                cached_text = plan['text_dump'].split('\n')
            material = detect_material_in_cad(cad_file_path, text_list=cached_text)
            detected_from_cad = True
            if material == "GENERIC":
                material = DEFAULT_MATERIAL
                detected_from_cad = False

        # Get part info from plan
        part_info = extract_part_info_from_plan(plan, material)

        # Populate part dimensions
        quote_data.part_dimensions = PartDimensions(
            length=part_info.length,
            width=part_info.width,
            thickness=part_info.thickness,
            volume=part_info.volume,
            area=part_info.area
        )

        # Populate material info
        density = get_material_density(material)
        material_lower = material.lower()
        if any(kw in material_lower for kw in ["aluminum", "aluminium", "6061", "mic6"]):
            material_family = "aluminum"
        elif any(kw in material_lower for kw in ["stainless", "304", "316", "17-4"]):
            material_family = "stainless"
        elif any(kw in material_lower for kw in ["steel", "p20", "a36", "1018"]):
            material_family = "steel"
        else:
            material_family = "aluminum"  # Default

        quote_data.material_info = MaterialInfo(
            material_name=material,
            material_family=material_family,
            density=density,
            detected_from_cad=detected_from_cad,
            is_default=(material == DEFAULT_MATERIAL and not material_override)
        )

        if verbose:
            print(f"  Dimensions: {part_info.length:.2f} x {part_info.width:.2f} x {part_info.thickness:.2f} in")
            print(f"  Material: {material} ({'detected' if detected_from_cad else 'default'})")
    else:
        # For punch parts, use the already-set dimensions and material
        material = quote_data.material_info.material_name
        part_info = type('PartInfo', (), {
            'length': quote_data.part_dimensions.length,
            'width': quote_data.part_dimensions.width,
            'thickness': quote_data.part_dimensions.thickness,
            'volume': quote_data.part_dimensions.volume,
            'area': quote_data.part_dimensions.area,
        })()

    # ========================================================================
    # STEP 3: Calculate direct costs (McMaster stock, scrap, pricing)
    # ========================================================================
    if verbose:
        print("[3/5] Calculating direct costs...")

    # Use default catalog if not specified
    if catalog_csv_path is None:
        catalog_csv_path = str(default_catalog_csv())

    # Calculate desired stock dimensions (with machining allowances)
    # These are the actual dimensions needed for the starting stock
    desired_L = part_info.length + 0.50   # +0.50" face milling allowance
    desired_W = part_info.width + 0.50    # +0.50" face milling allowance
    desired_T = part_info.thickness + 0.25  # +0.25" thickness allowance

    # Calculate scrap info
    # Pass desired dimensions explicitly to ensure catalog lookup uses correct values
    scrap_calc = calculate_total_scrap(
        cad_file_path=cad_file_path,
        material=material,
        desired_length=desired_L,
        desired_width=desired_W,
        desired_thickness=desired_T,
        part_length=part_info.length,
        part_width=part_info.width,
        part_thickness=part_info.thickness,
        catalog_csv_path=catalog_csv_path,
        verbose=verbose
    )

    # Calculate scrap value
    scrap_value_calc = calculate_scrap_value(
        scrap_weight_lbs=scrap_calc.total_scrap_weight,
        material=material,
        fallback_scrap_price_per_lb=0.50,  # $0.50/lb default scrap rate
        verbose=verbose
    )

    # Apply scrap value override if provided
    if scrap_value_override is not None:
        scrap_value_calc['scrap_value'] = scrap_value_override
        if verbose:
            print(f"  Using manual scrap value override: ${scrap_value_override:.2f}")

    # Get McMaster pricing
    # Note: McMaster stock dimensions are already calculated in scrap_calc from calculate_total_scrap()
    # We just need to get the part number for pricing lookup
    catalog_rows = load_mcmaster_catalog_rows(catalog_csv_path)

    mcmaster_result = pick_mcmaster_plate_sku(
        need_L_in=desired_L,
        need_W_in=desired_W,
        need_T_in=desired_T,
        material_key=material,
        catalog_rows=catalog_rows
    )

    mcmaster_part_num = None
    mcmaster_price = None
    price_is_estimated = False

    # Use price override if provided, otherwise lookup from McMaster
    if mcmaster_price_override is not None:
        mcmaster_price = mcmaster_price_override
        if verbose:
            print(f"  Using manual McMaster price override: ${mcmaster_price:.2f}")
    elif mcmaster_result:
        mcmaster_part_num = mcmaster_result.get('mcmaster_part')
        if mcmaster_part_num:
            mcmaster_price = get_mcmaster_price(mcmaster_part_num, quantity=quantity)

            # If direct pricing failed, try volume-based estimation
            if mcmaster_price is None:
                if verbose:
                    print(f"  No direct pricing available for {mcmaster_part_num}")
                    print(f"  Attempting volume-based price estimation...")

                from cad_quoter.pricing.DirectCostHelper import estimate_price_from_reference_part

                # Map material to McMaster catalog key for price estimation
                mcmaster_material = material_mapper.get_mcmaster_key(material) or material
                mcmaster_price = estimate_price_from_reference_part(
                    target_length=scrap_calc.mcmaster_length,
                    target_width=scrap_calc.mcmaster_width,
                    target_thickness=scrap_calc.mcmaster_thickness,
                    material=mcmaster_material,
                    catalog_csv_path=catalog_csv_path,
                    verbose=verbose
                )

                if mcmaster_price:
                    price_is_estimated = True
                    if verbose:
                        print(f"  ✓ Estimated price from volume: ${mcmaster_price:.2f}")

    # If we still don't have a price and no part was found, try volume-based estimation
    if mcmaster_price is None and not mcmaster_part_num:
        if verbose:
            print(f"  No McMaster part found in catalog")
            print(f"  Attempting volume-based price estimation...")

        from cad_quoter.pricing.DirectCostHelper import estimate_price_from_reference_part

        # Map material to McMaster catalog key for price estimation
        mcmaster_material = material_mapper.get_mcmaster_key(material) or material
        mcmaster_price = estimate_price_from_reference_part(
            target_length=scrap_calc.mcmaster_length,
            target_width=scrap_calc.mcmaster_width,
            target_thickness=scrap_calc.mcmaster_thickness,
            material=mcmaster_material,
            catalog_csv_path=catalog_csv_path,
            verbose=verbose
        )

        if mcmaster_price:
            price_is_estimated = True
            if verbose:
                print(f"  ✓ Estimated price from volume: ${mcmaster_price:.2f}")

    # If volume-based estimation also failed, try weight-based estimation using largest catalog part
    if mcmaster_price is None:
        if verbose:
            print(f"  Volume-based estimation failed")
            print(f"  Attempting weight-based price estimation from largest catalog part...")

        from cad_quoter.pricing.mcmaster_helpers import estimate_price_from_catalog_reference, load_mcmaster_catalog_rows

        # Load catalog rows
        catalog_rows = load_mcmaster_catalog_rows(path=catalog_csv_path)
        if verbose:
            print(f"  Loaded {len(catalog_rows)} catalog rows from {catalog_csv_path}")

        # Get material density
        density_lb_in3 = material_mapper.get_density_lb_in3(material)
        if density_lb_in3 is None or density_lb_in3 <= 0:
            density_lb_in3 = 0.10  # Default to aluminum density

        # Map material to McMaster catalog key
        mcmaster_material = material_mapper.get_mcmaster_key(material) or material

        # Estimate price based on weight
        estimated_price, source = estimate_price_from_catalog_reference(
            material_key=mcmaster_material,
            weight_lb=scrap_calc.mcmaster_weight,
            density_lb_in3=density_lb_in3,
            catalog_rows=catalog_rows,
            verbose=verbose
        )

        if estimated_price and estimated_price > 0:
            mcmaster_price = estimated_price
            price_is_estimated = True
            if verbose:
                print(f"  ✓ Estimated price from weight: ${mcmaster_price:.2f}")
                print(f"     ({source})")

    # Populate stock info
    # Use McMaster dimensions from scrap_calc (which already did the catalog lookup)
    quote_data.stock_info = StockInfo(
        desired_length=desired_L,
        desired_width=desired_W,
        desired_thickness=desired_T,
        desired_volume=desired_L * desired_W * desired_T,
        mcmaster_length=scrap_calc.mcmaster_length,
        mcmaster_width=scrap_calc.mcmaster_width,
        mcmaster_thickness=scrap_calc.mcmaster_thickness,
        mcmaster_volume=scrap_calc.mcmaster_length * scrap_calc.mcmaster_width * scrap_calc.mcmaster_thickness,
        mcmaster_part_number=mcmaster_part_num,
        mcmaster_price=mcmaster_price,
        price_is_estimated=price_is_estimated,
        mcmaster_weight=scrap_calc.mcmaster_weight,
        final_part_weight=scrap_calc.final_part_weight
    )

    # Populate scrap info
    quote_data.scrap_info = ScrapInfo(
        stock_prep_scrap=scrap_calc.stock_prep_scrap,
        face_milling_scrap=scrap_calc.face_milling_scrap,
        hole_drilling_scrap=scrap_calc.hole_drilling_scrap,
        total_scrap_volume=scrap_calc.total_scrap_volume,
        total_scrap_weight=scrap_calc.total_scrap_weight,
        scrap_percentage=scrap_calc.scrap_percentage,
        utilization_percentage=scrap_calc.utilization_percentage,
        scrap_price_per_lb=scrap_value_calc.get('scrap_price_per_lb'),
        scrap_value=scrap_value_calc.get('scrap_value', 0.0),
        scrap_price_source=scrap_value_calc.get('price_source', '')
    )

    # Calculate direct cost breakdown
    # NOTE: Keep total_material_cost in sync with printed line items (avoid 1¢ drift).
    # Round each component to 2 decimal places before summing to ensure the total
    # matches the sum of the displayed line items exactly.
    if mcmaster_price:
        tax = round(mcmaster_price * 0.07, 2)
        shipping = round(mcmaster_price * 0.125, 2)
        scrap_credit = round(scrap_value_calc.get('scrap_value', 0.0), 2)
        net_cost = round(mcmaster_price + tax + shipping - scrap_credit, 2)
    else:
        tax = shipping = scrap_credit = net_cost = 0.0

    quote_data.direct_cost_breakdown = DirectCostBreakdown(
        stock_cost=mcmaster_price or 0.0,
        tax=tax,
        shipping=shipping,
        scrap_credit=scrap_credit,
        net_material_cost=net_cost
    )

    if verbose:
        print(f"  McMaster: {mcmaster_part_num or 'N/A'}")
        print(f"  Stock price: ${mcmaster_price:.2f}" if mcmaster_price else "  Stock price: N/A")
        print(f"  Net material cost: ${net_cost:.2f}")

    # ========================================================================
    # STEP 4: Calculate machine hours
    # ========================================================================
    # Skip for punch parts - already calculated in punch extraction
    if is_punch and punch_data and "error" not in punch_data:
        if verbose:
            print("[4/5] Calculating machine hours...")
            print(f"  [PUNCH] Using punch-calculated machine hours: {quote_data.machine_hours.total_hours:.2f} hr")
            print(f"  Machine cost: ${quote_data.machine_hours.machine_cost:.2f}")
    else:
        if verbose:
            print("[4/5] Calculating machine hours...")

        # Use cached hole operations from plan if available (avoids redundant ODA conversion)
        if 'hole_operations_data' in plan and plan['hole_operations_data']:
            hole_table = plan['hole_operations_data']
            if verbose:
                print("  Using cached hole operations from plan")
        else:
            hole_table = extract_hole_operations_from_cad(cad_file_path)

        # Calculate total hole count (sum QTY field from each hole entry)
        holes_total = sum(int(hole.get('QTY', 1)) for hole in hole_table) if hole_table else 0

        # Initialize time accumulators
        total_drill_min = 0.0
        total_tap_min = 0.0
        total_cbore_min = 0.0
        total_cdrill_min = 0.0
        total_jig_grind_min = 0.0
        total_milling_min = 0.0
        total_grinding_min = 0.0
        total_edm_min = 0.0
        total_other_min = 0.0

        drill_ops = []
        tap_ops = []
        cbore_ops = []
        cdrill_ops = []
        jig_grind_ops = []

        if hole_table:
            times = estimate_hole_table_times(hole_table, material, part_info.thickness)

            # Convert to HoleOperation objects
            drill_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='drill',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time'],
                    sfm=g.get('sfm'),
                    ipr=g.get('ipr')
                )
                for g in times.get('drill_groups', [])
            ]

            tap_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='tap',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time'],
                    sfm=g.get('sfm'),
                    tpi=g.get('tpi')
                )
                for g in times.get('tap_groups', [])
            ]

            cbore_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='cbore',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time'],
                    sfm=g.get('sfm'),
                    side=g.get('side')
                )
                for g in times.get('cbore_groups', [])
            ]

            cdrill_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='cdrill',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time']
                )
                for g in times.get('cdrill_groups', [])
            ]

            jig_grind_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='jig_grind',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time']
                )
                for g in times.get('jig_grind_groups', [])
            ]

            # Accumulate hole operation times
            total_drill_min = times.get('total_drill_minutes', 0.0)
            total_tap_min = times.get('total_tap_minutes', 0.0)
            total_cbore_min = times.get('total_cbore_minutes', 0.0)
            total_cdrill_min = times.get('total_cdrill_minutes', 0.0)
            total_jig_grind_min = times.get('total_jig_grind_minutes', 0.0)

        # Calculate times for plan operations (squaring, face milling, EDM, etc.)
        plan_machine_times = estimate_machine_hours_from_plan(
            plan,
            material=material,
            plate_LxW=(part_info.length, part_info.width),
            thickness=part_info.thickness
        )

        # Add plan operation times to totals
        total_milling_min = plan_machine_times['breakdown_minutes'].get('milling', 0.0)
        total_grinding_min = plan_machine_times['breakdown_minutes'].get('grinding', 0.0)
        total_edm_min = plan_machine_times['breakdown_minutes'].get('edm', 0.0)
        total_other_min = plan_machine_times['breakdown_minutes'].get('other', 0.0)

        # Convert detailed operations to dataclass objects
        milling_ops = [
            MillingOperation(**op) for op in plan_machine_times.get('milling_operations', [])
        ]
        grinding_ops = [
            GrindingOperation(**op) for op in plan_machine_times.get('grinding_operations', [])
        ]

        # Calculate CMM inspection time (split between labor setup and machine checking)
        from cad_quoter.planning.process_planner import cmm_inspection_minutes
        cmm_breakdown = cmm_inspection_minutes(holes_total)
        cmm_setup_labor_min = cmm_breakdown['setup_labor_min']
        cmm_checking_machine_min = cmm_breakdown['checking_machine_min']
        cmm_holes_checked = cmm_breakdown['holes_checked']

        # Calculate grand totals (machine time only includes CMM checking, not setup)
        # Round each component to 2 decimal places before summing to ensure the total
        # matches the sum of the displayed line items exactly.
        total_drill_min = round(total_drill_min, 2)
        total_tap_min = round(total_tap_min, 2)
        total_cbore_min = round(total_cbore_min, 2)
        total_cdrill_min = round(total_cdrill_min, 2)
        total_jig_grind_min = round(total_jig_grind_min, 2)
        total_milling_min = round(total_milling_min, 2)
        total_grinding_min = round(total_grinding_min, 2)
        total_edm_min = round(total_edm_min, 2)
        total_other_min = round(total_other_min, 2)
        cmm_checking_machine_min = round(cmm_checking_machine_min, 2)

        grand_total_minutes = round(
            total_drill_min + total_tap_min + total_cbore_min +
            total_cdrill_min + total_jig_grind_min +
            total_milling_min + total_grinding_min + total_edm_min + total_other_min +
            cmm_checking_machine_min, 2
        )
        grand_total_hours = round(grand_total_minutes / 60.0, 2)

        quote_data.machine_hours = MachineHoursBreakdown(
            drill_operations=drill_ops,
            tap_operations=tap_ops,
            cbore_operations=cbore_ops,
            cdrill_operations=cdrill_ops,
            jig_grind_operations=jig_grind_ops,
            milling_operations=milling_ops,
            grinding_operations=grinding_ops,
            total_drill_minutes=total_drill_min,
            total_tap_minutes=total_tap_min,
            total_cbore_minutes=total_cbore_min,
            total_cdrill_minutes=total_cdrill_min,
            total_jig_grind_minutes=total_jig_grind_min,
            total_milling_minutes=total_milling_min,
            total_grinding_minutes=total_grinding_min,
            total_edm_minutes=total_edm_min,
            total_other_minutes=total_other_min,
            total_cmm_minutes=cmm_checking_machine_min,
            cmm_holes_checked=cmm_holes_checked,
            total_minutes=grand_total_minutes,
            total_hours=grand_total_hours,
            machine_cost=round(grand_total_hours * machine_rate, 2)
        )

        # Sanity check: warn if milling/overhead time is disproportionately high for small jobs
        primary_ops_min = total_drill_min + total_grinding_min + total_jig_grind_min + total_tap_min
        overhead_ops_min = total_milling_min + total_cbore_min + total_edm_min + total_other_min

        if primary_ops_min < 5.0 and overhead_ops_min > primary_ops_min and overhead_ops_min > 1.0:
            import logging
            logging.info(
                f"Machine time check: primary ops={primary_ops_min:.2f} min, "
                f"overhead ops (milling/cbore/edm/other)={overhead_ops_min:.2f} min. "
                f"High overhead ratio for small job - verify milling time is correct."
            )

        if verbose:
            print(f"  Machine hours: {grand_total_hours:.2f} hr")
            print(f"    - Drilling: {total_drill_min:.1f} min")
            print(f"    - Tapping: {total_tap_min:.1f} min")
            print(f"    - Milling (inc. squaring): {total_milling_min:.1f} min")
            print(f"    - Grinding (inc. wet grind): {total_grinding_min:.1f} min")
            print(f"    - EDM: {total_edm_min:.1f} min")
            print(f"    - CMM Inspection (checking only): {cmm_checking_machine_min:.1f} min")
            print(f"  Machine cost: ${quote_data.machine_hours.machine_cost:.2f}")

            # Also print the sanity check warning in verbose mode
            if primary_ops_min < 5.0 and overhead_ops_min > primary_ops_min and overhead_ops_min > 1.0:
                print(f"  WARNING: High overhead ratio - primary ops={primary_ops_min:.1f} min, "
                      f"overhead ops={overhead_ops_min:.1f} min")

        # ========================================================================
        # STEP 5: Calculate labor hours
        # ========================================================================
        if verbose:
            print("[5/5] Calculating labor hours...")

        ops = plan.get('ops', [])
        # Note: holes_total is already calculated earlier in STEP 4

        # Estimate labor inputs (simplified - could be more sophisticated)
        labor_inputs = LaborInputs(
            ops_total=len(ops),
            holes_total=holes_total,
            tool_changes=len(ops) * 2,  # Rough estimate
            fixturing_complexity=1,
            cmm_setup_min=cmm_setup_labor_min  # Add CMM setup to inspection labor
        )

        labor_result = compute_labor_minutes(labor_inputs)
        minutes = labor_result['minutes']

        # Round labor components for consistent display
        labor_total_min = round(minutes.get('Labor_Total', 0.0), 2)
        labor_total_hours = round(labor_total_min / 60.0, 2)
        quote_data.labor_hours = LaborHoursBreakdown(
            setup_minutes=round(minutes.get('Setup', 0.0), 2),
            programming_minutes=round(minutes.get('Programming', 0.0), 2),
            machining_steps_minutes=round(minutes.get('Machining_Steps', 0.0), 2),
            inspection_minutes=round(minutes.get('Inspection', 0.0), 2),
            finishing_minutes=round(minutes.get('Finishing', 0.0), 2),
            total_minutes=labor_total_min,
            total_hours=labor_total_hours,
            labor_cost=round(labor_total_hours * labor_rate, 2),
            ops_total=len(ops),
            holes_total=holes_total,
            tool_changes=len(ops) * 2
        )

        if verbose:
            print(f"  Labor hours: {quote_data.labor_hours.total_hours:.2f} hr")
            print(f"  Labor cost: ${quote_data.labor_hours.labor_cost:.2f}")

    # ========================================================================
    # STEP 6: Calculate cost summary with quantity-aware amortization
    # ========================================================================

    # Calculate amortized setup costs (spread across all parts)
    setup_labor = (quote_data.labor_hours.setup_minutes / 60.0) * labor_rate
    programming_labor = (quote_data.labor_hours.programming_minutes / 60.0) * labor_rate
    amortized_setup_cost = (setup_labor + programming_labor) / quantity

    # Calculate variable costs per unit (material, machining, inspection, finishing)
    material_cost_per_unit = quote_data.direct_cost_breakdown.net_material_cost
    machine_cost_per_unit = quote_data.machine_hours.machine_cost

    # Variable labor costs per unit (machining, inspection, finishing)
    machining_labor = (quote_data.labor_hours.machining_steps_minutes / 60.0) * labor_rate
    inspection_labor = (quote_data.labor_hours.inspection_minutes / 60.0) * labor_rate
    finishing_labor = (quote_data.labor_hours.finishing_minutes / 60.0) * labor_rate
    variable_labor_per_unit = machining_labor + inspection_labor + finishing_labor

    # Per-unit costs
    # Round each component to 2 decimal places before summing to ensure the total
    # matches the sum of the displayed line items exactly.
    per_unit_direct_cost = round(material_cost_per_unit, 2)
    per_unit_machine_cost = round(machine_cost_per_unit, 2)
    per_unit_labor_cost = round(amortized_setup_cost + variable_labor_per_unit, 2)
    per_unit_total_cost = round(per_unit_direct_cost + per_unit_machine_cost + per_unit_labor_cost, 2)

    # Total costs for all units
    total_direct_cost = round(per_unit_direct_cost * quantity, 2)
    total_machine_cost = round(per_unit_machine_cost * quantity, 2)
    total_labor_cost = round((setup_labor + programming_labor) + (variable_labor_per_unit * quantity), 2)
    total_total_cost = round(total_direct_cost + total_machine_cost + total_labor_cost, 2)

    # Margin and pricing
    per_unit_margin_amount = round(per_unit_total_cost * margin_rate, 2)
    per_unit_final_price = round(per_unit_total_cost + per_unit_margin_amount, 2)

    total_margin_amount = round(total_total_cost * margin_rate, 2)
    total_final_price = round(total_total_cost + total_margin_amount, 2)

    quote_data.cost_summary = CostSummary(
        # Per-unit costs
        direct_cost=per_unit_direct_cost,
        machine_cost=per_unit_machine_cost,
        labor_cost=per_unit_labor_cost,
        total_cost=per_unit_total_cost,
        # Total costs
        total_direct_cost=total_direct_cost,
        total_machine_cost=total_machine_cost,
        total_labor_cost=total_labor_cost,
        total_total_cost=total_total_cost,
        # Margin and pricing
        margin_rate=margin_rate,
        margin_amount=per_unit_margin_amount,
        final_price=per_unit_final_price,
        total_final_price=total_final_price
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*70}")
        if quantity > 1:
            print(f"Quantity: {quantity} parts")
            print(f"Per-unit cost: ${per_unit_total_cost:.2f}")
            print(f"Per-unit margin ({margin_rate:.0%}): ${per_unit_margin_amount:.2f}")
            print(f"Per-unit price: ${per_unit_final_price:.2f}")
            print(f"---")
            print(f"Total cost (all units): ${total_total_cost:.2f}")
            print(f"Total margin: ${total_margin_amount:.2f}")
            print(f"Total price: ${total_final_price:.2f}")
        else:
            print(f"Total cost: ${per_unit_total_cost:.2f}")
            print(f"Margin ({margin_rate:.0%}): ${per_unit_margin_amount:.2f}")
            print(f"Final price: ${per_unit_final_price:.2f}")
        print(f"{'='*70}\n")

    return quote_data


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def save_quote_data(
    quote_data: QuoteData,
    output_path: str | Path,
    pretty: bool = True
) -> None:
    """
    Save QuoteData to JSON file.

    Args:
        quote_data: QuoteData instance
        output_path: Path to save JSON file
        pretty: Use pretty formatting (indent=2)
    """
    indent = 2 if pretty else None
    quote_data.to_json(output_path, indent=indent)
    print(f"Saved quote data to: {output_path}")


def load_quote_data(input_path: str | Path) -> QuoteData:
    """
    Load QuoteData from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        QuoteData instance
    """
    return QuoteData.from_json(input_path)
