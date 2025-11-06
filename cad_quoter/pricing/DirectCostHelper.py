"""Direct cost calculation helper - extracts part size and material for cost estimation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Mapping
import os

# ============================================================================
# DEFAULT MATERIAL FOR TESTING
# ============================================================================
# This is used when no material is specified or auto-detected.
# Set to "aluminum MIC6" for McMaster catalog compatibility during testing.
DEFAULT_MATERIAL = "aluminum MIC6"  # <-- CHANGE THIS TO SET DEFAULT MATERIAL
# ============================================================================


@dataclass
class PartInfo:
    """Part information for direct cost calculations."""
    length: float = 0.0  # inches
    width: float = 0.0   # inches
    thickness: float = 0.0  # inches
    material: str = DEFAULT_MATERIAL  # Default to aluminum MIC6 for McMaster compatibility
    volume: float = 0.0  # cubic inches
    area: float = 0.0    # square inches (L × W)

    def __post_init__(self):
        """Calculate derived properties."""
        self.volume = self.length * self.width * self.thickness
        self.area = self.length * self.width


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_dim_overrides(dims_override: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """Return canonical overrides keyed by 'L', 'W', 'T'."""
    if not dims_override:
        return {}

    key_map = {
        "L": "L",
        "LENGTH": "L",
        "W": "W",
        "WIDTH": "W",
        "T": "T",
        "THK": "T",
        "THICK": "T",
        "THICKNESS": "T",
    }

    normalized: Dict[str, float] = {}
    for key, raw_value in dims_override.items():
        if raw_value is None:
            continue
        canonical = key_map.get(str(key).strip().upper())
        if not canonical:
            continue
        try:
            normalized[canonical] = float(raw_value)
        except (TypeError, ValueError):
            continue

    return normalized


def extract_part_info_from_plan(
    plan: Dict[str, Any],
    material: Optional[str] = None,
    *,
    dims_override: Optional[Mapping[str, Any]] = None,
) -> PartInfo:
    """
    Extract part size and material from a process plan.

    Args:
        plan: Process plan dict (from plan_job or plan_from_cad_file)
        material: Material override (if not provided, uses default)
        dims_override: Optional mapping with manual dimension overrides

    Returns:
        PartInfo with dimensions and material

    Example:
        >>> from cad_quoter.planning import plan_from_cad_file
        >>> plan = plan_from_cad_file("part.dxf")
        >>> part_info = extract_part_info_from_plan(plan, "17-4 PH Stainless")
        >>> print(f"Volume: {part_info.volume:.2f} cubic inches")
    """
    # Extract dimensions from plan
    dims = dict(plan.get('extracted_dims', {}))
    overrides = _normalize_dim_overrides(dims_override)

    if overrides:
        dims.update(overrides)
        plan.setdefault('extracted_dims', {}).update(overrides)

    length = _safe_float(dims.get('L', 0.0))
    width = _safe_float(dims.get('W', 0.0))
    thickness = _safe_float(dims.get('T', 0.0))

    # Use provided material or default
    part_material = material if material else DEFAULT_MATERIAL

    return PartInfo(
        length=length,
        width=width,
        thickness=thickness,
        material=part_material
    )


def extract_part_info_from_cad(
    cad_file_path: str | Path,
    material: Optional[str] = None,
    use_paddle_ocr: bool = True,
    auto_detect_material: bool = True,
    verbose: bool = False,
    *,
    dims_override: Optional[Mapping[str, Any]] = None,
) -> PartInfo:
    """
    Extract part size and material directly from CAD file using PaddleOCR.

    This function uses PaddleOCR to extract L×W×T dimensions from the CAD file,
    which provides accurate detection of dimension text annotations.

    Args:
        cad_file_path: Path to CAD file (DXF/DWG)
        material: Material name (if None and auto_detect_material=True, will auto-detect)
        use_paddle_ocr: Use PaddleOCR for dimension extraction (default: True)
        auto_detect_material: Auto-detect material from CAD text if not specified (default: True)
        verbose: Print extraction details
        dims_override: Optional mapping with manual dimension overrides

    Returns:
        PartInfo with dimensions (extracted via PaddleOCR) and material

    Example:
        >>> # Auto-detect material from CAD file
        >>> part_info = extract_part_info_from_cad("part.dxf")
        >>> print(f"Size: {part_info.length} x {part_info.width} x {part_info.thickness}")
        >>> print(f"Material: {part_info.material}")

        >>> # Or specify material explicitly
        >>> part_info = extract_part_info_from_cad("part.dxf", material="P20 Tool Steel")
    """
    from cad_quoter.planning import plan_from_cad_file

    # Auto-detect material if not specified
    if material is None and auto_detect_material:
        from cad_quoter.pricing.KeywordDetector import detect_material_in_cad
        material = detect_material_in_cad(cad_file_path)
        if verbose:
            print(f"Auto-detected material: {material}")

        # If auto-detection returns "GENERIC", use default material for McMaster compatibility
        if material == "GENERIC":
            if verbose:
                print(f"Using default material instead: {DEFAULT_MATERIAL}")
            material = DEFAULT_MATERIAL

    # Generate plan from CAD file - uses PaddleOCR by default for dimension extraction
    plan = plan_from_cad_file(
        cad_file_path,
        use_paddle_ocr=use_paddle_ocr,
        verbose=verbose,
        dims_override=dims_override,
    )

    # Extract part info from plan (dimensions come from PaddleOCR)
    return extract_part_info_from_plan(plan, material, dims_override=dims_override)


def get_part_dimensions(plan: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Extract just the dimensions (L, W, T) from a plan.

    Args:
        plan: Process plan dict

    Returns:
        Tuple of (length, width, thickness) in inches

    Example:
        >>> L, W, T = get_part_dimensions(plan)
    """
    dims = plan.get('extracted_dims', {})
    return (
        dims.get('L', 0.0),
        dims.get('W', 0.0),
        dims.get('T', 0.0)
    )


def extract_dimensions_with_paddle_ocr(cad_file_path: str | Path) -> Dict[str, float]:
    """
    Extract L×W×T dimensions directly from CAD file using PaddleOCR.

    This is a direct wrapper around extract_dimensions_from_cad() that explicitly
    uses PaddleOCR to detect dimension annotations in the CAD file.

    Args:
        cad_file_path: Path to CAD file (DXF/DWG)

    Returns:
        Dict with 'L', 'W', 'T' keys (length, width, thickness in inches)

    Example:
        >>> dims = extract_dimensions_with_paddle_ocr("part.dxf")
        >>> print(f"Length: {dims['L']}\", Width: {dims['W']}\", Thickness: {dims['T']}\"")
    """
    from cad_quoter.planning import extract_dimensions_from_cad

    # Use PaddleOCR to extract L×W×T from CAD file (returns tuple or None)
    dims_tuple = extract_dimensions_from_cad(cad_file_path)

    # Convert tuple to dict
    if dims_tuple:
        L, W, T = dims_tuple
        return {'L': L, 'W': W, 'T': T}
    else:
        return {'L': 0.0, 'W': 0.0, 'T': 0.0}


def calculate_material_volume(
    length: float,
    width: float,
    thickness: float
) -> float:
    """
    Calculate material volume in cubic inches.

    Args:
        length: Length in inches
        width: Width in inches
        thickness: Thickness in inches

    Returns:
        Volume in cubic inches
    """
    return length * width * thickness


def calculate_material_weight(
    volume_cubic_inches: float,
    density_lb_per_cubic_inch: float
) -> float:
    """
    Calculate material weight in pounds.

    Args:
        volume_cubic_inches: Volume in cubic inches
        density_lb_per_cubic_inch: Material density (lb/in³)

    Returns:
        Weight in pounds

    Example:
        >>> # Steel density ≈ 0.283 lb/in³
        >>> volume = 10.0  # cubic inches
        >>> weight = calculate_material_weight(volume, 0.283)
    """
    return volume_cubic_inches * density_lb_per_cubic_inch


# Material density lookup (lb/in³)
MATERIAL_DENSITIES = {
    "Aluminum 6061-T6": 0.098,
    "17-4 PH Stainless": 0.280,
    "P20 Tool Steel": 0.283,
    "Carbide": 0.540,
    "Ceramic": 0.145,
    "GENERIC": 0.283,  # Default to steel
}


def get_material_density(material: str) -> float:
    """
    Get material density in lb/in³.

    Args:
        material: Material name

    Returns:
        Density in lb/in³
    """
    return MATERIAL_DENSITIES.get(material, MATERIAL_DENSITIES["GENERIC"])


# McMaster-Carr integration functions

def get_mcmaster_part_number(
    length: float,
    width: float,
    thickness: float,
    material: str,
    catalog_csv_path: Optional[str] = None
) -> Optional[str]:
    """
    Get McMaster part number for material stock based on dimensions and material.

    Uses the catalog CSV to find the best-fitting stock size.

    Args:
        length: Length in inches
        width: Width in inches
        thickness: Thickness in inches
        material: Material name (e.g., "aluminum MIC6", "303 Stainless Steel")
        catalog_csv_path: Optional path to catalog.csv (defaults to resources/catalog.csv)

    Returns:
        McMaster part number string, or None if no match found

    Example:
        >>> part_num = get_mcmaster_part_number(6.0, 6.0, 0.25, "aluminum MIC6")
        >>> print(part_num)  # e.g., "2567N11"
    """
    from cad_quoter.pricing.mcmaster_helpers import (
        pick_mcmaster_plate_sku,
        load_mcmaster_catalog_rows
    )
    from cad_quoter.resources import default_catalog_csv

    # Use explicit path if not provided (don't rely on env var)
    if catalog_csv_path is None:
        catalog_csv_path = str(default_catalog_csv())

    # Load catalog
    catalog_rows = load_mcmaster_catalog_rows(catalog_csv_path)
    if not catalog_rows:
        return None

    # Normalize material name for lookup
    # The catalog uses keys like "MIC6", "303 Stainless Steel", "aluminum 5083", etc.
    material_key = material.strip()

    # Try to find a matching part
    result = pick_mcmaster_plate_sku(
        need_L_in=length,
        need_W_in=width,
        need_T_in=thickness,
        material_key=material_key,
        catalog_rows=catalog_rows
    )

    if result and "mcmaster_part" in result:
        return result["mcmaster_part"]

    return None


def get_mcmaster_price(
    part_number: str,
    quantity: int = 1
) -> Optional[float]:
    """
    Get price for a McMaster part number using the McMaster API.

    Args:
        part_number: McMaster part number (e.g., "2567N11")
        quantity: Quantity to quote (default: 1)

    Returns:
        Unit price as float, or None if unable to fetch price

    Example:
        >>> price = get_mcmaster_price("2567N11", quantity=1)
        >>> print(f"Price: ${price:.2f}")
    """
    from mcmaster_api import McMasterAPI, load_env

    try:
        # Load credentials from environment
        env = load_env()

        # Initialize API
        api = McMasterAPI(
            username=env["MCMASTER_USER"],
            password=env["MCMASTER_PASS"],
            pfx_path=env["MCMASTER_PFX_PATH"],
            pfx_password=env["MCMASTER_PFX_PASS"],
        )

        # Login and get price tiers
        api.login()
        tiers = api.get_price_tiers(part_number)

        if not tiers:
            return None

        # Find the appropriate tier for the requested quantity
        # Tiers are sorted by MinimumQuantity ascending
        selected_tier = None
        for tier in tiers:
            min_qty = tier.get("MinimumQuantity", 0)
            if min_qty <= quantity:
                selected_tier = tier
            else:
                break

        if selected_tier and "Amount" in selected_tier:
            amount = selected_tier["Amount"]
            if amount is not None:
                return float(amount)

        return None

    except Exception as e:
        # If API fails, return None (caller can handle fallback)
        print(f"Warning: Failed to fetch McMaster price for {part_number}: {e}")
        return None


def get_mcmaster_part_and_price(
    length: float,
    width: float,
    thickness: float,
    material: str,
    quantity: int = 1,
    catalog_csv_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get both McMaster part number and price for material stock.

    This is a convenience function that combines get_mcmaster_part_number()
    and get_mcmaster_price().

    Args:
        length: Length in inches
        width: Width in inches
        thickness: Thickness in inches
        material: Material name
        quantity: Quantity to quote (default: 1)
        catalog_csv_path: Optional path to catalog.csv

    Returns:
        Dict with keys:
            - "part_number": McMaster part number (str or None)
            - "price": Unit price (float or None)
            - "success": True if both part and price were found

    Example:
        >>> result = get_mcmaster_part_and_price(6.0, 6.0, 0.25, "aluminum MIC6")
        >>> if result["success"]:
        >>>     print(f"Part: {result['part_number']}, Price: ${result['price']:.2f}")
    """
    from cad_quoter.resources import default_catalog_csv

    # Use explicit path if not provided (don't rely on env var)
    if catalog_csv_path is None:
        catalog_csv_path = str(default_catalog_csv())

    # Get part number
    part_number = get_mcmaster_part_number(
        length=length,
        width=width,
        thickness=thickness,
        material=material,
        catalog_csv_path=catalog_csv_path
    )

    # Get price if we have a part number
    price = None
    if part_number:
        price = get_mcmaster_price(part_number, quantity=quantity)

    return {
        "part_number": part_number,
        "price": price,
        "success": part_number is not None and price is not None
    }


def get_material_cost_from_mcmaster(
    part_info: PartInfo,
    quantity: int = 1,
    catalog_csv_path: Optional[str] = None
) -> Optional[float]:
    """
    Get material cost directly from McMaster for a PartInfo object.

    Args:
        part_info: PartInfo with dimensions and material
        quantity: Quantity to quote (default: 1)
        catalog_csv_path: Optional path to catalog.csv

    Returns:
        Total material cost (unit price × quantity), or None if unavailable

    Example:
        >>> part_info = extract_part_info_from_cad("part.dxf")
        >>> cost = get_material_cost_from_mcmaster(part_info)
        >>> if cost:
        >>>     print(f"Material cost: ${cost:.2f}")
    """
    from cad_quoter.resources import default_catalog_csv

    # Use explicit path if not provided (don't rely on env var)
    if catalog_csv_path is None:
        catalog_csv_path = str(default_catalog_csv())

    result = get_mcmaster_part_and_price(
        length=part_info.length,
        width=part_info.width,
        thickness=part_info.thickness,
        material=part_info.material,
        quantity=quantity,
        catalog_csv_path=catalog_csv_path
    )

    if result["success"] and result["price"] is not None:
        return result["price"] * quantity

    return None


# ============================================================================
# SCRAP CALCULATION
# ============================================================================

@dataclass
class ScrapInfo:
    """Scrap calculation results."""
    # McMaster stock dimensions
    mcmaster_length: float = 0.0
    mcmaster_width: float = 0.0
    mcmaster_thickness: float = 0.0
    mcmaster_volume: float = 0.0

    # Desired starting stock dimensions (what we cut McMaster down to)
    desired_length: float = 0.0
    desired_width: float = 0.0
    desired_thickness: float = 0.0
    desired_volume: float = 0.0

    # Final part dimensions
    part_length: float = 0.0
    part_width: float = 0.0
    part_thickness: float = 0.0
    part_envelope_volume: float = 0.0
    part_final_volume: float = 0.0  # After holes

    # Scrap breakdown
    stock_prep_scrap: float = 0.0  # McMaster → Desired
    face_milling_scrap: float = 0.0  # Desired → Part envelope
    hole_drilling_scrap: float = 0.0  # Holes removed from part
    total_scrap_volume: float = 0.0  # Total material removed

    # Weight calculations (if material density provided)
    material: str = ""
    density: float = 0.0
    mcmaster_weight: float = 0.0
    final_part_weight: float = 0.0
    total_scrap_weight: float = 0.0

    # Percentages
    scrap_percentage: float = 0.0  # Scrap as % of McMaster stock
    utilization_percentage: float = 0.0  # Final part as % of McMaster stock


def calculate_stock_prep_scrap(
    mcmaster_length: float,
    mcmaster_width: float,
    mcmaster_thickness: float,
    desired_length: float,
    desired_width: float,
    desired_thickness: float
) -> float:
    """
    Calculate material removed when cutting McMaster stock down to desired starting size.

    This is the scrap generated during stock preparation (saw cutting, etc.)
    before any machining operations begin.

    Args:
        mcmaster_length: McMaster stock length in inches
        mcmaster_width: McMaster stock width in inches
        mcmaster_thickness: McMaster stock thickness in inches
        desired_length: Desired starting stock length in inches
        desired_width: Desired starting stock width in inches
        desired_thickness: Desired starting stock thickness in inches

    Returns:
        Volume of material removed in cubic inches

    Example:
        >>> # McMaster stock: 16" x 13" x 2.5"
        >>> # Desired size: 15.75" x 12.25" x 2.125"
        >>> scrap = calculate_stock_prep_scrap(16, 13, 2.5, 15.75, 12.25, 2.125)
        >>> print(f"Stock prep scrap: {scrap:.2f} in³")
    """
    mcmaster_vol = mcmaster_length * mcmaster_width * mcmaster_thickness
    desired_vol = desired_length * desired_width * desired_thickness
    return mcmaster_vol - desired_vol


def calculate_machining_scrap_from_cad(
    cad_file_path: str | Path,
    desired_length: float = None,
    desired_width: float = None,
    desired_thickness: float = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Calculate material removed during machining operations (face milling + holes).

    This uses the extract functions from process_planner.py to calculate:
    - Face milling scrap (desired stock → part envelope)
    - Hole drilling scrap (material removed by drilling)

    Args:
        cad_file_path: Path to CAD file (DXF/DWG)
        desired_length: Desired starting stock length (if None, uses part L + allowance)
        desired_width: Desired starting stock width (if None, uses part W + allowance)
        desired_thickness: Desired starting stock thickness (if None, uses part T + allowance)
        verbose: Print detailed output

    Returns:
        Dict with keys:
            - face_milling_scrap: Volume removed by face milling (in³)
            - hole_drilling_scrap: Volume removed by drilling holes (in³)
            - total_machining_scrap: Total material removed during machining (in³)
            - part_envelope_volume: Volume of part envelope (before holes) (in³)
            - part_final_volume: Volume of final part (after holes) (in³)
            - part_dimensions: Dict with L, W, T
            - desired_dimensions: Dict with L, W, T
            - holes: List of hole details

    Example:
        >>> result = calculate_machining_scrap_from_cad("part.dxf")
        >>> print(f"Face milling: {result['face_milling_scrap']:.2f} in³")
        >>> print(f"Holes: {result['hole_drilling_scrap']:.2f} in³")
    """
    from cad_quoter.planning.process_planner import (
        extract_dimensions_from_cad,
        extract_hole_table_from_cad
    )
    import math
    import re
    from fractions import Fraction

    cad_file_path = Path(cad_file_path)

    if verbose:
        print(f"Analyzing machining scrap for: {cad_file_path.name}")

    # Extract part dimensions
    dims = extract_dimensions_from_cad(cad_file_path)
    if dims:
        part_L, part_W, part_T = dims
    else:
        raise ValueError(f"Could not extract dimensions from {cad_file_path}")

    if verbose:
        print(f"Part dimensions: L={part_L:.3f}\", W={part_W:.3f}\", T={part_T:.3f}\"")

    # Calculate desired starting stock size (with machining allowances)
    if desired_length is None:
        desired_length = part_L + 0.25  # +0.25" for face milling
    if desired_width is None:
        desired_width = part_W + 0.25
    if desired_thickness is None:
        desired_thickness = part_T + 0.125  # +0.125" for thickness

    if verbose:
        print(f"Desired stock: L={desired_length:.3f}\", W={desired_width:.3f}\", T={desired_thickness:.3f}\"")

    # Calculate volumes
    desired_volume = desired_length * desired_width * desired_thickness
    part_envelope_volume = part_L * part_W * part_T

    # Face milling scrap (desired → part envelope)
    face_milling_scrap = desired_volume - part_envelope_volume

    # Extract holes and calculate drilling scrap
    hole_table = extract_hole_table_from_cad(cad_file_path)

    total_hole_volume = 0.0
    hole_details = []

    for hole in hole_table:
        hole_id = hole.get('HOLE', '?')
        ref_diam_str = hole.get('REF_DIAM', '')
        qty = int(hole.get('QTY', 1))
        description = (hole.get('DESCRIPTION', '') or '').upper()

        # Parse diameter
        s = ref_diam_str.replace("Ø", "").replace("∅", "").strip()
        if "/" in s:
            try:
                dia = float(Fraction(s))
            except:
                dia = 0.0
        else:
            try:
                dia = float(s)
            except:
                dia = 0.0

        # Determine depth
        if 'THRU' in description:
            depth = part_T
        else:
            depth_match = re.search(r'[Xx]\s*([0-9.]+)\s*DEEP', description)
            if depth_match:
                depth = float(depth_match.group(1))
            else:
                depth = part_T * 0.5  # Default

        # Calculate volume for this hole group (cylinder)
        radius = dia / 2
        volume_per_hole = math.pi * (radius ** 2) * depth
        total_vol = volume_per_hole * qty
        total_hole_volume += total_vol

        hole_details.append({
            'id': hole_id,
            'diameter': dia,
            'depth': depth,
            'qty': qty,
            'volume_per_hole': volume_per_hole,
            'total_volume': total_vol,
            'description': description
        })

    if verbose:
        print(f"Holes analyzed: {len(hole_table)} types, total volume: {total_hole_volume:.4f} in³")

    # Calculate final part volume (after holes)
    part_final_volume = part_envelope_volume - total_hole_volume

    # Total machining scrap
    total_machining_scrap = face_milling_scrap + total_hole_volume

    return {
        'face_milling_scrap': face_milling_scrap,
        'hole_drilling_scrap': total_hole_volume,
        'total_machining_scrap': total_machining_scrap,
        'part_envelope_volume': part_envelope_volume,
        'part_final_volume': part_final_volume,
        'part_dimensions': {'L': part_L, 'W': part_W, 'T': part_T},
        'desired_dimensions': {'L': desired_length, 'W': desired_width, 'T': desired_thickness},
        'holes': hole_details
    }


def calculate_total_scrap(
    cad_file_path: str | Path,
    material: str = None,
    mcmaster_length: float = None,
    mcmaster_width: float = None,
    mcmaster_thickness: float = None,
    desired_length: float = None,
    desired_width: float = None,
    desired_thickness: float = None,
    catalog_csv_path: Optional[str] = None,
    verbose: bool = False
) -> ScrapInfo:
    """
    Calculate total scrap from a CAD file including stock prep and machining.

    This is the main scrap calculation function that combines:
    1. Stock prep scrap (McMaster stock → Desired starting size)
    2. Machining scrap (Desired → Final part with holes)

    Formula:
        Total Scrap = (McMaster stock volume) - (Final part volume with holes)

    Or equivalently:
        Total Scrap = Stock Prep Scrap + Face Milling Scrap + Hole Drilling Scrap

    Args:
        cad_file_path: Path to CAD file (DXF/DWG)
        material: Material name (for density lookup and McMaster lookup)
        mcmaster_length: McMaster stock length (if None, looks up from catalog)
        mcmaster_width: McMaster stock width (if None, looks up from catalog)
        mcmaster_thickness: McMaster stock thickness (if None, looks up from catalog)
        desired_length: Desired starting stock length (if None, uses part L + allowance)
        desired_width: Desired starting stock width (if None, uses part W + allowance)
        desired_thickness: Desired starting stock thickness (if None, uses part T + allowance)
        catalog_csv_path: Path to McMaster catalog CSV
        verbose: Print detailed output

    Returns:
        ScrapInfo dataclass with complete scrap breakdown

    Example:
        >>> # Auto-lookup McMaster stock size
        >>> scrap = calculate_total_scrap("part.dxf", material="aluminum MIC6")
        >>> print(f"Total scrap: {scrap.total_scrap_volume:.2f} in³")
        >>> print(f"Scrap weight: {scrap.total_scrap_weight:.2f} lbs")
        >>> print(f"Material utilization: {scrap.utilization_percentage:.1f}%")

        >>> # Or specify McMaster stock dimensions manually
        >>> scrap = calculate_total_scrap(
        ...     "part.dxf",
        ...     material="P20 Tool Steel",
        ...     mcmaster_length=16.0,
        ...     mcmaster_width=13.0,
        ...     mcmaster_thickness=2.5
        ... )
    """
    from cad_quoter.resources import default_catalog_csv
    from cad_quoter.pricing.mcmaster_helpers import (
        pick_mcmaster_plate_sku,
        load_mcmaster_catalog_rows
    )
    from cad_quoter.planning.process_planner import extract_dimensions_from_cad

    cad_file_path = Path(cad_file_path)

    # Use default material if not specified
    if material is None:
        material = DEFAULT_MATERIAL

    if verbose:
        print(f"\n{'='*70}")
        print(f"TOTAL SCRAP CALCULATION: {cad_file_path.name}")
        print(f"Material: {material}")
        print(f"{'='*70}\n")

    # Get part dimensions
    dims = extract_dimensions_from_cad(cad_file_path)
    if not dims:
        raise ValueError(f"Could not extract dimensions from {cad_file_path}")

    part_L, part_W, part_T = dims

    # Calculate desired stock dimensions if not provided
    if desired_length is None:
        desired_length = part_L + 0.25
    if desired_width is None:
        desired_width = part_W + 0.25
    if desired_thickness is None:
        desired_thickness = part_T + 0.125

    # Look up McMaster stock size if not provided
    if mcmaster_length is None or mcmaster_width is None or mcmaster_thickness is None:
        if verbose:
            print("Looking up McMaster stock size from catalog...")

        if catalog_csv_path is None:
            catalog_csv_path = str(default_catalog_csv())

        catalog_rows = load_mcmaster_catalog_rows(catalog_csv_path)
        result = pick_mcmaster_plate_sku(
            need_L_in=desired_length,
            need_W_in=desired_width,
            need_T_in=desired_thickness,
            material_key=material,
            catalog_rows=catalog_rows
        )

        if result:
            mcmaster_length = result.get('stock_L_in', desired_length)
            mcmaster_width = result.get('stock_W_in', desired_width)
            mcmaster_thickness = result.get('stock_T_in', desired_thickness)

            if verbose:
                print(f"Found McMaster stock: {mcmaster_length}\" x {mcmaster_width}\" x {mcmaster_thickness}\"")
        else:
            # Fallback: use desired dimensions
            mcmaster_length = desired_length
            mcmaster_width = desired_width
            mcmaster_thickness = desired_thickness
            if verbose:
                print(f"No McMaster stock found, using desired dimensions")

    # Calculate machining scrap
    if verbose:
        print("\nCalculating machining scrap...")

    machining = calculate_machining_scrap_from_cad(
        cad_file_path,
        desired_length=desired_length,
        desired_width=desired_width,
        desired_thickness=desired_thickness,
        verbose=False
    )

    # Calculate stock prep scrap
    stock_prep_scrap = calculate_stock_prep_scrap(
        mcmaster_length, mcmaster_width, mcmaster_thickness,
        desired_length, desired_width, desired_thickness
    )

    # Calculate total scrap
    mcmaster_volume = mcmaster_length * mcmaster_width * mcmaster_thickness
    desired_volume = desired_length * desired_width * desired_thickness

    total_scrap_volume = (
        stock_prep_scrap +
        machining['face_milling_scrap'] +
        machining['hole_drilling_scrap']
    )

    # Verify math
    final_part_volume = machining['part_final_volume']
    assert abs((mcmaster_volume - total_scrap_volume) - final_part_volume) < 0.01, \
        "Scrap calculation error: volumes don't add up"

    # Calculate weights
    density = get_material_density(material)
    mcmaster_weight = mcmaster_volume * density
    final_part_weight = final_part_volume * density
    total_scrap_weight = total_scrap_volume * density

    # Calculate percentages
    scrap_percentage = (total_scrap_volume / mcmaster_volume * 100) if mcmaster_volume > 0 else 0
    utilization_percentage = (final_part_volume / mcmaster_volume * 100) if mcmaster_volume > 0 else 0

    if verbose:
        print(f"\nScrap Breakdown:")
        print(f"  Stock prep scrap: {stock_prep_scrap:.4f} in³")
        print(f"  Face milling scrap: {machining['face_milling_scrap']:.4f} in³")
        print(f"  Hole drilling scrap: {machining['hole_drilling_scrap']:.4f} in³")
        print(f"  TOTAL SCRAP: {total_scrap_volume:.4f} in³ ({scrap_percentage:.1f}%)")
        print(f"\nMaterial Utilization: {utilization_percentage:.1f}%")
        print(f"Scrap weight: {total_scrap_weight:.2f} lbs")
        print(f"{'='*70}\n")

    return ScrapInfo(
        mcmaster_length=mcmaster_length,
        mcmaster_width=mcmaster_width,
        mcmaster_thickness=mcmaster_thickness,
        mcmaster_volume=mcmaster_volume,
        desired_length=desired_length,
        desired_width=desired_width,
        desired_thickness=desired_thickness,
        desired_volume=desired_volume,
        part_length=part_L,
        part_width=part_W,
        part_thickness=part_T,
        part_envelope_volume=machining['part_envelope_volume'],
        part_final_volume=final_part_volume,
        stock_prep_scrap=stock_prep_scrap,
        face_milling_scrap=machining['face_milling_scrap'],
        hole_drilling_scrap=machining['hole_drilling_scrap'],
        total_scrap_volume=total_scrap_volume,
        material=material,
        density=density,
        mcmaster_weight=mcmaster_weight,
        final_part_weight=final_part_weight,
        total_scrap_weight=total_scrap_weight,
        scrap_percentage=scrap_percentage,
        utilization_percentage=utilization_percentage
    )


def calculate_scrap_value(
    scrap_weight_lbs: float,
    material: str,
    fallback_scrap_price_per_lb: Optional[float] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Calculate the dollar value of scrap using Wieland scrap prices.

    Args:
        scrap_weight_lbs: Weight of scrap in pounds
        material: Material name (e.g., "aluminum MIC6", "P20 Tool Steel", "17-4 PH Stainless")
        fallback_scrap_price_per_lb: Fallback scrap price if Wieland lookup fails
        verbose: Print detailed output

    Returns:
        Dict with keys:
            - scrap_weight_lbs: Weight of scrap in pounds
            - scrap_price_per_lb: Price per pound (USD/lb)
            - scrap_value: Total scrap value in dollars
            - price_source: Source of scrap price
            - material_family: Detected material family (aluminum, steel, stainless, etc.)

    Example:
        >>> value_info = calculate_scrap_value(21.19, "aluminum MIC6")
        >>> print(f"Scrap value: ${value_info['scrap_value']:.2f}")
        >>> print(f"Price source: {value_info['price_source']}")
    """
    from cad_quoter.pricing.wieland_scraper import get_scrap_price_per_lb

    # Determine material family from material string
    material_lower = material.lower()

    if any(kw in material_lower for kw in ["aluminum", "aluminium", "6061", "7075", "2024", "mic6"]):
        material_family = "aluminum"
    elif any(kw in material_lower for kw in ["stainless", "304", "316", "17-4", "17 4"]):
        material_family = "stainless"
    elif any(kw in material_lower for kw in ["steel", "p20", "a36", "1018", "1045", "tool steel"]):
        material_family = "steel"
    elif any(kw in material_lower for kw in ["copper", "cu", "c110"]):
        material_family = "copper"
    elif any(kw in material_lower for kw in ["brass", "bronze"]):
        material_family = "brass"
    elif any(kw in material_lower for kw in ["titanium", "ti"]):
        material_family = "titanium"
    else:
        material_family = "aluminum"  # Default

    if verbose:
        print(f"Detected material family: {material_family}")

    # Look up scrap price from Wieland
    scrap_price_per_lb = get_scrap_price_per_lb(material_family, fallback=fallback_scrap_price_per_lb)

    if scrap_price_per_lb is None:
        if verbose:
            print(f"Warning: No scrap price found for {material_family}")
        price_source = "No price available"
        scrap_value = 0.0
    else:
        price_source = f"Wieland {material_family} scrap"
        scrap_value = scrap_weight_lbs * scrap_price_per_lb

        if verbose:
            print(f"Scrap price: ${scrap_price_per_lb:.4f}/lb")
            print(f"Scrap value: ${scrap_value:.2f}")

    return {
        'scrap_weight_lbs': scrap_weight_lbs,
        'scrap_price_per_lb': scrap_price_per_lb,
        'scrap_value': scrap_value,
        'price_source': price_source,
        'material_family': material_family
    }


def calculate_total_scrap_with_value(
    cad_file_path: str | Path,
    material: str = None,
    mcmaster_length: float = None,
    mcmaster_width: float = None,
    mcmaster_thickness: float = None,
    desired_length: float = None,
    desired_width: float = None,
    desired_thickness: float = None,
    catalog_csv_path: Optional[str] = None,
    fallback_scrap_price_per_lb: Optional[float] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Calculate total scrap including dollar value using Wieland scrap prices.

    This is a convenience function that combines calculate_total_scrap() and
    calculate_scrap_value() to provide a complete scrap analysis.

    Args:
        cad_file_path: Path to CAD file (DXF/DWG)
        material: Material name (for density lookup and McMaster lookup)
        mcmaster_length: McMaster stock length (if None, looks up from catalog)
        mcmaster_width: McMaster stock width (if None, looks up from catalog)
        mcmaster_thickness: McMaster stock thickness (if None, looks up from catalog)
        desired_length: Desired starting stock length (if None, uses part L + allowance)
        desired_width: Desired starting stock width (if None, uses part W + allowance)
        desired_thickness: Desired starting stock thickness (if None, uses part T + allowance)
        catalog_csv_path: Path to McMaster catalog CSV
        fallback_scrap_price_per_lb: Fallback scrap price if Wieland lookup fails
        verbose: Print detailed output

    Returns:
        Dict with keys:
            - scrap_info: ScrapInfo dataclass with complete scrap breakdown
            - scrap_value_info: Dict with scrap value details

    Example:
        >>> result = calculate_total_scrap_with_value("part.dxf", material="aluminum MIC6")
        >>> scrap = result['scrap_info']
        >>> value = result['scrap_value_info']
        >>> print(f"Total scrap: {scrap.total_scrap_volume:.2f} in³ ({scrap.total_scrap_weight:.2f} lbs)")
        >>> print(f"Scrap value: ${value['scrap_value']:.2f}")
        >>> print(f"Net material cost: ${result['net_material_cost']:.2f}")
    """
    # Calculate scrap volume and weight
    scrap_info = calculate_total_scrap(
        cad_file_path=cad_file_path,
        material=material,
        mcmaster_length=mcmaster_length,
        mcmaster_width=mcmaster_width,
        mcmaster_thickness=mcmaster_thickness,
        desired_length=desired_length,
        desired_width=desired_width,
        desired_thickness=desired_thickness,
        catalog_csv_path=catalog_csv_path,
        verbose=verbose
    )

    # Calculate scrap value
    scrap_value_info = calculate_scrap_value(
        scrap_weight_lbs=scrap_info.total_scrap_weight,
        material=scrap_info.material,
        fallback_scrap_price_per_lb=fallback_scrap_price_per_lb,
        verbose=verbose
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"SCRAP VALUE SUMMARY")
        print(f"{'='*70}")
        print(f"Material: {scrap_info.material}")
        print(f"Material family: {scrap_value_info['material_family']}")
        print(f"\nScrap breakdown:")
        print(f"  Stock prep: {scrap_info.stock_prep_scrap:.4f} in³")
        print(f"  Face milling: {scrap_info.face_milling_scrap:.4f} in³")
        print(f"  Hole drilling: {scrap_info.hole_drilling_scrap:.4f} in³")
        print(f"  Total: {scrap_info.total_scrap_volume:.4f} in³ ({scrap_info.total_scrap_weight:.2f} lbs)")
        print(f"\nScrap value:")
        print(f"  Price: ${scrap_value_info['scrap_price_per_lb']:.4f}/lb")
        print(f"  Source: {scrap_value_info['price_source']}")
        print(f"  Total value: ${scrap_value_info['scrap_value']:.2f}")
        print(f"{'='*70}\n")

    return {
        'scrap_info': scrap_info,
        'scrap_value_info': scrap_value_info,
    }
