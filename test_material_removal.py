"""
Test script to calculate material removal from a CAD part.

This demonstrates how to use process_planner.py functions to:
1. Extract part dimensions
2. Extract hole information
3. Calculate total material removed
"""

import sys
from pathlib import Path
import math

# Add the cad_quoter module to path
sys.path.insert(0, str(Path(__file__).parent))

from cad_quoter.planning.process_planner import (
    extract_dimensions_from_cad,
    extract_hole_table_from_cad,
    plan_from_cad_file
)


def calculate_hole_volume(diameter: float, depth: float, qty: int = 1) -> float:
    """
    Calculate volume of material removed by drilling holes.

    Args:
        diameter: Hole diameter in inches
        depth: Hole depth in inches
        qty: Number of holes

    Returns:
        Volume in cubic inches
    """
    radius = diameter / 2
    volume_per_hole = math.pi * (radius ** 2) * depth
    return volume_per_hole * qty


def parse_diameter(ref_diam_str: str) -> float:
    """Parse diameter from string like 'Ø0.7500' or '1/2'."""
    import re
    from fractions import Fraction

    # Remove Ø symbol
    s = ref_diam_str.replace("Ø", "").replace("∅", "").strip()

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


def calculate_material_removal(
    cad_file: str,
    stock_length: float = None,
    stock_width: float = None,
    stock_thickness: float = None,
    verbose: bool = True
):
    """
    Calculate material removed from a part.

    Args:
        cad_file: Path to CAD file (DXF or DWG)
        stock_length: Starting stock length (if different from part L)
        stock_width: Starting stock width (if different from part W)
        stock_thickness: Starting stock thickness (if different from part T)
        verbose: Print detailed output

    Returns:
        Dictionary with material removal calculations
    """
    cad_file = Path(cad_file)

    if not cad_file.exists():
        raise FileNotFoundError(f"CAD file not found: {cad_file}")

    print(f"\n{'='*70}")
    print(f"MATERIAL REMOVAL ANALYSIS: {cad_file.name}")
    print(f"{'='*70}\n")

    # 1. Extract part dimensions
    print("1. Extracting dimensions...")
    dims = extract_dimensions_from_cad(cad_file)

    if dims:
        part_L, part_W, part_T = dims
        print(f"   Part dimensions: L={part_L:.3f}\", W={part_W:.3f}\", T={part_T:.3f}\"")
    else:
        print("   Warning: Could not extract dimensions automatically")
        # Use provided stock dimensions or defaults
        part_L = stock_length or 8.0
        part_W = stock_width or 4.0
        part_T = stock_thickness or 1.0
        print(f"   Using default dimensions: L={part_L:.3f}\", W={part_W:.3f}\", T={part_T:.3f}\"")

    # 2. Determine stock size
    if stock_length is None:
        stock_length = part_L + 0.25  # Add 0.25" for face milling allowance
    if stock_width is None:
        stock_width = part_W + 0.25
    if stock_thickness is None:
        stock_thickness = part_T + 0.125  # Add 0.125" for thickness allowance

    print(f"\n2. Stock dimensions:")
    print(f"   Stock: L={stock_length:.3f}\", W={stock_width:.3f}\", T={stock_thickness:.3f}\"")

    # 3. Calculate stock volume
    stock_volume = stock_length * stock_width * stock_thickness
    final_part_volume = part_L * part_W * part_T  # This is the solid volume before holes

    print(f"\n3. Volume calculations:")
    print(f"   Stock volume: {stock_volume:.4f} in³")
    print(f"   Part envelope volume: {final_part_volume:.4f} in³")

    # 4. Extract holes and calculate volume removed by drilling
    print(f"\n4. Analyzing holes...")
    hole_table = extract_hole_table_from_cad(cad_file)

    total_hole_volume = 0.0
    hole_details = []

    for hole in hole_table:
        hole_id = hole.get('HOLE', '?')
        ref_diam_str = hole.get('REF_DIAM', '')
        qty = int(hole.get('QTY', 1))
        description = (hole.get('DESCRIPTION', '') or '').upper()

        # Parse diameter
        dia = parse_diameter(ref_diam_str)

        # Determine depth
        if 'THRU' in description:
            depth = part_T
        else:
            # Try to extract depth from description (e.g., "X .25 DEEP")
            import re
            depth_match = re.search(r'[Xx]\s*([0-9.]+)\s*DEEP', description)
            if depth_match:
                depth = float(depth_match.group(1))
            else:
                depth = part_T * 0.5  # Default to half thickness

        # Calculate volume for this hole group
        hole_vol = calculate_hole_volume(dia, depth, qty)
        total_hole_volume += hole_vol

        hole_details.append({
            'id': hole_id,
            'diameter': dia,
            'depth': depth,
            'qty': qty,
            'volume_per_hole': hole_vol / qty if qty > 0 else 0,
            'total_volume': hole_vol,
            'description': description
        })

        if verbose:
            print(f"   Hole {hole_id}: Ø{dia:.4f}\" x {depth:.3f}\" deep, Qty={qty}")
            print(f"      Volume removed: {hole_vol:.4f} in³ ({hole_vol/qty:.4f} in³ each)")

    # 5. Calculate total material removed
    print(f"\n5. Material removal summary:")

    # Face milling removal (stock to part envelope)
    face_removal = stock_volume - final_part_volume
    print(f"   Face milling removal: {face_removal:.4f} in³")

    # Hole drilling removal
    print(f"   Hole drilling removal: {total_hole_volume:.4f} in³")

    # Total removal
    total_removal = face_removal + total_hole_volume
    print(f"   TOTAL MATERIAL REMOVED: {total_removal:.4f} in³")

    # Percentage removed
    removal_percent = (total_removal / stock_volume) * 100
    print(f"   Material removed: {removal_percent:.1f}% of stock")

    # Final part weight (if we know material density)
    final_volume_with_holes = final_part_volume - total_hole_volume
    print(f"   Final part volume: {final_volume_with_holes:.4f} in³")

    print(f"\n{'='*70}\n")

    return {
        'stock_volume': stock_volume,
        'stock_dimensions': {'L': stock_length, 'W': stock_width, 'T': stock_thickness},
        'part_envelope_volume': final_part_volume,
        'part_dimensions': {'L': part_L, 'W': part_W, 'T': part_T},
        'face_removal': face_removal,
        'hole_removal': total_hole_volume,
        'total_removal': total_removal,
        'removal_percent': removal_percent,
        'final_part_volume': final_volume_with_holes,
        'holes': hole_details
    }


def estimate_weight(volume_cubic_inches: float, material: str = "Steel") -> float:
    """
    Estimate weight based on volume and material.

    Common densities (lb/in³):
    - Steel: 0.283
    - Aluminum: 0.098
    - Stainless Steel: 0.286
    - Titanium: 0.163
    - Brass: 0.308
    """
    densities = {
        'steel': 0.283,
        'aluminum': 0.098,
        'stainless': 0.286,
        'titanium': 0.163,
        'brass': 0.308,
    }

    material_key = material.lower()
    for key in densities:
        if key in material_key:
            density = densities[key]
            return volume_cubic_inches * density

    # Default to steel
    return volume_cubic_inches * 0.283


if __name__ == "__main__":
    # Test with available CAD file
    cad_file = r"D:\CAD_Quoting_Tool\Cad Files\301_redacted.dxf"

    # Calculate material removal
    result = calculate_material_removal(
        cad_file,
        # Optional: provide custom stock dimensions
        # stock_length=9.0,
        # stock_width=5.0,
        # stock_thickness=2.5,
        verbose=True
    )

    # Estimate weight
    print("\n6. Weight estimation:")
    material = "Steel"
    stock_weight = estimate_weight(result['stock_volume'], material)
    final_weight = estimate_weight(result['final_part_volume'], material)
    chips_weight = estimate_weight(result['total_removal'], material)

    print(f"   Material: {material}")
    print(f"   Stock weight: {stock_weight:.2f} lbs")
    print(f"   Final part weight: {final_weight:.2f} lbs")
    print(f"   Chips/scrap: {chips_weight:.2f} lbs")
    print(f"   Material utilization: {(final_weight/stock_weight)*100:.1f}%")
