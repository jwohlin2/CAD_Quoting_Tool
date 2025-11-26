"""Shared helpers for locating McMaster plate SKUs from the catalog CSV."""

from __future__ import annotations

import csv
import os
from fractions import Fraction
from typing import Any, Mapping, Sequence

from cad_quoter.resources import default_catalog_csv
from cad_quoter.vendors.mcmaster_stock import parse_inches as _parse_inches
from cad_quoter.pricing.MaterialMapper import material_mapper


def load_mcmaster_catalog_rows(path: str | None = None) -> list[dict[str, Any]]:
    """Return rows from the McMaster stock catalog CSV."""

    csv_path = path or os.getenv("CATALOG_CSV_PATH") or str(default_catalog_csv())
    if not csv_path:
        return []
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader if row]
    except FileNotFoundError as e:
        print(f"[Catalog Load] ERROR: File not found: {csv_path}")
        print(f"  Exception: {e}")
        return []
    except Exception as e:
        print(f"[Catalog Load] ERROR: Failed to load catalog from {csv_path}")
        print(f"  Exception: {e}")
        return []


def _load_catalog_indexed_by_material() -> dict[str, list[dict[str, Any]]]:
    """Load catalog and pre-index by material for O(1) lookups.

    Returns a dictionary mapping normalized material keys to lists of matching rows.
    Each material is indexed under multiple variants (with/without underscores/spaces).
    """
    rows = load_mcmaster_catalog_rows()
    indexed: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        material_text = str(
            (row.get("material") or row.get("Material") or "")
        ).strip().lower()
        if not material_text:
            continue

        # Index under multiple normalized variants
        material_norm = material_text.replace("_", " ")
        material_compact = material_text.replace(" ", "").replace("_", "")

        for key in [material_text, material_norm, material_compact]:
            if key:
                if key not in indexed:
                    indexed[key] = []
                # Avoid duplicates in same key
                if row not in indexed[key]:
                    indexed[key].append(row)

    return indexed


def _get_material_candidates(
    target_key: str,
    catalog_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Get catalog rows matching a material key using indexed lookup.

    Falls back to linear search if custom catalog_rows are provided.
    """
    # If custom rows provided, fall back to linear search
    if catalog_rows is not None:
        rows = list(catalog_rows)
        target_lower = target_key.strip().lower()

        # Build variants once
        variants = {target_lower}
        if "_" in target_lower:
            variants.add(target_lower.replace("_", " "))
        if " " in target_lower:
            variants.add(target_lower.replace(" ", ""))
        variants = [v for v in variants if v]

        candidates = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            material_text = str(
                (row.get("material") or row.get("Material") or "")
            ).strip().lower()
            if not material_text:
                continue
            normalised_material = material_text.replace("_", " ")
            if any(variant in normalised_material for variant in variants):
                candidates.append(dict(row) if isinstance(row, Mapping) else row)
        return candidates

    # Use indexed lookup for default catalog (O(1) instead of O(N))
    indexed = _load_catalog_indexed_by_material()
    target_lower = target_key.strip().lower()

    # Try different variants of the target key
    variants_to_try = [
        target_lower,
        target_lower.replace("_", " "),
        target_lower.replace(" ", ""),
        target_lower.replace(" ", "_"),
    ]

    # Collect all matching rows (deduplicated)
    seen_ids: set[int] = set()
    candidates: list[dict[str, Any]] = []

    for variant in variants_to_try:
        for row in indexed.get(variant, []):
            row_id = id(row)
            if row_id not in seen_ids:
                seen_ids.add(row_id)
                candidates.append(row)

    return candidates


def _coerce_inches_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace("\u00a0", " ")
    if not text:
        return None

    try:
        parsed = float(_parse_inches(text))
    except Exception:
        parsed = None
    if parsed is not None and parsed > 0:
        return parsed

    try:
        return float(text)
    except Exception:
        pass

    try:
        return float(Fraction(text))
    except Exception:
        return None


def _pick_mcmaster_plate_sku_impl(
    need_L_in: float,
    need_W_in: float,
    need_T_in: float,
    *,
    material_key: str = "MIC6",
    catalog_rows: Sequence[Mapping[str, Any]] | None = None,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Return the smallest-area McMaster plate covering the requested envelope."""

    import math as _math

    if verbose:
        print(f"[McMaster Lookup] Searching for: {need_L_in:.3f} x {need_W_in:.3f} x {need_T_in:.3f} in")
        print(f"[McMaster Lookup] Material key: '{material_key}'")

    if not all(val > 0 for val in (need_L_in, need_W_in, need_T_in)):
        if verbose:
            print(f"[McMaster Lookup] ERROR: Invalid dimensions")
        return None

    rows = list(catalog_rows) if catalog_rows is not None else load_mcmaster_catalog_rows()
    if not rows:
        if verbose:
            print(f"[McMaster Lookup] ERROR: No catalog rows loaded")
        return None

    if verbose:
        print(f"[McMaster Lookup] Loaded {len(rows)} catalog rows")

    target_key = str(material_key or "").strip().lower()
    if not target_key:
        if verbose:
            print(f"[McMaster Lookup] ERROR: Empty material key")
        return None

    # Tolerance for thickness matching - allow rounding UP to next standard stock thickness
    # Standard stock comes in fixed increments (0.25", 0.5", etc.), so we need enough
    # tolerance to match the next available size (e.g., 2.125" need → 2.25" stock)
    # IMPORTANT: Stock must be >= needed thickness (never thinner)
    tolerance = 0.5  # Allow rounding up by up to 0.5" to next standard thickness
    candidates: list[dict[str, Any]] = []
    thickness_matches = 0

    # Use indexed lookup for material candidates (O(1) instead of O(N))
    material_rows = _get_material_candidates(target_key, catalog_rows)
    material_matches = len(material_rows)

    for row in material_rows:
        length = _coerce_inches_value(
            row.get("length_in")
            or row.get("L_in")
            or row.get("len_in")
            or row.get("length")
        )
        width = _coerce_inches_value(
            row.get("width_in")
            or row.get("W_in")
            or row.get("wid_in")
            or row.get("width")
        )
        thickness = _coerce_inches_value(
            row.get("thickness_in")
            or row.get("T_in")
            or row.get("thk_in")
            or row.get("thickness")
        )
        if (
            length is None
            or width is None
            or thickness is None
            or length <= 0
            or width <= 0
            or thickness <= 0
        ):
            continue
        # Stock thickness must be >= needed thickness (only round UP, never down)
        if thickness < need_T_in or (thickness - need_T_in) > tolerance:
            continue

        # Thickness matches within tolerance
        thickness_matches += 1

        part_no = str(
            row.get("mcmaster_part")
            or row.get("part")
            or row.get("sku")
            or ""
        ).strip()
        if not part_no:
            continue

        def _covers(a: float, b: float, A: float, B: float) -> bool:
            return (A >= a) and (B >= b)

        ok1 = _covers(need_L_in, need_W_in, length, width)
        ok2 = _covers(need_L_in, need_W_in, width, length)
        if not (ok1 or ok2):
            continue

        area = length * width
        overL1 = (length - need_L_in) if ok1 else _math.inf
        overW1 = (width - need_W_in) if ok1 else _math.inf
        overL2 = (width - need_L_in) if ok2 else _math.inf
        overW2 = (length - need_W_in) if ok2 else _math.inf
        over_L = min(overL1, overL2)
        over_W = min(overW1, overW2)

        # Determine which orientation to use and store dimensions accordingly
        # If ok1, use original orientation (length covers need_L, width covers need_W)
        # If ok2, use rotated orientation (width covers need_L, length covers need_W)
        # Prefer ok1 if both work (less material waste in original orientation)
        if ok1:
            stock_L = float(length)
            stock_W = float(width)
        else:  # ok2
            # Swap dimensions to reflect the rotated orientation
            stock_L = float(width)
            stock_W = float(length)

        candidates.append(
            {
                "len_in": stock_L,
                "wid_in": stock_W,
                "thk_in": float(thickness),
                "mcmaster_part": part_no,
                "area": float(area),
                "overL": float(over_L),
                "overW": float(over_W),
                "source": row.get("source") or "mcmaster-catalog",
            }
        )

    if verbose:
        print(f"[McMaster Lookup] Material matches: {material_matches}")
        print(f"[McMaster Lookup] Thickness matches (within {tolerance} in): {thickness_matches}")
        print(f"[McMaster Lookup] Final candidates (covers L×W): {len(candidates)}")

    if not candidates:
        if verbose:
            print(f"[McMaster Lookup] ERROR: No matching catalog entries found")
        return None

    candidates.sort(key=lambda c: (c["area"], max(c["overL"], c["overW"])))
    best = candidates[0]

    # CRITICAL FIX: Ensure the selected stock is NEVER smaller than required
    # (Fix for rod stock rounding down bug - applies to plate stock too for consistency)
    # This is a safety check to catch any edge cases that slip through the filtering
    best_L = float(best["len_in"])
    best_W = float(best["wid_in"])
    best_T = float(best["thk_in"])

    # Check if stock covers the required dimensions (either orientation)
    covers_as_is = (best_L >= need_L_in and best_W >= need_W_in)
    covers_rotated = (best_L >= need_W_in and best_W >= need_L_in)
    thickness_ok = best_T >= need_T_in

    if not ((covers_as_is or covers_rotated) and thickness_ok):
        if verbose:
            print(f"[McMaster Lookup] WARNING: Best candidate is smaller than required!")
            print(f"  Candidate: {best_L:.3f} × {best_W:.3f} × {best_T:.3f} in")
            print(f"  Required: {need_L_in:.3f} × {need_W_in:.3f} × {need_T_in:.3f} in")
            print(f"  Rejecting this candidate to prevent impossible stock sizes")
        return None  # Better to return None than return stock that's too small

    if verbose:
        print(f"[McMaster Lookup] FOUND: {best_L} × {best_W} × {best_T} in")
        print(f"[McMaster Lookup] Part: {best['mcmaster_part']}")

    return {
        "stock_L_in": best_L,
        "stock_W_in": best_W,
        "stock_T_in": best_T,
        "mcmaster_part": best["mcmaster_part"],
        "source": best.get("source") or "mcmaster-catalog",
    }


def pick_mcmaster_cylindrical_sku(
    need_diam_in: float,
    need_length_in: float,
    *,
    material_key: str = "303 Stainless Steel",
    catalog_rows: Sequence[Mapping[str, Any]] | None = None,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Return the smallest McMaster cylindrical stock covering the requested dimensions.

    For cylindrical parts (guide posts, spring pins, etc.), we match by diameter and length
    instead of plate dimensions (L × W × T).

    Args:
        need_diam_in: Required diameter in inches
        need_length_in: Required length in inches
        material_key: Material identifier (e.g., "303 Stainless Steel")
        catalog_rows: Optional pre-loaded catalog rows
        verbose: Print debug information

    Returns:
        Dict with stock info or None if no match found
    """
    if verbose:
        print(f"[Cylindrical Lookup] Searching for: diam={need_diam_in:.3f} in, length={need_length_in:.3f} in")
        print(f"[Cylindrical Lookup] Material key: '{material_key}'")

    if not (need_diam_in > 0 and need_length_in > 0):
        if verbose:
            print(f"[Cylindrical Lookup] ERROR: Invalid dimensions")
        return None

    rows = list(catalog_rows) if catalog_rows is not None else load_mcmaster_catalog_rows()
    if not rows:
        if verbose:
            print(f"[Cylindrical Lookup] ERROR: No catalog rows loaded")
        return None

    # Map material to McMaster catalog key
    mcmaster_material = material_mapper.get_mcmaster_key(material_key) or material_key
    target_key = str(mcmaster_material or "").strip().lower()

    if not target_key:
        if verbose:
            print(f"[Cylindrical Lookup] ERROR: Empty material key")
        return None

    # Get material candidates
    material_rows = _get_material_candidates(target_key, catalog_rows)
    material_matches = len(material_rows)

    if verbose:
        print(f"[Cylindrical Lookup] Material matches: {material_matches}")

    candidates: list[dict[str, Any]] = []

    # Tolerance for diameter matching (allow rounding up to next standard size)
    # For small diameters (<2"), use 0.5" tolerance to reach next standard size
    # For larger diameters (>=2"), use 0.25" tolerance
    diam_tolerance = 0.5 if need_diam_in < 2.0 else 0.25
    length_tolerance = 12.0  # Allow up to 12" longer stock (round bar comes in standard lengths like 12", 36")

    for row in material_rows:
        # Check for diam_in column (cylindrical parts)
        diameter = _coerce_inches_value(
            row.get("diam_in") or row.get("diameter_in") or row.get("diameter")
        )

        # Skip if no diameter specified (not a cylindrical part)
        if diameter is None or diameter <= 0:
            continue

        length = _coerce_inches_value(
            row.get("length_in")
            or row.get("L_in")
            or row.get("len_in")
            or row.get("length")
        )

        if length is None or length <= 0:
            continue

        # Stock must be >= needed dimensions (only round UP, never down)
        if diameter < need_diam_in or (diameter - need_diam_in) > diam_tolerance:
            continue
        if length < need_length_in or (length - need_length_in) > length_tolerance:
            continue

        part_no = str(
            row.get("mcmaster_part")
            or row.get("part")
            or row.get("sku")
            or ""
        ).strip()

        if not part_no:
            continue

        # Calculate material waste metrics
        import math as _math
        stock_volume = _math.pi * (diameter / 2.0) ** 2 * length
        needed_volume = _math.pi * (need_diam_in / 2.0) ** 2 * need_length_in
        waste_volume = stock_volume - needed_volume

        candidates.append({
            "diam_in": float(diameter),
            "len_in": float(length),
            "mcmaster_part": part_no,
            "waste_volume": float(waste_volume),
            "stock_volume": float(stock_volume),
            "source": row.get("source") or "mcmaster-catalog",
        })

    if verbose:
        print(f"[Cylindrical Lookup] Final candidates: {len(candidates)}")

    if not candidates:
        if verbose:
            print(f"[Cylindrical Lookup] ERROR: No matching cylindrical parts found")
        return None

    # Sort by waste volume (smallest waste first)
    candidates.sort(key=lambda c: c["waste_volume"])
    best = candidates[0]

    # CRITICAL FIX: Ensure the selected stock is NEVER smaller than required
    # (Fix for rod stock rounding down bug - T1769-219 & T1769-134)
    # This is a safety check to catch any edge cases that slip through the filtering
    best_diam = float(best["diam_in"])
    best_len = float(best["len_in"])

    if best_diam < need_diam_in or best_len < need_length_in:
        if verbose:
            print(f"[Cylindrical Lookup] WARNING: Best candidate is smaller than required!")
            print(f"  Candidate: diam={best_diam:.3f} in, length={best_len:.3f} in")
            print(f"  Required: diam={need_diam_in:.3f} in, length={need_length_in:.3f} in")
            print(f"  Rejecting this candidate to prevent impossible stock sizes")
        return None  # Better to return None than return stock that's too small

    if verbose:
        print(f"[Cylindrical Lookup] FOUND: diam={best_diam:.3f} in, length={best_len:.3f} in")
        print(f"[Cylindrical Lookup] Part: {best['mcmaster_part']}")

    return {
        "stock_diam_in": best_diam,
        "stock_L_in": best_len,
        "mcmaster_part": best["mcmaster_part"],
        "source": best.get("source") or "mcmaster-catalog",
    }


def pick_mcmaster_plate_sku(
    need_L_in: float,
    need_W_in: float,
    need_T_in: float,
    *,
    material_key: str = "MIC6",
    catalog_rows: Sequence[Mapping[str, Any]] | None = None,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Return the smallest-area McMaster plate covering the requested envelope."""

    # Map material to McMaster catalog key (e.g., "316" -> "303 Stainless Steel", "A2" -> "Tool Steel A2")
    mcmaster_material = material_mapper.get_mcmaster_key(material_key) or material_key

    return _pick_mcmaster_plate_sku_impl(
        need_L_in,
        need_W_in,
        need_T_in,
        material_key=mcmaster_material,
        catalog_rows=catalog_rows,
        verbose=verbose,
    )


def find_largest_catalog_part_for_material(
    material_key: str,
    max_thickness_in: float = 6.0,
    catalog_rows: Sequence[Mapping[str, Any]] | None = None,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Find the largest available catalog part for a given material.

    This is useful for estimating prices for oversized parts by finding
    the closest available catalog part and using its $/lb as a reference.

    Args:
        material_key: Material identifier (e.g., "aluminum MIC6", "303 Stainless Steel")
        max_thickness_in: Maximum thickness to consider (default 6.0 in) - helps find planar plates
        catalog_rows: Optional pre-loaded catalog rows
        verbose: Print debug information

    Returns:
        Dict with catalog part info (len_in, wid_in, thk_in, part), or None if not found
    """
    rows = list(catalog_rows) if catalog_rows is not None else load_mcmaster_catalog_rows()
    if not rows:
        if verbose:
            print(f"[Largest Part Lookup] ERROR: No catalog rows loaded")
        return None

    target_key = str(material_key or "").strip().lower()
    if not target_key:
        if verbose:
            print(f"[Largest Part Lookup] ERROR: Empty material key")
        return None

    # Find all matching material entries with reasonable thickness
    matches = []
    for row in rows:
        mat = str(row.get("material", "")).strip().lower()
        if target_key not in mat:
            continue

        len_in = _coerce_inches_value(row.get("length_in"))
        wid_in = _coerce_inches_value(row.get("width_in"))
        thk_in = _coerce_inches_value(row.get("thickness_in"))
        part = str(row.get("part", "")).strip()

        # Skip if thickness is too large (avoids very thick blocks)
        if thk_in and thk_in > max_thickness_in:
            continue

        if len_in and wid_in and thk_in and len_in > 0 and wid_in > 0 and thk_in > 0:
            area = float(len_in) * float(wid_in)
            volume = area * float(thk_in)
            matches.append({
                "len_in": float(len_in),
                "wid_in": float(wid_in),
                "thk_in": float(thk_in),
                "part": part,
                "area": area,
                "volume": volume,
            })

    if not matches:
        if verbose:
            print(f"[Largest Part Lookup] ERROR: No matching parts found for material '{material_key}' with thickness <= {max_thickness_in} in")
        return None

    # Sort by area (largest planar dimension), then by volume
    matches.sort(key=lambda x: (x["area"], x["volume"]), reverse=True)
    largest = matches[0]

    if verbose:
        print(f"[Largest Part Lookup] Found largest part for '{material_key}':")
        print(f"  {largest['len_in']:.2f} × {largest['wid_in']:.2f} × {largest['thk_in']:.3f} in")
        print(f"  Part: {largest['part']}")
        print(f"  Area: {largest['area']:.2f} in²")
        print(f"  Volume: {largest['volume']:.2f} in³")

    return largest


def estimate_price_from_catalog_reference(
    material_key: str,
    weight_lb: float,
    density_lb_in3: float = 0.10,  # Default for aluminum
    catalog_rows: Sequence[Mapping[str, Any]] | None = None,
    verbose: bool = False,
) -> tuple[float | None, str]:
    """Estimate price for an oversized part using the largest catalog part's $/lb.

    Args:
        material_key: Material identifier
        weight_lb: Weight of the part in pounds
        density_lb_in3: Material density in lb/in³
        catalog_rows: Optional pre-loaded catalog rows
        verbose: Print debug information

    Returns:
        Tuple of (estimated_price, source_description) or (None, "") if cannot estimate
    """
    # Map material to McMaster catalog key
    mcmaster_material = material_mapper.get_mcmaster_key(material_key) or material_key

    # Find the largest catalog part for this material
    largest_part = find_largest_catalog_part_for_material(
        mcmaster_material,
        catalog_rows=catalog_rows,
        verbose=verbose
    )

    if not largest_part:
        if verbose:
            print(f"[Price Estimate] Cannot estimate - no catalog parts found for '{material_key}'")
        return None, ""

    # Try to get price for the largest catalog part
    try:
        from cad_quoter.vendors.mcmaster_stock import lookup_sku_and_price_for_mm

        part_num = largest_part.get("part")
        if not part_num:
            if verbose:
                print(f"[Price Estimate] No part number for largest catalog item")
            return None, ""

        # Get price for the catalog part
        sku, price_each, uom, dims = lookup_sku_and_price_for_mm(
            mcmaster_material,
            largest_part["len_in"] * 25.4,  # Convert to mm
            largest_part["wid_in"] * 25.4,
            largest_part["thk_in"] * 25.4,
            qty=1,
        )

        if not price_each or price_each <= 0:
            if verbose:
                print(f"[Price Estimate] No valid price returned for part {part_num}")
            return None, ""

        # Calculate weight of the catalog part
        catalog_volume_in3 = largest_part["len_in"] * largest_part["wid_in"] * largest_part["thk_in"]
        catalog_weight_lb = catalog_volume_in3 * density_lb_in3

        if catalog_weight_lb <= 0:
            if verbose:
                print(f"[Price Estimate] Invalid catalog part weight: {catalog_weight_lb}")
            return None, ""

        # Calculate $/lb from catalog part
        price_per_lb = float(price_each) / catalog_weight_lb

        # Estimate price for oversized part
        estimated_price = price_per_lb * weight_lb

        source = f"Estimated from largest catalog part ({part_num}) @ ${price_per_lb:.2f}/lb"

        if verbose:
            print(f"[Price Estimate] Catalog part: {part_num}")
            print(f"  Catalog price: ${price_each:.2f}")
            print(f"  Catalog weight: {catalog_weight_lb:.1f} lb")
            print(f"  Price per lb: ${price_per_lb:.2f}/lb")
            print(f"  Your part weight: {weight_lb:.1f} lb")
            print(f"  Estimated price: ${estimated_price:.2f}")

        return estimated_price, source

    except Exception as e:
        if verbose:
            print(f"[Price Estimate] Error during estimation: {e}")
        return None, ""


def resolve_mcmaster_plate_for_quote(
    need_L_in: float | None,
    need_W_in: float | None,
    need_T_in: float | None,
    *,
    material_key: str,
    stock_L_in: float | None = None,
    stock_W_in: float | None = None,
    stock_T_in: float | None = None,
    catalog_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return a McMaster plate candidate using quote needs and existing stock sizing."""

    candidate: dict[str, Any] | None = None

    if need_L_in and need_W_in and need_T_in:
        try:
            candidate = pick_mcmaster_plate_sku(
                float(need_L_in),
                float(need_W_in),
                float(need_T_in),
                material_key=material_key,
                catalog_rows=catalog_rows,
            )
        except Exception:
            candidate = None

    if candidate:
        return candidate

    if stock_L_in and stock_W_in and stock_T_in:
        try:
            return pick_mcmaster_plate_sku(
                float(stock_L_in),
                float(stock_W_in),
                float(stock_T_in),
                material_key=material_key,
                catalog_rows=catalog_rows,
            )
        except Exception:
            return None

    return None


def get_qty_one_tier(tiers: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """
    Find the tier that applies at quantity 1 (if it exists).

    Args:
        tiers: List of price tier dictionaries from McMaster API

    Returns:
        The first tier where MinimumQuantity <= 1, or None if no such tier exists
    """
    if not tiers:
        return None

    for tier in tiers:
        min_qty = tier.get("MinimumQuantity")
        if min_qty is not None and min_qty <= 1:
            return tier

    return None


def compute_price_per_cubic_inch(
    tiers: Sequence[Mapping[str, Any]],
    stock_volume_cuin: float
) -> float | None:
    """
    Turn McMaster's pricing into a price per cubic inch of material.

    Args:
        tiers: List of price tier dictionaries from McMaster API
        stock_volume_cuin: The cubic inches of the McMaster piece you're buying

    Returns:
        Price per cubic inch ($/in³), or None if cannot be computed
    """
    if not tiers or stock_volume_cuin <= 0:
        return None

    # First, check if there's a qty-1 tier
    qty_one_tier = get_qty_one_tier(tiers)

    if qty_one_tier is not None:
        base_price = qty_one_tier.get("Amount")
    else:
        # Use the "largest price" tier (smallest minimum quantity)
        # Tiers should already be sorted by MinimumQuantity ascending
        base_price = tiers[0].get("Amount") if tiers else None

    if base_price is None or not isinstance(base_price, (int, float)):
        return None

    return float(base_price) / float(stock_volume_cuin)


def estimate_price_for_part_from_volume(
    tiers: Sequence[Mapping[str, Any]],
    stock_volume_cuin: float,
    part_volume_cuin: float
) -> float | None:
    """
    Use the $/in³ to estimate a price for your part.

    Args:
        tiers: List of price tier dictionaries from McMaster API
        stock_volume_cuin: Volume of the McMaster stock piece in cubic inches
        part_volume_cuin: Volume of your finished part in cubic inches

    Returns:
        Estimated part price, or None if cannot be computed
    """
    if part_volume_cuin <= 0:
        return None

    price_per_cuin = compute_price_per_cubic_inch(tiers, stock_volume_cuin)

    if price_per_cuin is None:
        return None

    return price_per_cuin * float(part_volume_cuin)


def collect_available_plate_thicknesses(
    catalog_rows: Sequence[Mapping[str, Any]]
) -> list[float]:
    """
    Extract and deduplicate all available plate thicknesses from catalog rows.

    Args:
        catalog_rows: List of catalog row dictionaries

    Returns:
        Sorted list of unique thickness values in inches
    """
    thicknesses = set()

    for row in catalog_rows:
        if not isinstance(row, Mapping):
            continue

        thickness = _coerce_inches_value(
            row.get("thickness_in")
            or row.get("T_in")
            or row.get("thk_in")
            or row.get("thickness")
        )

        if thickness is not None and thickness > 0:
            thicknesses.add(thickness)

    return sorted(thicknesses)


def estimate_mcmaster_shipping(weight_lb: float) -> float:
    """Rough McMaster shipping estimator (weight-based only).

    Args:
        weight_lb: Weight of the material in pounds

    Returns:
        Estimated shipping cost in dollars
    """
    # 1) Tiny packages
    if weight_lb <= 1.25:
        return 11.00

    # 2) Regular parcel
    if weight_lb <= 60:
        ship = 13.50 + 0.15 * weight_lb
        ship = max(11.00, ship)
        return round(ship, 2)

    # 3) Heavy / freight-ish
    ship = 0.35 * weight_lb
    return round(ship, 2)


__all__ = [
    "load_mcmaster_catalog_rows",
    "_coerce_inches_value",
    "pick_mcmaster_plate_sku",
    "pick_mcmaster_cylindrical_sku",
    "resolve_mcmaster_plate_for_quote",
    "get_qty_one_tier",
    "compute_price_per_cubic_inch",
    "estimate_price_for_part_from_volume",
    "collect_available_plate_thicknesses",
    "estimate_mcmaster_shipping",
]
