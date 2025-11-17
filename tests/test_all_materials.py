#!/usr/bin/env python
"""Test scrap pricing for all common materials in AppV7."""

import sys

MATERIALS = [
    "303 stainless steel",
    "316",
    "52100",
    "6061",
    "7075",
    "A2",
    "Ceramic",
    "Grade 2 Titanium",
    "Grade 5 Titanium",
    "Hokotol",
    "Low-carbon Steel",
    "Mild Steel",
    "Ti-6AI-4V",
    "Tool Steel A2",
    "VM-15M",
    "Aluminum 5083",
    "aluminum MIC6",
]


def test_material_detection_and_pricing():
    """Test that all materials get detected and priced correctly."""
    from cad_quoter.pricing.DirectCostHelper import calculate_scrap_value

    print("=" * 80)
    print(" MATERIAL SCRAP PRICING TEST ".center(80))
    print("=" * 80)
    print()
    print(f"{'Material':<25} {'Family':<15} {'Price ($/lb)':<12} {'Source'}")
    print("-" * 80)

    results = []
    for material in MATERIALS:
        try:
            result = calculate_scrap_value(
                scrap_weight_lbs=45.0,
                material=material,
                fallback_scrap_price_per_lb=0.50,
                verbose=False
            )

            family = result['material_family']
            price = result['scrap_price_per_lb']
            source = result['price_source']

            # Abbreviate source for display
            if "ScrapMetalBuyers" in source:
                source_display = "SMB"
            elif "Wieland" in source:
                source_display = "Wieland"
            elif "house_rate" in source.lower():
                source_display = "house rate"
            else:
                source_display = source[:30]

            print(f"{material:<25} {family:<15} ${price:<11.4f} {source_display}")

            results.append({
                "material": material,
                "family": family,
                "price": price,
                "source": source,
                "success": True
            })

        except Exception as e:
            print(f"{material:<25} ERROR: {str(e)[:40]}")
            results.append({
                "material": material,
                "success": False,
                "error": str(e)
            })

    print()
    print("=" * 80)

    # Summary
    success_count = sum(1 for r in results if r.get("success", False))
    print(f"Results: {success_count}/{len(MATERIALS)} materials processed successfully")
    print()

    # Check for issues
    issues = []

    # Check aluminum materials
    aluminum_materials = [r for r in results if r.get("success") and "aluminum" in r["material"].lower()]
    for r in aluminum_materials:
        if r["family"] != "aluminum":
            issues.append(f"[ERROR] {r['material']}: Detected as '{r['family']}' instead of 'aluminum'")
        elif "Wieland" not in r["source"]:
            issues.append(f"[WARN] {r['material']}: Using {r['source']} instead of Wieland (expected for aluminum)")

    # Check stainless materials
    stainless_materials = [r for r in results if r.get("success") and "stainless" in r["material"].lower()]
    for r in stainless_materials:
        if r["family"] != "stainless":
            issues.append(f"[ERROR] {r['material']}: Detected as '{r['family']}' instead of 'stainless'")

    # Check titanium materials
    titanium_materials = [r for r in results if r.get("success") and ("titanium" in r["material"].lower() or "ti-6" in r["material"].lower())]
    for r in titanium_materials:
        if r["family"] != "titanium":
            issues.append(f"[ERROR] {r['material']}: Detected as '{r['family']}' instead of 'titanium'")

    # Check steel materials
    steel_materials = [r for r in results if r.get("success") and "steel" in r["material"].lower() and "stainless" not in r["material"].lower()]
    for r in steel_materials:
        if r["family"] not in ["steel", "stainless"]:
            issues.append(f"[ERROR] {r['material']}: Detected as '{r['family']}' instead of 'steel'")

    # Check specific steel alloys
    steel_alloys = ["52100", "A2", "Hokotol", "VM-15M"]
    for material_name in steel_alloys:
        r = next((r for r in results if r.get("success") and r["material"] == material_name), None)
        if r and r["family"] != "steel":
            issues.append(f"[ERROR] {r['material']}: Detected as '{r['family']}' instead of 'steel'")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("[OK] All materials detected correctly!")

    print()

    # Show price source breakdown
    print("Price Source Breakdown:")
    wieland_count = sum(1 for r in results if r.get("success") and "Wieland" in r.get("source", ""))
    smb_count = sum(1 for r in results if r.get("success") and "ScrapMetalBuyers" in r.get("source", ""))
    house_rate_count = sum(1 for r in results if r.get("success") and "house_rate" in r.get("source", "").lower())

    print(f"  Wieland:        {wieland_count}")
    print(f"  SMB:            {smb_count}")
    print(f"  House rate:     {house_rate_count}")
    print()

    return len(issues) == 0


if __name__ == "__main__":
    success = test_material_detection_and_pricing()
    sys.exit(0 if success else 1)
