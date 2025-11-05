"""
example_cad_to_plan.py
======================
Simple example showing how to generate a process plan from a CAD file.

This is the most straightforward way to use the integrated system.
"""

from cad_quoter.planning.process_planner import plan_from_cad_file
import json

# Example 1: Basic usage with verbose output
def example_basic():
    """Generate plan with full output."""
    print("Example 1: Basic Plan Generation")
    print("=" * 60)

    plan = plan_from_cad_file(
        "Cad Files/301.dxf",  # Your CAD file
        verbose=True           # Show extraction progress
    )

    print("\nPlan generated!")
    print(f"  Operations: {len(plan['ops'])}")
    print(f"  Holes found: {plan.get('extracted_holes', 0)}")

    if "extracted_dims" in plan:
        dims = plan["extracted_dims"]
        print(f"  Dimensions: {dims['L']:.3f}\" × {dims['W']:.3f}\" × {dims['T']:.3f}\"")

    return plan


# Example 2: Without PaddleOCR (faster)
def example_no_ocr():
    """Generate plan without using PaddleOCR (if you already know dimensions)."""
    print("\nExample 2: Without PaddleOCR")
    print("=" * 60)

    plan = plan_from_cad_file(
        "Cad Files/301.dxf",
        use_paddle_ocr=False,  # Skip OCR (faster)
        verbose=False
    )

    print(f"Plan generated: {len(plan['ops'])} operations")
    return plan


# Example 3: Specify family explicitly
def example_with_family():
    """Force a specific part family."""
    print("\nExample 3: Force Part Family")
    print("=" * 60)

    plan = plan_from_cad_file(
        "Cad Files/301.dxf",
        fallback_family="punch",  # Force punch family
        verbose=False
    )

    print(f"Plan generated for 'punch' family: {len(plan['ops'])} operations")
    return plan


# Example 4: Access individual components
def example_components():
    """Access dimensions, holes, and text separately."""
    print("\nExample 4: Individual Components")
    print("=" * 60)

    from cad_quoter.planning.process_planner import (
        extract_dimensions_from_cad,
        extract_hole_table_from_cad,
        extract_all_text_from_cad
    )

    file_path = "Cad Files/301.dxf"

    # Get dimensions
    dims = extract_dimensions_from_cad(file_path)
    if dims:
        L, W, T = dims
        print(f"Dimensions: {L:.3f}\" × {W:.3f}\" × {T:.3f}\"")

    # Get hole table
    holes = extract_hole_table_from_cad(file_path)
    print(f"Holes: {len(holes)} entries")
    for hole in holes[:3]:  # Show first 3
        print(f"  {hole['HOLE']}: {hole['REF_DIAM']} × {hole['QTY']}")

    # Get all text
    all_text = extract_all_text_from_cad(file_path)
    print(f"Text: {len(all_text)} records")


# Example 5: Save plan to JSON
def example_save_json():
    """Generate plan and save to JSON file."""
    print("\nExample 5: Save Plan to JSON")
    print("=" * 60)

    plan = plan_from_cad_file("Cad Files/301.dxf", verbose=False)

    output_file = "debug/example_plan.json"
    with open(output_file, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"Plan saved to: {output_file}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CAD FILE → PROCESS PLAN EXAMPLES")
    print("=" * 60)

    # Run examples
    try:
        example_basic()
        # example_no_ocr()
        # example_with_family()
        # example_components()
        # example_save_json()
    except FileNotFoundError:
        print("\nNote: Update the file path in the examples to point to your CAD file!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
