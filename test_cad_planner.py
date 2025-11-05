"""
test_cad_planner.py
===================
Demo script for the integrated CAD file → process plan workflow.

This demonstrates the plan_from_cad_file() function which:
1. Extracts dimensions (L, W, T) using PaddleOCR
2. Extracts hole table using geo_dump
3. Auto-detects part family from text
4. Generates complete process plan

Usage:
    python test_cad_planner.py "path/to/file.dxf"
"""

import sys
import json
from pathlib import Path

# Add cad_quoter to path
cad_quoter_dir = Path(__file__).resolve().parent
if str(cad_quoter_dir) not in sys.path:
    sys.path.insert(0, str(cad_quoter_dir))

from cad_quoter.planning.process_planner import plan_from_cad_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_cad_planner.py <cad_file.dxf>")
        print("\nExample:")
        print('  python test_cad_planner.py "Cad Files/301.dxf"')
        print('  python test_cad_planner.py "debug/T1769.dxf" --no-ocr')
        return 1

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 1

    # Check if --no-ocr flag is present
    use_paddle_ocr = "--no-ocr" not in sys.argv

    print("=" * 70)
    print("CAD FILE -> PROCESS PLAN")
    print("=" * 70)
    print(f"Input: {file_path}")
    print(f"Using PaddleOCR: {use_paddle_ocr}")
    print()

    try:
        # Generate process plan from CAD file
        plan = plan_from_cad_file(
            file_path,
            fallback_family="die_plate",
            use_paddle_ocr=use_paddle_ocr,
            verbose=True
        )

        print("\n" + "=" * 70)
        print("PROCESS PLAN SUMMARY")
        print("=" * 70)

        # Show extracted info
        if "extracted_dims" in plan:
            dims = plan["extracted_dims"]
            print(f"Dimensions: L={dims['L']:.3f}\" x W={dims['W']:.3f}\" x T={dims['T']:.3f}\"")

        unique_holes = plan.get('extracted_holes', 0)
        hole_ops = plan.get('extracted_hole_operations', 0)
        if hole_ops > unique_holes:
            print(f"Holes: {unique_holes} unique -> {hole_ops} operations")
        else:
            print(f"Holes: {unique_holes} entries")
        print(f"Generated Operations: {len(plan['ops'])}")
        print()

        # Show operations
        print("Operations:")
        for i, op in enumerate(plan["ops"], 1):
            op_name = op.get("op", "unknown")
            details = {k: v for k, v in op.items() if k != "op"}
            if details:
                print(f"  {i:2d}. {op_name:30s} {details}")
            else:
                print(f"  {i:2d}. {op_name}")

        # Show fixturing
        if plan.get("fixturing"):
            print("\nFixturing:")
            for fix in plan["fixturing"]:
                print(f"  - {fix}")

        # Show QA
        if plan.get("qa"):
            print("\nQA:")
            for qa in plan["qa"]:
                print(f"  - {qa}")

        # Show warnings
        if plan.get("warnings"):
            print("\nWarnings:")
            for warn in plan["warnings"]:
                print(f"  - {warn}")

        # Show directs
        if plan.get("directs"):
            print("\nDirect Costs:")
            for key, value in plan["directs"].items():
                print(f"  {key}: {value}")

        # Save to JSON file
        output_file = Path("debug") / f"{file_path.stem}_plan.json"
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(plan, f, indent=2)

        print("\n" + "=" * 70)
        print(f"Plan saved to: {output_file}")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
