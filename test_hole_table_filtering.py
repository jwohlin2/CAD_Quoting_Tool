"""
Test script to verify hole table filtering in paddle_dims_extractor.py
"""

import sys
from pathlib import Path

# Add tools directory to path
tools_dir = Path(__file__).parent / "tools"
sys.path.insert(0, str(tools_dir))

from paddle_dims_extractor import PaddleOCRDimensionExtractor, DrawingRenderer
from PIL import Image

def test_extraction(drawing_file: str, verbose: bool = True):
    """Test dimension extraction with hole table filtering."""

    drawing_path = Path(drawing_file)

    if not drawing_path.exists():
        print(f"[ERROR] File not found: {drawing_path}")
        return

    print(f"\n{'='*70}")
    print(f"Testing: {drawing_path.name}")
    print(f"{'='*70}\n")

    try:
        # Step 1: Render to PNG
        print("[1/2] Rendering CAD file to PNG...")
        output_dir = Path("debug")
        output_dir.mkdir(exist_ok=True)

        png_path = output_dir / f"{drawing_path.stem}_test_render.png"

        renderer = DrawingRenderer(verbose=False)
        renderer.render(str(drawing_path), str(png_path))
        print(f"  Rendered to: {png_path}")

        # Step 2: Extract dimensions
        print(f"\n[2/2] Extracting dimensions with hole table filtering...")
        extractor = PaddleOCRDimensionExtractor(verbose=verbose)

        cropped_path = output_dir / f"{drawing_path.stem}_test_cropped.png"

        with Image.open(png_path) as img:
            img = img.convert("RGB")
            dims = extractor.extract(img, save_cropped_path=str(cropped_path))

        if dims:
            print(f"\n{'='*70}")
            print(f"[SUCCESS] Dimensions extracted:")
            print(f"{'='*70}")
            print(f"  Length:    {dims.length:.4f} in")
            print(f"  Width:     {dims.width:.4f} in")
            print(f"  Thickness: {dims.thickness:.4f} in")
            print(f"  Method:    {dims.method}")
            print(f"  Confidence: {dims.confidence}")

            if dims.all_numbers:
                print(f"\n  All detected numbers (top 15):")
                for i, num in enumerate(dims.all_numbers[:15], 1):
                    print(f"    {i:2d}. {num:.4f}")

            print(f"{'='*70}\n")
        else:
            print(f"\n[FAIL] No dimensions extracted")
            print(f"  Check the debug output above for filtering details")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


def main():
    """Test on multiple files."""

    # Test files
    test_files = [
        "Cad Files/T1769-201.dwg",
        "Cad Files/T1769-104.dwg",
        "Cad Files/T1769-143.dwg",
        "Cad Files/301.dwg",
    ]

    # Find available test files
    available_files = [f for f in test_files if Path(f).exists()]

    if not available_files:
        print("[ERROR] No test files found. Please specify a CAD file path.")
        print("\nUsage:")
        print('  python test_hole_table_filtering.py "path/to/drawing.dwg"')
        return

    # Test each file
    for test_file in available_files:
        test_extraction(test_file, verbose=True)
        print("\n" + "="*70)
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific file
        test_extraction(sys.argv[1], verbose=True)
    else:
        # Test default files
        main()
