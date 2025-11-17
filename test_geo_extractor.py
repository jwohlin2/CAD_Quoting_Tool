#!/usr/bin/env python3
"""Test geo_extractor with 301_redacted.dwg"""

import json
import sys
from pathlib import Path
from cad_quoter.geo_extractor import (
    open_doc,
    collect_all_text,
    rebuild_structured_rows,
)

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

def test_geo_extractor():
    """Test geo_extractor with the specified DWG file"""

    dwg_path = Path(r"C:\Users\John Wohlin\Downloads\composinonimizer\T1769-219.dwg")

    print(f"Testing geo_extractor with: {dwg_path}")
    print("=" * 80)

    # Step 1: Open the document
    print("\n[1] Opening document...")
    try:
        doc = open_doc(dwg_path)
        print(f"[OK] Document opened successfully")
        print(f"  Layouts available: {[layout.name for layout in doc.layouts]}")
    except Exception as e:
        print(f"[FAIL] Failed to open document: {e}")
        return

    # Step 2: Collect all text
    print("\n[2] Collecting all text from document...")
    try:
        all_text = collect_all_text(
            doc,
            layouts=None,  # All layouts
            include_layers=None,  # All layers
            exclude_layers=None,
            min_height=0.0,
            max_block_depth=3
        )
        print(f"[OK] Collected {len(all_text)} text records")

        # Show summary by entity type
        etype_counts = {}
        for rec in all_text:
            etype = rec.get("etype", "UNKNOWN")
            etype_counts[etype] = etype_counts.get(etype, 0) + 1

        print(f"\n  Text by entity type:")
        for etype, count in sorted(etype_counts.items()):
            print(f"    {etype}: {count}")

        # Show summary by layout
        layout_counts = {}
        for rec in all_text:
            layout = rec.get("layout", "UNKNOWN")
            layout_counts[layout] = layout_counts.get(layout, 0) + 1

        print(f"\n  Text by layout:")
        for layout, count in sorted(layout_counts.items()):
            print(f"    {layout}: {count}")

        # Show first 10 text samples
        print(f"\n  First 10 text samples:")
        for i, rec in enumerate(all_text[:10]):
            text = rec.get("text", "")[:60]  # Truncate long text
            layer = rec.get("layer", "?")
            etype = rec.get("etype", "?")
            print(f"    [{i+1}] ({etype} on {layer}): {text}")

    except Exception as e:
        print(f"[FAIL] Failed to collect text: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Look for hole-related text (proxy entities)
    print("\n[3] Looking for hole-related text (PROXYTEXT)...")
    hole_texts = [rec for rec in all_text if rec.get("etype") == "PROXYTEXT"]

    if hole_texts:
        print(f"[OK] Found {len(hole_texts)} PROXYTEXT records")
        print(f"\n  Hole table text fragments:")
        for i, rec in enumerate(hole_texts[:20]):  # Show first 20
            text = rec.get("text", "")
            print(f"    [{i+1}] {text}")
    else:
        print(f"  No PROXYTEXT records found")

    # Step 4: Try to rebuild structured hole rows
    print("\n[4] Attempting to rebuild structured hole rows...")
    try:
        all_text_strings = [rec.get("text", "") for rec in all_text]
        structured_rows = rebuild_structured_rows(all_text_strings)

        if structured_rows:
            print(f"[OK] Rebuilt {len(structured_rows)} structured hole rows")
            print(f"\n  Structured hole data:")
            for i, row in enumerate(structured_rows[:10]):  # Show first 10
                ref = row.get("ref")
                diam = row.get("diam")
                tap = row.get("tap_thread")
                ops = row.get("ops", [])
                qty = row.get("qty")
                depth = row.get("depth")
                side = row.get("side")

                print(f"    [{i+1}] Ref: {ref}, Diam: {diam}, Tap: {tap}, Ops: {ops}, Qty: {qty}, Depth: {depth}, Side: {side}")
        else:
            print(f"  No structured hole rows found")
    except Exception as e:
        print(f"[FAIL] Failed to rebuild structured rows: {e}")
        import traceback
        traceback.print_exc()

    # Step 5: Save output to JSON for inspection
    print("\n[5] Saving output to JSON...")
    try:
        output_file = Path(r"D:\CAD_Quoting_Tool\geo_extractor_test_output.json")
        with open(output_file, 'w') as f:
            json.dump({
                "file": str(dwg_path),
                "total_text_records": len(all_text),
                "all_text": all_text,
                "structured_holes": structured_rows if 'structured_rows' in locals() else []
            }, f, indent=2)
        print(f"[OK] Output saved to: {output_file}")
    except Exception as e:
        print(f"[FAIL] Failed to save output: {e}")

    print("\n" + "=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    test_geo_extractor()
