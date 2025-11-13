"""
test_text_extraction.py
========================
Quick test script to demonstrate text extraction from CAD files.
"""

from pathlib import Path
from cad_quoter.geo_dump import extract_all_text_from_file

# Example 1: Get all text from a CAD file
def example_basic():
    """Extract all text with full metadata."""
    file_path = Path("Cad Files/301_redacted.dxf")  # Adjust path as needed

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    all_text = extract_all_text_from_file(file_path)

    print(f"Found {len(all_text)} text records\n")

    # Show first 10 records
    for i, record in enumerate(all_text[:10], 1):
        print(f"{i}. '{record['text']}'")
        print(f"   Layout: {record['layout']}, Layer: {record['layer']}, Type: {record['etype']}")
        print(f"   Position: ({record['x']:.2f}, {record['y']:.2f}), Height: {record['height']:.2f}")
        print()


# Example 2: Get just the text strings (no metadata)
def example_text_only():
    """Extract just the text strings."""
    file_path = Path("Cad Files/301_redacted.dxf")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    all_text = extract_all_text_from_file(file_path)
    text_only = [r["text"] for r in all_text]

    print(f"All text strings ({len(text_only)} total):")
    print("\n".join(text_only[:20]))  # Show first 20


# Example 3: Filter by layout
def example_model_space_only():
    """Extract text from Model space only."""
    file_path = Path("Cad Files/301_redacted.dxf")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    model_text = extract_all_text_from_file(file_path, layouts=["Model"])

    print(f"Found {len(model_text)} text records in Model space")
    for record in model_text[:5]:
        print(f"  - {record['text']}")


# Example 4: Exclude specific layers
def example_exclude_layers():
    """Extract text but exclude dimension and titleblock layers."""
    file_path = Path("Cad Files/301_redacted.dxf")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    filtered = extract_all_text_from_file(
        file_path,
        exclude_layers=["DIM.*", "TITLE.*", "BORDER.*"]  # Regex patterns
    )

    print(f"Found {len(filtered)} text records (after filtering)")


# Example 5: Save all text to a file
def example_save_to_file():
    """Save all text to a plain text file."""
    file_path = Path("Cad Files/301_redacted.dxf")
    output_path = Path("debug/all_text_dump.txt")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    all_text = extract_all_text_from_file(file_path)

    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_text:
            f.write(f"{record['text']}\n")

    print(f"Saved {len(all_text)} text records to: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Text Extraction Examples")
    print("=" * 60)

    # Run the examples
    print("\n--- Example 1: Basic extraction with metadata ---")
    example_basic()

    print("\n--- Example 2: Text strings only ---")
    example_text_only()

    print("\n--- Example 3: Model space only ---")
    example_model_space_only()

    print("\n--- Example 4: Exclude specific layers ---")
    example_exclude_layers()

    print("\n--- Example 5: Save to file ---")
    example_save_to_file()
