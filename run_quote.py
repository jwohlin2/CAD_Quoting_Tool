from cad_quoter.pricing.QuoteDataHelper import extract_quote_data_from_cad, save_quote_data
from pathlib import Path

def main():
    """
    Main function to extract quote data from a CAD file and save it.
    """
    # Path to your CAD file
    cad_file = r"D:\CAD_Quoting_Tool\Cad Files\301_redacted.dwg"

    print(f"Processing CAD file: {cad_file}")

    # Create an output directory if it doesn't exist
    output_dir = Path("debug")
    output_dir.mkdir(exist_ok=True)

    # Extract quote data (this runs ODA conversion, OCR, calculations)
    quote_data = extract_quote_data_from_cad(
        cad_file_path=cad_file,
        machine_rate=90.0,      # $/hr
        labor_rate=90.0,        # $/hr
        margin_rate=0.15,       # 15%
        material_override=None, # or "17-4 PH Stainless Steel"
        dimension_override=None,# or (10.0, 8.0, 0.5) for L, W, T
        verbose=True            # Added for better feedback
    )

    # Save to JSON file in the debug directory
    output_file = output_dir / "test_quote_output.json"
    save_quote_data(quote_data, output_file, pretty=True)

    print(f"\nFinal price: ${quote_data.cost_summary.final_price:.2f}")

if __name__ == "__main__":
    main()
