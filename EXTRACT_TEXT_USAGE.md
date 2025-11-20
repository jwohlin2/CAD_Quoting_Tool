# CAD Text Extraction - Usage Guide

## Standalone Sidecar Script (For Testing)

I created **`extract_cad_text_sidecar.py`** - a simple standalone script for extracting all text from CAD files.

### Quick Start

```bash
# Simple text-only output
python extract_cad_text_sidecar.py "Cad Files/301_redacted.dxf" --text-only

# Human-readable format (default)
python extract_cad_text_sidecar.py "Cad Files/301_redacted.dxf"

# JSON format (with full metadata)
python extract_cad_text_sidecar.py "Cad Files/301_redacted.dxf" --format json

# CSV format
python extract_cad_text_sidecar.py "Cad Files/301_redacted.dxf" --format csv

# Save to file
python extract_cad_text_sidecar.py "Cad Files/301_redacted.dxf" --format json -o output.json

# Explore deeper into blocks (default is 5 levels)
python extract_cad_text_sidecar.py "Cad Files/301_redacted.dxf" --block-depth 10
```

### What it Extracts

The script extracts ALL text from CAD files using the same method as the main application:

- **TEXT** entities
- **MTEXT** entities (with proper plain text conversion)
- **TABLE** entities (cell by cell)
- **ACAD_PROXY_ENTITY** (for HOLE TABLEs and AutoCAD Mechanical objects)
- **DIMENSION** entities
- **MLEADER** entities
- **ATTRIB/ATTDEF** (block attributes)
- Text inside **INSERT** blocks (recursively, up to specified depth)

### Special Features

- Decodes Unicode escape sequences (e.g., `\U+2205` → `∅`)
- Normalizes MTEXT formatting codes (e.g., `\P` → newlines)
- Handles both **DWG** and **DXF** files
- Explores nested blocks to find all text
- Merges proxy entity fragments (important for HOLE TABLEs)

---

## Existing Tools in the Codebase

### 1. Full-Featured Extractor: `geo_dump.py`

This is the production tool used by the application:

```bash
# Extract all text and save to CSV/JSONL
python -m cad_quoter.geo_dump "Cad Files/301_redacted.dxf"

# This creates in the ./debug directory:
# - dxf_text_dump.csv       # All text records
# - dxf_text_dump.jsonl     # Same as JSON Lines
# - hole_table_structured.csv  # Parsed hole table
# - hole_table_ops.csv      # Machining operations
# - stock_dims.csv          # Inferred stock dimensions
```

### 2. Python API

You can also use the extraction functions in your own Python code:

```python
from cad_quoter.geo_dump import extract_all_text_from_file

# Extract all text
text_records = extract_all_text_from_file("drawing.dxf")

# Each record is a dict with:
# - layout: Layout name
# - layer: Layer name  
# - etype: Entity type (TEXT, MTEXT, etc.)
# - text: The actual text content
# - x, y: Position
# - height: Text height
# - rotation: Rotation angle
# - in_block: Whether inside a block
# - depth: Block nesting level
# - block_path: List of parent block names
```

---

## How It Works (Under the Hood)

### Architecture

```
CAD File (DWG/DXF)
    ↓
geo_extractor.py        # Opens file, walks entities, extracts raw text
    ↓
Text normalization      # Decodes Unicode, strips formatting codes
    ↓
geo_dump.py            # Finds HOLE TABLEs, structures data
    ↓
hole_operations.py     # Breaks down into machining ops
    ↓
Output (CSV/JSON)
```

### Key Files

1. **`cad_quoter/geo_extractor.py`** (lines 1-696)
   - Low-level text extractor
   - Handles all entity types
   - Opens DWG/DXF files
   - Recursive block exploration

2. **`cad_quoter/geo_dump.py`** (lines 1-672)
   - Higher-level API
   - HOLE TABLE parsing
   - CSV/JSONL output
   - Stock dimension inference

3. **`cad_quoter/geometry/dxf_text.py`** (lines 1-92)
   - Simple text extraction helper
   - Used by some older code

4. **`extract_cad_text_sidecar.py`** (NEW - for testing)
   - Standalone script
   - Multiple output formats
   - Easy to use for debugging

---

## Testing

```bash
# Test with DXF file
python extract_cad_text_sidecar.py "Cad Files/301_redacted.dxf" --text-only | head -20

# Test with DWG file (requires ODA File Converter)
python extract_cad_text_sidecar.py "Cad Files/301_redacted.dwg" --format json

# Count text records
python extract_cad_text_sidecar.py "Cad Files/301_redacted.dxf" --text-only | wc -l
```

---

## Dependencies

- **ezdxf**: DXF/DWG reading (install with `pip install ezdxf`)
- **ODA File Converter**: Optional, for DWG files (converts DWG → DXF)

