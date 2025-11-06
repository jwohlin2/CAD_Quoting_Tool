# CAD File → Process Plan Integration

This integration automatically generates process plans from CAD files using geo_dump (for text/holes) and PaddleOCR (for dimensions).

## Quick Start

```python
from cad_quoter.planning.process_planner import plan_from_cad_file

# Generate plan from CAD file (DXF or DWG)
plan = plan_from_cad_file("path/to/file.dxf", verbose=True)

# Access the plan
print(f"Operations: {len(plan['ops'])}")
for op in plan["ops"]:
    print(f"  {op['op']}: {op}")
```

## Command Line Demo

```bash
# With PaddleOCR dimensions
python test_cad_planner.py "Cad Files/301.dxf"

# Without PaddleOCR (faster, but no dimensions)
python test_cad_planner.py "Cad Files/301.dxf" --no-ocr
```

## What It Does

The `plan_from_cad_file()` function performs these steps automatically:

1. **Extract Dimensions (L×W×T)** using PaddleOCR
   - Renders CAD file to high-res PNG
   - Uses OCR to find dimension text
   - Returns dimensions in inches

2. **Extract Hole Table** using geo_dump
   - Reads CAD file entities directly (TEXT, MTEXT, PROXY_ENTITY)
   - Finds and parses HOLE TABLE
   - Returns structured hole data

3. **Extract All Text** using geo_dump
   - Gets all text from all layouts/layers/blocks
   - Used for auto-detecting part family

4. **Auto-Detect Part Family**
   - Looks for keywords in text:
     - "hole table", "c'bore" → `die_plate`
     - "punch detail", "bearing land" → `punch`
     - "bushing", "id grind" → `bushing_id_critical`
     - "shear blade", "knife" → `shear_blade`
   - Falls back to specified default if no match

5. **Convert Hole Table to Operations**
   - Parses hole descriptions (TAP, C'BORE, THRU, etc.)
   - Maps to operation types (tapped, counterbore, thru, etc.)
   - Extracts depths, sides, tolerances

6. **Generate Process Plan**
   - Selects operations based on geometry & tolerances
   - Determines machining sequence
   - Adds fixturing and QA requirements

## API Reference

### `plan_from_cad_file(file_path, fallback_family="die_plate", use_paddle_ocr=True, verbose=False, override_dims=None)`

**Parameters:**
- `file_path`: Path to DXF or DWG file
- `fallback_family`: Family to use if auto-detection fails (default: `"die_plate"`)
- `use_paddle_ocr`: Whether to use PaddleOCR for dimensions (default: `True`)
- `verbose`: Print extraction progress (default: `False`)
- `override_dims`: Optional `dict` for manually overriding extracted dimensions (keys `L`, `W`, `T`)

**Returns:** Process plan dict with keys:
```python
{
    "ops": [...],              # List of operations
    "fixturing": [...],        # Fixturing requirements
    "qa": [...],               # QA checks
    "warnings": [...],         # Warnings/notes
    "directs": {...},          # Direct cost flags
    "source_file": "...",      # Input file path
    "extracted_dims": {...},   # Final L, W, T (after applying overrides)
    "extracted_holes": 0       # Number of holes found
}
```

## Lower-Level Functions

If you need more control, use the individual extraction functions:

### Extract Dimensions Only
```python
from cad_quoter.planning.process_planner import extract_dimensions_from_cad

dims = extract_dimensions_from_cad("file.dxf")
if dims:
    L, W, T = dims
    print(f"Dimensions: {L}\" × {W}\" × {T}\"")
```

### Extract Hole Table Only
```python
from cad_quoter.planning.process_planner import extract_hole_table_from_cad

holes = extract_hole_table_from_cad("file.dxf")
for hole in holes:
    print(f"{hole['HOLE']}: {hole['REF_DIAM']} × {hole['QTY']} - {hole['DESCRIPTION']}")
```

### Extract All Text Only
```python
from cad_quoter.planning.process_planner import extract_all_text_from_cad

all_text = extract_all_text_from_cad("file.dxf")
print(f"Found {len(all_text)} text records")
```

## Supported Hole Types

The integration automatically recognizes these hole types:

| Description Pattern | Mapped Type | Additional Fields |
|---------------------|-------------|-------------------|
| "TAP", "5/8-11" | `tapped` | `depth` (if "X .25 DEEP") |
| "C'BORE", "COUNTERBORE" | `counterbore` | `depth`, `side` (FRONT/BACK) |
| "C'DRILL", "COUNTERDRILL" | `c_drill` | `depth`, `side` |
| "JIG GRIND", "±.0001" | `post_bore` | `tol=0.0002` |
| "DOWEL" + "PRESS" | `dowel_press` | - |
| "DOWEL" + "SLIP" | `dowel_slip` | - |
| "THRU" | `thru` | - |

## Part Families

Currently supported families:
- `die_plate` - Main production family
- `punch` - Punch details
- `bushing_id_critical` - Bushings with tight ID tolerance
- `cam_or_hemmer` - Cam and hemmer components
- `shear_blade` - Shear blades and knives
- `flat_die_chaser` - Die chasers
- `pm_compaction_die` - PM compaction dies
- `pilot_punch` - Pilot punches
- `extrude_hone` - Extrude hone operations

## Troubleshooting

### PaddleOCR Not Working
If dimension extraction fails, you can disable it:
```python
plan = plan_from_cad_file("file.dxf", use_paddle_ocr=False)
```

Then manually provide dimensions:
```python
from cad_quoter.planning.process_planner import plan_job

params = {
    "plate_LxW": (8.72, 3.247),
    "T": 0.5,
    "hole_sets": [...]
}
plan = plan_job("die_plate", params)
```

### Hole Table Not Found
Make sure your CAD file has:
- A text entity containing "HOLE TABLE"
- HOLE, REF DIAM, QTY, DESCRIPTION columns
- Hole data below the header

You can test extraction separately:
```python
from cad_quoter.geo_dump import extract_hole_table_from_file

holes = extract_hole_table_from_file("file.dxf")
print(f"Found {len(holes)} holes")
```

### Family Not Detected
If auto-detection fails, specify the family explicitly:
```python
plan = plan_from_cad_file("file.dxf", fallback_family="punch")
```

Or use the lower-level API:
```python
from cad_quoter.planning.process_planner import plan_job

params = {...}  # Your params
plan = plan_job("punch", params)
```

## Performance Notes

- **With PaddleOCR**: 5-15 seconds per file (rendering + OCR)
- **Without PaddleOCR**: 1-2 seconds per file (text extraction only)

For batch processing, consider disabling PaddleOCR and providing dimensions separately.
