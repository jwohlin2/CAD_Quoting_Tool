# DWG Punch Extraction - Usage Guide

## Overview

The DWG Punch Extraction system automatically extracts manufacturing features from 2D punch drawings (DWG/DXF) to enable accurate quoting without requiring STEP files.

**Status**: ✅ Fully Implemented and Integrated

## Quick Start

### Basic Usage

```python
from pathlib import Path
from cad_quoter.geometry.dwg_punch_extractor import extract_punch_features

# Extract features from a DXF file
dxf_path = Path("my_punch_drawing.dxf")
summary = extract_punch_features(dxf_path)

# Access extracted features
print(f"Family: {summary.family}")
print(f"Material: {summary.material_callout}")
print(f"Length: {summary.overall_length_in}\"")
print(f"Max OD: {summary.max_od_or_width_in}\"")
print(f"Taps: {summary.tap_count}")
```

### Generate Manufacturing Plan

```python
from cad_quoter.planning.process_planner import plan_job

# Use the integrated planner
plan = plan_job("Punches", {
    "dxf_path": "my_punch.dxf",
    # Optional: override any extracted values
    "material": "A2",
})

# Access plan details
for op in plan["ops"]:
    print(f"Operation: {op['op']}")

for qa_check in plan["qa"]:
    print(f"QA: {qa_check}")
```

## Implementation Components

### 1. Feature Extraction Module

**File**: `cad_quoter/geometry/dwg_punch_extractor.py`

**Key Functions**:

- `extract_punch_features(dxf_path, text_lines=None)` - Main entry point
- `extract_punch_features_from_dxf(dxf_path, text_dump)` - Core extraction
- `extract_geometry_envelope(dxf_path)` - Geometry-based dimensions
- `extract_dimensions(dxf_path)` - Dimension entities with tolerances
- `classify_punch_family(text_dump)` - Family/shape classification
- `detect_material(text_dump)` - Material detection
- `parse_holes_from_text(text_dump)` - Hole/tap parsing

**Data Structure**:

```python
@dataclass
class PunchFeatureSummary:
    # Classification
    family: str                    # round_punch, pilot_pin, form_punch, etc.
    shape_type: str                # round or rectangular

    # Size/Envelope
    overall_length_in: float
    max_od_or_width_in: float
    body_width_in: Optional[float]
    body_thickness_in: Optional[float]

    # Operations Drivers
    num_ground_diams: int
    total_ground_length_in: float
    tap_count: int
    tap_summary: List[Dict]
    num_chamfers: int
    num_small_radii: int

    # Pain/Quality Flags
    has_perp_face_grind: bool
    has_3d_surface: bool
    has_polish_contour: bool
    has_no_step_permitted: bool
    has_gdt: bool
    min_dia_tol_in: Optional[float]
    min_len_tol_in: Optional[float]

    # Material
    material_callout: Optional[str]

    # Metadata
    confidence_score: float
    warnings: List[str]
```

### 2. Punch Planner Module

**File**: `cad_quoter/planning/punch_planner.py`

**Key Functions**:

- `create_punch_plan(params)` - Generate manufacturing plan from features
- `planner_punches_enhanced(params)` - Integration point for process_planner

**Plan Structure**:

```python
{
    "ops": [
        {"op": "stock_procurement", "material": "A2", ...},
        {"op": "saw_to_length", ...},
        {"op": "rough_turning", ...},
        {"op": "heat_treat", ...},
        {"op": "od_grind", ...},
        {"op": "tap", "size": "5/16-18", "depth_in": 0.80, ...},
    ],
    "fixturing": [
        "Use collet or 3-jaw chuck for turning",
        ...
    ],
    "qa": [
        "Verify OAL: 6.990\"",
        "Critical diameter tolerance: ±0.0001\"",
        ...
    ],
    "warnings": [...],
    "directs": {"hardware": False, "outsourced": True, ...}
}
```

### 3. Process Planner Integration

**File**: `cad_quoter/planning/process_planner.py`

**Changes**: Enhanced `planner_punches()` function to use the new punch planner with automatic feature extraction.

**Usage**:

```python
from cad_quoter.planning.process_planner import plan_job

# With DXF path (automatic extraction)
plan = plan_job("Punches", {"dxf_path": "punch.dxf"})

# With manual params
plan = plan_job("Punches", {
    "overall_length_in": 6.5,
    "max_od_or_width_in": 0.75,
    "num_ground_diams": 3,
    "material": "A2",
})

# With PunchFeatureSummary
plan = plan_job("Punches", summary.__dict__)
```

## Extracted Features

### A. Classification

From title block and text analysis:

- **Family**: `round_punch`, `pilot_pin`, `form_punch`, `die_section`, `guide_post`, `bushing`
- **Shape**: `round` or `rectangular`
- **Material**: `A2`, `D2`, `M2`, `O1`, `S7`, `CARBIDE`, etc.

### B. Size/Envelope

From geometry bbox and dimensions:

- `overall_length_in` - Total length
- `max_od_or_width_in` - Largest diameter or width
- `body_width_in` / `body_thickness_in` - For rectangular parts
- `num_ground_diams` - Count of distinct diameters

### C. Operations Drivers

From text patterns:

- **Grinding**: `num_ground_diams`, `has_perp_face_grind`
- **Form work**: `has_3d_surface`, `has_polish_contour`, `form_complexity_level`
- **Holes**: `tap_count`, `tap_summary` (size, depth)
- **Edge work**: `num_chamfers`, `num_small_radii`

### D. Quality/Pain Flags

From notes and tolerances:

- `min_dia_tol_in` - Tightest diameter tolerance
- `min_len_tol_in` - Tightest length tolerance
- `has_no_step_permitted` - No step allowed between diameters
- `has_sharp_edges` - Sharp edges required
- `has_gdt` - GD&T callouts present

## Text Patterns Recognized

### Material Detection

```
A2, A6, A10
D2, D3
M2, M4
O1, S7, H13
CARBIDE
440C, 17-4
4140, 4340
```

### Hole/Tap Patterns

```
5/16-18 TAP X .80 DEEP
1/4-20 TAP
Ø.250 THRU
Ø.125 X .50 DP
#7 DRILL (.201) X .75 DEEP
```

### Chamfer/Radius Patterns

```
(2) .010 X 45°
.015 X 45° TYP
R.005 TYP
.003 R
```

### Tolerance Patterns

```
±0.001
+0.0000-0.0002
1.500 ±.005
Ø.7504 +.0000 -.0001
```

### Quality Flags

```
POLISH CONTOUR
NO STEP PERMITTED
SHARP EDGES
THIS SURFACE PERPENDICULAR TO CENTERLINE
GD&T PER ASME Y14.5
```

## Testing

### Unit Tests

**File**: `tests/geometry/test_dwg_punch_extractor.py`

Run tests (when pytest is available):

```bash
pytest tests/geometry/test_dwg_punch_extractor.py -v
```

### Examples

**File**: `examples/dwg_punch_extraction_example.py`

Run examples:

```bash
PYTHONPATH=/home/user/CAD_Quoting_Tool python examples/dwg_punch_extraction_example.py
```

**Examples Included**:

1. Text-based feature extraction
2. Manufacturing plan generation
3. Integrated process planning
4. Form punch with 3D contour

## Architecture

```
┌─────────────────┐
│   DWG/DXF File  │
└────────┬────────┘
         │
         ├─── ezdxf ──────────────► Geometry (bbox, dimensions)
         │
         ├─── geo_extractor ──────► Text records
         │
         ▼
┌──────────────────────────────┐
│  dwg_punch_extractor.py      │
│  - classify_punch_family()   │
│  - detect_material()         │
│  - parse_holes_from_text()   │
│  - extract_dimensions()      │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│   PunchFeatureSummary        │
│  (comprehensive features)    │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│   punch_planner.py           │
│  - create_punch_plan()       │
│  - _add_grinding_ops()       │
│  - _add_hole_ops()           │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│   process_planner.py         │
│  - planner_punches()         │
│  - plan_job("Punches", ...)  │
└────────┬─────────────────────┘
         │
         ▼
    Manufacturing Plan
    {ops, fixturing, qa, warnings, directs}
```

## Integration Points

### 1. CAD Quoting Workflow

The punch extractor integrates seamlessly with the existing CAD quoting flow:

```python
# In your quoting workflow
from cad_quoter.geometry.dwg_punch_extractor import extract_punch_features
from cad_quoter.planning.process_planner import plan_job

# Extract features
features = extract_punch_features(dxf_path)

# Generate plan
plan = plan_job("Punches", features.__dict__)

# Continue with pricing/quoting...
```

### 2. Existing Text Extraction

Uses the existing `geo_extractor.py` infrastructure:

```python
from cad_quoter.geo_extractor import open_doc, collect_all_text

doc = open_doc(dxf_path)
text_records = list(collect_all_text(doc))
```

### 3. Hole/Tap Parsing

Complements the existing `hole_table_parser.py` for hole tables with free-text hole parsing.

## Configuration

No special configuration required. The system works out-of-the-box with:

- ✅ `ezdxf>=1.0` (already in requirements.txt)
- ✅ Existing `geo_extractor` infrastructure
- ✅ Integrated with `process_planner`

## Limitations & Future Enhancements

### Current Limitations

1. **Geometry extraction** - Currently uses simple bbox; could be enhanced with:
   - Layer filtering (ignore title block layers)
   - Profile loop detection for actual part outline
   - WEDM path length calculation

2. **Dimension parsing** - Basic tolerance extraction; could add:
   - Geometric tolerance (GD&T) symbol parsing
   - Dimension location mapping (which diameter at which position)

3. **Ground diameter estimation** - Currently counts distinct diameter values; could:
   - Group diameters by location
   - Estimate length of each diameter section
   - Detect taper/form transitions

### Future Enhancements

- [ ] **Layer-based filtering** for cleaner geometry extraction
- [ ] **Profile path detection** using edgeminer/edgesmith
- [ ] **WEDM path length** calculation for 2D profiles
- [ ] **Drawing view detection** to identify detail views vs main views
- [ ] **Symbol recognition** for GD&T and surface finish
- [ ] **Machine learning** for form complexity classification

## Troubleshooting

### Issue: No dimensions extracted

**Cause**: DXF may not have DIMENSION entities (text-only dimensions)

**Solution**: The system falls back to geometry bbox and text parsing

### Issue: Wrong family classification

**Cause**: Ambiguous or missing title block text

**Solution**: Provide explicit `family` parameter:

```python
plan = plan_job("Punches", {
    "dxf_path": "punch.dxf",
    "family": "pilot_pin",  # override classification
})
```

### Issue: Zero dimensions

**Cause**: DXF file not accessible or malformed

**Solution**: Check `summary.warnings` for errors

```python
summary = extract_punch_features(dxf_path)
if summary.warnings:
    print("Warnings:", summary.warnings)
if summary.confidence_score < 0.5:
    print("Low confidence extraction!")
```

## References

- **Implementation Plan**: `docs/DWG_Punch_Extraction_Readme.md`
- **Usage Examples**: `examples/dwg_punch_extraction_example.py`
- **Unit Tests**: `tests/geometry/test_dwg_punch_extractor.py`
- **Punch Planner**: `cad_quoter/planning/punch_planner.py`
- **Feature Extractor**: `cad_quoter/geometry/dwg_punch_extractor.py`

## Support

For issues or questions:

1. Check the examples: `examples/dwg_punch_extraction_example.py`
2. Review test cases: `tests/geometry/test_dwg_punch_extractor.py`
3. Check `summary.warnings` for extraction issues
4. Review `summary.confidence_score` (< 0.5 = low quality)

---

**Last Updated**: 2025-11-17
**Version**: 1.0
**Status**: Production Ready ✅
