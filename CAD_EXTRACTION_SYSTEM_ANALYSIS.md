# CAD Extraction System - Comprehensive Analysis

## Executive Summary

The CAD Quoting Tool has a **text-focused CAD extraction system** that:
- ✅ Extracts text content from DXF/DWG files (text, dimensions, tables, annotations)
- ✅ Extracts geometric features from STEP/IGES files (via OpenCASCADE/OCCT)
- ✅ Parses hole table specifications and operations
- ❌ Does NOT extract geometry from DXF/DWG files (circles, lines, polylines)
- ❌ Does NOT extract actual dimension values or geometry-based metrics from 2D files

---

## 1. CAD EXTRACTION MODULES & FILES

### Core Extraction Files:

#### A. Text Extraction (DXF/DWG Focus)
**File**: `/home/user/CAD_Quoting_Tool/cad_quoter/geo_extractor.py` (22KB)
- **Purpose**: Extract ALL human-visible text from DXF/DWG files
- **Key Data Structure**: `TextRecord` (dataclass with 10 fields)
- **Supported Entity Types**: TEXT, MTEXT, ATTRIB, ATTDEF, DIMENSION, MLEADER, TABLE, ACAD_PROXY_ENTITY
- **Outputs**: dxf_text_dump.csv, dxf_text_dump.jsonl

#### B. 3D Feature Extraction (STEP/IGES Focus)
**File**: `/home/user/CAD_Quoting_Tool/cad_quoter/geometry/__init__.py` (1380 lines)
- **Purpose**: Extract geometric features from STEP/IGES/BREP/STL files
- **Key Functions**:
  - `extract_features_with_occ()` - Main extraction for STEP/IGES
  - `enrich_geo_occ()` - OCC-based feature calculation
  - `enrich_geo_stl()` - STL mesh analysis
  - `load_cad_any()` - Multi-format loader

#### C. DXF Enrichment & Processing
**File**: `/home/user/CAD_Quoting_Tool/cad_quoter/geometry/dxf_enrich.py` (600+ lines)
- **Purpose**: Process DXF content, extract holes, material, dimensions
- **Key Functions**:
  - `harvest_hole_geometry()` - Extract hole features from DXF
  - `harvest_hole_table()` - Parse hole table structures
  - `harvest_outline_metrics()` - Extract part outline
  - `harvest_plate_dimensions()` - Extract plate/stock dimensions
  - `detect_units_scale()` - Unit detection

#### D. Hole Table Parsing
**File**: `/home/user/CAD_Quoting_Tool/cad_quoter/geometry/hole_table_parser.py` (200+ lines)
- **Data Structure**: `HoleRow` (dataclass: ref, qty, features, raw_desc)
- **Purpose**: Parse structured hole table data from text
- **Supports**: Letter drills (A-Z), number drills (#1-#20), metric threads, inch fractions

#### E. Hole Operations Processing
**File**: `/home/user/CAD_Quoting_Tool/tools/hole_ops.py` (300+ lines)
- **Data Structure**: `HoleSpec` (dataclass: name, ref, qty, value, aliases)
- **Purpose**: Convert hole table rows into atomic machining operations
- **Operations**: DRILL, TAP, COUNTERBORE, COUNTERSINK, THRU, FROM FRONT/BACK

#### F. Main Dump/Organization
**File**: `/home/user/CAD_Quoting_Tool/cad_quoter/geo_dump.py` (300+ lines)
- **Purpose**: Organize extracted text into structured tables
- **Outputs**: hole_table_structured.csv, hole_table_ops.csv

#### G. Geometry Fallbacks (Lightweight)
**File**: `/home/user/CAD_Quoting_Tool/cad_quoter/geometry_fallbacks.py` (200+ lines)
- **Purpose**: Provide geometry metrics without heavy OCCT
- **Functions**:
  - `map_geo_to_double_underscore()` - Convert GEO-* to GEO__ format
  - `collect_geo_features_from_df()` - Extract from dataframes
  - `update_variables_df_with_geo()` - Update variable tables

### Supporting Files:

- `/home/user/CAD_Quoting_Tool/cad_quoter/geometry/occ_compat.py` - OCCT compatibility layer
- `/home/user/CAD_Quoting_Tool/cad_quoter/geometry/dxf_text.py` - Text extraction helpers
- `/home/user/CAD_Quoting_Tool/cad_quoter/geometry/hole_table_adapter.py` - Hole table adapters
- `/home/user/CAD_Quoting_Tool/cad_quoter/geometry/mtext_utils.py` - MTEXT formatting
- `/home/user/CAD_Quoting_Tool/cad_quoter/vendors/occt_core.py` - OCCT core exports
- `/home/user/CAD_Quoting_Tool/cad_quoter/vendors/_occt_base.py` - OCCT base layer

---

## 2. FEATURES CURRENTLY EXTRACTED FROM CAD FILES

### A. FROM DXF/DWG FILES (Text-Based)

#### Text Content Extraction:
| Feature | Status | Details |
|---------|--------|---------|
| TEXT entities | ✅ | Basic text, insertion point, height, rotation |
| MTEXT entities | ✅ | Formatted text, plain_text() conversion |
| ATTRIB/ATTDEF | ✅ | Attributes, insertion points |
| DIMENSION entities | ⚠️ | Text only (not measurement values) |
| MLEADER | ✅ | Leader text/annotations |
| TABLE/ACAD_TABLE | ✅ | Cell contents extraction |
| ACAD_PROXY_ENTITY | ✅ | AutoCAD Mechanical hole tables |

#### Text Processing Features:
- Unicode decoding (`\U+XXXX` → Unicode characters)
- Symbol conversion (`%%c` → Ø, `%%d` → °, `%%p` → ±)
- MTEXT control code stripping (`\P` → newlines)
- Proxy fragment merging (reassemble split text)
- Layer and layout organization
- Block nesting support (up to 8 levels deep)

#### Hole Table Features (Text-Parsed):
| Feature | Extraction | Quality |
|---------|-----------|---------|
| Reference Letter | ✅ | A-Z parsing works |
| Diameter | ⚠️ | Regex-based, often fails |
| Tap Thread Spec | ⚠️ | Partial matches |
| Depth | ⚠️ | "X N DEEP" pattern matching |
| Side (FRONT/BACK) | ✅ | "FROM FRONT/BACK" detection |
| Quantity | ⚠️ | Limited pattern matching |
| Operations | ⚠️ | CBORE, CDRILL, TAP, THRU detection |

### B. FROM STEP/IGES/BREP/STL FILES (Geometry-Based)

#### 3D Geometry Features:
| Feature | Extraction | Value |
|---------|-----------|-------|
| Bounding Box (X,Y,Z) | ✅ | `GEO-01_Length_mm`, `GEO-02_Width_mm`, `GEO-03_Height_mm` |
| Volume | ✅ | `GEO-Volume_mm3` |
| Surface Area | ✅ | `GEO-SurfaceArea_mm2` |
| Face Count | ✅ | `Feature_Face_Count` |
| Surface Type Distribution | ✅ | Planar, Cylindrical, Freeform areas |
| Face Normals | ✅ | Unique normal directions |
| Minimum Wall Thickness | ✅ | `GEO_MinWall_mm` (for parallel planes) |
| Thin Wall Detection | ✅ | `GEO_ThinWall_Present` (< 1mm) |
| WEDM Path Length | ✅ | `GEO_WEDM_PathLen_mm` (perimeter at Z-sections) |
| Deburr Edge Length | ✅ | `GEO_Deburr_EdgeLen_mm` (sharp edges > 175°) |
| Hole Groups | ✅ | Cylinder diameter, depth, through/blind, count |
| Complexity Score | ✅ | `GEO_Complexity_0to100` (faces/volume ratio) |
| 3-Axis Accessibility | ✅ | `GEO_3Axis_Accessible_Pct` (face normal alignment) |
| Center of Mass | ✅ | `GEO_CenterOfMass` (X,Y,Z) |
| Turning Score | ✅ | `GEO_Turning_Score_0to1` (cylindrical ratio) |
| Max OD/Length | ✅ | `GEO_MaxOD_mm`, `GEO_Length_mm` |
| Largest Planar Face | ✅ | `GEO_LargestPlane_Area_mm2` |

#### STL-Specific Features:
- Mesh validity checking
- Face count
- Volume calculation
- Surface area
- Complexity metrics
- 3-axis accessibility

---

## 3. DATA STRUCTURES FOR STORING FEATURES

### A. Text Records (DXF Extraction)
```python
@dataclass(frozen=True)
class TextRecord:
    layout: str           # "Model", "SHEET (B)", etc.
    layer: str            # Layer name
    etype: str            # Entity type: TEXT, MTEXT, TABLE, etc.
    text: str             # Extracted text content
    x: float              # Insertion point X
    y: float              # Insertion point Y
    height: float         # Text height
    rotation: float       # Rotation angle (degrees)
    in_block: bool        # If within a block
    depth: int            # Block nesting depth
    block_path: Tuple     # Path of block hierarchy
```

### B. Hole Row (Parsed Hole Table)
```python
@dataclass
class HoleRow:
    ref: str                          # Reference letter (A, B, C...)
    qty: int                          # Quantity
    features: List[Dict[str, Any]]    # Machining operations
    raw_desc: str                     # Original text description
```

### C. Hole Spec (Operation Definition)
```python
@dataclass
class HoleSpec:
    name: str              # "Hole A", etc.
    ref: str               # Reference letter
    qty: str               # Quantity string
    value: float           # Diameter in inches
    aliases: set[str]      # Alternative representations
```

### D. Geometry Features Dictionary (OCC Output)
```python
{
    # Dimensions (mm)
    "GEO-01_Length_mm": float,
    "GEO-02_Width_mm": float,
    "GEO-03_Height_mm": float,
    
    # Volume & Area
    "GEO-Volume_mm3": float,
    "GEO-SurfaceArea_mm2": float,
    
    # Face Analysis
    "Feature_Face_Count": int,
    "GEO_Area_Planar_mm2": float,
    "GEO_Area_Cyl_mm2": float,
    "GEO_Area_Freeform_mm2": float,
    "GEO_LargestPlane_Area_mm2": float,
    
    # Wall Analysis
    "GEO_MinWall_mm": float | None,
    "GEO_ThinWall_Present": bool,
    
    # Machining Metrics
    "GEO_WEDM_PathLen_mm": float,
    "GEO_Deburr_EdgeLen_mm": float,
    
    # Hole Analysis
    "GEO_Hole_Groups": List[{
        "dia_mm": float,
        "depth_mm": float,
        "through": bool,
        "count": int
    }],
    
    # Manufacturing Properties
    "GEO_Complexity_0to100": float,
    "GEO_3Axis_Accessible_Pct": float,
    "GEO_Setup_UniqueNormals": int,
    "GEO_Turning_Score_0to1": float,
    "GEO_MaxOD_mm": float,
    
    # Location
    "GEO_CenterOfMass": List[float],  # [X, Y, Z]
    
    # Backend info
    "OCC_Backend": str  # "OCC" or "trimesh (STL)"
}
```

### E. Extracted Geometry Features (Legacy Format - GEO__)
```python
{
    "GEO__BBox_X_mm": float,
    "GEO__BBox_Y_mm": float,
    "GEO__BBox_Z_mm": float,
    "GEO__MaxDim_mm": float,
    "GEO__MinDim_mm": float,
    "GEO__Stock_Thickness_mm": float,
    "GEO__Volume_mm3": float,
    "GEO__SurfaceArea_mm2": float,
    "GEO__Face_Count": float,
    "GEO__WEDM_PathLen_mm": float,
    "GEO__Area_to_Volume": float,
}
```

---

## 4. CAD FILE FORMATS SUPPORTED

### Fully Supported (with Geometry Extraction):
| Format | Extension | Library | Level | Capabilities |
|--------|-----------|---------|-------|--------------|
| STEP | .step, .stp | OCCT | Full | Complete geometry extraction |
| IGES | .iges, .igs | OCCT | Full | Complete geometry extraction |
| BREP | .brep | OCCT | Full | OpenCASCADE native format |
| STL | .stl | trimesh | Full | Mesh analysis only |

### Partially Supported (Text Only):
| Format | Extension | Library | Level | Capabilities |
|--------|-----------|---------|-------|--------------|
| DXF | .dxf | ezdxf | Text | Text, dimensions, hole tables |
| DWG | .dwg | ezdxf (with ODA) | Text | Converts to DXF first |

### Supported DXF Versions:
- AutoCAD R12 through recent versions
- Supports all DXF entity types including AutoCAD Mechanical extensions

---

## 5. LIBRARIES & TOOLS BEING USED

### CAD Processing Libraries:

#### A. OCCT (Open Cascade Technology) - 3D Solid Model Analysis
**Location**: `/cad_quoter/vendors/occt*.py`
**Bindings**: 
- Primary: `OCP` (pythonocc-core wheels)
- Fallback: `pythonocc-core`

**Core Classes Used**:
```
STEPControl_Reader        - STEP file reading
IGESControl_Reader        - IGES file reading
TopoDS_Shape              - Base shape representation
TopoDS_Face, TopoDS_Edge  - Geometric primitives
TopExp_Explorer           - Shape traversal
BRepAdaptor_Surface       - Surface analysis
GeomAbs_*                 - Geometry type enumerations
GProp_GProps              - Mass properties (volume, area, center)
ShapeFix_Shape            - Shape healing
BRepCheck_Analyzer        - Validity checking
```

**Capabilities**:
- Read STEP, IGES, BREP formats
- Traverse shape topology (solids, shells, faces, edges)
- Calculate volumes, surface areas, mass properties
- Analyze surface types (planar, cylindrical, conical, etc.)
- Extract face normals and accessibility analysis

#### B. ezdxf - DXF/DWG Text Extraction
**Location**: `/cad_quoter/vendors/` (vendored)
**Version**: 1.0+
**Key Imports**:
```
ezdxf.readfile()          - DXF reading
ezdxf.recover.readfile()  - Recovery mode for malformed files
ProxyGraphic              - AutoCAD Mechanical proxy parsing
```

**Capabilities**:
- Read/parse DXF files
- Recover corrupted DXF structures
- Access all entity types (TEXT, MTEXT, CIRCLE, etc.)
- Extract proxy graphics from AutoCAD Mechanical
- Support for blocks and attributes

#### C. trimesh - STL Mesh Analysis
**Version**: 3.22+
**Capabilities**:
- Load STL mesh files
- Calculate mesh metrics (volume, area, face count)
- Analyze mesh sections (cross-sections)
- Extract face normals and edge angles
- Compute center of mass

#### D. ODA File Converter - DWG Conversion
**Env Variables**:
- `ODA_CONVERTER_EXE` - Path to ODAFileConverter
- `DWG2DXF_EXE` - Path to custom DWG converter

**Purpose**: Convert DWG → DXF before text extraction

#### E. Regular Expression Processing
**Location**: Throughout codebase
**Key Patterns**:
- Diameter tokens: `Ø\s*(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?|\.\d+)`
- Tap specs: `#\s*\d{1,2}-\d+|M\d+(?:\.\d+)?x\d+(?:\.\d+)?`
- Depth: `[Xx]\s*([0-9.]+)\s*DEEP`
- Side: `FROM\s+(FRONT|BACK)`
- Quantities: `\((\d+)\)`
- Material: `\b(MATERIAL|MAT)\b[:\s]*([A-Z0-9\-\s/\.]+)`

---

## 6. EXTRACTION PIPELINE ARCHITECTURE

### Two Parallel Paths:

#### Path 1: 2D CAD Files (DXF/DWG) - TEXT FOCUS
```
DXF/DWG File
    ↓
ezdxf.readfile()
    ↓
iter_text_records() → Extract TEXT, MTEXT, DIMENSION, TABLE, PROXY entities
    ↓
TextRecord dataclass
    ↓
dxf_text_dump.csv/jsonl
    ↓
harvest_hole_geometry() / harvest_hole_table()
    ↓
parse_hole_table_lines() → HoleRow objects
    ↓
hole_ops.explode_rows_to_operations()
    ↓
hole_table_structured.csv / hole_table_ops.csv
```

#### Path 2: 3D CAD Files (STEP/IGES/BREP/STL) - GEOMETRY FOCUS
```
STEP/IGES/BREP/STL File
    ↓
read_step_or_iges_or_brep() OR trimesh.load()
    ↓
TopoDS_Shape / Mesh object
    ↓
enrich_geo_occ() / enrich_geo_stl()
    ↓
Extract geometric features:
  - Bounding box, volume, surface area
  - Face analysis (planar/cylindrical/freeform)
  - Hole detection (from cylindrical faces)
  - Wall thickness, accessibility
  - Complexity metrics
    ↓
Geometry Dictionary (GEO-* format)
    ↓
map_geo_to_double_underscore()
    ↓
GEO__ metrics for quoting engine
```

---

## 7. CURRENT EXTRACTION CAPABILITIES

### ✅ WORKING WELL:

1. **Text Extraction from DXF/DWG**
   - All text entity types captured
   - Unicode/symbol handling robust
   - Proxy entity text reconstruction
   - Multi-layout support

2. **Geometry Feature Extraction from STEP/IGES**
   - Bounding box calculations accurate
   - Volume/surface area reliable
   - Face classification (planar/cylindrical/freeform)
   - Hole detection functional for simple geometries
   - Complexity scoring works

3. **Hole Table Row Parsing**
   - Letter and number drill lookups accurate
   - Reference letters extracted
   - Basic operation detection (TAP, CBORE, etc.)
   - Quantity parsing from standard patterns

4. **Unit Detection & Conversion**
   - Detects INSUNITS header (inches, mm, feet, etc.)
   - Proper conversion factors applied
   - Falls back gracefully on missing data

### ⚠️ PARTIALLY WORKING:

1. **Dimension Value Extraction from DXF**
   - Gets dimension TEXT but not measurement values
   - No dimension type classification
   - No leader/extension line geometry

2. **Hole Diameter Parsing**
   - Regex patterns inconsistent
   - Fails on malformed text
   - No validation against actual geometry

3. **Material Property Extraction**
   - Gets material code if in text
   - No structured material database lookup
   - No hardness/density information

4. **Hole Table Structure Detection**
   - Works for standard AutoCAD Mechanical tables
   - Fails on custom table formats
   - Proxy text often merged as single string

### ❌ NOT IMPLEMENTED:

1. **Geometry Extraction from DXF/DWG**
   - ❌ Circle/arc positions (no hole locations)
   - ❌ Polyline profiles (no part outline)
   - ❌ Line geometry (no edge data)
   - ❌ Spatial relationships between entities

2. **Dimension Analysis**
   - ❌ Measured values (only text captured)
   - ❌ Dimension validation
   - ❌ Tolerance application
   - ❌ Dimension type classification

3. **BOM Extraction**
   - ❌ Bill of Materials parsing
   - ❌ Part number lookups
   - ❌ Assembly structure

4. **Title Block Extraction**
   - ❌ Drawing number
   - ❌ Revision history
   - ❌ Author/date metadata
   - ❌ Document control information

5. **Advanced Features**
   - ❌ Feature recognition (pockets, bosses, fillets)
   - ❌ GD&T extraction
   - ❌ Assembly relationships
   - ❌ Part hierarchies

---

## 8. MAJOR GAPS vs. COMPREHENSIVE REQUIREMENTS

### Gap Analysis Table:

| Capability | Current | Required | Gap Severity |
|-----------|---------|----------|--------------|
| **STEP/IGES File Support** | ✅ Full | ✅ Yes | ✓ CLOSED |
| **3D Geometry Analysis** | ✅ Comprehensive | ✅ Required | ✓ CLOSED |
| **Hole Detection (3D)** | ✅ From cylinders | ✅ Required | ✓ CLOSED |
| **DXF Geometry Extraction** | ❌ None | ✅ Required | **CRITICAL** |
| **Circle/Hole Locations (DXF)** | ❌ None | ✅ Required | **CRITICAL** |
| **Actual Dimension Values** | ❌ Text only | ✅ Required | **HIGH** |
| **Part Profile/Outline** | ❌ None | ✅ Important | **HIGH** |
| **Material Properties** | ⚠️ Text only | ✅ Required | **HIGH** |
| **Hole Location Matching** | ❌ None | ✅ Important | **MEDIUM** |
| **Feature Recognition** | ❌ None | ✅ Nice-to-have | **MEDIUM** |
| **BOM Parsing** | ❌ None | ⚠️ Nice-to-have | **LOW** |
| **Title Block Data** | ❌ None | ⚠️ Nice-to-have | **LOW** |

### Critical Missing Piece:
**DXF files contain NO geometry data in current extraction.**

The system reads all text but ignores:
- CIRCLE entities (actual hole positions)
- LWPOLYLINE/POLYLINE entities (part outlines)
- LINE entities (edge geometry)
- ARC entities (curved edges)

**Impact**: For a manufacturing quoting tool using DXF drawings:
- Cannot determine actual hole positions (x, y coordinates)
- Cannot calculate part profile area
- Cannot validate text dimensions against geometry
- Cannot generate CNC programs
- Cannot estimate machining time from geometry

---

## 9. RECENT WORK & DOCUMENTATION

### Latest Analysis (Commit: 8f054bf)
**File**: `/home/user/CAD_Quoting_Tool/docs/cad_extraction_analysis.md`

This document (511 lines) provides:
- Detailed breakdown of what's extracted vs. missing
- Comparison to actual drawing content
- Specific code recommendations for improvements
- Prioritized implementation roadmap

### Key Findings from Analysis:
1. System excels at text extraction but misses geometry
2. Hole table parsing needs improvement for consistency
3. STEP/IGES support is comprehensive and working well
4. DXF support needs geometry extraction layer

---

## 10. TESTING & VALIDATION

### Test Files Located:
- `/tests/test_geo_extractor.py` - Main extraction tests
- `/tests/unit/test_geo_extractor_fallback.py` - Fallback tests
- `/tests/geometry/test_geometry_utils.py` - Geometry utility tests
- `/tests/test_hole_table_adapter.py` - Hole table tests

### Test Capabilities:
- Text record extraction validation
- Hole table parsing verification
- Proxy entity handling
- Unicode/symbol conversion
- Operation classification (TAP, DRILL, etc.)

### Known Test Cases:
- Hole table row stitching and splitting
- Operation kind classification
- Quantity extraction
- Side detection (FRONT/BACK)
- Material property extraction

---

## Summary Table

| Aspect | Status | Details |
|--------|--------|---------|
| **Text Extraction** | ✅ Robust | DXF/DWG text fully supported |
| **3D Geometry** | ✅ Comprehensive | STEP/IGES/BREP/STL well-implemented |
| **Hole Features** | ✅ Partial | Works for 3D, limited for 2D |
| **Dimension Values** | ⚠️ Limited | Text only, no measurement data |
| **2D Geometry** | ❌ Missing | No circle, line, polyline extraction |
| **Material Data** | ⚠️ Basic | Text parsing only, no properties |
| **Performance** | ✅ Good | Lazy loading, multi-format support |
| **Extensibility** | ✅ Good | Modular architecture, clear interfaces |

