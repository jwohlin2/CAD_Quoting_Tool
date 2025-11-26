# CAD Text Extraction Files - Redundancy Analysis

## Summary of Findings

**REDUNDANT FILES IDENTIFIED:**
1. `cad_quoter/geometry/dxf_text.py` - **OLD/SUPERSEDED** by geo_extractor.py
2. `examples/debug_punch_extraction.py` - **PARTIAL REDUNDANCY** with extract_cad_text_sidecar.py

---

## Complete File Inventory

### Core Text Extraction (Keep These)

#### 1. `cad_quoter/geo_extractor.py` (696 lines)
**Purpose:** Low-level CAD text extraction engine
**Status:** ✅ **KEEP** - Core functionality
**Features:**
- Opens DWG/DXF files
- Extracts TEXT, MTEXT, TABLE, PROXY_ENTITY, DIMENSION, MLEADER
- Recursive block exploration
- Unicode decoding
- Proxy entity fragment merging (for HOLE TABLEs)

**Used by:**
- geo_dump.py
- All production code
- extract_cad_text_sidecar.py
- debug_punch_extraction.py
- Tests

---

#### 2. `cad_quoter/geo_dump.py` (672 lines)
**Purpose:** High-level text extraction + HOLE TABLE parsing
**Status:** ✅ **KEEP** - Production tool
**Features:**
- Full text dump to CSV/JSONL
- HOLE TABLE detection and parsing
- Machining operations extraction
- Stock dimension inference
- CLI interface: `python -m cad_quoter.geo_dump`

**API Functions:**
- `extract_all_text_from_file(path)` - Public API
- `extract_hole_table_from_file(path)`
- `extract_hole_operations_from_file(path)`

---

### Potentially Redundant Files

#### 3. `cad_quoter/geometry/dxf_text.py` (91 lines) ⚠️
**Purpose:** Simple text extraction helper
**Status:** 🔴 **LIKELY REDUNDANT** - Superseded by geo_extractor.py

**Why it might be redundant:**
- Much simpler/older implementation than geo_extractor.py
- Only extracts TEXT/MTEXT from modelspace + layouts
- Doesn't handle PROXY_ENTITY, TABLE, DIMENSION, blocks deeply
- Only used in `cad_quoter/geometry/__init__.py` export (legacy compatibility)
- **NOT actively called** in main application code paths

**Function:**
- `extract_text_lines_from_dxf(path)` → Returns list of text strings

**Recommendation:**
- ✅ Can likely be **REMOVED** or **DEPRECATED**
- If removed, update `cad_quoter/geometry/__init__.py` to remove export
- Check if any external code depends on it first
- Consider keeping for backward compatibility if unsure

---

### Testing/Debug Scripts

#### 4. `extract_cad_text_sidecar.py` (224 lines) - NEW
**Purpose:** Standalone testing script for TEXT EXTRACTION ONLY
**Status:** ✅ **KEEP** - New testing tool
**Features:**
- Multiple output formats (text-only, JSON, CSV, human-readable)
- Uses geo_extractor.py internally
- CLI interface for quick testing
- Simple and focused on text extraction only

---

#### 5. `examples/debug_punch_extraction.py` (297 lines)
**Purpose:** Debug script for PUNCH FEATURES (includes text extraction)
**Status:** ⚠️ **PARTIAL REDUNDANCY**

**Overlap with extract_cad_text_sidecar.py:**
- Lines 59-86: Text extraction code (REDUNDANT - duplicates sidecar functionality)
- Saves text dump to debug/[filename]_text_dump.txt

**Unique functionality (NOT redundant):**
- Punch-specific feature extraction (classification, material, ops)
- Dimension text resolution testing
- Tolerance detection
- Full punch extraction validation
- Manufacturing feature analysis

**Recommendation:**
- ✅ **KEEP** but could be refactored to use extract_cad_text_sidecar.py
- The text extraction portion could call the sidecar script instead
- Or use `geo_dump.extract_all_text_from_file()` API directly

---

#### 6. `examples/dwg_punch_extraction_example.py` (261 lines)
**Purpose:** Example/demo of punch extraction workflow
**Status:** ✅ **KEEP** - Documentation/examples
**Redundancy:** None - uses API functions, doesn't duplicate extraction logic

---

#### 7. `examples/extract_cad_features_example.py` (313 lines)
**Purpose:** Example/demo of comprehensive feature extraction
**Status:** ✅ **KEEP** - Documentation/examples
**Redundancy:** None - high-level examples only

---

### Test Files (Keep All)

#### 8. `tests/test_geo_extractor.py` (217 lines)
**Status:** ✅ **KEEP** - Unit tests for geo_extractor

#### 9. `tests/test_geo_dump.py` (27 lines)
**Status:** ✅ **KEEP** - Unit tests for geo_dump

---

### Special Purpose Tools

#### 10. `tools/paddle_dims_extractor.py` (58K lines)
**Purpose:** OCR-based dimension extraction using PaddleOCR
**Status:** ✅ **KEEP** - Different approach (OCR vs DXF parsing)
**Redundancy:** None - uses image OCR, not DXF text entities

#### 11. `cad_quoter/geometry/mtext_normalizer.py`
**Purpose:** MTEXT formatting code normalization
**Status:** ✅ **KEEP** - Specialized utility used by extractors

---

## Recommendations

### 1. Remove (Safe to Delete)
- ✅ `cad_quoter/geometry/dxf_text.py` (91 lines)
  - Superseded by geo_extractor.py
  - Not actively used in production code
  - Only exists for legacy export compatibility
  - **Action:** Delete file and remove from `cad_quoter/geometry/__init__.py`

### 2. Refactor (Optional Improvement)
- ⚠️ `examples/debug_punch_extraction.py` (lines 59-86)
  - Replace text extraction code with:
    ```python
    from cad_quoter.geo_dump import extract_all_text_from_file
    text_records = extract_all_text_from_file(dxf_path)
    text_lines = [rec["text"] for rec in text_records]
    ```
  - This removes ~30 lines of duplicate code

### 3. Keep Everything Else
- All other files serve distinct purposes

---

## Usage Comparison

### If you want to extract text from a CAD file:

**Option 1: Production API (Recommended)**
```python
from cad_quoter.geo_dump import extract_all_text_from_file
records = extract_all_text_from_file("drawing.dxf")
```

**Option 2: CLI Tool**
```bash
python -m cad_quoter.geo_dump "drawing.dxf"
```

**Option 3: Testing Script**
```bash
python extract_cad_text_sidecar.py "drawing.dxf" --format json
```

**Option 4: OLD/DEPRECATED (Don't use)**
```python
from cad_quoter.geometry import extract_text_lines_from_dxf  # OLD, limited
texts = extract_text_lines_from_dxf("drawing.dxf")
```

---

## Action Items

### Immediate (Safe to do now):
1. ✅ Delete `cad_quoter/geometry/dxf_text.py`
2. ✅ Remove it from `cad_quoter/geometry/__init__.py` exports
3. ✅ Update tests/conftest.py to remove the mock

### Optional (Improvement):
4. ⚠️ Refactor `examples/debug_punch_extraction.py` to use API instead of duplicating extraction code

### Testing before deletion:
```bash
# Search for any external usage
grep -r "extract_text_lines_from_dxf" --include="*.py" | grep -v test | grep -v __init__
grep -r "from.*dxf_text" --include="*.py" | grep -v test
```

---

## File Size Summary

```
Core Extraction:
  geo_extractor.py:               696 lines ✅ KEEP
  geo_dump.py:                    672 lines ✅ KEEP
  dxf_text.py:                     91 lines 🔴 DELETE

Testing/Debug:
  extract_cad_text_sidecar.py:    224 lines ✅ KEEP (new)
  debug_punch_extraction.py:      297 lines ⚠️  REFACTOR
  dwg_punch_extraction_example:   261 lines ✅ KEEP
  extract_cad_features_example:   313 lines ✅ KEEP

Tests:
  test_geo_extractor.py:          217 lines ✅ KEEP
  test_geo_dump.py:                27 lines ✅ KEEP

Special:
  paddle_dims_extractor.py:     1,500+ lines ✅ KEEP (different tech)
  mtext_normalizer.py:            ~500 lines ✅ KEEP (utility)
```

---

## Conclusion

**1 file can be safely deleted:**
- `cad_quoter/geometry/dxf_text.py` - Superseded by geo_extractor.py

**1 file could be improved (optional):**
- `examples/debug_punch_extraction.py` - Refactor to use API

**Estimated cleanup:** ~120 lines of redundant code removed
