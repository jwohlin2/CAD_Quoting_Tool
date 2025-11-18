# Cost Helper Modules Consolidation Analysis

## Executive Summary

**DirectCostHelper.py** and **QuoteDataHelper.py** contain significantly overlapping dataclasses and functions that represent the same domain concepts in different ways. This analysis identifies:

- **3 duplicate/conflicting dataclasses**: `PartInfo`/`PartDimensions`, `ScrapInfo` (2 versions), and implicit material handling
- **Architectural inconsistency**: DirectCostHelper mixes concerns; QuoteDataHelper correctly separates them
- **Code duplication**: Identical volume/area calculation logic in `__post_init__` methods
- **Conversion overhead**: QuoteDataHelper must transform between dataclass representations

---

## Current Structure Analysis

### File 1: DirectCostHelper.py (1,298 lines)

**Primary Purpose**: Direct cost calculations (McMaster pricing, scrap value)

**Dataclasses** (2):
```
PartInfo (6 fields)
├── length, width, thickness (dimensions)
├── material (string, default="aluminum MIC6")
├── volume, area (derived in __post_init__)
└── Use: Extraction functions return this; used in cost calculations

ScrapInfo (29 fields - HEAVYWEIGHT)
├── McMaster stock dims (mcmaster_length/width/thickness/volume)
├── Desired starting dims (desired_length/width/thickness/volume)
├── Part envelope dims (part_length/width/thickness/envelope_volume/final_volume)
├── Scrap breakdown (stock_prep, face_milling, hole_drilling, total_volume)
├── Material info (material, density)
├── Weights (mcmaster_weight, final_part_weight, total_scrap_weight)
├── Percentages (scrap_percentage, utilization_percentage)
└── NOTE: No scrap value fields!
```

**Key Functions**:
- `extract_part_info_from_plan(plan, material)` → PartInfo
- `extract_part_info_from_cad()` → PartInfo
- `calculate_total_scrap()` → ScrapInfo
- `calculate_scrap_value()` → Dict (not dataclass!)
- McMaster API functions: `get_mcmaster_part_number()`, `get_mcmaster_price()`

### File 2: QuoteDataHelper.py (1,547 lines)

**Primary Purpose**: Unified quote data aggregation; serialization to JSON

**Dataclasses** (8):
```
PartDimensions (5 fields)
├── length, width, thickness (dimensions only - NO material!)
├── volume, area (derived in __post_init__)
└── Use: Stored in QuoteData

MaterialInfo (5 fields) ← NEW, separates concerns
├── material_name, material_family
├── density, detected_from_cad, is_default
└── Use: Stored in QuoteData

StockInfo (11 fields) ← NEW, separates stock from scrap
├── Desired stock (desired_length/width/thickness/volume)
├── McMaster stock (mcmaster_length/width/thickness/volume)
├── McMaster pricing (part_number, price, price_is_estimated)
├── Weights (mcmaster_weight, final_part_weight)
└── Use: Stored in QuoteData

ScrapInfo (10 fields - MINIMAL, CLEAN)
├── Scrap breakdown (stock_prep, face_milling, hole_drilling, total_volume/weight)
├── Percentages (scrap_percentage, utilization_percentage)
├── Scrap value (price_per_lb, value, price_source) ← ADDED
└── Use: Stored in QuoteData

Plus: HoleOperation, MillingOperation, GrindingOperation, MachineHoursBreakdown,
      LaborHoursBreakdown, CostSummary, DirectCostBreakdown, QuoteData
```

**Key Functions**:
- `extract_quote_data_from_cad()` → QuoteData (main orchestrator)
- Imports & uses 7 functions from DirectCostHelper

---

## Detailed Comparison

### 1. PartInfo vs PartDimensions - DUPLICATE

**DirectCostHelper.PartInfo** (20 lines):
```python
@dataclass
class PartInfo:
    length: float = 0.0
    width: float = 0.0
    thickness: float = 0.0
    material: str = DEFAULT_MATERIAL  # ← Includes material
    volume: float = 0.0
    area: float = 0.0
    
    def __post_init__(self):
        self.volume = self.length * self.width * self.thickness
        self.area = self.length * self.width
```

**QuoteDataHelper.PartDimensions** (15 lines):
```python
@dataclass
class PartDimensions:
    length: float = 0.0
    width: float = 0.0
    thickness: float = 0.0
    volume: float = 0.0
    area: float = 0.0
    
    def __post_init__(self):
        if self.volume == 0.0:
            self.volume = self.length * self.width * self.thickness
        if self.area == 0.0:
            self.area = self.length * self.width
```

**Key Differences**:
- PartInfo includes `material`; PartDimensions doesn't
- PartDimensions has smarter `__post_init__` (checks if already set)
- Identical calculation logic for volume/area (DUPLICATION)

**Current Usage in QuoteDataHelper** (line 947-956):
```python
part_info = extract_part_info_from_plan(plan, material)

quote_data.part_dimensions = PartDimensions(
    length=part_info.length,
    width=part_info.width,
    thickness=part_info.thickness,
    volume=part_info.volume,
    area=part_info.area
)
# Material handled separately in MaterialInfo
```

**Problem**: Data conversion overhead + inconsistent material handling

---

### 2. ScrapInfo - TWO CONFLICTING VERSIONS

**DirectCostHelper.ScrapInfo** (37 lines, 29 fields):
```python
class ScrapInfo:
    # McMaster stock (3 fields)
    mcmaster_length/width/thickness, mcmaster_volume
    
    # Desired starting stock (4 fields)
    desired_length/width/thickness, desired_volume
    
    # Final part (5 fields)
    part_length/width/thickness, part_envelope_volume, part_final_volume
    
    # Scrap breakdown (4 fields)
    stock_prep_scrap, face_milling_scrap, hole_drilling_scrap, total_scrap_volume
    
    # Material & density (2 fields)
    material, density
    
    # Weights (3 fields)
    mcmaster_weight, final_part_weight, total_scrap_weight
    
    # Percentages (2 fields)
    scrap_percentage, utilization_percentage
    
    # MISSING: scrap_value, scrap_price_per_lb, price_source
```

**QuoteDataHelper.ScrapInfo** (17 lines, 10 fields):
```python
class ScrapInfo:
    # Scrap breakdown (4 fields)
    stock_prep_scrap, face_milling_scrap, hole_drilling_scrap, total_scrap_volume
    
    # Scrap weight & percentages (3 fields)
    total_scrap_weight, scrap_percentage, utilization_percentage
    
    # Scrap value (3 fields)
    scrap_price_per_lb, scrap_value, scrap_price_source
```

**Key Issues**:
- DirectCostHelper version has 3× the fields (bloated, violates SRP)
- Contains dimensions + material that should be elsewhere
- Missing scrap value fields (those come from separate dict return)
- QuoteDataHelper version is cleaner but incomplete (missing dimensions from other dataclasses)

**Current Usage in QuoteDataHelper** (lines 1010-1029):
```python
# Get DirectCostHelper.ScrapInfo (29 fields)
scrap_calc = calculate_total_scrap(...)  # Returns DirectCostHelper.ScrapInfo

# Get scrap value as Dict (not dataclass!)
scrap_value_calc = calculate_scrap_value(...)  # Returns Dict

# Map DirectCostHelper.ScrapInfo → QuoteDataHelper.ScrapInfo (lines 1168-1179)
quote_data.scrap_info = ScrapInfo(
    stock_prep_scrap=scrap_calc.stock_prep_scrap,
    face_milling_scrap=scrap_calc.face_milling_scrap,
    hole_drilling_scrap=scrap_calc.hole_drilling_scrap,
    total_scrap_volume=scrap_calc.total_scrap_volume,
    total_scrap_weight=scrap_calc.total_scrap_weight,
    scrap_percentage=scrap_calc.scrap_percentage,
    utilization_percentage=scrap_calc.utilization_percentage,
    scrap_price_per_lb=scrap_value_calc.get('scrap_price_per_lb'),
    scrap_value=scrap_value_calc.get('scrap_value', 0.0),
    scrap_price_source=scrap_value_calc.get('price_source', '')
)
```

**Problem**: Unnecessary data mapping + mixing return types (dataclass vs dict)

---

### 3. Material Information - IMPLICIT vs EXPLICIT

**DirectCostHelper** (scattered):
- Material passed as string throughout
- Only `get_material_density(material: str)` function
- No material family classification

**QuoteDataHelper.MaterialInfo** (new):
```python
@dataclass
class MaterialInfo:
    material_name: str = "GENERIC"
    material_family: str = "aluminum"  # ← EXPLICIT classification
    density: float = 0.098
    detected_from_cad: bool = False
    is_default: bool = False
```

**Better approach**: MaterialInfo explicitly tracks metadata

---

## Dependency Flow

### Current (Problematic)
```
Tests & AppV7
    ↓
QuoteDataHelper.extract_quote_data_from_cad()
    ↓ imports & uses
DirectCostHelper functions (7 imports):
    - extract_part_info_from_plan → PartInfo
    - get_mcmaster_part_number
    - get_mcmaster_price
    - calculate_total_scrap → ScrapInfo (29 fields!)
    - calculate_scrap_value → Dict
    - get_material_density
    - DEFAULT_MATERIAL
    ↓
QuoteDataHelper then:
    - Maps PartInfo → PartDimensions + MaterialInfo
    - Maps DirectCostHelper.ScrapInfo → QuoteDataHelper.ScrapInfo
    - Merges Dict scrap_value_calc into ScrapInfo
```

**Issues**:
1. Requires callers to map between 3 representations of same data
2. PartInfo includes material but PartDimensions doesn't (inconsistent API)
3. ScrapInfo in DirectCostHelper mixed with unrelated data (violation of Single Responsibility)
4. Function returns mix of dataclasses and dicts

---

## Duplicate Code Analysis

### Calculation Logic Duplication

**DirectCostHelper.PartInfo.__post_init__** (lines 30-33):
```python
def __post_init__(self):
    self.volume = self.length * self.width * self.thickness
    self.area = self.length * self.width
```

**QuoteDataHelper.PartDimensions.__post_init__** (lines 38-43):
```python
def __post_init__(self):
    if self.volume == 0.0:
        self.volume = self.length * self.width * self.thickness
    if self.area == 0.0:
        self.area = self.length * self.width
```

**Status**: EXACT DUPLICATE (with minor conditional check difference)

---

## Callers Analysis

### Direct Imports of DirectCostHelper
- **AppV7.py** (line 640): Not imported directly, only uses QuoteDataHelper
- **tests/test_all_materials.py** (line 29): Imports `calculate_scrap_value`
- **tests/test_volume_pricing.py** (line 7): Imports `estimate_price_from_reference_part`
- **tests/unit/test_plan_dimension_overrides.py** (line 8): Imports `extract_part_info_from_cad`
- **QuoteDataHelper.py** (line 647): Imports 7 functions (MAJOR DEPENDENCY)

### Direct Imports of QuoteDataHelper
- **AppV7.py** (line 640): Imports `extract_quote_data_from_cad`
- **AppV7.py** (line 1814): Imports `save_quote_data`
- **Tests**: `test_all_materials.py`, `test_volume_pricing.py`
- **README.md**: Example usage shows `extract_quote_data_from_cad`

---

## Recommended Consolidation Approach

### OPTION 1: Consolidate into QuoteDataHelper (RECOMMENDED)
**Rationale**: QuoteDataHelper's design is cleaner (separation of concerns), it's the consumer, and it's the newer pattern.

**Steps**:
1. **Keep QuoteDataHelper as primary location**
2. **Move these helper functions from DirectCostHelper → QuoteDataHelper**:
   - `extract_part_info_from_plan()` (but rename to internal or deprecate)
   - `extract_part_info_from_cad()`
   - `extract_dimensions_with_paddle_ocr()`
   - `calculate_material_volume()`, `calculate_material_weight()`
   - `get_material_density()` (or keep as thin wrapper)
   - McMaster functions: `get_mcmaster_part_number()`, `get_mcmaster_price()`, etc.
   - `calculate_stock_prep_scrap()`
   - `calculate_machining_scrap_from_cad()`
   - `calculate_total_scrap()`
   - `calculate_scrap_value()`

3. **Keep DirectCostHelper for backward compatibility** (tests still use it):
   - Deprecate gradually
   - Import functions from QuoteDataHelper and re-export

### OPTION 2: Consolidate into DirectCostHelper
**Rationale**: Keeps "helper" functions close together

**Issues**: 
- Requires redesigning 29-field ScrapInfo
- Less clean separation of concerns
- QuoteDataHelper becomes too large

### Recommended: OPTION 1 with phased approach

---

## Detailed Unification Strategy

### Phase 1: Create Unified Data Model in QuoteDataHelper

**Step 1.1: Enhance PartInfo (in QuoteDataHelper)**
```python
@dataclass
class PartInfo:  # Replaces both versions
    """Complete part information for extraction and cost calculation."""
    length: float = 0.0
    width: float = 0.0
    thickness: float = 0.0
    volume: float = 0.0
    area: float = 0.0
    material: str = "aluminum MIC6"  # ← Keep from DirectCostHelper version
    
    def __post_init__(self):
        """Calculate derived properties if not provided."""
        if self.volume == 0.0:
            self.volume = self.length * self.width * self.thickness
        if self.area == 0.0:
            self.area = self.length * self.width
```

**Why**: Keeps dimension + material together (more intuitive); uses QuoteDataHelper's smarter __post_init__

---

**Step 1.2: Create Unified ScrapInfo (replace both versions)**
```python
@dataclass
class ScrapCalculation:  # New umbrella for complete scrap data
    """Complete scrap calculation results."""
    # Part dimensions
    part_dimensions: PartDimensions  # Reference to part_dimensions
    
    # Scrap breakdown
    stock_prep_scrap: float = 0.0
    face_milling_scrap: float = 0.0
    hole_drilling_scrap: float = 0.0
    total_scrap_volume: float = 0.0
    total_scrap_weight: float = 0.0
    
    # Percentages
    scrap_percentage: float = 0.0
    utilization_percentage: float = 0.0
    
    # Scrap value
    scrap_price_per_lb: Optional[float] = None
    scrap_value: float = 0.0
    scrap_price_source: str = ""

@dataclass  
class StockInfo:  # Keep this, refactor only
    """McMaster stock information and pricing."""
    desired_length: float = 0.0
    desired_width: float = 0.0
    desired_thickness: float = 0.0
    desired_volume: float = 0.0
    
    mcmaster_length: float = 0.0
    mcmaster_width: float = 0.0
    mcmaster_thickness: float = 0.0
    mcmaster_volume: float = 0.0
    mcmaster_part_number: Optional[str] = None
    mcmaster_price: Optional[float] = None
    price_is_estimated: bool = False
    
    mcmaster_weight: float = 0.0
    final_part_weight: float = 0.0
```

**Why**: 
- Separates stock dimensions/pricing (StockInfo) from scrap breakdown (ScrapInfo)
- Each class has single responsibility
- ScrapCalculation can reference PartDimensions instead of duplicating
- Cleaner than DirectCostHelper's 29-field monster

---

**Step 1.3: Update MaterialInfo** (already good)
```python
@dataclass
class MaterialInfo:
    material_name: str = "GENERIC"
    material_family: str = "aluminum"
    density: float = 0.098
    detected_from_cad: bool = False
    is_default: bool = False
```

---

### Phase 2: Move & Refactor Functions

**Step 2.1: Move calculation functions**
```python
# In QuoteDataHelper.py, add these (currently in DirectCostHelper):

def extract_part_info_from_plan(
    plan: Dict[str, Any],
    material: Optional[str] = None
) -> PartInfo:
    """Extract part info from process plan."""
    # Move code from DirectCostHelper line 36-71

def extract_part_info_from_cad(
    cad_file_path: str | Path,
    material: Optional[str] = None,
    use_paddle_ocr: bool = True,
    auto_detect_material: bool = True,
    verbose: bool = False
) -> PartInfo:
    """Extract part info directly from CAD."""
    # Move code from DirectCostHelper line 74-130

# Also move:
# - calculate_material_volume()
# - calculate_material_weight()
# - get_material_density()
# - get_mcmaster_part_number()
# - get_mcmaster_price()
# - get_mcmaster_part_and_price()
# - get_material_cost_from_mcmaster()
# - estimate_price_from_reference_part()
# - calculate_stock_prep_scrap()
# - calculate_machining_scrap_from_cad()
# - calculate_total_scrap() ← REFACTOR to return new ScrapCalculation
# - calculate_scrap_value() ← REFACTOR to return updated ScrapInfo
# - calculate_total_scrap_with_value()
```

**Step 2.2: Update calculate_total_scrap() return type**

**Current** (DirectCostHelper, returns DirectCostHelper.ScrapInfo with 29 fields):
```python
def calculate_total_scrap(...) -> ScrapInfo:  # 29 fields!
    ...
    return ScrapInfo(
        mcmaster_length=...,
        mcmaster_width=...,
        mcmaster_thickness=...,
        mcmaster_volume=...,
        desired_length=...,
        # ... 25 more fields
    )
```

**New** (returns StockInfo + ScrapCalculation):
```python
def calculate_total_scrap(...) -> Tuple[StockInfo, ScrapCalculation]:
    """Calculate scrap, return organized data."""
    # ... calculation logic ...
    
    stock_info = StockInfo(
        desired_length=...,
        desired_width=...,
        desired_thickness=...,
        mcmaster_length=...,
        # ... etc
    )
    
    scrap_info = ScrapCalculation(
        part_dimensions=part_dims,
        stock_prep_scrap=...,
        # ... only scrap-specific fields
    )
    
    return stock_info, scrap_info
```

**Why**: 
- Cleaner API (caller doesn't need to know about 29-field monster)
- Better separation of concerns
- Makes it obvious what data is for what purpose

---

**Step 2.3: Update calculate_scrap_value()**

**Current** (returns Dict):
```python
def calculate_scrap_value(...) -> Dict[str, Any]:
    return {
        'scrap_weight_lbs': ...,
        'scrap_price_per_lb': ...,
        'scrap_value': ...,
        'price_source': ...,
        'material_family': ...
    }
```

**New** (updates ScrapCalculation dataclass):
```python
def calculate_scrap_value(
    scrap_info: ScrapCalculation,
    material: str,
    fallback_scrap_price_per_lb: Optional[float] = None,
    verbose: bool = False
) -> ScrapCalculation:  # Returns updated object
    """Calculate and update scrap value in place."""
    # ... pricing logic ...
    
    scrap_info.scrap_price_per_lb = price
    scrap_info.scrap_value = total_value
    scrap_info.scrap_price_source = source
    
    return scrap_info
```

**Why**: 
- Eliminates Dict return type inconsistency
- All scrap data in one place
- Easier to serialize/deserialize

---

### Phase 3: Update QuoteDataHelper.extract_quote_data_from_cad()

**Current** (lines 1010-1029):
```python
scrap_calc = calculate_total_scrap(...)  # Returns DirectCostHelper.ScrapInfo
scrap_value_calc = calculate_scrap_value(...)  # Returns Dict

# Manual mapping (lines 1168-1179)
quote_data.scrap_info = ScrapInfo(
    stock_prep_scrap=scrap_calc.stock_prep_scrap,
    face_milling_scrap=scrap_calc.face_milling_scrap,
    # ... 8 more field mappings ...
)
```

**New**:
```python
# Returns properly separated data
stock_info, scrap_calc = calculate_total_scrap(...)

# Calculate scrap value (now updates in place)
scrap_calc = calculate_scrap_value(scrap_calc, material, ...)

# Direct assignment (no mapping needed!)
quote_data.stock_info = stock_info
quote_data.scrap_info = scrap_calc
```

**Benefit**: Eliminates 10+ lines of field-by-field mapping

---

### Phase 4: Backward Compatibility (DirectCostHelper)

**Option A: Deprecation path**
```python
# DirectCostHelper.py - keep for tests

# Deprecate entire module:
"""
DEPRECATED: This module is deprecated. Use QuoteDataHelper instead.

All functionality has been moved to cad_quoter.pricing.QuoteDataHelper.
Direct imports will be removed in v2.0.
"""

# Re-export from new location for tests:
from cad_quoter.pricing.QuoteDataHelper import (
    calculate_scrap_value,
    estimate_price_from_reference_part,
    extract_part_info_from_cad,
    # ... etc for all test imports
)
```

**Option B: Slim it down**
- Keep only the most-used functions
- Move internal helpers to QuoteDataHelper
- Update tests to import from QuoteDataHelper

---

## Migration Path for Callers

### For Tests
```python
# BEFORE
from cad_quoter.pricing.DirectCostHelper import calculate_scrap_value

# AFTER
from cad_quoter.pricing.QuoteDataHelper import calculate_scrap_value
```

No code changes needed if re-exports are maintained.

### For AppV7.py
Already uses QuoteDataHelper exclusively - no changes needed!

### For New Code
```python
# WRONG (old way)
from cad_quoter.pricing.DirectCostHelper import extract_part_info_from_plan

# RIGHT (new way)
from cad_quoter.pricing.QuoteDataHelper import extract_quote_data_from_cad
# or for lower-level access:
from cad_quoter.pricing.QuoteDataHelper import extract_part_info_from_plan
```

---

## Summary of Changes

| Item | Before | After | Benefit |
|------|--------|-------|---------|
| **PartInfo duplicates** | 2 versions (PartInfo + PartDimensions) | 1 unified PartInfo | Single source of truth |
| **ScrapInfo** | 29-field monster in DirectCostHelper; 10-field slimmed in QuoteDataHelper | Unified ScrapCalculation + StockInfo (11 fields total, clear separation) | Better SRP, cleaner API |
| **Material handling** | Scattered as strings | Explicit MaterialInfo dataclass | Consistent type safety |
| **Return types** | Mix of dataclasses & dicts | Consistent dataclasses | Easier to serialize, type-safe |
| **Location** | Split across 2 modules | Unified in QuoteDataHelper | Single module to understand |
| **Calculation duplicates** | `__post_init__` logic repeated | Single implementation | DRY principle |
| **Mapping overhead** | 10+ field assignments to convert between types | Direct assignments | Better performance |
| **Module size** | DirectCostHelper: 1298 lines (mixed concerns) | QuoteDataHelper: ~1900 lines (but cohesive) | Easier to maintain |

---

## Implementation Checklist

**Phase 1: Unified Data Model**
- [ ] Enhance PartInfo with material field (keep QuoteDataHelper version)
- [ ] Create unified ScrapCalculation dataclass
- [ ] Validate PartDimensions still works with new PartInfo
- [ ] Update MaterialInfo (minimal changes)

**Phase 2: Function Migration**
- [ ] Move extraction functions from DirectCostHelper → QuoteDataHelper
- [ ] Move calculation functions from DirectCostHelper → QuoteDataHelper  
- [ ] Refactor calculate_total_scrap() to return (StockInfo, ScrapCalculation)
- [ ] Refactor calculate_scrap_value() to return ScrapCalculation

**Phase 3: Update Consumers**
- [ ] Update extract_quote_data_from_cad() to use new return types
- [ ] Remove manual field-by-field mapping (save ~10 lines)
- [ ] Update tests to use new signatures

**Phase 4: Backward Compatibility**
- [ ] Re-export all DirectCostHelper functions from QuoteDataHelper
- [ ] Mark DirectCostHelper as deprecated
- [ ] Update tests (minimal changes - mostly imports)

**Phase 5: Testing**
- [ ] Run all existing tests
- [ ] Update any that reference internal structures
- [ ] Add integration tests for new consolidated module

---

## Risk Assessment

**Low Risk**:
- DirectCostHelper tests mostly use public functions
- Re-exports allow gradual migration
- QuoteDataHelper already has the right patterns

**Medium Risk**:
- Need to ensure ScrapCalculation reference to PartDimensions works
- Need to validate all function signatures
- Need to test JSON serialization of new dataclasses

**Mitigation**:
- Create feature branch for consolidation
- Run full test suite
- Keep DirectCostHelper as deprecated module for 1-2 releases
- Document migration in CHANGELOG

---

## Conclusion

Consolidating these modules will:
1. **Reduce cognitive load**: Single source of truth for cost concepts
2. **Improve maintainability**: Clear separation of concerns
3. **Eliminate conversions**: Direct assignments instead of field mapping
4. **Standardize API**: Consistent return types (dataclasses, not dicts)
5. **Enable better testing**: Smaller, focused functions

**Recommended approach**: OPTION 1 - Consolidate into QuoteDataHelper with phased deprecation of DirectCostHelper.

