# Cost Helper Consolidation - Quick Reference

## Three Duplicate/Conflicting Dataclasses

### 1. PartInfo Duplication
```
DUPLICATE FIELDS (6 total, 5 identical):
├── length, width, thickness, volume, area (EXACT DUPLICATE)
└── material (only in DirectCostHelper)

DUPLICATE LOGIC (__post_init__):
volume = length × width × thickness
area = length × width
```

**Location**: 
- DirectCostHelper.py line 20
- QuoteDataHelper.py line 30 (as PartDimensions)

**Solution**: Unify into single PartInfo with material field

---

### 2. ScrapInfo Conflict (WORST OFFENDER)
```
DirectCostHelper version:     29 FIELDS (bloated)
├── McMaster dims (4)
├── Desired dims (4)
├── Part dims (5)
├── Material info (2)
├── Weights (3)
├── Scrap breakdown (4)
├── Percentages (2)
└── MISSING scrap value fields!

QuoteDataHelper version:      10 FIELDS (incomplete)
├── Scrap breakdown (4)
├── Weights (1)
├── Percentages (2)
└── Scrap value (3)

BETTER: Split into StockInfo (11 fields) + ScrapInfo (10 fields)
```

**Location**:
- DirectCostHelper.py line 645 (29 fields)
- QuoteDataHelper.py line 80 (10 fields)

**Problem**: QuoteDataHelper must map between versions (10+ field assignments!)

**Solution**: Keep both, separate concerns clearly

---

### 3. Material Handling (IMPLICIT vs EXPLICIT)
```
DirectCostHelper:
├── Material as string parameter only
└── get_material_density(material: str) function

QuoteDataHelper:
├── MaterialInfo dataclass
├── material_name: str
├── material_family: str (aluminum, steel, stainless, etc.)
├── density: float
├── detected_from_cad: bool
└── is_default: bool

BETTER: QuoteDataHelper approach (explicit, type-safe)
```

**Location**:
- DirectCostHelper.py: scattered throughout
- QuoteDataHelper.py line 47

**Solution**: Use QuoteDataHelper's MaterialInfo design

---

## Current Usage Map

```
Callers:
  - AppV7.py → QuoteDataHelper.extract_quote_data_from_cad()
  - Tests → DirectCostHelper functions directly
  - README → QuoteDataHelper examples

QuoteDataHelper dependencies:
  - Imports 7 functions FROM DirectCostHelper
  - Converts between dataclass versions
  - Field-by-field mapping (10+ assignments)
```

---

## Code Duplication Summary

### __post_init__ Logic Duplication
**DirectCostHelper.PartInfo** (lines 30-33):
```python
def __post_init__(self):
    self.volume = self.length * self.width * self.thickness
    self.area = self.length * self.width
```

**QuoteDataHelper.PartDimensions** (lines 38-43):
```python
def __post_init__(self):
    if self.volume == 0.0:
        self.volume = self.length * self.width * self.thickness
    if self.area == 0.0:
        self.area = self.length * self.width
```

**Status**: EXACT DUPLICATE (line 31 = lines 41-42, line 32 = lines 43)

---

## Recommended Consolidation Approach

### Option 1: Consolidate into QuoteDataHelper (RECOMMENDED)

**Why**:
- Cleaner architecture (separation of concerns)
- Consumer/orchestrator module
- Better dataclass design already proven
- Newer pattern

**Implementation**:
1. Move unified PartInfo to QuoteDataHelper (add material field)
2. Keep StockInfo + ScrapInfo separation (already good in QuoteDataHelper)
3. Move 15 functions from DirectCostHelper → QuoteDataHelper
4. Update calculate_total_scrap() and calculate_scrap_value() return types
5. DirectCostHelper becomes thin re-export wrapper for backward compatibility

**Result**:
- Single source of truth for all cost/pricing concepts
- Eliminates 10+ field assignments in data mapping
- Consistent return types (no more dict mixing)
- Better performance (no intermediate 29-field object)

---

## Import Path Changes

### Before (After Consolidation)
```python
# Tests - no change needed (re-exports maintained)
from cad_quoter.pricing.DirectCostHelper import calculate_scrap_value

# New code - can use direct source
from cad_quoter.pricing.QuoteDataHelper import calculate_scrap_value

# Main API (no change)
from cad_quoter.pricing.QuoteDataHelper import extract_quote_data_from_cad
```

---

## Data Flow Improvement

### Current (Problematic)
```
extract_quote_data_from_cad()
  ├─ extract_part_info_from_plan() → PartInfo (DirectCostHelper)
  │   └─ Map to PartDimensions (QuoteDataHelper)
  ├─ calculate_total_scrap() → ScrapInfo (29 fields, DirectCostHelper)
  │   └─ Extract 9 fields, discard 20 fields
  ├─ calculate_scrap_value() → Dict
  │   └─ Merge 3 fields into ScrapInfo
  └─ Final: 10+ field assignments to create QuoteDataHelper.ScrapInfo
```

### After Consolidation (Clean)
```
extract_quote_data_from_cad()
  ├─ extract_part_info_from_plan() → PartInfo (unified)
  │   └─ Direct assignment
  ├─ calculate_total_scrap() → (StockInfo, ScrapInfo)
  │   └─ Direct assignment (no mapping!)
  ├─ calculate_scrap_value() → ScrapInfo (updated)
  │   └─ Direct assignment
  └─ Final: 3 direct assignments, no field mapping overhead
```

---

## Files Involved

### Primary
1. **cad_quoter/pricing/DirectCostHelper.py** (1,298 lines)
   - Contains: 2 dataclasses, 23 functions
   - After: 70 lines (re-export wrapper)
   - Lost: Everything moves to QuoteDataHelper

2. **cad_quoter/pricing/QuoteDataHelper.py** (1,547 lines)
   - Contains: 8 dataclasses, 2 main functions
   - After: 2,547 lines (adds 15 functions + enhanced dataclasses)
   - Gained: All cost/pricing logic consolidated

### Tests
- **tests/test_all_materials.py**: Imports `calculate_scrap_value`
- **tests/test_volume_pricing.py**: Imports `estimate_price_from_reference_part`
- **tests/unit/test_plan_dimension_overrides.py**: Imports `extract_part_info_from_cad`
- All continue to work with re-exports (zero code change)

### App
- **AppV7.py**: Already uses QuoteDataHelper (zero change needed)

---

## Benefits at a Glance

| Issue | Before | After | Impact |
|-------|--------|-------|--------|
| PartInfo duplicates | 2 versions | 1 unified | Single source of truth |
| ScrapInfo bloat | 29 fields | 10+11 fields | Better SRP |
| Material handling | Scattered strings | MaterialInfo class | Type safe |
| Return type mixing | Dataclass + Dict | Dataclass only | Consistent API |
| Data mapping | 10+ assignments | 3 direct assignments | Better performance |
| Module organization | Split concerns | Unified, cohesive | Easier maintenance |

---

## Implementation Phases

### Phase 1: Data Model (1 day)
- Enhance PartInfo with material
- Keep StockInfo + ScrapInfo as-is
- Validate all dataclass changes

### Phase 2: Function Migration (2-3 days)
- Move extract functions
- Move calculation functions
- Refactor return types

### Phase 3: Consumer Update (1-2 days)
- Update extract_quote_data_from_cad()
- Remove field mapping (save ~10 lines)
- Test with real data

### Phase 4: Backward Compatibility (0.5 days)
- Convert DirectCostHelper to re-exports
- Mark as deprecated
- Verify tests pass

### Phase 5: Testing & QA (2-3 days)
- Full test suite
- JSON serialization validation
- Quote accuracy verification

**Total Time**: 1-2 weeks

---

## Recommendation

**Proceed with Option 1**: Consolidate into QuoteDataHelper

**Justification**:
1. Eliminates duplicate PartInfo (same 5 fields, identical logic)
2. Fixes bloated ScrapInfo (29→10 fields with proper separation)
3. Unifies material handling pattern
4. Removes 10+ field assignments in hot path
5. Establishes single source of truth
6. Maintains backward compatibility with re-exports
7. Low risk: tests already use public functions

---

## Success Criteria

- [ ] All tests pass (no code change needed with re-exports)
- [ ] Quote generation produces identical results
- [ ] Code coverage maintained or improved
- [ ] No external API breakage
- [ ] Documentation updated
- [ ] Single PartInfo in codebase
- [ ] No more conflicting ScrapInfo versions

