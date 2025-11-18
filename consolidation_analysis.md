# Process Rendering Modules Consolidation Analysis

## Executive Summary

The two modules `process_cost_renderer.py` and `process_view.py` represent different rendering contexts but share significant overlapping functionality. Both handle bucket canonicalization, cost rendering, and label/display utilities. A consolidation should unify the shared logic while maintaining backward compatibility for existing callers.

---

## Current Structure

### 1. `cad_quoter/pricing/process_cost_renderer.py` (206 lines)

**Purpose**: Pricing engine's process cost rendering for display and analysis

**Public API**:
- `ORDER` - Tuple of bucket keys in display order
- `HIDE_IN_COST` - Frozenset of buckets to hide from cost totals
- `canonicalize_costs()` - Main function to fold raw process costs into canonical buckets
- `render_process_costs()` - Render costs into a table object with hours, rates, and amounts

**Key Functions**:
| Function | Purpose | Responsibility |
|----------|---------|-----------------|
| `canonicalize_costs()` | Normalize costs to canonical buckets | Input validation, aliasing, planner meta filtering, misc hiding |
| `render_process_costs()` | Populate table object with rendered rows | Bucket ordering, rate lookup, drilling minute special handling |
| `_canonicalize_minutes()` | Normalize minutes detail to canonical buckets | Input validation, canonicalization |
| `_emit_cost_row()` | Generic table row appending | Polymorphic table interface support |
| `_iter_items()` | Parse various input formats | Handles Mapping, Iterable, None |
| `_to_float()` | Safe float coercion | Exception handling |

**Dependencies**:
- Imports from `process_buckets`: `PROCESS_BUCKETS`, `bucket_label`, `canonical_bucket_key`, `flatten_rates`, `lookup_rate`, `normalize_bucket_key`
- Uses `os.environ` for DEBUG_MISC flag
- No external state dependencies

---

### 2. `cad_quoter/pricing/process_view.py` (411 lines)

**Purpose**: Planner UI cost recording and metadata management

**Public API**:
- `render_process_costs` (re-exported from process_cost_renderer)
- `_ProcessCostTableRecorder` - Table recording class with row management and metadata handling
- `_ProcessRowRecord` - Dataclass for row records
- `_merge_process_meta()` - Merge metadata entries
- `_fold_process_meta()` - Build folded metadata structure
- `_merge_applied_process_entries()` - Merge applied process entries
- `_fold_applied_process()` - Build applied process structure
- `_lookup_process_meta()` - Find metadata for a key with fallback variants

**Key Classes & Functions**:
| Item | Purpose | Responsibility |
|------|---------|-----------------|
| `_ProcessRowRecord` | Record a single process row | Store: name, hours, rate, total, canon_key |
| `_ProcessCostTableRecorder` | Custom table interface for recording | Manage row state, apply metadata overrides, rate display |
| `_merge_process_meta()` | Merge two metadata dicts | Combine minutes/hr/cost/rate, dedupe notes |
| `_fold_process_meta()` | Organize metadata by canonical key | Build lookup structure, alias mapping |
| `_merge_applied_process_entries()` | Merge applied process entries | Combine entries, merge notes |
| `_fold_applied_process()` | Apply aliases to process structure | Reorganize by canonical key |
| `_lookup_process_meta()` | Find metadata with variants | Try key, canon key, space/underscore variants |

**Dependencies**:
- Imports from `process_cost_renderer`: `render_process_costs`
- Imports from `planner_render`: `_canonical_bucket_key`, `_display_rate_for_row`
- Uses `safe_float` from `domain_models.values`
- Complex metadata management with detailed fallback chains

---

## Key Findings: Overlapping vs. Unique Functionality

### A. Bucket/Label Canonicalization Logic

**DUPLICATE ISSUE**: Both files implement independent canonicalization pipelines

**In `process_cost_renderer.py`**:
```python
canon = canonical_bucket_key(key)  # From process_buckets
if not canon:
    continue
```

**In `process_view.py`**:
```python
canon_key = _canonical_bucket_key(label_str)  # From planner_render (different impl!)
if canon_key:
    override_label = self.canon_to_display_label.get(canon_key)
```

**Problem**: `process_view.py` imports `_canonical_bucket_key` from `planner_render` instead of using the shared `canonical_bucket_key` from `process_buckets`. This creates a maintenance burden and potential inconsistency.

### B. Input Normalization

**DUPLICATE**: Both implement similar input parsing:

**process_cost_renderer.py**:
```python
def _iter_items(data: Mapping[str, Any] | Iterable[Any] | None) -> Iterable[tuple[Any, Any]]:
def _to_float(value: Any) -> float | None:
```

**process_view.py**:
```python
# Inline equivalents in add_row():
hours_val = float(hours or 0.0)  # with exception handling
rate_val = float(rate or 0.0)
```

**Status**: Partially duplicate; could be unified via shared utilities

### C. Rate Display Calculation

**DUPLICATE**: Both handle rate display logic

**process_cost_renderer.py**:
```python
rate = lookup_rate(key, flat_rates, normalized_rates)
if rate <= 0 and hours > 0:
    rate = amount / hours
```

**process_view.py**:
```python
rate_display = _display_rate_for_row(
    record_canon or display_label,
    cfg=self.cfg,
    render_state=self.bucket_state,
    hours=hours_val,
)
```

**Status**: Different levels of abstraction; process_view adds planner-specific configuration

### D. Metadata Management (UNIQUE to process_view)

**UNIQUE**: process_view has no equivalent in process_cost_renderer:
- `_merge_process_meta()` - Combines hours, cost, rate, extra fields
- `_fold_process_meta()` - Creates lookup structure with aliases
- `_merge_applied_process_entries()` - Handles applied process entries
- `_fold_applied_process()` - Reorganizes by canonical key
- `_lookup_process_meta()` - Multi-strategy lookup with variants

**Use Case**: The planner tracks detailed per-bucket metadata (base_extra, minutes, rates, notes) that needs to be merged from multiple sources with fallback variants.

### E. Misc/Planner Filtering

**DIFFERENT APPROACH**:

**process_cost_renderer.py**:
```python
skip_keys: frozenset[str] = PROCESS_BUCKETS.planner_meta if skip_planner_meta else frozenset()
if skip_planner_meta and (norm_key in skip_keys or norm_key.startswith("planner_")):
    continue
```

**process_view.py**:
```python
def _is_planner_meta(key: str) -> bool:
    canonical_key = _canonical_bucket_key(key)
    if not canonical_key:
        return False
    return canonical_key.startswith("planner_") or canonical_key == "planner_total"
```

**Status**: Independent implementations; should consolidate

---

## Import/Usage Analysis

### Files That Import These Modules

#### Direct Imports:
1. **`cad_quoter/pricing/__init__.py`** (lines 20-24):
   - Imports: `ORDER`, `canonicalize_costs`, `render_process_costs` from `process_cost_renderer`
   - Re-exports publicly
   - **Usage**: Public API for the pricing module

2. **`cad_quoter/pricing/planner_render.py`** (lines 25-27):
   - Imports: `canonicalize_costs` as `_shared_canonicalize_costs`
   - **Usage**: Line 1346-1350 wraps it with planner-specific options
   ```python
   def canonicalize_costs(process_costs: ...):
       return _shared_canonicalize_costs(
           process_costs,
           skip_planner_meta=True,
           hide_misc_under=50.0,
       )
   ```

3. **`cad_quoter/pricing/process_view.py`** (line 7):
   - Imports: `render_process_costs`
   - **Usage**: Re-exported in `__all__`

#### Test Coverage:
1. **`tests/pricing/test_process_cost_renderer.py`** (lines 4):
   - Tests: `canonicalize_costs`, `render_process_costs`
   - Coverage: Cost grouping, planner meta filtering, misc hiding, rate handling
   - **Status**: Good test coverage for core functions

2. **`tests/app/test_planner_render.py`**:
   - Tests planner-specific functionality
   - Uses classes from `planner_render.py`
   - May use `process_view.py` indirectly

### Dependency Graph

```
┌─────────────────────────────────────┐
│ process_buckets.py                  │
│ (canonical_bucket_key, etc.)        │
└────────────┬────────────────────────┘
             │
       ┌─────┴──────────────────────────────┐
       │                                    │
       v                                    v
process_cost_renderer.py        planner_render.py
(render_process_costs,          (_canonical_bucket_key,
 canonicalize_costs)            _display_rate_for_row)
       │                               │
       │                          ┌────┘
       │                          │
       └──────────┬───────────────┘
                  │
                  v
         process_view.py
    (_ProcessCostTableRecorder,
     _merge_process_meta, etc.)
                  │
                  v
            __init__.py (exports)
```

**Critical Insight**: 
- `process_view.py` currently imports `render_process_costs` from `process_cost_renderer` AND `_canonical_bucket_key` from `planner_render`
- This creates a circular-ish dependency: `process_view` imports from both siblings
- `process_view` is not imported anywhere directly (checked with grep)
- Only `render_process_costs` is re-exported, but the metadata functions are internal (_prefix)

---

## Duplicate Canonicalization/Bucket Logic Summary

### Issue 1: Dual Canonicalization Functions

**Current State**:
```
process_buckets.canonical_bucket_key()    ← Canonical, public
         ↓ (used by)
process_cost_renderer._canonicalize_costs()
         ↓ (used by)
planner_render.canonicalize_costs()
         ↓ (wraps with options)
planner_render._canonical_bucket_key()    ← Local re-implementation
         ↑ (used by)
process_view._ProcessCostTableRecorder
```

**Problem**: `planner_render._canonical_bucket_key()` shadows the shared utility

### Issue 2: Metadata Lookup Variants

`process_view._lookup_process_meta()` implements sophisticated fallback:
```python
candidates: list[str] = []
base = str(key or "").lower()
canon = _canonical_bucket_key(key)
variants = [...]  # space/underscore conversions
```

This logic is **not found** in `process_cost_renderer` because it doesn't handle metadata.

### Issue 3: Planner Meta Filtering

Both filter planner meta but differently:
- `process_cost_renderer`: Checks against `PROCESS_BUCKETS.planner_meta` frozenset
- `process_view`: Checks if key starts with "planner_" OR equals "planner_total"

These should be unified to use the same source of truth.

---

## Recommended Consolidation Strategy

### Phase 1: Unify Base Utilities (process_cost_renderer)

**Action**: Move shared utilities to `process_cost_renderer` and dedup

**Target Functions**:
- Keep `_iter_items()`, `_to_float()` as private helpers
- Add `_is_planner_meta_key()` to centralize planner meta checking
- Ensure all canonicalization goes through shared `process_buckets` functions

**Code Addition**:
```python
def _is_planner_meta_key(key: str) -> bool:
    """Check if key represents a planner metadata entry."""
    norm = normalize_bucket_key(key)
    if not norm:
        return False
    return norm in PROCESS_BUCKETS.planner_meta or norm.startswith("planner_")

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce value to float with default."""
    try:
        return float(value)
    except (TypeError, ValueError, AttributeError):
        return default
```

**Remove**:
- Remove `_to_float()` or unify with `_safe_float()` naming

### Phase 2: Consolidate process_view Into Planner Context

**Action**: Move `_ProcessCostTableRecorder` and metadata functions to a new module or keep in `process_view` but import utilities from `process_cost_renderer`

**Updated Imports for process_view.py**:
```python
from cad_quoter.pricing.process_cost_renderer import (
    render_process_costs,
    _is_planner_meta_key,  # New shared function
    _safe_float,           # New shared function
)
from cad_quoter.pricing.process_buckets import (
    canonical_bucket_key,  # Use shared canonical, not planner_render._canonical
)
```

**Update logic in `_ProcessCostTableRecorder.add_row()`**:
```python
canon_key = self.label_to_canon.get(label_str)
if canon_key is None:
    canon_key = canonical_bucket_key(label_str)  # Use shared, not planner_render
```

### Phase 3: Rationalize Rate Display

**Action**: Keep `_display_rate_for_row()` in `planner_render.py` (it's planner-specific) but ensure it uses shared utilities

**No change needed** - it's appropriately scoped

### Phase 4: Metadata Management (Keep in process_view)

**Action**: These are planner-specific, no consolidation needed:
- `_merge_process_meta()`
- `_fold_process_meta()`
- `_merge_applied_process_entries()`
- `_fold_applied_process()`
- `_lookup_process_meta()`

**Reason**: No equivalent in pricing renderer; unique to planner UI

---

## Migration Path for Callers

### Step 1: Update process_view.py Imports

**Before**:
```python
from cad_quoter.pricing.process_cost_renderer import render_process_costs
from cad_quoter.pricing.planner_render import (
    _canonical_bucket_key,
    _display_rate_for_row,
)
```

**After**:
```python
from cad_quoter.pricing.process_cost_renderer import (
    render_process_costs,
    _is_planner_meta_key,
    _safe_float,
    canonicalize_costs,
)
from cad_quoter.pricing.process_buckets import (
    canonical_bucket_key,
    normalize_bucket_key,
)
from cad_quoter.pricing.planner_render import (
    _display_rate_for_row,
)
```

### Step 2: Update _ProcessCostTableRecorder

**Lines 72 → Update canon_key lookup**:
```python
# OLD:
canon_key = _canonical_bucket_key(label_str)

# NEW:
canon_key = canonical_bucket_key(label_str)
```

**Lines 91 → Update canon_key fallback**:
```python
# OLD:
record_canon = canon_key or _canonical_bucket_key(display_label)

# NEW:
record_canon = canon_key or canonical_bucket_key(display_label)
```

**Lines 124 → Update variant lookup**:
```python
# OLD:
alt_key = _canonical_bucket_key(record_canon)

# NEW:
alt_key = canonical_bucket_key(record_canon)
```

**Lines 196 → Update variant lookup**:
```python
canonical_key = _canonical_bucket_key(key)

# NEW:
canonical_key = canonical_bucket_key(key)
```

### Step 3: Update _is_planner_meta() Function

**Before** (line 195-199):
```python
def _is_planner_meta(key: str) -> bool:
    canonical_key = _canonical_bucket_key(key)
    if not canonical_key:
        return False
    return canonical_key.startswith("planner_") or canonical_key == "planner_total"
```

**After**:
```python
def _is_planner_meta(key: str) -> bool:
    return _is_planner_meta_key(key)  # Delegate to shared implementation
```

### Step 4: No Changes to callers of process_view

**Reason**: `process_view.py` is not directly imported anywhere. Only `render_process_costs` is re-exported from `__init__.py`, and that function signature doesn't change.

### Step 5: Public API Verification

**In `process_view.py` __all__**:
```python
__all__ = [
    "render_process_costs",  # ← Still imported, re-exported
    "_ProcessCostTableRecorder",
    "_ProcessRowRecord",
    "_merge_process_meta",
    "_fold_process_meta",
    "_merge_applied_process_entries",
    "_fold_applied_process",
    "_lookup_process_meta",
]
```

**No change needed** - these remain the same

---

## Consolidation Benefits

| Benefit | Impact |
|---------|--------|
| **Single Source of Truth** | All bucket canonicalization through `process_buckets.canonical_bucket_key()` |
| **Reduced Code Duplication** | Remove duplicate implementations of float coercion, planner meta checking |
| **Easier Maintenance** | Bug fixes and enhancements in one place |
| **Clearer Dependencies** | Remove cross-imports between `process_view` and `planner_render` |
| **Backward Compatible** | No public API changes; only internal refactoring |
| **Test Coverage** | Existing tests continue to pass; no new tests needed |

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Circular import | Unlikely; we're moving toward `process_buckets` as single source |
| Function signature changes | None - only internal consolidation |
| Behavioral changes | Use shared utilities instead of duplicates; should be identical |
| Test failures | Existing tests validate behavior preservation |

---

## Files to Modify

1. **`cad_quoter/pricing/process_cost_renderer.py`**
   - ADD: `_is_planner_meta_key()` function
   - Rename: `_to_float()` → `_safe_float()` (or keep both)
   - ADD: Module docstring clarifying shared utilities

2. **`cad_quoter/pricing/process_view.py`**
   - UPDATE: Import `canonical_bucket_key` from `process_buckets`
   - UPDATE: Import `_is_planner_meta_key` from `process_cost_renderer`
   - UPDATE: 4 locations where `_canonical_bucket_key()` is called
   - SIMPLIFY: `_is_planner_meta()` function (delegate to shared)
   - No changes to `__all__` or public API

3. **`tests/pricing/test_process_cost_renderer.py`**
   - No changes (tests remain valid)

4. **Documentation**
   - Update docstrings to clarify shared utilities
   - No breaking changes to document

---

## Next Steps

1. **Code Review** - Review this analysis with team
2. **Implement Phase 1** - Add shared utilities to `process_cost_renderer`
3. **Implement Phase 2** - Update `process_view` imports and calls
4. **Testing** - Run full test suite to validate no regressions
5. **Final Review** - Ensure clean imports and no circular dependencies
6. **Commit** - Create single commit: "Consolidate bucket canonicalization logic in process modules"

