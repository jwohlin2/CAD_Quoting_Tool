# Duplicate Functions and Logic Summary

## Quick Reference: Overlapping Code Blocks

### 1. Float Coercion Utility

**process_cost_renderer.py (lines 37-41)**:
```python
def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None
```

**process_view.py (lines 79-90, inline)**:
```python
try:
    hours_val = float(hours or 0.0)
except Exception:
    hours_val = 0.0
try:
    rate_val = float(rate or 0.0)
except Exception:
    rate_val = 0.0
try:
    cost_val = float(cost or 0.0)
except Exception:
    cost_val = 0.0
```

**Recommendation**: Use shared `_safe_float(value, default=0.0)` in process_cost_renderer; import and use in process_view

---

### 2. Canonicalization Lookup

**process_cost_renderer.py (line 68)**:
```python
canon = canonical_bucket_key(key)  # From process_buckets (CORRECT)
if not canon:
    continue
```

**process_view.py (lines 72, 91, 124, 196)**:
```python
canon_key = _canonical_bucket_key(label_str)  # From planner_render (WRONG!)
# ... repeated 4 times in different contexts
```

**Problem**: process_view uses `planner_render._canonical_bucket_key()` instead of `process_buckets.canonical_bucket_key()`

**Recommendation**: Update all 4 locations to use `canonical_bucket_key` from `process_buckets`

---

### 3. Planner Meta Detection

**process_cost_renderer.py (lines 61, 66)**:
```python
skip_keys: frozenset[str] = PROCESS_BUCKETS.planner_meta if skip_planner_meta else frozenset()
if skip_planner_meta and (norm_key in skip_keys or norm_key.startswith("planner_")):
    continue
```

**process_view.py (lines 195-199)**:
```python
def _is_planner_meta(key: str) -> bool:
    canonical_key = _canonical_bucket_key(key)
    if not canonical_key:
        return False
    return canonical_key.startswith("planner_") or canonical_key == "planner_total"
```

**Issue**: Two independent implementations with slightly different logic (frozenset check vs. string check)

**Recommendation**: Extract as `_is_planner_meta_key()` in process_cost_renderer; use consistently

---

### 4. Input Parsing (Partial Duplicate)

**process_cost_renderer.py (lines 25-34)**:
```python
def _iter_items(data: Mapping[str, Any] | Iterable[Any] | None) -> Iterable[tuple[Any, Any]]:
    if isinstance(data, Mapping):
        return data.items()
    if isinstance(data, Iterable):
        items: list[tuple[Any, Any]] = []
        for entry in data:
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                items.append((entry[0], entry[1]))
        return items
    return []
```

**process_view.py**: No equivalent helper; logic is inlined in add_row()

**Status**: Low-priority duplicate; `_iter_items` is specific to cost parsing

---

## Unique Functionality (process_view only)

These functions have **no equivalent** in process_cost_renderer and shouldn't be consolidated:

### Metadata Merging (lines 202-299)
```python
def _merge_process_meta(existing, incoming) -> dict[str, Any]:
    """Merge two metadata dicts with complex field logic"""
    # Combines: minutes, hr, base_extra, cost, rate, notes
    # Total 85 lines of intricate merging logic
```

### Metadata Folding (lines 302-327)
```python
def _fold_process_meta(meta_source) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Build lookup structure from metadata source"""
    # Creates canonical mapping + alias mapping
```

### Applied Process Entry Merging (lines 330-355)
```python
def _merge_applied_process_entries(entries) -> dict[str, Any]:
    """Merge applied process entries with note deduplication"""
```

### Applied Process Folding (lines 358-380)
```python
def _fold_applied_process(applied_source, alias_map) -> dict[str, Any]:
    """Organize applied process by canonical key"""
```

### Metadata Lookup with Variants (lines 383-410)
```python
def _lookup_process_meta(process_meta, key) -> Mapping[str, Any] | None:
    """Multi-strategy lookup with space/underscore variants"""
    # Tries: key, canon_key, space_variant, underscore_variant
```

**Reason for Keeping**: These are planner-specific metadata management features that have no use case in the pricing renderer.

---

## Cross-Module Dependency Issues

### Issue: process_view imports from planner_render

**Current (line 8-11)**:
```python
from cad_quoter.pricing.planner_render import (
    _canonical_bucket_key,
    _display_rate_for_row,
)
```

**Problem**: 
- `_canonical_bucket_key` shadows the shared utility from `process_buckets`
- Creates circular-ish dependency: process_view ← process_cost_renderer ← process_buckets
- AND process_view ← planner_render ← process_buckets
- AND planner_render ← process_cost_renderer

**Recommended Fix**:
```python
from cad_quoter.pricing.process_cost_renderer import (
    render_process_costs,
    _is_planner_meta_key,
)
from cad_quoter.pricing.process_buckets import (
    canonical_bucket_key,
)
from cad_quoter.pricing.planner_render import (
    _display_rate_for_row,  # Keep this, it's planner-specific
)
```

---

## Impact Analysis by Module

### process_cost_renderer.py
**Lines of Duplicate Code**: ~30 lines (float coercion)  
**Complexity**: Low - basic utilities  
**Consolidation Difficulty**: Easy  
**Changes Required**: +2 new functions, rename one

### process_view.py
**Lines of Duplicate Code**: ~60 lines (float coercion, canonicalization)  
**Complexity**: High - tight integration with metadata system  
**Consolidation Difficulty**: Medium  
**Changes Required**: 4 import updates, 4 call-site updates, 1 function simplification

### planner_render.py
**Lines to Update**: ~30 lines  
**Complexity**: Medium - uses _canonical_bucket_key extensively  
**Consolidation Difficulty**: Low  
**Changes Required**: Update import; keep everything else as-is (it's a wrapper)

---

## Code Metrics

| Module | Total Lines | Duplicate Lines | % Duplicate | Public Functions | Private Helpers |
|--------|-------------|-----------------|-------------|------------------|-----------------|
| process_cost_renderer | 206 | 30 | 15% | 2 | 4 |
| process_view | 411 | 60 | 14% | 7 | 6 |
| **Combined** | 617 | 90 | 14% | 9 | 10 |

---

## Consolidation Impact on Public API

### No Breaking Changes
✓ `render_process_costs()` signature unchanged  
✓ `canonicalize_costs()` signature unchanged  
✓ `ORDER`, `HIDE_IN_COST` constants unchanged  
✓ All test files remain valid  

### Internal Refactoring Only
✓ New internal functions: `_is_planner_meta_key()`, `_safe_float()`  
✓ Updated imports in process_view.py  
✓ No changes to __all__ lists  
✓ No changes to callers outside these modules  

---

## Estimated Effort

| Task | Lines | Time | Risk |
|------|-------|------|------|
| Phase 1: Add shared utilities | +15 | 15 min | Low |
| Phase 2: Update imports | +5 changes | 15 min | Low |
| Phase 3: Test & verify | - | 30 min | Medium |
| **Total** | +20 net | **60 min** | **Low** |

---

## Implementation Checklist

- [ ] Add `_is_planner_meta_key()` to process_cost_renderer
- [ ] Add `_safe_float()` to process_cost_renderer (or rename `_to_float()`)
- [ ] Update process_view.py imports (remove planner_render._canonical_bucket_key)
- [ ] Update process_view.py line 72: `canonical_bucket_key()` 
- [ ] Update process_view.py line 91: `canonical_bucket_key()`
- [ ] Update process_view.py line 124: `canonical_bucket_key()`
- [ ] Update process_view.py line 196: `canonical_bucket_key()`
- [ ] Simplify process_view._is_planner_meta() to delegate
- [ ] Run `pytest tests/pricing/test_process_cost_renderer.py`
- [ ] Run full test suite
- [ ] Create commit with clear message
- [ ] Request code review

