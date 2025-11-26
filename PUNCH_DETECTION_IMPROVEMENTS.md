# Enhanced Punch Detection Heuristics

## Problem Statement

Simple round punch/turned parts (guide posts, form punches, pilot pins) were being misclassified as plates and getting **150-240 min labor** instead of the correct **15-40 min labor** for simple turned parts.

**Root Cause:** Parts with filenames like "T1769-134.dwg" or "T1769-219.dwg" don't contain punch keywords, so they failed text-based detection and defaulted to the heavy plate-style labor formulas.

## Solution: Multi-Heuristic Detection with Confidence Scoring

The enhanced `detect_punch_drawing()` function now uses **three complementary detection methods**:

### 1. Text-Based Detection (Enhanced)
**What it does:** Searches filename and drawing text for punch-related keywords

**Enhancements:**
- ✓ **Part number patterns**: Detects "PART 2", "PART 6", "DETAIL 14", "ITEM 7" (+3 points)
- ✓ **Filename keywords**: "PUNCH", "PILOT", "PIN", "FORM", "GUIDE POST" (+5 points)
- ✓ **Text indicators**: "FORM PUNCH", "PIERCING PUNCH", "PILOT PIN" (+4 points)
- ✓ **Exclusions**: Rejects "PUNCH PLATE", "PUNCH SHOE", "PUNCH HOLDER", etc.

### 2. Geometry-Based Detection (NEW)
**What it does:** Analyzes part dimensions to detect cylindrical punch shapes

**Signals:**
- ✓ **Round geometry**: Two dimensions similar (within 30%) → likely round part (+2 points)
  - Example: 1.5" × 1.5" × 6.0" (pilot diameter ≈ shank diameter)
- ✓ **High aspect ratio**: L/D > 2.5 → punch/pin shape (+2 points)
  - Example: 6.0" / 1.5" = 4.0 aspect ratio
- ✓ **Small cross-section**: max(W,T) < 3.0" → typical punch size (+1 point)

### 3. Feature-Based Detection (NEW)
**What it does:** Examines machining operations and features in the process plan

**Signals:**
- ✓ **Grinding operations**: OD/face grinding suggests turned part (+1 point)
- ✓ **Few holes**: < 3 holes typical for punches (+1 point)
- ✗ **EDM windows**: Wire EDM windows indicate plate part (-3 points)
- ✗ **Many holes**: > 10 holes indicate plate part (-2 points)

## Confidence Scoring System

Each signal contributes points to a confidence score:
- **Text signals**: +3 to +5 points (strong)
- **Geometry signals**: +1 to +2 points each
- **Feature signals**: +1 point or -2/-3 points (negative for plates)

**Decision threshold:** score ≥ 3 → classify as PUNCH

## Example: How It Works

### Scenario 1: T1769-134.dwg (Form Punch - Part 2)
```
Dimensions: 3.2" × 0.75" × 0.75"
Text: "PART 2\nMATERIAL: A2\nHARDNESS: 58-60 HRC"
Operations: rough_turn, finish_turn, od_grind

Detection signals:
  ✓ Part number "PART 2" → +3 points
  ✓ Round geometry (0.75 ≈ 0.75) → +2 points
  ✓ High aspect ratio (L/D = 4.3) → +2 points
  ✓ Small diameter (0.75") → +1 point
  ✓ Grinding operations → +1 point
  ✓ No holes → +1 point

Total score: 10 points → PUNCH ✓
Labor time: 25 min (correct!)
```

### Scenario 2: T1769-201.dwg (Stripper Plate)
```
Dimensions: 10.0" × 8.0" × 0.5"
Text: "A 201 1 A2 STRIPPER 46-48 ROCK"
Operations: square_up_mill, drill_patterns, wedm_windows
Holes: 88
Windows: 2 EDM windows

Detection signals:
  ✗ Large aspect ratio (L/W = 1.25) → 0 points
  ✗ EDM windows → -3 points
  ✗ Many holes (88) → -2 points

Total score: -5 points → PLATE ✓
Labor time: 180 min (correct!)
```

## Benefits

1. **Automatic detection** - No manual override needed for most parts
2. **Robust** - Multiple signals catch punches even when filenames lack keywords
3. **Conservative** - Exclusion patterns prevent false positives
4. **Transparent** - Confidence scoring explains why each decision was made
5. **Accurate** - Catches parts like "T1769-134" that previously failed detection

## Test Results

All 14 test cases pass:
- ✓ Filename-based detection
- ✓ Text-based detection
- ✓ Part number detection (NEW)
- ✓ Geometry-based detection (NEW)
- ✓ Feature-based detection (NEW)
- ✓ Combined multi-signal detection
- ✓ Exclusion patterns (punch plates, holders, shoes)

## Impact on Labor Estimates

**Before:** Simple round punches → 150-240 min labor (plate formulas)
**After:** Simple round punches → 15-40 min labor (punch formulas)

**Labor reduction:** 80-90% for correctly identified punch parts!

## Files Modified

- `cad_quoter/pricing/QuoteDataHelper.py`
  - Enhanced `detect_punch_drawing()` with multi-heuristic detection
  - Updated calling code to pass `plan` parameter

## Testing

Run the test suite:
```bash
python3 test_enhanced_punch_detection.py
```

Expected output: All tests PASS ✓
