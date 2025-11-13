# Punch Planner — Estimator Stubs & Integration

This README documents the **time‑estimation hooks** for the new `planner_punch(...)` process family and how to wire them into your existing planner/estimator loop. It matches the decision tree you provided and keeps the math simple and consistent with your global rules (e.g., grinding minutes = volume × 3.0 × grind_factor).

---

## TL;DR
- **Wire EDM (profile)**: `minutes = wire_perimeter_in × mins_per_in(material_group, thickness_band)`
- **Surface/Face Grind**: use your global rule `minutes = (L×W×stock_removed_total) × 3.0 × grind_factor` (if stock is known) or fall back to area × min_per_sq_in × factor.
- **OD Grind (round pierce punches)**: use **stock‑based volume** removal or a **circumference × length** rate if you add it to the CSV.
- **Defaults**: when a CSV value is missing, fall back to reasonable constants so the estimate never breaks.

---

## CSV Expectations
These fields can live alongside your existing speeds/feeds rows. Keep names flexible; the stub will check multiple aliases:

### Wire EDM
- `wire_mins_per_in` (base) — or per‑band keys like `wire_mpi_thin`, `wire_mpi_med`, `wire_mpi_thick`
- Optional per‑material overrides via `material_group`.

### Grinding
- `grind_factor` (aka `grind_material_factor` or `material_factor`) — multiplier.
- Optional: `grind_min_per_sq_in` (face), `grind_min_per_circ_in` (OD) if you prefer area/circumference‑based timing.

> If not present, the stubs will default to sensible values (e.g., `grind_factor=1.0`, `wire_mins_per_in=0.8`).

---

## Drop‑in Stubs (Python)
Paste these near your other estimators. They **don’t** mutate your plan; they only compute minutes from op/meta.

```python
from math import pi

# ---------- CSV helpers (adapt to your get_speeds_feeds) -------------------

def _csv_row(material: str, material_group: str, operation_hint: str = "Generic") -> dict:
    # Try material first, then group, then fallback Generic
    return (
        get_speeds_feeds(material=material, operation=operation_hint)
        or get_speeds_feeds(material=material_group or "", operation=operation_hint)
        or get_speeds_feeds(material="GENERIC", operation=operation_hint)
        or {}
    )

# ---------- Wire EDM --------------------------------------------------------

def _wire_mins_per_in(material: str, material_group: str, thickness_in: float) -> float:
    row = _csv_row(material, material_group, operation_hint="Wire_EDM")
    # Generic fallbacks / bands
    base = _try_float(row.get("wire_mins_per_in"), default=None)
    if base is not None:
        return max(0.01, base)
    # Banding by thickness (tweak thresholds as needed)
    band = (
        "wire_mpi_thin" if thickness_in <= 0.125 else
        "wire_mpi_med"  if thickness_in <= 0.500 else
        "wire_mpi_thick"
    )
    return max(0.01, _try_float(row.get(band), default=0.9))  # default conservative


def estimate_wire_edm_minutes(op: dict, material: str, material_group: str) -> float:
    perim = float(op.get("wire_profile_perimeter_in", 0.0) or 0.0)
    thk   = float(op.get("thickness_in", 0.0) or 0.0)
    mpi   = _wire_mins_per_in(material, material_group, thk)
    return max(0.0, perim * mpi)

# ---------- Grinding --------------------------------------------------------

MIN_PER_CUIN_GRIND = 3.0  # global rule you set


def _grind_factor(material: str, material_group: str) -> float:
    row = (
        _csv_row(material, material_group, operation_hint="Grinding")
        or _csv_row(material, material_group, operation_hint="Generic")
    )
    for k in ("grind_factor", "grind_material_factor", "material_factor"):
        val = row.get(k)
        if val not in (None, ""):
            try:
                f = float(val)
                if f > 0:
                    return f
            except Exception:
                pass
    return 1.0


def estimate_face_grind_minutes(length_in: float, width_in: float, stock_removed_total_in: float,
                                material: str, material_group: str, faces: int = 2) -> float:
    # Your canonical formula (faces currently top+bottom → 1 pair)
    volume_cuin = (length_in * width_in * stock_removed_total_in) * max(1, faces // 2)
    return max(0.0, volume_cuin * MIN_PER_CUIN_GRIND * _grind_factor(material, material_group))


def estimate_od_grind_minutes(meta: dict, material: str, material_group: str,
                              prefer_circ_model: bool = False) -> float:
    """Two models:
    1) Volume model (default): minutes = volume_removed × 3.0 × grind_factor
    2) Circumference model (optional): minutes = (circumference × length) × min_per_circ_in × factor
    Pick via prefer_circ_model or CSV presence of grind_min_per_circ_in.
    """
    factor = _grind_factor(material, material_group)
    # Check for circumference rate in CSV if user wants it
    row = _csv_row(material, material_group, operation_hint="Grinding")
    circ_rate = _try_float(row.get("grind_min_per_circ_in"), default=None)

    if prefer_circ_model and circ_rate:
        circ = float(meta.get("od_grind_circumference_in", 0.0) or 0.0)
        length = float(meta.get("od_length_in", 0.0) or 0.0)
        return max(0.0, circ * length * circ_rate * factor)

    # Default volume model
    vol = float(meta.get("od_grind_volume_removed_cuin", 0.0) or 0.0)
    if vol <= 0.0:
        # If volume not present, try to infer from D, T, and radial stock (very rough fallback)
        D = _try_float(meta.get("diameter"), default=0.0)
        T = _try_float(meta.get("thickness"), default=0.0)
        stock = _try_float(meta.get("stock_allow_radial"), default=0.003)
        if D > 0 and T > 0 and stock > 0:
            r = D / 2.0
            vol = pi * (r*r - (r - stock)**2) * T
    return max(0.0, vol * MIN_PER_CUIN_GRIND * factor)

# ---------- Utility ---------------------------------------------------------

def _try_float(x, default=0.0):
    try:
        v = float(x)
        # Treat NaN/inf as default
        if v != v or v == float("inf") or v == float("-inf"):
            return default
        return v
    except Exception:
        return default
```

---

## Dispatcher Integration
In your `estimate_machine_hours_from_plan(...)` loop, add handlers for punch ops. Keep names aligned with the ops emitted by `planner_punch(...)`.

```python
elif op_name == "Wire_EDM_profile":
    minutes += estimate_wire_edm_minutes(op, material, material_group)

elif op_name in ("Grind_faces",):
    # If you carry explicit face stock, pass it; otherwise use a conservative default (e.g., 0.004–0.008 in total)
    stock_total = _try_float(op.get("stock_removed_total"), default=0.006)
    minutes += estimate_face_grind_minutes(L, W, stock_total, material, material_group, faces=2)

elif op_name in ("Grind_length", "Grind_OD", "OD_grind_rough", "OD_grind_finish"):
    # Use OD model when round punches are in play (planner_punch stores meta)
    minutes += estimate_od_grind_minutes(plan.get("meta", {}), material, material_group)

# Optional: rough milling/turning hooks if you want time here too
elif op_name in ("Mill_rough_profile", "Mill_Turn_rough", "Turn_rough"):
    minutes += estimate_generic_rough_minutes(op, material, material_group)  # simple placeholder
```

> You can leave `estimate_generic_rough_minutes` as a pass‑through (0.0) until you’re ready to populate it with your milling/turning rates.

---

## Planner → Estimator Data Flow (what the estimator expects)
Ensure `planner_punch(...)` emits the following fields when possible:

- **Wire EDM**: `wire_profile_perimeter_in`, `thickness_in` (already included in the punch planner drop‑in)
- **OD Grind** (round pierce punches): meta keys `od_grind_volume_removed_cuin`, `od_grind_circumference_in`, `od_length_in`
- **Face Grind**: if you know the total stock to remove on faces, set `op.stock_removed_total`; else the estimator will use a conservative default.

---

## Sanity Checks
- Round pierce punch (steel, not carbide), D=0.500, T=2.000, stock_allow_radial=0.003
  - Planner should emit: Turn_rough → OD_grind_rough → OD_grind_finish → Grind_length
  - Estimator should compute `od_grind_volume_removed_cuin ≈ π[(0.25)^2 − (0.247)^2]×2.0` and multiply by `3.0 × grind_factor`.
- Non‑round form punch with tiny inside radii → Wire_EDM_profile present with `wire_profile_perimeter_in ≈ 2(L+W)` and thickness.
- Coin punch with 3D surface → 3D_mill_rough + EDM_burn_or_3D_finish + micron_finish + grind faces/length.

---

## Notes
- These stubs deliberately avoid deep coupling to your speeds/feeds parser; they rely on your existing `get_speeds_feeds(...)` to fetch rows and then probe for a few well‑named columns.
- Keep defaults conservative so estimates are stable even when the CSV lacks a value.
- When you formalize thickness bands for wire EDM, document the band keys in the CSV to avoid guesswork.

