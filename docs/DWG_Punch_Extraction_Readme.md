# DWG Punch Extraction – Implementation Guide

## Overview

Goal: make **DWG-only punch drawings** usable for automated quoting.

Right now, the system:

- Converts DWG → DXF.
- Extracts **all text** (TEXT / MTEXT / DIMENSION display strings / TABLE text).
- Partially parses **hole tables** into `HoleSpec` / `HoleOp`.

But for a DWG punch with **no STEP file**, the quoting engine still only sees a **bag of strings** – it has no idea how big the punch is, how many ground diameters it has, or how nasty the tolerances are.

This README explains:

1. What we need to extract from DWG punches.
2. The gaps in the current system.
3. How to close those gaps using `ezdxf` and some text/dimension processing.
4. A target `PunchFeatureSummary` structure for the planner.

---

## 1. What we need from a DWG punch

For quoting, a punch drawing only has to provide a small set of knobs:

### A. Classification

- `family`: `round_punch`, `pilot_pin`, `form_punch`, `die_section`, etc.
- `shape_type`: `round` or `rectangular`.

Driven by:

- Title block and notes (keywords: `PUNCH`, `PIN`, `INSERT`, `SECTION`, `DIE`, `COIN`, `FORM`).
- Presence of Ø dimensions vs rectangular outline.

### B. Size / stock (envelope)

- `overall_length_in`
- `max_od_or_width_in` (largest diameter or width/thickness)
- For rectangular punches/sections:
  - `body_width_in`
  - `body_thickness_in`
- Optional: `form_length_in` – length of contoured nose region.

### C. Ops drivers

- Grinding / turning:
  - `num_ground_diams`
  - `total_ground_length_in`
  - `has_perp_face_grind` (e.g. “THIS SURFACE PERPENDICULAR TO CENTERLINE”)
- Form nose:
  - `has_3d_surface` (form/coin/polish contour)
  - `form_complexity_level` (0–3)
- Holes & undercuts:
  - `tap_count` + simple `tap_summary` (size + depth)
  - `num_undercuts`
- Edge work:
  - `num_chamfers`
  - `num_small_radii` (tiny R that are slow to grind/EDM)

### D. Pain / quality level

- Tightest diameter tolerance: `min_dia_tol_in`
- Tightest length tolerance: `min_len_tol_in`
- Flags:
  - `has_polish_contour`
  - `has_no_step_permitted`
  - `has_sharp_edges`
  - `has_gdt` (any GD&T frames present)

Everything above can be derived from **DXF geometry + DIMENSION measurements + text**. No OCR needed.

---

## 2. Current gaps for DWG punches

### Gap A – No 2D geometry

We currently **do not** read LINE / ARC / CIRCLE / LWPOLYLINE / SPLINE from DXF.

- No envelope bbox from geometry.
- No sense of which shapes are the part vs border, etc.
- No path length for WEDM, no contour info.

### Gap B – No structured dimensions / tolerances

We store DIMENSION **display text only**, not:

- Numeric measurement (`get_measurement()`).
- Type (linear vs aligned vs diameter).
- Units (in vs mm via `$INSUNITS`).
- Parsed tolerances (±.0001, +.0000/–.0002).

### Gap C – No “punch feature summary” layer for 2D

There is no module that takes:

- DXF geometry + DIMENSION info + text

and returns a structured `PunchFeatureSummary` for the planner.

### Gap D – Material & pain flags not mapped

We read material and notes as free text, but:

- Do not map `A2`, `D2`, `M2`, `CARBIDE`, etc. → material groups (P2/H1/C1/etc.).
- Do not summarize notes like `POLISH CONTOUR`, `NO STEP PERMITTED`, `SHARP`.

### Gap E – Holes/taps from non-table text are flaky

We have a hole/table parser, but hole/tap notes on the view like:

- `5/16-18 TAP X .80 DEEP`

are not consistently turned into clean `HoleSpec`s.

---

## 3. Library choice: `ezdxf`

Use [`ezdxf`](https://pypi.org/project/ezdxf/) for DXF geometry and dimension semantics.

Key features we need:

- Read DXF: `doc = ezdxf.readfile(path)`; `msp = doc.modelspace()`.
- Query entities: `msp.query("LINE ARC CIRCLE LWPOLYLINE SPLINE DIMENSION")`.
- Bounding boxes: `from ezdxf.bbox import bounding_box`.
- Dimension values: `dim.get_measurement()` + `dim.dimtype`.
- Dimension text: `dim.get_text()` (or `dim.dxf.text`).
- Units: `doc.header.get("$INSUNITS")`, `doc.header.get("$MEASUREMENT")`.

---

## 4. Implementation plan

### 4.1 Add minimal geometry extraction (bbox)

**Goal:** get a usable envelope from a DWG punch.

1. Read DXF:

```python
import ezdxf
from ezdxf.bbox import bounding_box

doc = ezdxf.readfile(dxf_path)
msp = doc.modelspace()
```

2. Select outline entities (first pass: everything non-text):

```python
outline = msp.query("LINE ARC CIRCLE LWPOLYLINE POLYLINE SPLINE")
bbox = bounding_box(outline)
(min_x, min_y, _), (max_x, max_y, _) = bbox.extmin, bbox.extmax
overall_length = max_x - min_x
overall_width  = max_y - min_y
```

3. Normalize units using `$INSUNITS` (1=inches, 4=mm, etc.) and store as inches.

This fills:

- `overall_length_in`
- `max_od_or_width_in` (use width from bbox; refine later if needed).

Later refinement (optional):

- Use layer filters (e.g. ignore border/titleblock layers).
- Use `edgeminer`/`edgesmith` to find the actual profile loop and WEDM path length.

---

### 4.2 Dimension mining (numeric measurements + tolerances)

1. Iterate DIMENSION entities:

```python
dims = []
for dim in msp.query("DIMENSION"):
    meas = dim.get_measurement()     # numeric length/angle
    text = dim.get_text()            # displayed text (may include Ø, TYP, etc.)
    dimtype = dim.dimtype            # linear, diameter, etc.
    dims.append((meas, text, dimtype))
```

2. Normalize `meas` to inches (same `$INSUNITS` logic as above).

3. Derive key values:

- `overall_length_in`:
  - Largest linear or aligned dimension (`dimtype` in {0,1}) in modelspace.
- `max_od_in`:
  - Largest dimension with `dimtype` == diameter or whose text contains `Ø`.
- Optional: body shank sizes:
  - Second-largest diameter for round punches.
  - Width/thickness dims near the base for rectangular punches.

4. Tolerances:

- Scan **all dimension text strings** for patterns:
  - `±0.000X`
  - `+0.0000-0.000X`
- Track smallest absolute tolerance for:
  - Any diameter → `min_dia_tol_in`
  - Any linear length → `min_len_tol_in`

(This can be done purely on text; the numeric `meas` is only for the nominal.)

---

### 4.3 Build `PunchFeatureSummary` for 2D

Create a data structure like:

```python
@dataclass
class PunchFeatureSummary:
    family: str                # "round_punch" | "pilot_pin" | ...
    shape_type: str            # "round" | "rectangular"

    overall_length_in: float
    max_od_or_width_in: float
    body_width_in: float | None = None
    body_thickness_in: float | None = None

    num_ground_diams: int = 0
    total_ground_length_in: float = 0.0
    has_perp_face_grind: bool = False

    has_3d_surface: bool = False
    form_complexity_level: int = 0

    tap_count: int = 0
    tap_summary: list[dict] = field(default_factory=list)  # {size, depth_in}
    num_undercuts: int = 0
    num_chamfers: int = 0
    num_small_radii: int = 0

    min_dia_tol_in: float | None = None
    min_len_tol_in: float | None = None
    has_polish_contour: bool = False
    has_no_step_permitted: bool = False
    has_sharp_edges: bool = False
    has_gdt: bool = False
```

Then implement `extract_punch_features_from_dxf(doc, text_dump) -> PunchFeatureSummary`:

#### Pass 1 – Classification & material (text only)

- Search `text_dump` for keywords:
  - `PUNCH`, `PIN`, `INSERT`, `SECTION`, `DIE`, `COIN`, `FORM`.
- Decide:
  - `family` and `shape_type` (round vs rectangular).
- Map material strings (`A2`, `D2`, `M2`, `CARBIDE`, etc.) into your existing material groups; store separately so the planner can pick grind factors.

#### Pass 2 – Size from geometry + DIMENSION

- Use bbox from §4.1 for a rough `overall_length_in` and `max_od_or_width_in`.
- Use dimension mining from §4.2 as a sanity check and to refine:
  - `overall_length_in`
  - `max_od_or_width_in`
  - `body_width_in`, `body_thickness_in` (rectangular parts).

Optional later: group cylindrical features (for round punches) by diameter to estimate `num_ground_diams` and `total_ground_length_in`.  
For now, `num_ground_diams` can be approximated by counting **distinct diameter values** from DIMENSIONs that contain `Ø`.

#### Pass 3 – Ops & pain flags (text + dims)

From full text dump:

- Chamfers:
  - Regex like `\(\d+\)\s*\.0\d+\s*X\s*45` → increment `num_chamfers` by the quantity.
- Small radii:
  - Regex `R\.00\d+` or `\.00\d+\s*R` → increment `num_small_radii`.
- Nose / 3D form:
  - If any text contains `POLISH`, `POLISH CONTOUR`, `FORM`, `COIN`, set `has_3d_surface=True`.
  - `form_complexity_level` can be a simple heuristic based on the **count of radius/profile dimensions** within the detail view (for v1, just 1–3 based on total number of “R”/“Ø” dims in nose region).
- Pain words:
  - `has_polish_contour` if any line has `POLISH`.
  - `has_no_step_permitted` if any line has `NO STEP PERMITTED`.
  - `has_sharp_edges` if any line has `SHARP`.
  - `has_gdt` if any line looks like a GD&T frame (e.g. contains characteristic symbols or bracketed feature-control strings).

From dimension text:

- Update `min_dia_tol_in` / `min_len_tol_in` as described in 4.2.

---

### 4.4 Upgrade hole/tap parsing on non-table text

We already have a hole/table parser that can produce `HoleSpec` / `HoleOp` from table rows.  
We need to reuse the same logic on free text.

Implementation:

1. For every TEXT / MTEXT / DIMENSION display string in the drawing:

   - Run a “loose” hole/tap parser with regexes like:
     - `(?P<size>\d+/\d+-\d+)\s*TAP\s*X\s*(?P<depth>[\d\.]+)\s*DEEP`
     - `Ø(?P<dia>[\d\.]+)\s*(THRU|X\s*(?P<depth>[\d\.]+)\s*DP)`

2. Normalize matches into `HoleSpec` instances:

```python
HoleSpec(
    size="5/16-18",
    depth_in=0.80,
    thru=False,
    qty=1,
    op="tap",
)
```

3. Summarize into `PunchFeatureSummary`:

- `tap_count` = number of `HoleSpec` where `op == "tap"`.
- `tap_summary` = list of `{size, depth_in}` dicts for the planner.

XY locations aren’t important for these parts; size + depth is enough for time.

---

## 5. Wiring into the planner

Once `PunchFeatureSummary` exists, the punch process planner can:

- Detect `family` + `shape_type` to pick the correct process family.
- Use `overall_length_in` + `max_od_or_width_in` to size stock.
- Use `num_ground_diams`, `total_ground_length_in` for grind time.
- Use `tap_count`, `tap_summary` for drilling/tapping time.
- Use `has_3d_surface`, `form_complexity_level` for 3D milling / EDM hours.
- Use `min_dia_tol_in`, `min_len_tol_in`, and pain flags to scale grind & inspection.

**Contract:** given just a DWG/DXF + existing text dump,  
`extract_punch_features_from_dxf()` must be able to populate this summary well enough that the quote is in the right ballpark without a STEP file.

That’s the target for this implementation.
