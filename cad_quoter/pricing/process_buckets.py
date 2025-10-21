"""Shared helpers for process cost bucket normalisation and labelling."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
import re
from typing import Any, Dict, Iterable as _Iterable, Mapping as _Mapping, Tuple

from cad_quoter.rates import OP_TO_LABOR, OP_TO_MACHINE, rate_for_role


@dataclass(frozen=True)
class _BucketSpec:
    """Configuration for a canonical process bucket."""

    key: str
    label: str
    rate_aliases: tuple[str, ...] = ()
    hide_in_cost: bool = False


_BASE_BUCKET_SPECS: tuple[_BucketSpec, ...] = (
    _BucketSpec("milling", "Milling", ("MillingRate",)),
    _BucketSpec(
        "drilling",
        "Drilling",
        ("DrillingRate", "CncVertical", "CncVerticalRate", "cnc_vertical"),
    ),
    _BucketSpec("counterbore", "Counterbore", ("CounterboreRate", "DrillingRate")),
    _BucketSpec("tapping", "Tapping", ("TappingRate", "DrillingRate")),
    _BucketSpec(
        "grinding",
        "Grinding",
        ("GrindingRate", "SurfaceGrindRate", "ODIDGrindRate", "JigGrindRate"),
    ),
    _BucketSpec("wire_edm", "Wire EDM", ("WireEDMRate", "EDMRate")),
    _BucketSpec("sinker_edm", "Sinker EDM", ("SinkerEDMRate", "EDMRate")),
    _BucketSpec("finishing_deburr", "Finishing/Deburr", ("FinishingRate", "DeburrRate")),
    _BucketSpec("saw_waterjet", "Saw/Waterjet", ("SawWaterjetRate", "SawRate", "WaterjetRate")),
    _BucketSpec("inspection", "Inspection", ("InspectionRate",)),
    _BucketSpec(
        "toolmaker_support",
        "Toolmaker Support",
        ("ToolmakerRate", "ToolAndDieMakerRate", "LaborRate"),
    ),
    _BucketSpec("fixture_build_amortized", "Fixture Build (amortized)", ("FixtureBuildRate",)),
    _BucketSpec(
        "programming_amortized",
        "Programming (per part)",
        ("ProgrammingRate", "EngineerRate", "ProgrammerRate"),
    ),
    _BucketSpec(
        "misc",
        "Misc",
        ("LaborRate", "MachineRate", "DefaultLaborRate", "DefaultMachineRate"),
        hide_in_cost=True,
    ),
)


# Additional buckets are surfaced by the planner UI but are not part of the
# pricing renderer's default order.  They still benefit from the same labelling
# and rate heuristics.
_EXTRA_BUCKET_SPECS: dict[str, _BucketSpec] = {
    "programming": _BucketSpec(
        "programming",
        "Programming",
        ("ProgrammingRate", "EngineerRate", "ProgrammerRate"),
    ),
    "fixture_build": _BucketSpec("fixture_build", "Fixture Build", ("FixtureBuildRate",)),
    "countersink": _BucketSpec("countersink", "Countersink", ("CounterSinkRate", "DrillingRate")),
}


ORDER: tuple[str, ...] = tuple(spec.key for spec in _BASE_BUCKET_SPECS)
"""Canonical ordering for pricing process buckets."""


PLANNER_BUCKET_ORDER: tuple[str, ...] = (
    "programming",
    "programming_amortized",
    "fixture_build",
    "fixture_build_amortized",
    "milling",
    "drilling",
    "counterbore",
    "countersink",
    "tapping",
    "saw_waterjet",
    "wire_edm",
    "sinker_edm",
    "grinding",
    "finishing_deburr",
    "inspection",
)
"""Display order for planner bucketised cost summaries."""


_LABEL_MAP: dict[str, str] = {spec.key: spec.label for spec in _BASE_BUCKET_SPECS}
_LABEL_MAP.update({key: spec.label for key, spec in _EXTRA_BUCKET_SPECS.items()})

_RATE_ALIAS_KEYS: dict[str, tuple[str, ...]] = {
    spec.key: spec.rate_aliases for spec in _BASE_BUCKET_SPECS if spec.rate_aliases
}
_RATE_ALIAS_KEYS.update(
    {
        key: spec.rate_aliases
        for key, spec in _EXTRA_BUCKET_SPECS.items()
        if spec.rate_aliases
    }
)

HIDE_IN_COST: frozenset[str] = frozenset(
    {"planner_total", "planner_labor", "planner_machine"}
    | {spec.key for spec in _BASE_BUCKET_SPECS if spec.hide_in_cost}
)
"""Buckets that should be hidden from rendered totals."""


_ALIAS_MAP: dict[str, str] = {
    "machining": "milling",
    "mill": "milling",
    "cnc_milling": "milling",
    "cnc": "milling",
    "turning": "misc",
    "cnc_turning": "misc",
    "wire_edm": "wire_edm",
    "wireedm": "wire_edm",
    "wire-edm": "wire_edm",
    "wire edm": "wire_edm",
    "wire_edm_windows": "wire_edm",
    "wire_edm_outline": "wire_edm",
    "wire_edm_open_id": "wire_edm",
    "wire_edm_cam_slot_or_profile": "wire_edm",
    "wire_edm_id_leave": "wire_edm",
    "wedm": "wire_edm",
    "edm": "wire_edm",
    "sinker_edm": "sinker_edm",
    "sinkeredm": "sinker_edm",
    "sinker-edm": "sinker_edm",
    "sinker edm": "sinker_edm",
    "ram_edm": "sinker_edm",
    "ramedm": "sinker_edm",
    "ram-edm": "sinker_edm",
    "sinker_edm_finish_burn": "sinker_edm",
    "lap": "grinding",
    "lapping": "grinding",
    "lapping_honing": "grinding",
    "honing": "grinding",
    "deburr": "finishing_deburr",
    "deburring": "finishing_deburr",
    "finishing": "finishing_deburr",
    "finishing_misc": "finishing_deburr",
    "finishing_deburr": "finishing_deburr",
    "saw": "saw_waterjet",
    "waterjet": "saw_waterjet",
    "saw_waterjet": "saw_waterjet",
    "inspection": "inspection",
    "inspect": "inspection",
    "quality": "inspection",
    "counter_bore": "counterbore",
    "counter_boring": "counterbore",
    "counterbore": "counterbore",
    "counter_sink": "drilling",
    "countersink": "drilling",
    "csk": "drilling",
    "tap": "tapping",
    "taps": "tapping",
    "tapping": "tapping",
    "drill": "drilling",
    "drilling": "drilling",
    "assembly": "misc",
    "packaging": "misc",
    "ehs_compliance": "misc",
    "machine": "misc",
    "labor": "misc",
    "planner_machine": "misc",
    "planner_labor": "misc",
    "planner_misc": "misc",
}


__all__ = [
    "ORDER",
    "PLANNER_BUCKET_ORDER",
    "HIDE_IN_COST",
    "bucket_label",
    "bucketize",
    "canonical_bucket_key",
    "flatten_rates",
    "lookup_rate",
    "normalize_bucket_key",
]


def normalize_bucket_key(name: Any) -> str:
    """Return a stable, lowercase/underscored representation of *name*."""

    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


def canonical_bucket_key(
    name: Any,
    *,
    allowed: Iterable[str] | None = None,
    default: str = "misc",
) -> str | None:
    """Return the canonical bucket key for *name*.

    When *allowed* is provided the normalised key is accepted before any alias
    lookups.  This lets callers opt into planner-specific buckets (such as
    countersink) without affecting the pricing renderer's canonicalisation.
    """

    norm = normalize_bucket_key(name)
    if not norm:
        return None
    if norm == "planner_total":
        return None

    allowed_keys: Tuple[str, ...]
    if allowed is None:
        allowed_keys = ORDER
    else:
        allowed_keys = tuple(allowed)
        if norm in allowed_keys:
            return norm

    alias = _ALIAS_MAP.get(norm)
    if alias:
        norm = alias

    if norm in allowed_keys:
        return norm

    if norm.startswith("planner_"):
        return default

    return default


def bucket_label(key: str) -> str:
    """Return the display label for a canonical bucket key."""

    if not key:
        return ""
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    return key.replace("_", " ").title()


def _rate_aliases_for(key: str) -> tuple[str, ...]:
    norm = normalize_bucket_key(key)
    if not norm:
        return ()
    aliases = _RATE_ALIAS_KEYS.get(norm)
    if aliases:
        return aliases
    return ()


def flatten_rates(
    rates: Mapping[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return flat and normalised rate lookups."""

    flat: dict[str, float] = {}
    normalized: dict[str, float] = {}

    if not isinstance(rates, Mapping):
        return flat, normalized

    def _walk(mapping: Mapping[str, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                _walk(value)
                continue
            try:
                amount = float(value)
            except Exception:
                continue
            key_str = str(key)
            flat[key_str] = amount
            norm = normalize_bucket_key(key_str)
            if norm and norm not in normalized:
                normalized[norm] = amount

    _walk(rates)
    return flat, normalized


def _rate_candidates(key: str) -> tuple[str, ...]:
    norm = normalize_bucket_key(key)
    if not norm:
        return ()
    pieces = [part for part in norm.split("_") if part]
    camel = "".join(piece.title() for piece in pieces)
    spaced = " ".join(piece.title() for piece in pieces)
    candidates = [
        key,
        norm,
        camel,
        f"{camel}Rate",
        f"{camel}_rate",
        spaced,
        f"{spaced} Rate",
        f"{spaced} rate",
    ]
    candidates.extend(_rate_aliases_for(norm))
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def lookup_rate(
    key: str,
    flat: Mapping[str, float],
    normalized: Mapping[str, float],
    *,
    fallbacks: Iterable[str] = ("labor", "labor_rate", "machine", "machine_rate"),
) -> float:
    """Return the rate for *key* using flattened/normalised rate maps."""

    for candidate in _rate_candidates(key):
        if candidate in flat:
            return flat[candidate]
        norm = normalize_bucket_key(candidate)
        if norm and norm in normalized:
            return normalized[norm]
    for fallback in fallbacks:
        norm = normalize_bucket_key(fallback)
        if norm and norm in normalized:
            return normalized[norm]
    return 0.0


# ---------------------------------------------------------------------------
# Planner bucketisation helpers
# ---------------------------------------------------------------------------


TAP_MINUTES_PER_HOLE = 0.3
CBORE_MINUTES_PER_SIDE = 0.15
CSK_MINUTES_PER_SIDE = 0.12
JIG_GRIND_MINUTES_PER_FEATURE = 15.0


OP_TO_BUCKET: Dict[str, str] = {
    "cnc_rough_mill": "milling",
    "finish_mill_windows": "milling",
    "thread_mill": "tapping",
    "drill_patterns": "drilling",
    "drill_ream_bore": "drilling",
    "drill_ream_dowel_press": "drilling",
    "rigid_tap": "tapping",
    "counterbore_holes": "counterbore",
    "waterjet_or_saw_blanks": "saw_waterjet",
    "surface_grind_faces": "grinding",
    "surface_or_profile_grind_bearing": "grinding",
    "profile_or_surface_grind_wear_faces": "grinding",
    "profile_grind_cutting_edges_and_angles": "grinding",
    "match_grind_set_for_gap_and_parallelism": "grinding",
    "blanchard_grind_pre": "grinding",
    "jig_bore_or_jig_grind_coaxial_bores": "grinding",
    "jig_grind_ID_to_size_and_roundness": "grinding",
    "visual_contour_grind": "grinding",
    "wire_edm_windows": "wire_edm",
    "wire_edm_outline": "wire_edm",
    "sinker_edm_finish_burn": "sinker_edm",
    "edge_break": "finishing_deburr",
    "lap_bearing_land": "finishing_deburr",
    "lap_ID": "finishing_deburr",
    "abrasive_flow_polish": "finishing_deburr",
}


INSPECTION_BASE_MIN = 6.0
INSPECTION_PER_OP_MIN = 0.6
INSPECTION_PER_HOLE_MIN = 1.0
INSPECTION_FRACTION_OF_TOTAL = 0.05


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_rate_for_role(rates: Dict[str, Dict[str, float]], role: str) -> float:
    try:
        return rate_for_role(rates, role)
    except Exception:
        return float(rates.get("labor", {}).get(role, 0.0))


def _resolve_bucket_for_op(op: str) -> str:
    if not op:
        return "milling"

    bucket = OP_TO_BUCKET.get(op)
    if bucket:
        return bucket

    op_lower = op.lower()
    if any(token in op_lower for token in ("counterbore", "c'bore")):
        return "counterbore"
    if any(token in op_lower for token in ("countersink", "csk")):
        return "countersink"
    if any(
        token in op_lower
        for token in (
            "rigid_tap",
            "rigid tap",
            "thread_mill",
            "thread mill",
            "tap",
        )
    ):
        return "tapping"

    if any(token in op_lower for token in ("mill", "pocket")) or (
        "profile" in op_lower
        and "grind" not in op_lower
        and "edm" not in op_lower
    ):
        if "thread_mill" not in op_lower and "thread mill" not in op_lower:
            return "milling"

    machine = OP_TO_MACHINE.get(op, "").lower()
    if machine:
        if "grind" in machine:
            return "grinding"
        if "edm" in machine:
            return "wire_edm" if "wire" in machine else "sinker_edm"
        if "counterbore" in machine:
            return "counterbore"
        if "countersink" in machine or "csk" in machine:
            return "countersink"
        if "drill" in machine:
            return "drilling"
        if "tap" in machine:
            return "tapping"
        if "waterjet" in machine or "saw" in machine:
            return "saw_waterjet"

    role = OP_TO_LABOR.get(op, "").lower()
    if role:
        if "inspect" in role:
            return "inspection"
        if any(token in role for token in ("deburr", "lap", "finish")):
            return "finishing_deburr"

    return "milling"


def _iter_geom_list(value: Any) -> _Iterable[Any]:
    if isinstance(value, (list, tuple, set)):
        return value
    return ()


def bucketize(
    planner_pricing: Dict[str, Any],
    rates_two_bucket: Dict[str, Dict[str, float]],
    nre: Dict[str, float],
    *,
    qty: int,
    geom: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate planner pricing into user-facing cost buckets."""

    qty_int = int(qty) if isinstance(qty, (int, float)) else 1
    if qty_int <= 0:
        qty_int = 1

    buckets: Dict[str, Dict[str, float]] = {
        key: {"minutes": 0.0, "labor$": 0.0, "machine$": 0.0, "total$": 0.0}
        for key in PLANNER_BUCKET_ORDER
    }

    def add(bucket: str, minutes: float, machine_cost: float, labor_cost: float) -> None:
        canon = canonical_bucket_key(bucket, allowed=PLANNER_BUCKET_ORDER, default="milling")
        if not canon:
            return
        entry = buckets.setdefault(
            canon, {"minutes": 0.0, "labor$": 0.0, "machine$": 0.0, "total$": 0.0}
        )
        entry["minutes"] += minutes
        entry["machine$"] += machine_cost
        entry["labor$"] += labor_cost
        entry["total$"] += machine_cost + labor_cost

    line_items = planner_pricing.get("line_items", [])
    for li in line_items if isinstance(line_items, list) else []:
        if not isinstance(li, dict):
            continue
        op_key = str(li.get("op") or "")
        minutes = _as_float(li.get("minutes"))
        machine_cost = _as_float(li.get("machine_cost"))
        labor_cost = _as_float(li.get("labor_cost"))
        bucket_name = _resolve_bucket_for_op(op_key)
        if bucket_name == "drilling" and machine_cost > 0 and labor_cost <= 0:
            labor_cost = machine_cost
            machine_cost = 0.0
        add(bucket_name, minutes, machine_cost, labor_cost)

    programming_min = _as_float(nre.get("programming_min"))
    fixture_min = _as_float(nre.get("fixture_min"))

    programmer_rate = (
        _safe_rate_for_role(rates_two_bucket, "Programmer")
        or _safe_rate_for_role(rates_two_bucket, "Engineer")
    )
    if programming_min > 0:
        labor_cost = programmer_rate * (programming_min / 60.0)
        add("programming", programming_min, 0.0, labor_cost)
        if qty_int > 1:
            per_min = programming_min / qty_int
            add(
                "programming_amortized",
                per_min,
                0.0,
                programmer_rate * (per_min / 60.0),
            )

    fixture_rate = (
        _safe_rate_for_role(rates_two_bucket, "FixtureBuilder")
        or _safe_rate_for_role(rates_two_bucket, "Toolmaker")
        or _safe_rate_for_role(rates_two_bucket, "Machinist")
    )
    if fixture_min > 0:
        labor_cost = fixture_rate * (fixture_min / 60.0)
        add("fixture_build", fixture_min, 0.0, labor_cost)
        if qty_int > 1:
            per_min = fixture_min / qty_int
            add(
                "fixture_build_amortized",
                per_min,
                0.0,
                fixture_rate * (per_min / 60.0),
            )

    def _coerce_count(value: Any) -> int:
        if value in (None, ""):
            return 0
        return max(0, int(_as_float(value)))

    def _count_features(entries: _Iterable[Any]) -> int:
        total = 0
        for entry in entries:
            if isinstance(entry, dict):
                qty = entry.get("qty") or entry.get("count") or entry.get("quantity")
                qty_int = _coerce_count(qty)
                if qty_int > 0:
                    total += qty_int
                    continue
            total += 1
        return total

    hole_features = list(_iter_geom_list(geom.get("drill")))
    tapped_count = _coerce_count(geom.get("tapped_count"))
    cbore_entries = list(_iter_geom_list(geom.get("counterbore")))
    op_count = sum(1 for _ in line_items if isinstance(line_items, list))

    hole_feature_qty = _count_features(hole_features)
    cbore_qty = _count_features(cbore_entries)

    fallback_counts = [
        _coerce_count(geom.get("hole_count")),
        _coerce_count(geom.get("hole_count_geom")),
        _coerce_count(geom.get("hole_count_table")),
    ]

    derived = geom.get("derived") if isinstance(geom.get("derived"), dict) else {}
    fallback_counts.extend(
        [
            _coerce_count(derived.get("hole_count")),
            _coerce_count(derived.get("hole_count_geom")),
        ]
    )

    hole_groups = list(_iter_geom_list(geom.get("hole_groups")))
    hole_groups_qty = _count_features(hole_groups)
    if hole_groups_qty > 0:
        fallback_counts.append(hole_groups_qty)

    if fallback_counts:
        hole_feature_qty = max([hole_feature_qty] + [cnt for cnt in fallback_counts if cnt > 0])

    inspection_min = INSPECTION_BASE_MIN
    inspection_min += op_count * INSPECTION_PER_OP_MIN
    inspection_min += (hole_feature_qty + tapped_count + cbore_qty) * INSPECTION_PER_HOLE_MIN

    total_process_minutes = sum(entry["minutes"] for entry in buckets.values())
    floor_minutes = total_process_minutes * INSPECTION_FRACTION_OF_TOTAL
    inspection_min = max(inspection_min, floor_minutes)

    inspector_rate = _safe_rate_for_role(rates_two_bucket, "Inspector")
    add("inspection", inspection_min, 0.0, inspector_rate * (inspection_min / 60.0))

    def _ops_totals_map(data: _Mapping[str, Any] | None) -> _Mapping[str, Any]:
        if isinstance(data, _Mapping):
            totals = data.get("totals")
            if isinstance(totals, _Mapping):
                return totals
        return {}

    geom_mapping: _Mapping[str, Any] | None = geom if isinstance(geom, dict) else None
    ops_summary = geom_mapping.get("ops_summary") if geom_mapping else None
    ops_totals = _ops_totals_map(ops_summary if isinstance(ops_summary, _Mapping) else None)

    def _ops_total(*keys: str) -> float:
        total = 0.0
        for key in keys:
            if isinstance(ops_totals, _Mapping):
                total += _as_float(ops_totals.get(key))
        return total

    def _sum_minutes_from_details(entries: _Iterable[Any] | None) -> float:
        total = 0.0
        if isinstance(entries, (list, tuple, set)):
            for entry in entries:
                if isinstance(entry, dict):
                    total += _as_float(entry.get("total_minutes"))
        return total

    fallback_minutes: Dict[str, float] = {}

    tap_minutes = 0.0
    if geom_mapping:
        tap_minutes = _as_float(geom_mapping.get("tap_minutes_hint"))
        if tap_minutes <= 0.0:
            tap_minutes = _sum_minutes_from_details(geom_mapping.get("tap_details"))
    if tap_minutes <= 0.0:
        tap_count = max(float(tapped_count), _ops_total("tap_front", "tap_back"))
        if tap_count > 0.0:
            tap_minutes = tap_count * TAP_MINUTES_PER_HOLE
    if tap_minutes > 0.0:
        fallback_minutes["tapping"] = float(tap_minutes)

    cbore_minutes = 0.0
    if geom_mapping:
        cbore_minutes = _as_float(geom_mapping.get("cbore_minutes_hint"))
    if cbore_minutes <= 0.0:
        cbore_count = max(float(cbore_qty), _ops_total("cbore_front", "cbore_back"))
        if cbore_count > 0.0:
            cbore_minutes = cbore_count * CBORE_MINUTES_PER_SIDE
    if cbore_minutes > 0.0:
        fallback_minutes["counterbore"] = float(cbore_minutes)

    csk_minutes = _ops_total("csk_front", "csk_back") * CSK_MINUTES_PER_SIDE
    if csk_minutes > 0.0:
        fallback_minutes["countersink"] = float(csk_minutes)

    jig_minutes = _ops_total("jig_grind") * JIG_GRIND_MINUTES_PER_FEATURE
    if jig_minutes > 0.0:
        fallback_minutes["grinding"] = float(jig_minutes)

    drilling_entry = buckets.get("drilling")
    if drilling_entry and fallback_minutes:
        drilling_minutes = float(drilling_entry.get("minutes") or 0.0)
        if drilling_minutes > 0.0:
            original_machine = float(drilling_entry.get("machine$") or 0.0)
            original_labor = float(drilling_entry.get("labor$") or 0.0)
            allocated_minutes = 0.0
            allocated_machine = 0.0
            allocated_labor = 0.0
            for name in ("tapping", "counterbore", "countersink", "grinding"):
                minutes = fallback_minutes.get(name, 0.0)
                if minutes <= 0.0:
                    continue
                existing = buckets.get(name)
                if existing and (
                    existing.get("minutes", 0.0) > 0.01 or abs(existing.get("total$", 0.0)) > 0.01
                ):
                    continue
                remaining = drilling_minutes - allocated_minutes
                if remaining <= 0.0:
                    break
                allocate = min(minutes, remaining)
                if allocate <= 0.0:
                    continue
                share = allocate / drilling_minutes if drilling_minutes else 0.0
                machine_alloc = original_machine * share
                labor_alloc = original_labor * share
                allocated_minutes += allocate
                allocated_machine += machine_alloc
                allocated_labor += labor_alloc
                add(name, allocate, machine_alloc, labor_alloc)
            if allocated_minutes > 0.0:
                drilling_entry["minutes"] = max(0.0, drilling_entry["minutes"] - allocated_minutes)
                drilling_entry["machine$"] = max(0.0, drilling_entry["machine$"] - allocated_machine)
                drilling_entry["labor$"] = max(0.0, drilling_entry["labor$"] - allocated_labor)
                drilling_entry["total$"] = max(
                    0.0,
                    drilling_entry["total$"] - (allocated_machine + allocated_labor),
                )

    cleaned_buckets: Dict[str, Dict[str, float]] = {}
    totals = {"minutes": 0.0, "machine$": 0.0, "labor$": 0.0, "total$": 0.0}

    for key in PLANNER_BUCKET_ORDER:
        entry = buckets.get(key)
        if not entry:
            continue
        if entry["minutes"] <= 0.01 and abs(entry["total$"]) <= 0.01:
            continue
        rounded = {
            "minutes": round(entry["minutes"], 2),
            "machine$": round(entry["machine$"], 2),
            "labor$": round(entry["labor$"], 2),
            "total$": round(entry["total$"], 2),
        }
        cleaned_buckets[bucket_label(key)] = rounded
        totals["minutes"] += rounded["minutes"]
        totals["machine$"] += rounded["machine$"]
        totals["labor$"] += rounded["labor$"]
        totals["total$"] += rounded["total$"]

    totals = {key: round(value, 2) for key, value in totals.items()}

    return {"buckets": cleaned_buckets, "totals": totals}

