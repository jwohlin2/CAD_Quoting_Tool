"""Rule-based decision tree for generating machining quotes.

This module provides a lightweight alternative to the legacy heuristics in
``appV5`` by combining a handful of geometric features with canned
process-planning rules.  The intent is not to perfectly match the historical
outputs but to provide a deterministic, inspectable baseline that can evolve
independently from the UI monolith.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping

import pandas as pd

from cad_quoter.costing_glue import op_cost as _op_cost
from cad_quoter.domain_models.materials import (
    DEFAULT_MATERIAL_DISPLAY,
    MATERIAL_DENSITY_G_CC_BY_KEYWORD,
    MATERIAL_DENSITY_G_CC_BY_KEY,
    MATERIAL_DISPLAY_BY_KEY,
    MATERIAL_OTHER_KEY,
    normalize_material_key,
)
from cad_quoter.domain_models.values import coerce_float_or_none
from cad_quoter.pricing import (
    ensure_material_backup_csv,
    load_backup_prices_csv,
)


MM_PER_INCH = 25.4
CM3_PER_IN3 = 16.387064


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


@dataclass(slots=True)
class DecisionTreeInputs:
    """Minimal feature vector consumed by the decision tree."""

    qty: int
    volume_cm3: float
    thickness_in: float
    material: str
    profile_length_mm: float | None = None
    hole_count: int = 0

    @property
    def material_display(self) -> str:
        normalized = normalize_material_key(self.material)
        if normalized == MATERIAL_OTHER_KEY:
            return DEFAULT_MATERIAL_DISPLAY
        return MATERIAL_DISPLAY_BY_KEY.get(normalized, self.material or DEFAULT_MATERIAL_DISPLAY)

    @property
    def volume_in3(self) -> float:
        return max(self.volume_cm3, 0.0) / CM3_PER_IN3

    @property
    def area_in2(self) -> float:
        thickness = max(self.thickness_in, 0.0)
        vol = self.volume_in3
        if thickness > 1e-6:
            area = vol / thickness
        else:
            area = vol ** (2.0 / 3.0)
        return max(area, 0.0)

    @property
    def profile_length_in(self) -> float:
        if self.profile_length_mm and self.profile_length_mm > 0:
            return self.profile_length_mm / MM_PER_INCH
        area = self.area_in2
        if area <= 0:
            return 0.0
        side = math.sqrt(area)
        return 4.0 * side


def _first_match(
    df: pd.DataFrame,
    patterns: Iterable[str],
    *,
    numeric: bool = False,
    text: bool = False,
) -> float | str | None:
    columns = list(getattr(df, "columns", []))
    if "Item" not in columns or "Example Values / Options" not in columns:
        return None
    items = df["Item"].astype(str)
    values = df["Example Values / Options"]
    for pattern in patterns:
        mask = items.str.contains(pattern, case=False, regex=True, na=False)
        try:
            has_any = bool(mask.any())
        except Exception:
            has_any = any(bool(flag) for flag in mask)
        if not has_any:
            continue
        filtered = (raw for raw, flag in zip(values, mask) if flag)
        for raw in filtered:
            if numeric:
                number = coerce_float_or_none(raw)
                if number is not None:
                    return float(number)
            elif text:
                text_val = str(raw).strip()
                if text_val:
                    return text_val
    return None


def extract_inputs_from_dataframe(
    df: pd.DataFrame,
    *,
    geo: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> DecisionTreeInputs | None:
    """Extract the minimal feature vector required by the decision tree."""

    params = params or {}
    qty_val = _first_match(df, (r"\bqty\b", r"quantity"), numeric=True)
    if qty_val is None:
        qty_val = _to_float(params.get("Quantity"), 1.0)
    qty = int(qty_val or 1)
    if qty <= 0:
        qty = 1

    volume_cm3 = _first_match(
        df,
        (r"net\s*volume", r"volume\s*\(cm\^?3\)", r"volume_cm3"),
        numeric=True,
    )
    if volume_cm3 is None and geo:
        for key in ("volume_cm3", "net_volume_cm3"):
            volume_cm3 = coerce_float_or_none(geo.get(key))
            if volume_cm3 is not None:
                break
        if volume_cm3 is None:
            volume_mm3 = coerce_float_or_none(geo.get("volume_mm3"))
            if volume_mm3 is not None:
                volume_cm3 = float(volume_mm3) / 1000.0
    if volume_cm3 is None:
        return None

    thickness_in = _first_match(df, (r"thickness\s*\(in\)", r"thickness_in"), numeric=True)
    if thickness_in is None:
        thickness_mm = _first_match(df, (r"thickness\s*\(mm\)", r"thickness_mm"), numeric=True)
        if thickness_mm is not None:
            thickness_in = float(thickness_mm) / MM_PER_INCH
    if thickness_in is None and geo:
        thickness_in = coerce_float_or_none(geo.get("thickness_in"))
        if thickness_in is None:
            thickness_mm = coerce_float_or_none(geo.get("thickness_mm"))
            if thickness_mm is not None:
                thickness_in = float(thickness_mm) / MM_PER_INCH
    if thickness_in is None:
        thickness_in = 1.0

    profile_length_mm = _first_match(df, (r"profile\s*(?:perimeter|length)\s*\(mm\)", r"profile_length_mm"), numeric=True)
    if profile_length_mm is None and geo:
        for key in ("profile_length_mm", "perimeter_mm"):
            profile_length_mm = coerce_float_or_none(geo.get(key))
            if profile_length_mm is not None:
                break

    hole_count_val = _first_match(df, (r"hole\s*count", r"number\s*of\s*holes"), numeric=True)
    if hole_count_val is None and geo:
        for key in ("hole_count", "derived_hole_count"):
            hole_count_val = coerce_float_or_none(geo.get(key))
            if hole_count_val is not None:
                break
    hole_count = int(hole_count_val or 0)
    if hole_count < 0:
        hole_count = 0

    material_text = _first_match(
        df,
        (r"material\s*(?:name|grade|alloy)", r"material"),
        text=True,
    )
    if (not material_text) and geo:
        material_text = str(geo.get("material") or "")
    material = material_text or params.get("DefaultMaterial", DEFAULT_MATERIAL_DISPLAY)

    return DecisionTreeInputs(
        qty=qty,
        volume_cm3=float(volume_cm3),
        thickness_in=float(thickness_in),
        material=str(material or DEFAULT_MATERIAL_DISPLAY),
        profile_length_mm=float(profile_length_mm) if profile_length_mm is not None else None,
        hole_count=hole_count,
    )


def _lookup_density(material: str) -> float:
    normalized = normalize_material_key(material)
    if normalized and normalized in MATERIAL_DENSITY_G_CC_BY_KEY:
        return MATERIAL_DENSITY_G_CC_BY_KEY[normalized]
    candidates = [normalize_material_key(material), normalized.replace(" ", "") if normalized else ""]
    for token in candidates:
        if token in MATERIAL_DENSITY_G_CC_BY_KEYWORD:
            return MATERIAL_DENSITY_G_CC_BY_KEYWORD[token]
    lowered = material.lower()
    if any(tag in lowered for tag in ("plastic", "peek", "acetal", "delrin")):
        return 1.45
    if "magnesium" in lowered or "az" in lowered:
        return 1.8
    return MATERIAL_DENSITY_G_CC_BY_KEY.get(normalize_material_key(DEFAULT_MATERIAL_DISPLAY), 7.85)


def _safe_op_cost(op: dict[str, Any], rates: Mapping[str, Mapping[str, float]], minutes: float) -> float:
    try:
        return _op_cost(op, {"labor": dict(rates.get("labor", {})), "machine": dict(rates.get("machine", {}))}, minutes)
    except Exception:
        return 0.0


class DecisionTreeQuoteEngine:
    """Evaluate the rule-based decision tree and compute costs."""

    def __init__(self, params: Mapping[str, Any], rates: Mapping[str, Mapping[str, float]]) -> None:
        self.params = dict(params)
        self.rates = {
            "labor": {str(k): float(v) for k, v in (rates.get("labor", {}) or {}).items()},
            "machine": {str(k): float(v) for k, v in (rates.get("machine", {}) or {}).items()},
        }

    # ---- scenario classification -------------------------------------------------

    def _classify(self, inputs: DecisionTreeInputs) -> str:
        t = max(inputs.thickness_in, 0.0)
        vol = max(inputs.volume_cm3, 0.0)
        if t <= 0.75 and vol <= 400:
            return "thin_plate"
        if t >= 1.5 or vol >= 1500:
            return "heavy_plate"
        return "block"

    def _scrap_factor(self, scenario: str) -> float:
        if scenario == "heavy_plate":
            return 1.18
        if scenario == "block":
            return 1.12
        return 1.08

    # ---- operation estimation ----------------------------------------------------

    def _chip_minutes(self, inputs: DecisionTreeInputs, *, base: float, mrr: float) -> float:
        volume = inputs.volume_in3 * self._scrap_factor(self._classify(inputs))
        if mrr <= 1e-6:
            return base
        return max(base, volume / mrr)

    def _profile_minutes(self, inputs: DecisionTreeInputs, *, base: float, rate: float) -> float:
        length = inputs.profile_length_in
        if length <= 0:
            length = 4.0 * math.sqrt(max(inputs.area_in2, 0.0))
        return max(base, length * rate)

    def _inspection_minutes(self, inputs: DecisionTreeInputs, *, base: float, scale: float) -> float:
        area = inputs.area_in2
        return max(base, area * scale)

    def _program_minutes(self, inputs: DecisionTreeInputs, *, base: float, scale: float) -> float:
        area = inputs.area_in2
        return max(base, area * scale)

    def _estimate_operations(self, scenario: str, inputs: DecisionTreeInputs) -> list[tuple[str, float]]:
        if scenario == "thin_plate":
            return [
                ("program_estimate", self._program_minutes(inputs, base=20.0, scale=0.4)),
                ("cnc_rough_mill", self._chip_minutes(inputs, base=12.0, mrr=1.8)),
                ("finish_mill_windows", self._profile_minutes(inputs, base=8.0, rate=0.35)),
                ("edge_break", max(4.0, inputs.profile_length_in * 0.2)),
                ("stability_check_after_ops", self._inspection_minutes(inputs, base=6.0, scale=0.12)),
            ]
        if scenario == "heavy_plate":
            return [
                ("program_estimate", self._program_minutes(inputs, base=35.0, scale=0.55)),
                ("blanchard_grind_pre", self._inspection_minutes(inputs, base=18.0, scale=0.25)),
                ("cnc_rough_mill", self._chip_minutes(inputs, base=22.0, mrr=1.2)),
                ("finish_mill_windows", self._profile_minutes(inputs, base=12.0, rate=0.28)),
                ("surface_grind_faces", self._inspection_minutes(inputs, base=15.0, scale=0.22)),
                ("edge_break", max(6.0, inputs.profile_length_in * 0.18)),
                ("stability_check_after_ops", self._inspection_minutes(inputs, base=10.0, scale=0.16)),
            ]
        hole_factor = max(inputs.hole_count, 0)
        return [
            ("program_estimate", self._program_minutes(inputs, base=28.0, scale=0.5)),
            ("cnc_rough_mill", self._chip_minutes(inputs, base=16.0, mrr=1.5)),
            ("finish_mill_windows", self._profile_minutes(inputs, base=10.0, rate=0.32)),
            ("spot_drill_all", max(5.0, hole_factor * 0.4)),
            ("drill_patterns", max(6.0, hole_factor * 0.6)),
            ("edge_break", max(4.0, inputs.profile_length_in * 0.18)),
            ("stability_check_after_ops", self._inspection_minutes(inputs, base=8.0, scale=0.14)),
        ]

    # ---- public API --------------------------------------------------------------

    def quote(self, inputs: DecisionTreeInputs) -> dict[str, Any]:
        ensure_material_backup_csv()
        scenario = self._classify(inputs)
        operations = self._estimate_operations(scenario, inputs)

        qty = max(inputs.qty, 1)
        op_details: list[dict[str, Any]] = []
        process_costs: dict[str, float] = {}
        process_minutes: dict[str, float] = {}
        total_process_cost = 0.0

        for op_name, minutes_per_part in operations:
            minutes_per_part = max(float(minutes_per_part or 0.0), 0.0)
            total_minutes = minutes_per_part * qty
            cost_total = _safe_op_cost({"op": op_name}, self.rates, total_minutes)
            total_process_cost += cost_total
            process_costs[op_name] = cost_total
            process_minutes[op_name] = total_minutes
            cost_per_part = cost_total / qty if qty else cost_total
            op_details.append(
                {
                    "op": op_name,
                    "minutes_per_part": minutes_per_part,
                    "minutes_total": total_minutes,
                    "cost_total": cost_total,
                    "cost_per_part": cost_per_part,
                }
            )

        density = _lookup_density(inputs.material_display)
        mass_kg_per_part = (inputs.volume_cm3 * density) / 1000.0
        mass_kg_per_part *= self._scrap_factor(scenario)

        price_per_kg: float | None = None
        source = ""
        try:
            backup_table = load_backup_prices_csv()
        except Exception:
            backup_table = {}
        if backup_table:
            key = normalize_material_key(inputs.material_display)
            record = backup_table.get(key)
            if not record:
                if "stainless" in key:
                    record = backup_table.get("stainless steel")
                elif "steel" in key:
                    record = backup_table.get("steel")
                elif "alum" in key:
                    record = backup_table.get("aluminum")
            if record and record.get("usd_per_kg") is not None:
                try:
                    price_per_kg = float(record["usd_per_kg"])
                    source = "backup_csv"
                except Exception:
                    price_per_kg = None
                    source = ""
        if price_per_kg is None or not math.isfinite(price_per_kg) or price_per_kg <= 0:
            fallback = _to_float(self.params.get("MaterialOther"), 50.0) or 50.0
            price_per_kg = float(fallback)
            source = "params.MaterialOther"

        material_cost_total = mass_kg_per_part * qty * price_per_kg
        material_cost_per_part = material_cost_total / qty if qty else material_cost_total

        labor_cost = total_process_cost
        direct_costs = labor_cost + material_cost_total

        overhead_pct = _to_float(self.params.get("OverheadPct"), 0.0) or 0.0
        ga_pct = _to_float(self.params.get("GA_Pct"), 0.0) or 0.0
        contingency_pct = _to_float(self.params.get("ContingencyPct"), 0.0) or 0.0
        expedite_pct = _to_float(self.params.get("ExpeditePct"), 0.0) or 0.0
        margin_pct = _to_float(self.params.get("MarginPct"), 0.0) or 0.0

        with_overhead = direct_costs * (1.0 + overhead_pct)
        with_ga = with_overhead * (1.0 + ga_pct)
        with_cont = with_ga * (1.0 + contingency_pct)
        with_expedite = with_cont * (1.0 + expedite_pct)
        price_total = with_expedite * (1.0 + margin_pct)

        unit_price = price_total / qty if qty else price_total

        totals = {
            "labor_cost": labor_cost,
            "material_cost_total": material_cost_total,
            "direct_costs": direct_costs,
            "with_overhead": with_overhead,
            "with_ga": with_ga,
            "with_contingency": with_cont,
            "with_expedite": with_expedite,
            "price_total": price_total,
            "unit_price": unit_price,
        }

        return {
            "scenario": scenario,
            "inputs": asdict(inputs),
            "operations": op_details,
            "process_costs": process_costs,
            "process_minutes": process_minutes,
            "direct_costs": {"Material": material_cost_total},
            "material": {
                "material": inputs.material_display,
                "density_g_cc": density,
                "scrap_factor": self._scrap_factor(scenario),
                "mass_kg_per_part": mass_kg_per_part,
                "price_per_kg": price_per_kg,
                "source": source or "decision_tree",
                "cost_total": material_cost_total,
                "cost_per_part": material_cost_per_part,
            },
            "totals": totals,
        }


def generate_decision_tree_quote(
    df: pd.DataFrame,
    params: Mapping[str, Any],
    rates: Mapping[str, Mapping[str, float]],
    *,
    geo: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """High level helper used by the legacy entry point."""

    inputs = extract_inputs_from_dataframe(df, geo=geo, params=params)
    if inputs is None:
        return None
    engine = DecisionTreeQuoteEngine(params=params, rates=rates)
    return engine.quote(inputs)


__all__ = [
    "DecisionTreeInputs",
    "DecisionTreeQuoteEngine",
    "extract_inputs_from_dataframe",
    "generate_decision_tree_quote",
]

