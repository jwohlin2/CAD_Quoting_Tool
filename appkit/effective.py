from __future__ import annotations

import copy
import math
from typing import Any

from cad_quoter.domain import canonicalize_pass_through_map

from appkit.effective_helpers import (
    compute_effective_state,
    ensure_accept_flags,
    reprice_with_effective,
)


def effective_to_overrides(effective: dict, baseline: dict | None = None) -> dict:
    baseline = baseline or {}
    out: dict[str, Any] = {}
    mults = (
        effective.get("process_hour_multipliers")
        if isinstance(effective.get("process_hour_multipliers"), dict)
        else {}
    )
    if mults:
        cleaned = {
            k: float(v)
            for k, v in mults.items()
            if v is not None
            and not math.isclose(float(v), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        }
        if cleaned:
            out["process_hour_multipliers"] = cleaned
    adders = (
        effective.get("process_hour_adders")
        if isinstance(effective.get("process_hour_adders"), dict)
        else {}
    )
    if adders:
        cleaned_add = {
            k: float(v)
            for k, v in adders.items()
            if v is not None and not math.isclose(float(v), 0.0, abs_tol=1e-6)
        }
        if cleaned_add:
            out["process_hour_adders"] = cleaned_add
    passes = (
        effective.get("add_pass_through")
        if isinstance(effective.get("add_pass_through"), dict)
        else {}
    )
    if passes:
        canonical_passes = canonicalize_pass_through_map(passes)
        cleaned_pass = {
            k: float(v)
            for k, v in canonical_passes.items()
            if not math.isclose(float(v), 0.0, abs_tol=1e-6)
        }
        if cleaned_pass:
            out["add_pass_through"] = cleaned_pass
    scrap_eff = effective.get("scrap_pct")
    scrap_base = baseline.get("scrap_pct")
    if scrap_eff is not None and (
        scrap_base is None
        or not math.isclose(float(scrap_eff), float(scrap_base or 0.0), abs_tol=1e-6)
    ):
        out["scrap_pct_override"] = float(scrap_eff)
    setups_eff = effective.get("setups")
    fixture_eff = effective.get("fixture")
    if setups_eff is not None or fixture_eff is not None:
        out["setup_recommendation"] = {}
        if setups_eff is not None:
            out["setup_recommendation"]["setups"] = setups_eff
        if fixture_eff is not None:
            out["setup_recommendation"]["fixture"] = fixture_eff
    numeric_keys = {
        "fixture_build_hr": (0.0, None),
        "soft_jaw_hr": (0.0, None),
        "soft_jaw_material_cost": (0.0, None),
        "handling_adder_hr": (0.0, None),
        "cmm_minutes": (0.0, None),
        "in_process_inspection_hr": (0.0, None),
        "fai_prep_hr": (0.0, None),
        "packaging_hours": (0.0, None),
        "packaging_flat_cost": (0.0, None),
        "shipping_cost": (0.0, None),
    }
    for key, (_default, _) in numeric_keys.items():
        eff_val = effective.get(key)
        base_val = baseline.get(key) if isinstance(baseline, dict) else None
        if eff_val is None:
            continue
        if base_val is None or not math.isclose(
            float(eff_val), float(base_val or 0.0), rel_tol=1e-6, abs_tol=1e-6
        ):
            out[key] = float(eff_val)
    bool_keys = ["fai_required"]
    for key in bool_keys:
        eff_val = effective.get(key)
        base_val = baseline.get(key) if isinstance(baseline, dict) else None
        if eff_val is None:
            continue
        if base_val is None or bool(eff_val) != bool(base_val):
            out[key] = bool(eff_val)
    text_keys = ["shipping_hint"]
    for key in text_keys:
        eff_val = effective.get(key)
        base_val = baseline.get(key) if isinstance(baseline, dict) else None
        if eff_val is None:
            continue
        if (base_val or "") != (eff_val or ""):
            out[key] = eff_val
    if effective.get("operation_sequence"):
        out["operation_sequence"] = list(effective["operation_sequence"])
    if isinstance(effective.get("drilling_strategy"), dict):
        out["drilling_strategy"] = copy.deepcopy(effective["drilling_strategy"])
    return out


__all__ = [
    "ensure_accept_flags",
    "compute_effective_state",
    "reprice_with_effective",
    "effective_to_overrides",
]
