
from typing import Dict, List, Optional, Tuple
import pandas as pd, re

KEYWORDS = {
    "programming":        [r"Programming", r"Simulation", r"Verification", r"Tool Library", r"Process Sheet"],
    "wedm":               [r"Wire EDM Burn Time", r"\bWEDM\b", r"Wire EDM"],
    "sinker":             [r"Sinker EDM", r"Ram EDM", r"Electrode Burn"],
    "electrode_hours":    [r"Electrode Manufacturing Time"],
    "grind":              [r"Surface Grind", r"\bGrinding\b", r"Pre-Op Grinding"],
    "jig":                [r"Jig Grind"],
    "lap":                [r"Lapping", r"Honing", r"Polishing"],
    "inspection":         [r"\bInspection\b", r"CMM Run"],
    "cmm_programming":    [r"CMM Programming"],
    "finish":             [r"Bead Blasting", r"Sanding", r"Masking", r"Passivation", r"Laser Marking"],
    "saw":                [r"Sawing", r"Waterjet"],
    "fixture":            [r"Fixture Build"],
    "assembly":           [r"Assembly", r"Touch-up"],
    "material_credit":    [r"Material Scrap / Remnant Value"],
    "scrap_pct_row":      [r"Expected Scrap Rate"],
}

DEFAULT_RATES = {
    "ProgrammingRate":   120.0,
    "WireEDMRate":       140.0,
    "SinkerEDMRate":     150.0,
    "SurfaceGrindRate":  120.0,
    "JigGrindRate":      150.0,
    "LappingRate":       130.0,
    "InspectionRate":    110.0,
    "FinishingRate":     100.0,
    "SawWaterjetRate":   100.0,
    "FixtureBuildRate":  120.0,
    "AssemblyRate":      110.0,
    "ElectrodeBuildRate":120.0,
}

DEFAULT_PARAMS = {
    "ConsumablesFlat": 35.0,
    "NRE_FixturesEtc": 0.0,
    "MaterialOther": 50.0,
    "OverheadPct": 0.15,
    "MarginPct": 0.35,
}

def _sum_hours(df: pd.DataFrame, patterns: List[str]) -> float:
    """Sum 'Example Values / Options' where Data Type is 'Number' and Item matches any pattern."""
    if df.empty:
        return 0.0
    dtype = df['Data Type / Input Method'].astype(str).str.strip().str.lower()
    mask_number = dtype.eq('number')
    vals = pd.to_numeric(df.loc[mask_number, 'Example Values / Options'], errors='coerce').fillna(0.0)
    items = df.loc[mask_number, 'Item'].astype(str)
    if vals.empty:
        return 0.0
    mask = False
    for pat in patterns:
        m = items.str.contains(pat, case=False, regex=True, na=False)
        mask = m if isinstance(mask, bool) else (mask | m)
    if isinstance(mask, bool):
        return 0.0
    return float(vals.loc[mask].sum())

def _first_number(df: pd.DataFrame, patterns: List[str], default: float = 0.0) -> float:
    """Return first numeric 'Example Values / Options' where Item matches patterns (any Data Type)."""
    items = df['Item'].astype(str)
    mask = False
    for pat in patterns:
        m = items.str.contains(pat, case=False, regex=True, na=False)
        mask = m if isinstance(mask, bool) else (mask | m)
    if isinstance(mask, bool):
        return default
    vals = pd.to_numeric(df.loc[mask, 'Example Values / Options'], errors='coerce').dropna()
    return float(vals.iloc[0]) if not vals.empty else default

def compute_quote_from_df(df: pd.DataFrame,
                          rates: Optional[Dict[str, float]] = None,
                          params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Compute quoted price and a detailed breakdown from a variables DataFrame."""
    rates = {**DEFAULT_RATES, **(rates or {})}
    params = {**DEFAULT_PARAMS, **(params or {})}

    buckets = {}
    for key in ["programming","wedm","sinker","electrode_hours","grind","jig","lap",
                "inspection","cmm_programming","finish","saw","fixture","assembly"]:
        buckets[key] = _sum_hours(df, KEYWORDS[key])

    material_credit = _sum_hours(df, KEYWORDS["material_credit"])
    scrap_pct_val = _first_number(df, KEYWORDS["scrap_pct_row"], default=0.0)
    scrap_mult = 1.0 + (scrap_pct_val / 100.0)

    # Costs
    costs = {}
    costs["programming"] = buckets["programming"] * rates["ProgrammingRate"]
    costs["wedm"]        = buckets["wedm"] * rates["WireEDMRate"]
    costs["sinker"]      = buckets["sinker"] * rates["SinkerEDMRate"] + buckets["electrode_hours"] * rates["ElectrodeBuildRate"]
    costs["grind"]       = buckets["grind"] * rates["SurfaceGrindRate"]
    costs["jig"]         = buckets["jig"] * rates["JigGrindRate"]
    costs["lap"]         = buckets["lap"] * rates["LappingRate"]
    costs["inspection"]  = (buckets["inspection"] + buckets["cmm_programming"]) * rates["InspectionRate"]
    costs["finish"]      = buckets["finish"] * rates["FinishingRate"]
    costs["saw"]         = buckets["saw"] * rates["SawWaterjetRate"]
    costs["fixture"]     = buckets["fixture"] * rates["FixtureBuildRate"]
    costs["assembly"]    = buckets["assembly"] * rates["AssemblyRate"]

    labor_subtotal_post_scrap = scrap_mult * sum(costs.values())

    material_subtotal = params["MaterialOther"] + material_credit
    base_subtotal = labor_subtotal_post_scrap + material_subtotal + params["ConsumablesFlat"] + params["NRE_FixturesEtc"]
    with_overhead = base_subtotal * (1.0 + params["OverheadPct"])
    price = with_overhead * (1.0 + params["MarginPct"])

    out = {
        "scrap_pct": scrap_pct_val,
        **{f"{k}_hours": v for k,v in buckets.items()},
        **{f"{k}_cost": v for k,v in costs.items()},
        "material_credit": material_credit,
        "material_subtotal": material_subtotal,
        "labor_subtotal_post_scrap": labor_subtotal_post_scrap,
        "base_subtotal": base_subtotal,
        "with_overhead": with_overhead,
        "price": price,
    }
    return out

def compute_quote_from_excel(path: str, sheet: str = "Sheet1",
                             rates: Optional[Dict[str, float]] = None,
                             params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    df = pd.read_excel(path, sheet_name=sheet)
    return compute_quote_from_df(df, rates=rates, params=params)
