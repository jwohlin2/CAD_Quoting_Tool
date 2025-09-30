
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd, re, math

# ---------- Keyword buckets (extendable) ----------
BUCKET_KEYWORDS = {
    "programming":        [r"Programming", r"Simulation", r"Verification", r"Tool Library", r"Setup Sheet", r"Process Sheet", r"Traveler Creation", r"DFM Review"],
    "wedm":               [r"Wire EDM Burn Time", r"\bWEDM\b", r"Wire EDM"],
    "sinker":             [r"Sinker EDM", r"Ram EDM", r"Electrode Burn"],
    "electrodes":         [r"Electrode Manufacturing Time"],
    "grind":              [r"Surface Grind", r"\bGrinding\b", r"Pre-Op Grinding", r"Blank Squaring"],
    "jig":                [r"Jig Grind", r"\bOD/ID Grind"],
    "lap":                [r"Lapping", r"Honing", r"Polishing"],
    "inspection":         [r"\bInspection\b", r"CMM Run"],
    "cmm_programming":    [r"CMM Programming"],
    "finish":             [r"Bead Blasting", r"Sanding", r"Masking", r"Passivation", r"Laser Marking", r"Deburring", r"Edge Break"],
    "saw":                [r"Sawing", r"Waterjet", r"Blank Saw"],
    "fixture":            [r"Fixture Build", r"Fixture Design"],
    "assembly":           [r"\bAssembly\b", r"Touch-up", r"Precision Fitting", r"Support for Assembly"],
    # milling/turning buckets (from Calculated Value or future hours)
    "milling":            [r"Milling", r"Roughing Cycle", r"Semi-Finishing", r"Finishing Cycle"],
    "turning":            [r"Turning", r"Threading", r"Cut-Off / Parting", r"ID Boring / Drilling"],
}

# ---------- Default rates & params (override via args or sheet "Lookup Value (Rate)" / "Lookup Value (Percentage)") ----------
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
    "Milling3AxisRate":   90.0,
    "Milling5AxisRate":  150.0,
    "TurningRate":       100.0,
    "ToolmakerRate":     145.0,
    "PMRate":            100.0,
    "SetupRate":         110.0,
    "BlankHandlingRate": 100.0,
}

DEFAULT_PARAMS = {
    "ConsumablesFlat": 35.0,
    "NRE_FixturesEtc": 0.0,
    "MaterialOther": 50.0,
    "OverheadPct": 0.15,
    "MarginPct": 0.35,
    # heuristics to convert "Number of X" into hours/cost
    "HoursPerMillingSetup": 0.50,
    "HoursPerUniqueToolSetup": 0.05,
    "HoursPerBlankHandling": 0.03,
    # checkbox & dropdown effects
    "ExpeditePct": 0.15,
    "FAIR_Hours": 1.0,
    "SourceInspection_Hours": 1.0,
    "DFMReview_Hours": 0.5,
    "PrecisionFitting_Hours": 0.5,
    "AssemblyDoc_Hours": 0.5,
    "LaserMark_Hours": 0.2,
    "Passivation_Hours": 0.25,
    "Tumble_Hours": 0.3,
    "CrateNRE": 75.0,
    "CFM_HandlingFee": 25.0,
    "CheckFixtureNRE": 250.0,
    "LiveTooling_Hours": 0.3,
    "SubSpindle_Hours": 0.2,
    "FormToolNRE": 150.0,
    # dropdown percentage tweaks
    "PaymentTerms_Net60_Pct": 0.01,
    "PaymentTerms_Net90_Pct": 0.02,
    "Relationship_New_Pct": 0.05,        # +margin
    "Relationship_KeyAccount_Pct": -0.05 # -margin
}

# ---------- Helper matchers ----------
def _is_number_row(row) -> bool:
    return str(row['Data Type / Input Method']).strip().lower() == "number" and pd.notnull(row['Example Values / Options'])

def _sum_if(df: pd.DataFrame, patterns: List[str]) -> float:
    if df.empty: return 0.0
    items = df['Item'].astype(str)
    vals  = pd.to_numeric(df['Example Values / Options'], errors='coerce').fillna(0.0)
    mask_any = False
    for p in patterns:
        m = items.str.contains(p, case=False, regex=True, na=False)
        mask_any = m if isinstance(mask_any, bool) else (mask_any | m)
    return float(vals.loc[mask_any].sum()) if not isinstance(mask_any, bool) else 0.0

def _first_numeric(df: pd.DataFrame, patterns: List[str], default: float = 0.0) -> float:
    if df.empty: return default
    items = df['Item'].astype(str)
    mask_any = False
    for p in patterns:
        m = items.str.contains(p, case=False, regex=True, na=False)
        mask_any = m if isinstance(mask_any, bool) else (mask_any | m)
    if isinstance(mask_any, bool): return default
    vals = pd.to_numeric(df.loc[mask_any, 'Example Values / Options'], errors='coerce').dropna()
    return float(vals.iloc[0]) if not vals.empty else default

def _bucket_hours(df_num: pd.DataFrame, keywords: Dict[str, List[str]]) -> Dict[str, float]:
    out = {k:0.0 for k in keywords.keys()}
    for key, pats in keywords.items():
        out[key] = _sum_if(df_num, pats)
    return out

def _override_rates_from_lookup(df_lookup_rate: pd.DataFrame, rates: Dict[str, float]) -> Dict[str, float]:
    # Map item names to our rate keys using loose matching
    mapping = [
        (r"Surface Grinder Rate", "SurfaceGrindRate"),
        (r"OD/ID Grinder Rate", "JigGrindRate"),
        (r"Wire EDM", "WireEDMRate"),
        (r"Sinker EDM|Ram EDM", "SinkerEDMRate"),
        (r"Primary Lathe|Turning Center Rate", "TurningRate"),
    ]
    for _, row in df_lookup_rate.iterrows():
        item = str(row["Item"])
        val  = pd.to_numeric(row["Example Values / Options"], errors="coerce")
        if pd.isna(val): continue
        for pat, key in mapping:
            if re.search(pat, item, flags=re.IGNORECASE):
                rates[key] = float(val)
    return rates

def _extract_pct_from_formula(text: str) -> Optional[float]:
    # Pull the first multiplier like "* 0.04" -> 0.04
    if not isinstance(text, str): return None
    m = re.search(r"\*\s*([0-9]*\.?[0-9]+)", text)
    if m:
        return float(m.group(1))
    return None

def _eval_expr(expr: str, context: Dict[str, float]) -> Optional[float]:
    # VERY SAFE EVAL: only allow names present in context, numbers, + - * / ( )
    if not isinstance(expr, str): return None
    # Replace variable-like tokens with context lookups
    # Build allowed names set
    allowed = {k: float(v) for k, v in context.items() if v is not None}
    # Reject anything sketchy
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expr):  # no function calls
        return None
    # Allow ^ as ** for users
    expr2 = expr.replace("^", "**")
    try:
        val = eval(expr2, {"__builtins__":{}}, allowed)
        return float(val)
    except Exception:
        return None

def compute_full_quote(xlsx_path: str,
                       sheet: str = "Sheet1",
                       user_choices: Optional[Dict[str, Any]] = None,
                       geo_context: Optional[Dict[str, float]] = None,
                       rate_overrides: Optional[Dict[str, float]] = None,
                       param_overrides: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Full-coverage calculator:
    - Numbers: summed into buckets by keywords; plus generic rules for hours/costs not matched.
    - Checkboxes: taken from user_choices['checkbox'][<Variable ID or Item>] = True/False.
    - Dropdowns: taken from user_choices['dropdown'][<Variable ID or Item>] = selected string.
    - Lookup (Rate/Percentage): used to override rates and compute extra overhead percentages.
    - Calculated Values: expressions evaluated with geo_context (e.g., Perimeter, Thickness, SurfaceArea_cm2, etc.).
    Returns a dict with 'price', components, and 'unused_rows' list for transparency.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    # Normalize categories
    data_type = df["Data Type / Input Method"].astype(str).str.strip().str.lower()
    df_num = df[data_type.eq("number")].copy()
    df_checkbox = df[data_type.eq("checkbox")].copy()
    df_dropdown = df[data_type.eq("dropdown")].copy()
    df_lookup_rate = df[data_type.eq("lookup value (rate)")].copy()
    df_lookup_pct  = df[data_type.eq("lookup value (percentage)")].copy()
    df_lookup      = df[data_type.eq("lookup value")].copy()
    df_calc        = df[data_type.eq("calculated value")].copy()

    # Initialize rates/params
    rates  = {**DEFAULT_RATES, **(rate_overrides or {})}
    params = {**DEFAULT_PARAMS, **(param_overrides or {})}

    # Override rates from lookup rate rows
    rates = _override_rates_from_lookup(df_lookup_rate, rates)

    # Bucket known hours
    buckets = _bucket_hours(df_num, BUCKET_KEYWORDS)

    # Heuristics: capture generic hours/cost not matched into buckets
    used_mask = pd.Series(False, index=df_num.index)
    for k, pats in BUCKET_KEYWORDS.items():
        pat = re.compile("|".join(pats), flags=re.IGNORECASE)
        used_mask |= df_num["Item"].astype(str).str.contains(pat)

    generic_rows = df_num[~used_mask].copy()

    # Treat various "hours" by role or by keyword
    def add_generic(row):
        itm = str(row["Item"])
        val = float(pd.to_numeric(row["Example Values / Options"], errors="coerce") or 0.0)
        # Profit Margin override
        if re.search(r"Profit Margin", itm, flags=re.IGNORECASE):
            params["MarginPct"] = val/100.0
            return ("meta", 0.0)
        # Expected Scrap Rate handled later
        if re.search(r"Expected Scrap Rate", itm, flags=re.IGNORECASE):
            return ("meta", 0.0)
        # Material surcharge: treat as percentage on material subtotal later
        if re.search(r"Surcharge", itm, flags=re.IGNORECASE):
            params["MaterialSurchargePct"] = val/100.0
            return ("meta", 0.0)
        # Material driven direct costs (BOM, fixture material, MOQ)
        if re.search(r"Hardware|BOM|Fixture Material|Material MOQ", itm, flags=re.IGNORECASE):
            return ("material_cost", val)
        # Freight / shipping / lead time surcharges treated as logistics direct cost
        if re.search(r"Freight|Shipping|Lead Time", itm, flags=re.IGNORECASE):
            return ("freight_cost", val)
        # Tooling consumables
        if re.search(r"Tooling Cost|Grinding Wheel", itm, flags=re.IGNORECASE):
            return ("tooling_cost", val)
        # Remaining numeric costs
        if re.search(r"Cost", itm, flags=re.IGNORECASE):
            return ("other_direct_cost", val)
        # Role / time keywords
        if re.search(r"Project Manager Hours", itm, flags=re.IGNORECASE):
            return ("pm_hours", val)
        if re.search(r"Tool & Die Maker|Precision Fitting", itm, flags=re.IGNORECASE):
            return ("toolmaker_hours", val)
        if re.search(r"Number of Milling Setups", itm, flags=re.IGNORECASE):
            hours = val * params["HoursPerMillingSetup"]
            return ("setup_hours", hours)
        if re.search(r"Number of Unique Tools", itm, flags=re.IGNORECASE):
            hours = val * params["HoursPerUniqueToolSetup"]
            return ("programming_hours", hours)
        if re.search(r"Number of Blanks Required", itm, flags=re.IGNORECASE):
            hours = val * params["HoursPerBlankHandling"]
            return ("blank_handling_hours", hours)
        # Fallback: try to map generic 'Labor|Time|Hours' to a sensible bucket by additional keywords
        if re.search(r"\b(Labor|Time|Hours)\b", itm, flags=re.IGNORECASE):
            if re.search(r"Assembly", itm, flags=re.IGNORECASE):
                return ("assembly_hours", val)
            if re.search(r"Inspection|CMM", itm, flags=re.IGNORECASE):
                return ("inspection_hours", val)
            if re.search(r"Grind", itm, flags=re.IGNORECASE):
                return ("grind_hours", val)
            if re.search(r"EDM", itm, flags=re.IGNORECASE):
                # choose sinker if present, else wedm
                if "Sinker" in itm or "Ram" in itm:
                    return ("sinker_hours", val)
                return ("wedm_hours", val)
            if re.search(r"Saw|Waterjet", itm, flags=re.IGNORECASE):
                return ("saw_hours", val)
            if re.search(r"Lap|Hone|Polish|Deburr|Edge", itm, flags=re.IGNORECASE):
                return ("lap_hours", val)
            if re.search(r"Program|Process|DFM|Simulation|Verification|Tool Library|Traveler", itm, flags=re.IGNORECASE):
                return ("programming_hours", val)
        # If we truly don't know, treat as PM hours (conservative)
        return ("pm_hours", val)

    generic_contrib = {
        "pm_hours":0.0, "toolmaker_hours":0.0, "setup_hours":0.0, "programming_hours":0.0,
        "blank_handling_hours":0.0, "assembly_hours":0.0, "inspection_hours":0.0, "grind_hours":0.0,
        "sinker_hours":0.0, "wedm_hours":0.0, "saw_hours":0.0, "lap_hours":0.0,
        "material_cost":0.0, "freight_cost":0.0, "tooling_cost":0.0, "other_direct_cost":0.0
    }

    for _, row in generic_rows.iterrows():
        kind, amount = add_generic(row)
        if kind in generic_contrib:
            generic_contrib[kind] += float(amount)
        # meta handled via params, ignore

    # Merge generic hours back into buckets
    buckets["programming"] += generic_contrib["programming_hours"]
    buckets["wedm"]        += generic_contrib["wedm_hours"]
    buckets["sinker"]      += generic_contrib["sinker_hours"]
    buckets["electrodes"]  += 0.0
    buckets["grind"]       += generic_contrib["grind_hours"]
    buckets["jig"]         += 0.0
    buckets["lap"]         += generic_contrib["lap_hours"]
    buckets["inspection"]  += generic_contrib["inspection_hours"]
    buckets["cmm_programming"] += 0.0
    buckets["finish"]      += 0.0
    buckets["saw"]         += generic_contrib["saw_hours"]
    buckets["fixture"]     += 0.0
    buckets["assembly"]    += generic_contrib["assembly_hours"]
    # extra buckets
    milling_hours = 0.0
    turning_hours = 0.0

    # Add setup/toolmaker/PM/blank handling hours priced at their own rates
    extra_role_hours = {
        "pm": generic_contrib["pm_hours"],
        "toolmaker": generic_contrib["toolmaker_hours"],
        "setup": generic_contrib["setup_hours"],
        "blank": generic_contrib["blank_handling_hours"]
    }

    # ---------- Calculated Values ----------
    geo = {**(geo_context or {})}
    calc_hours = {
        "wedm":0.0, "sinker":0.0, "milling":0.0, "turning":0.0
    }
    missing_calc = []
    for _, row in df_calc.iterrows():
        name = str(row["Item"])
        expr = row["Example Values / Options"]
        val = None
        # Evaluate numeric if possible
        pct = None
        try:
            # try numeric literal first
            val = float(expr)
        except Exception:
            pass
        if val is None:
            val = None
            # Evaluate simple arithmetic with provided geometry context
            val = None
            if isinstance(expr, str):
                # Replace common variable names to context as-is
                try:
                    # Very limited evaluation
                    if re.search(r"[A-Za-z_]", expr):
                        # ensure required names exist
                        needed = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
                        missing = [n for n in needed if n not in geo]
                        if missing:
                            missing_calc.append({"item":name, "needs":sorted(missing), "expr":expr})
                            continue
                    val = eval(expr.replace("^","**"), {"__builtins__":{}}, {k:float(v) for k,v in geo.items()})
                except Exception:
                    missing_calc.append({"item":name, "needs":["<parse error>"], "expr":expr})
                    continue
        if val is None:
            continue
        # Route to buckets by name
        if re.search(r"WEDM", name, flags=re.IGNORECASE):
            calc_hours["wedm"] += float(val)
        elif re.search(r"Sinker", name, flags=re.IGNORECASE):
            calc_hours["sinker"] += float(val)
        elif re.search(r"Roughing|Semi-Finishing|Finishing|Milling", name, flags=re.IGNORECASE):
            calc_hours["milling"] += float(val)
        elif re.search(r"Turning|Threading|Cut-Off|Parting|Boring|Drilling", name, flags=re.IGNORECASE):
            calc_hours["turning"] += float(val)

    # add calc hours
    buckets["wedm"]   += calc_hours["wedm"]
    buckets["sinker"] += calc_hours["sinker"]
    milling_hours     += calc_hours["milling"]
    turning_hours     += calc_hours["turning"]

    # ---------- Lookup Percentage rows as extra overhead adders ----------
    overhead_add_pct = 0.0
    overhead_details = []
    for _, row in df_lookup_pct.iterrows():
        txt = str(row["Example Values / Options"])
        pct = None
        # try numeric direct
        try:
            v = float(txt)
            pct = v if v < 1.0 else v/100.0
        except Exception:
            pct = None
        if pct is None:
            pct = _extract_pct_from_formula(txt) or 0.0
        overhead_add_pct += pct
        overhead_details.append((row["Item"], pct))

    # ---------- Dropdown rules ----------
    choices = user_choices.get("dropdown", {}) if user_choices else {}

    # Primary Milling Machine Rate -> affects milling rate
    pmr = choices.get("Primary Milling Machine Rate") or choices.get("MIL-01")
    if isinstance(pmr, str):
        if "5-Axis" in pmr: rates["MillingRate"] = rates["Milling5AxisRate"]
        elif "3-Axis" in pmr: rates["MillingRate"] = rates["Milling3AxisRate"]
        elif "Lathe" in pmr:  rates["TurningRate"] = rates["TurningRate"]
        elif "Wire EDM" in pmr: rates["WireEDMRate"] = rates["WireEDMRate"]
    else:
        rates["MillingRate"] = rates.get("Milling3AxisRate", 90.0)

    # Payment Terms
    pt = choices.get("Payment Terms") or choices.get("PM-06")
    payment_pct = 0.0
    if pt == "Net 60": payment_pct += DEFAULT_PARAMS["PaymentTerms_Net60_Pct"]
    if pt == "Net 90": payment_pct += DEFAULT_PARAMS["PaymentTerms_Net90_Pct"]

    # Customer Relationship -> tweak margin
    rel = choices.get("Customer Relationship") or choices.get("PM-02")
    margin_adj = 0.0
    if rel == "New":        margin_adj += DEFAULT_PARAMS["Relationship_New_Pct"]
    if rel == "Key Account":margin_adj += DEFAULT_PARAMS["Relationship_KeyAccount_Pct"]

    # Manual Deburring / Edge Break
    deb = choices.get("Manual Deburring / Edge Break Labor") or choices.get("FIN-01")
    if deb == "Standard Edge Break":
        buckets["finish"] += 0.2
    elif deb == "Full Cosmetic":
        buckets["finish"] += 0.6

    # Outsourced Plating / Coating Cost
    coat = choices.get("Outsourced Plating / Coating Cost") or choices.get("FIN-07")
    coating_cost = 0.0
    if coat == "Anodize": coating_cost += 3.0
    elif coat == "Black Oxide": coating_cost += 2.0
    elif coat == "Nickel Plate": coating_cost += 6.0
    # Heat Treat
    ht = choices.get("Outsourced Heat Treat Cost") or choices.get("FIN-06")
    heat_cost = 0.0
    if ht == "Harden & Temper": heat_cost += 8.0
    elif ht == "Anneal": heat_cost += 5.0
    elif ht == "Nitride": heat_cost += 12.0

    # Packaging
    pkg = choices.get("Packaging Method") or choices.get("PM-11")
    packaging_cost = 0.0
    if pkg == "Custom Crate": packaging_cost += DEFAULT_PARAMS["CrateNRE"]
    elif pkg == "ESD Bagging": packaging_cost += 5.0

    # Shipment speed
    ship = choices.get("Shipment Speed") or choices.get("PM-12")
    expedite_pct_dropdown = 0.0
    if ship == "Rush": expedite_pct_dropdown += 0.05

    # Material starting condition -> multiplier for milling/grinding removal
    msc = choices.get("Material Starting Condition") or choices.get("MAT-03")
    removal_multiplier = 1.0
    if msc == "Pre-Hardened": removal_multiplier = 1.15

    # Solid model quality -> extra programming hours
    smq = choices.get("Solid Model Quality") or choices.get("ENG-09")
    if smq == "Needs Repair":
        buckets["programming"] += 0.5

    # Fixture design hours (if "Yes")
    fdh = choices.get("Fixture Design Hours") or choices.get("ENG-05")
    if fdh == "Yes":
        buckets["fixture"] += 0.5

    # RFQ Completeness
    rfq = choices.get("RFQ Completeness") or choices.get("PM-03")
    if rfq == "Incomplete":
        buckets["programming"] += 0.5

    # Revision Control Complexity
    revc = choices.get("Revision Control Complexity") or choices.get("PM-04")
    if revc == "High":
        extra_role_hours["pm"] += 0.3

    # Certifications
    cert = choices.get("Required Certifications (ITAR, AS9100)") or choices.get("PM-05")
    cert_admin_cost = 0.0
    if cert == "ITAR": cert_admin_cost += 50.0
    if cert == "AS9100": cert_admin_cost += 75.0

    # ---------- Checkboxes ----------
    checks = user_choices.get("checkbox", {}) if user_choices else {}
    expedite_pct_checkbox = DEFAULT_PARAMS["ExpeditePct"] if checks.get("Expedite Request") or checks.get("PM-07") else 0.0

    if checks.get("Customer-Furnished Material (CFM)") or checks.get("PM-10"):
        # Remove material-other but add handling
        params["MaterialOther"] = 0.0
        packaging_cost += DEFAULT_PARAMS["CFM_HandlingFee"]

    if checks.get("First Article Inspection Report (FAIR) Labor") or checks.get("QC-05"):
        buckets["inspection"] += DEFAULT_PARAMS["FAIR_Hours"]

    if checks.get("Source Inspection Requirement") or checks.get("QC-06"):
        buckets["inspection"] += DEFAULT_PARAMS["SourceInspection_Hours"]

    if checks.get("Gauge / Check Fixture NRE") or checks.get("QC-07"):
        params["NRE_FixturesEtc"] += DEFAULT_PARAMS["CheckFixtureNRE"]

    if checks.get("Precision Fitting Labor (Toolmaker)") or checks.get("ASM-04"):
        extra_role_hours["toolmaker"] += DEFAULT_PARAMS["PrecisionFitting_Hours"]

    if checks.get("Complex Assembly Documentation") or checks.get("ASM-07"):
        extra_role_hours["pm"] += DEFAULT_PARAMS["AssemblyDoc_Hours"]

    if checks.get("Laser Marking / Engraving Time") or checks.get("FIN-09"):
        buckets["finish"] += DEFAULT_PARAMS["LaserMark_Hours"]

    if checks.get("Passivation / Cleaning") or checks.get("FIN-10"):
        buckets["finish"] += DEFAULT_PARAMS["Passivation_Hours"]

    if checks.get("Tumbling / Vibratory Finishing Time") or checks.get("FIN-02"):
        buckets["finish"] += DEFAULT_PARAMS["Tumble_Hours"]

    # Live tooling / Sub-spindle / Form tool effects
    if checks.get("Live Tooling / Mill-Turn Ops Time") or checks.get("TRN-03"):
        turning_hours += DEFAULT_PARAMS["LiveTooling_Hours"]
    if checks.get("Sub-Spindle Utilization") or checks.get("TRN-04"):
        turning_hours += DEFAULT_PARAMS["SubSpindle_Hours"]
    if checks.get("Form Tool Requirement") or checks.get("TRN-10"):
        params["NRE_FixturesEtc"] += DEFAULT_PARAMS["FormToolNRE"]

    # ---------- Compute costs ----------
    # Scrap %
    scrap_pct = _first_numeric(df_num, [r"Expected Scrap Rate"], default=0.0)/100.0
    scrap_mult = 1.0 + scrap_pct

    # Add any Calculated milling/turning hours to their buckets
    # Apply material starting condition multiplier to milling/grinding
    milling_hours *= removal_multiplier

    # Role-specific hours -> costs
    pm_cost = extra_role_hours["pm"] * DEFAULT_RATES["PMRate"]
    toolmk_cost = extra_role_hours["toolmaker"] * DEFAULT_RATES["ToolmakerRate"]
    setup_cost = extra_role_hours["setup"] * DEFAULT_RATES["SetupRate"]
    blank_cost = extra_role_hours["blank"] * DEFAULT_RATES["BlankHandlingRate"]

    # Convert hour buckets to costs
    costs = {}
    costs["programming"] = buckets["programming"] * rates["ProgrammingRate"]
    costs["wedm"]        = buckets["wedm"] * rates["WireEDMRate"]
    costs["sinker"]      = buckets["sinker"] * rates["SinkerEDMRate"] + buckets["electrodes"] * rates["ElectrodeBuildRate"]
    costs["grind"]       = buckets["grind"] * rates["SurfaceGrindRate"]
    costs["jig"]         = buckets["jig"] * rates["JigGrindRate"]
    costs["lap"]         = buckets["lap"] * rates["LappingRate"]
    costs["inspection"]  = (buckets["inspection"] + buckets["cmm_programming"]) * rates["InspectionRate"]
    costs["finish"]      = buckets["finish"] * rates["FinishingRate"]
    costs["saw"]         = buckets["saw"] * rates["SawWaterjetRate"]
    costs["fixture"]     = buckets["fixture"] * rates["FixtureBuildRate"]
    costs["assembly"]    = buckets["assembly"] * rates["AssemblyRate"]
    costs["milling"]     = milling_hours * rates.get("MillingRate", rates["Milling3AxisRate"])
    costs["turning"]     = turning_hours * rates["TurningRate"]

    role_costs = pm_cost + toolmk_cost + setup_cost + blank_cost

    # Direct numeric costs picked up earlier
    direct_costs = (
        generic_contrib["freight_cost"]
        + generic_contrib["tooling_cost"]
        + generic_contrib["other_direct_cost"]
        + coating_cost + heat_cost + packaging_cost + cert_admin_cost
    )

    labor_subtotal = sum(costs.values()) + role_costs
    labor_post_scrap = labor_subtotal * scrap_mult

    # Material subtotal + surcharge (on positive portion only)
    material_base = params["MaterialOther"] + generic_contrib["material_cost"]
    material_surcharge_pct = params.get("MaterialSurchargePct", 0.0)
    material_surcharge = max(material_base, 0.0) * material_surcharge_pct

    material_credit = _sum_if(df_num, [r"Material Scrap / Remnant Value"])
    material_post_scrap = material_base * scrap_mult
    material_subtotal = material_post_scrap + material_surcharge + material_credit

    # Base subtotal + consumables + NRE + overhead (base OverheadPct + lookup adders + payment terms)
    base_subtotal = labor_post_scrap + material_subtotal + params["ConsumablesFlat"] + params["NRE_FixturesEtc"] + direct_costs

    # Overhead percent
    overhead_pct_total = params["OverheadPct"] + overhead_add_pct + payment_pct
    with_overhead = base_subtotal * (1.0 + overhead_pct_total)

    # Margin (base + relationship adj + expedite dropdown + expedite checkbox)
    margin_pct_total = params["MarginPct"] + margin_adj + expedite_pct_dropdown + expedite_pct_checkbox
    price = with_overhead * (1.0 + margin_pct_total)

    # Diagnostics: which numeric rows were not used at all
    def used_row(row) -> bool:
        itm = str(row["Item"])
        # If it matched a bucket keyword
        for pats in BUCKET_KEYWORDS.values():
            if any(re.search(p, itm, flags=re.IGNORECASE) for p in pats):
                return True
        # Or if it hit generic rules
        if re.search(r"Profit Margin|Surcharge|Cost|Freight|MOQ|Material Vendor|Fixture Material|Tooling Cost|Lead Time|Hardware|BOM|Project Manager Hours|Tool & Die Maker|Precision Fitting|Number of Milling Setups|Number of Unique Tools|Number of Blanks Required|Labor|Time|Hours", itm, flags=re.IGNORECASE):
            return True
        return False

    numeric_unused = df_num[~df_num.apply(used_row, axis=1)].copy()
    # Build and return breakdown
    return {
        "price": price,
        "components": {
            "labor_subtotal": labor_subtotal,
            "labor_post_scrap": labor_post_scrap,
            "scrap_pct": scrap_pct,
            "material_subtotal": material_subtotal,
            "material_base": material_base,
            "material_surcharge_pct": material_surcharge_pct,
            "material_surcharge": material_surcharge,
            "material_credit": material_credit,
            "material_post_scrap": material_post_scrap,
            "material_sheet_inputs": generic_contrib["material_cost"],
            "freight_costs": generic_contrib["freight_cost"],
            "tooling_costs": generic_contrib["tooling_cost"],
            "other_direct_costs": generic_contrib["other_direct_cost"],
            "direct_costs": direct_costs,
            "base_subtotal": base_subtotal,
            "overhead_pct_total": overhead_pct_total,
            "with_overhead": with_overhead,
            "margin_pct_total": margin_pct_total
        },
        "hours_buckets": buckets | {"milling": milling_hours, "turning": turning_hours},
        "cost_buckets": costs,
        "role_hours": extra_role_hours,
        "overhead_adders": overhead_details,
        "unused_numeric_rows": numeric_unused.to_dict(orient="records"),
        "missing_calculated_inputs": missing_calc,
    }
