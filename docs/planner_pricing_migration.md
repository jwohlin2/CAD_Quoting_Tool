# Planner Pricing Migration Guide

This guide documents how to replace the legacy worksheet-based process costing in `appV5.py` with the new planner-powered pricing path. The intent is to give Codex (and future maintainers) a clear recipe for swapping between the two implementations during the transition period.

## Overview

1. Convert any legacy flat-rate overrides (`{"MillingRate": 85.0, ...}`) into the new two-bucket structure expected by the planner pricing engine. Use `cad_quoter.rates.migrate_flat_to_two_bucket` for this conversion.
2. Build the planner input payload from the quote worksheet. The planner needs three pieces of information:
   - `family`: planner family identifier such as `"die_plate"` or `"punch"`.
   - `planner_params`: the logical inputs gathered from UI variables (material, tolerances, etc.).
   - `geom`: the geometric aggregates derived from CAD (perimeters, areas, hole lists, and the time-model inputs).
3. Pass the payload to `planner_pricing.price_with_planner`. This function is the single workhorse that runs the planner, calculates cycle times, and returns a transparent cost breakdown.
4. Persist the pricing results back onto the quote dictionary. Store the qualitative plan (`planner_plan`), line-item breakdown (`process_line_items`), machine vs. labor totals (`process_costs`), total minutes, and subtotal process cost so that downstream insurance/markup logic continues to work unchanged.
5. Keep the rest of the quote math identical. Insurance, markup, and margin should continue to consume `quote["process_costs"]` and `quote["subtotal_process_cost"]` as they do today.

## Sample Implementation

Below is a template that can replace the legacy `validate_quote_before_pricing` helper in `appV5.py` (or whichever module is currently accumulating process costs by multiplying hours × rate):

```python
from planner_pricing import price_with_planner
from rates import migrate_flat_to_two_bucket


def validate_quote_before_pricing(quote: Dict[str, Any], overrides_flat: Dict[str, float]) -> Dict[str, Any]:
    # 1) Convert old UI keys to two-bucket rates
    rates = migrate_flat_to_two_bucket(overrides_flat)

    # 2) Build the planner inputs from the quote worksheet (family + params + geom)
    family = quote["family"]              # e.g., "die_plate", "punch"
    params = quote["planner_params"]      # the logic inputs we defined (material, tolerances, etc.)
    geom   = quote["geom"]                # perimeters, areas, hole lists, etc. (see time_models header)

    # 3) Use the planner as THE workhorse
    priced = price_with_planner(family, params, geom, rates, oee=quote.get("oee", 0.85))

    # 4) Persist transparent breakdown
    quote["planner_plan"] = priced["plan"]            # qualitative ops + fixturing + QA
    quote["process_line_items"] = priced["line_items"]
    quote["process_costs"] = {
        "Machine": priced["totals"]["machine_cost"],
        "Labor":   priced["totals"]["labor_cost"],
    }
    quote["process_minutes"] = priced["totals"]["minutes"]
    quote["subtotal_process_cost"] = priced["totals"]["total_cost"]

    # 5) Keep downstream math (insurance, markup, etc.) the same
    return quote
```

## Transitional Flag (Optional)

If you need to keep the legacy worksheet logic available while rolling out planner pricing, gate the new path behind a runtime flag:

```python
if settings.USE_PLANNER_PRICING:
    quote = validate_quote_before_pricing(quote, overrides_flat)
else:
    quote = validate_quote_before_pricing_legacy(quote, overrides_flat)
```

Wire the flag to an environment variable or configuration setting so you can toggle planner pricing per deployment.

Following these steps lets you toggle between the historical hour × rate approach and the planner-driven costing without touching the downstream financial calculations.
