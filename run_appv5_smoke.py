import appV5


def main() -> int:
    result = {
        "price": 10.0,
        "breakdown": {
            "qty": 1,
            "totals": {
                "labor_cost": 0.0,
                "direct_costs": 0.0,
                "subtotal": 0.0,
                "with_expedite": 0.0,
                "with_margin": 0.0,
            },
            "material": {},
            "nre_detail": {
                "programming": {"per_lot": 150.0, "prog_hr": 1.0, "prog_rate": 75.0},
                "fixture": {
                    "per_lot": 80.0,
                    "build_hr": 0.5,
                    "build_rate": 60.0,
                    "mat_cost": 20.0,
                    "labor_cost": 30.0,
                },
            },
            "nre": {"programming_per_lot": 150.0},
            "process_costs": {"grinding": 300.0},
            "process_meta": {"grinding": {"hr": 1.5, "rate": 120.0, "base_extra": 200.0}},
            "labor_cost_details": {"Grinding": "1.50 hr @ $120.00/hr; includes $200.00 extras"},
            "direct_cost_details": {},
            "bucket_view": {
                "buckets": {"grinding": {"minutes": 90.0, "labor$": 300.0, "machine$": 0.0}},
                "order": ["grinding"],
            },
        },
    }

    text = appV5.render_quote(result, currency="$", show_zeros=False)
    # Basic sanity checks from tests
    if not text or not isinstance(text, str):
        print("Empty or invalid render output")
        return 1
    if "Quote Summary" not in text:
        print("Expected Quote Summary header not found")
        return 1
    if "Cost Breakdown" not in text:
        print("Expected Cost Breakdown section not found")
        return 1
    # Ensure NRE per-lot appears for single-qty
    if "Programming & Eng (per lot)" not in text:
        print("Expected Programming & Eng (per lot) not found")
        return 1
    print("OK appV5.render_quote smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

