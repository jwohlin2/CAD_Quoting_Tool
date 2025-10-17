from quote_renderer import render_quote


def main() -> int:
    data = {
        "summary": {
            "qty": 2,
            "final_price": 100,
            "unit_price": 50,
            "subtotal_before_margin": 80,
            "margin_pct": 0.25,
        },
        "price_drivers": [
            {
                "label": "Cycle — roughing",
                "detail": "Contains\tcolor \x1b[31mred\x1b[0m text",
            }
        ],
        "cost_breakdown": {"Labor": 60, "Material": 20},
    }

    rendered = render_quote(data)

    # Simple assertions mirroring tests/test_quote_renderer.py
    if "\t" in rendered:
        print("Found tab in output")
        return 1
    if "\x1b" in rendered:
        print("Found ANSI escape in output")
        return 1
    if not all(ord(ch) < 128 for ch in rendered):
        print("Found non-ASCII character in output")
        return 1
    if "Cycle - roughing" not in rendered:
        print("Expected normalized label not found")
        return 1
    if "color red text" not in rendered:
        print("Expected sanitized detail not found")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

