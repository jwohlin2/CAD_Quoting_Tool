from appV5 import render_quote


def test_render_quote_sanitizes_special_characters() -> None:
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
                "label": "Cycle – roughing",
                "detail": "Contains\tcolor \x1b[31mred\x1b[0m text",
            }
        ],
        "cost_breakdown": {"Labor": 60, "Material": 20},
    }

    rendered = render_quote(data)

    assert "\t" not in rendered
    assert "\x1b" not in rendered
    allowed_unicode = {"×", "–", "≥", "≤"}
    assert all(ord(ch) < 128 or ch in allowed_unicode for ch in rendered)
    assert "Cycle – roughing" in rendered
    assert "color red text" in rendered
