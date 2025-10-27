from __future__ import annotations

from cad_quoter.render import RenderState, render_pass_through


def _build_state(payload: dict, *, currency: str = "$", show_zeros: bool = False) -> RenderState:
    return RenderState(payload, currency=currency, show_zeros=show_zeros, divider="-" * 5, page_width=60)


def test_render_pass_through_updates_header_and_placeholders() -> None:
    payload = {
        "breakdown": {
            "pass_through": {
                "Material": 0.0,
                "Heat Treat": 250.0,
                "Shipping": 45.0,
                "vendor_items": {"Fixture": 125.0},
            },
            "vendor_items": {"CMM": 180.0},
            "direct_cost_details": {
                "Heat Treat": "Includes rush service",
                "vendor items": "Fixture and outside inspection",
            },
            "pass_basis": {
                "Heat Treat": {"basis": "Quoted vendor turnaround"},
                "Material": {"basis": "Tool steel billet pricing"},
            },
            "material": {
                "total_material_cost": 320.0,
            },
            "materials_direct": 0.0,
            "materials": [
                {"label": "Tool steel billet", "amount": 0.0},
            ],
            "cost_breakdown": {"NRE": 150.0},
        }
    }

    state = _build_state(payload)

    summary_builder = state.new_section("Quote Summary")
    summary_builder.add_line(state.divider)
    total_line_index = summary_builder.add_row("Direct Costs:", 0.0, indent="  ")
    summary_record = summary_builder.finalize()
    state.register_placeholder(
        "direct_cost_total",
        summary_record.index,
        total_line_index,
        formatter=lambda value: state.format_row("Direct Costs:", value, indent="  "),
    )

    section_lines = render_pass_through(state)

    assert section_lines[0] == "Pass-Through & Direct Costs (Total: $920.00)"
    assert state.sections[1].title == section_lines[0]
    assert state.sections[1].doc_section is not None
    assert state.sections[1].doc_section.title == section_lines[0]

    expected_total_line = state.format_row("Direct Costs:", 920.0, indent="  ")
    assert state.sections[0].lines[total_line_index] == expected_total_line

    direct_costs = state.pricing["direct_costs"]
    assert direct_costs["vendor items"] == 305.0  # 125 + 180
    assert direct_costs["Heat Treat"] == 250.0

    assert payload["breakdown"]["vendor_items_total"] == 305.0
    assert payload["breakdown"]["total_direct_costs"] == 920.0
    assert payload["breakdown"]["pass_through_total"] == 295.0


def test_render_pass_through_material_warning_flag() -> None:
    payload = {
        "breakdown": {
            "pass_through": {
                "Material": 0.0,
                "Shipping": 60.0,
            },
            "materials": [
                {"label": "Plate", "amount": 0.0},
            ],
            "materials_direct": 0.0,
            "material": {},
            "direct_cost_details": {},
        }
    }

    state = _build_state(payload)

    lines = render_pass_through(state)

    assert any("⚠ MATERIALS MISSING" in line for line in lines)
    assert state.warning_flags["material_warning"] is True
    assert payload["breakdown"]["material_warning_needed"] is True

    header = lines[0]
    assert header.endswith("$60.00)")

