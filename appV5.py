from cad_quoter.utils.render_utils import (
    fmt_hours,
    fmt_money,
    fmt_percent,
    fmt_range,
    format_currency,
    format_dimension,
    format_hours,
    format_hours_with_rate,
    format_percent,
)
        currency_formatter = lambda x: fmt_money(x, "$")  # pragma: no cover
    hours_text = fmt_hours(hr_val)
        hours_text = fmt_hours(hr_val)
                    credit_display = f"-{fmt_money(scrap_credit, currency)}"
                        f"{_format_weight_lb_oz(scrap_credit_mass_g)} × {fmt_money(scrap_credit_unit_price_lb, currency)} / lb"
                    detail_args.append(f"Soft jaw prep {fmt_hours(soft_jaw_hr)}")
                f"{label} hours capped at {fmt_hours(24.0, decimals=0)} for single-piece quote (was {fmt_hours(numeric_value)})."
            f"Subtotal (per-hole × qty) ............... {subtotal_min:.2f} min  ({fmt_hours(subtotal_min/60.0)})"
            f"TOTAL DRILLING (with toolchange) ........ {subtotal_min + tool_add:.2f} min  ({fmt_hours((subtotal_min + tool_add)/60.0)})"
                    return fmt_range(
                        min_float,
                        max_float,
                        formatter=lambda val: fmt.format(float(val)),
                    )
            return fmt_range(
                min_val,
                max_val,
                formatter=lambda val: fmt.format(float(val)),
            )
        fixture_plan_desc = f"{fmt_hours(fb)} build"
            fixture_plan_desc = f"{fmt_hours(fixture_build_hr)} build"
        fixture_notes.append(
            f"Soft jaw prep +{fmt_hours(soft_jaw_hr_override, unit='h')}{_source_suffix('soft_jaw_hr')}"
        )
        fixture_notes.append(
            f"Fixture build set to {fmt_hours(fixture_build_override, unit='h')}{_source_suffix('fixture_build_hr')}"
        )
        fixture_notes.append(
            f"Soft jaw stock +{fmt_money(soft_jaw_cost_override, '$')}{_source_suffix('soft_jaw_material_cost')}"
        )
            f"+{fmt_hours(handling_override, unit='h')} handling{_source_suffix('handling_adder_hr')}",
        )
        llm_notes.append(
            f"Handling adder +{fmt_hours(handling_override, unit='h')}{_source_suffix('handling_adder_hr')}"
            f"Packaging set to {fmt_hours(packaging_hours_override, unit='h')}{_source_suffix('packaging_hours')}",
        entry["notes"].append(
            f"set to {fmt_money(packaging_flat_override, '$')}{_source_suffix('packaging_flat_cost')}"
        )
        entry["notes"].append(
            f"set to {fmt_money(shipping_override, '$')}{_source_suffix('shipping_cost')}"
        )
                f"scrap {fmt_percent(old_scrap)} → {fmt_percent(scrap_after)}{suffix}"
                f"Scrap {fmt_percent(old_scrap)} → {fmt_percent(scrap_after)}{suffix}"
        entry["notes"].append(f"+{fmt_hours(add_hr)}")
        llm_notes.append(f"{_friendly_process(actual)} +{fmt_hours(add_hr)}{suffix}")
        entry["notes"].append(f"+{fmt_money(add_val, '$')}")
        llm_notes.append(f"{actual_label}: +{fmt_money(add_val, '$')}{suffix}")
            llm_notes.append(f"Contingency set to {fmt_percent(ContingencyPct)}{suffix}")
            fixture_bits.append(f"Soft jaw prep {fmt_hours(soft_jaw_hr)}")
            drift_display = fmt_money(abs(drift_amount), "$")
            rendered_display = fmt_money(planner_rendered_total, "$")
            planner_display = fmt_money(planner_totals_cost_float, "$")
                f"{drift_display}: rendered {rendered_display} vs planner {planner_display}"
            f"Note: labor total adjusted (expected {fmt_money(expected_labor_total, '$')})."
                f"⚠️ Labor totals drifted by {fmt_money(drift_amount, '$')}: "
                f"rendered {fmt_money(rendered_labor_total, '$')} vs expected {fmt_money(expected_display, '$')}"
            detail_bits.append(f"Applied {fmt_percent(insurance_pct)} of labor + directs")
            detail_bits.append(f"Markup {fmt_percent(vendor_markup)} on vendors + shipping")
            details.append(
                f"Programmer {fmt_hours(prog_detail['prog_hr'])} @ {fmt_money(prog_detail.get('prog_rate', 0), '$')}/hr"
            )
            details.append(
                f"Engineer {fmt_hours(prog_detail['eng_hr'])} @ {fmt_money(prog_detail.get('eng_rate', 0), '$')}/hr"
            )
            details.append(
                f"Build {fmt_hours(fix_detail['build_hr'])} @ {fmt_money(fix_detail.get('build_rate', 0), '$')}/hr"
            )
        f"  Cost makeup: material {fmt_money(material_total_for_why, currency)}; labor & machine "
        f"{fmt_money(breakdown.get('labor_cost_rendered', labor_cost), currency)}."
            if lo != hi:
                material_signals.append(
                    f"Hardness target {fmt_range(lo, hi, formatter=lambda v: f'{float(v):.0f}', unit='HRC')}"
                )
            else:
                material_signals.append(f"Hardness target {float(lo):.0f} HRC")
                f"Labor {fmt_money(labor_cost, '$')} ({fmt_percent(driver_primary['pct_of_subtotal'])}) and directs "
                f"{fmt_money(direct_costs, '$')} ({fmt_percent(driver_secondary['pct_of_subtotal'])}) drive cost. "
                f"Quote Generated! Final Price: {fmt_money(res.get('price', 0), '$')} (chars: simp={simp_len}, full={full_len})"
