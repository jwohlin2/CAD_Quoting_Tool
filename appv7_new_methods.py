    def _generate_direct_costs_report(self) -> str:
        """Generate formatted direct costs report using QuoteData."""
        self.direct_cost_total = None
        if not self.cad_file_path:
            return "No CAD file loaded. Please load a CAD file first."

        try:
            # Get cached QuoteData (avoids redundant ODA/OCR)
            quote_data = self._get_or_create_quote_data()

            # Extract data from QuoteData
            part_dims = quote_data.part_dimensions
            material_info = quote_data.material_info
            stock_info = quote_data.stock_info
            scrap_info = quote_data.scrap_info
            cost_breakdown = quote_data.direct_cost_breakdown

            # Check if we have valid price data
            if stock_info.mcmaster_price is None:
                price_str = "Price N/A"
            else:
                price_str = f"${stock_info.mcmaster_price:,.2f}"

            # Format the report
            report = []
            report.append("DIRECT COSTS")
            report.append("=" * 74)
            report.append(f"Material used: {material_info.material_name}")
            report.append(f"  Required Stock: {stock_info.desired_length:.2f} × {stock_info.desired_width:.2f} × {stock_info.desired_thickness:.2f} in")
            report.append(f"  Rounded to catalog: {stock_info.mcmaster_length:.2f} × {stock_info.mcmaster_width:.2f} × {stock_info.mcmaster_thickness:.3f}")
            report.append(f"  Starting Weight: {self._format_weight(stock_info.mcmaster_weight)}")
            report.append(f"  Net Weight: {self._format_weight(stock_info.final_part_weight)}")
            report.append(f"  Scrap Percentage: {scrap_info.scrap_percentage:.1f}%")
            report.append(f"  Scrap Weight: {self._format_weight(scrap_info.total_scrap_weight)}")

            if scrap_info.scrap_price_per_lb is not None:
                report.append(f"  Scrap Price: ${scrap_info.scrap_price_per_lb:.2f} / lb")
            else:
                report.append(f"  Scrap Price: N/A")

            report.append("")
            report.append("")

            # Cost breakdown
            report.append(f"  Stock Piece (McMaster part {stock_info.mcmaster_part_number or 'N/A'})".ljust(60) + f"{price_str:>14}")

            if stock_info.mcmaster_price is not None:
                report.append(f"  Tax".ljust(60) + f"+${cost_breakdown.tax:>12.2f}")
                report.append(f"  Shipping".ljust(60) + f"+${cost_breakdown.shipping:>12.2f}")

            if cost_breakdown.scrap_credit > 0:
                scrap_credit_line = f"  Scrap Credit @ Wieland ${scrap_info.scrap_price_per_lb:.2f}/lb × {scrap_info.scrap_percentage:.1f}%"
                report.append(f"{scrap_credit_line.ljust(60)}-${cost_breakdown.scrap_credit:>12.2f}")

            report.append(" " * 60 + "-" * 14)

            if stock_info.mcmaster_price is not None:
                self.direct_cost_total = cost_breakdown.net_material_cost
                report.append(f"  Total Material Cost :".ljust(60) + f"${cost_breakdown.net_material_cost:>13.2f}")
            else:
                self.direct_cost_total = None
                report.append(f"  Total Material Cost :".ljust(60) + "Price N/A")

            report.append("")

            return "\n".join(report)

        except Exception as e:
            self.direct_cost_total = None
            import traceback
            return f"Error generating direct costs report:\n{str(e)}\n\n{traceback.format_exc()}"

    def _generate_machine_hours_report(self) -> str:
        """Generate formatted machine hours report using QuoteData."""
        self.machine_cost_total = None
        if not self.cad_file_path:
            return "No CAD file loaded. Please load a CAD file first."

        try:
            # Get cached QuoteData (avoids redundant ODA/OCR)
            quote_data = self._get_or_create_quote_data()

            machine_hours = quote_data.machine_hours

            if not machine_hours.drill_operations and not machine_hours.tap_operations:
                return "No hole operations found in CAD file."

            # Format helper functions
            def format_drill_group(op):
                return (f"Hole {op.hole_id} | Dia {op.diameter:.4f}\" x {op.qty} | "
                        f"depth {op.depth:.3f}\" | {op.sfm:.0f} sfm | "
                        f"{op.ipr:.4f} ipr | t/hole {op.time_per_hole:.2f} min | "
                        f"group {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_jig_grind_group(op):
                return (f"Hole {op.hole_id} | Dia {op.diameter:.4f}\" x {op.qty} | "
                        f"depth {op.depth:.3f}\" | "
                        f"t/hole {op.time_per_hole:.2f} min | "
                        f"group {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_tap_group(op):
                return (f"Hole {op.hole_id} | Dia {op.diameter:.4f}\" x {op.qty} | "
                        f"depth {op.depth:.3f}\" | {op.tpi} TPI | "
                        f"t/hole {op.time_per_hole:.2f} min | "
                        f"group {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_cbore_group(op):
                return (f"Hole {op.hole_id} | Dia {op.diameter:.4f}\" x {op.qty} | "
                        f"depth {op.depth:.3f}\" | {op.sfm:.0f} sfm | "
                        f"t/hole {op.time_per_hole:.2f} min | "
                        f"group {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_cdrill_group(op):
                return (f"Hole {op.hole_id} | Dia {op.diameter:.4f}\" x {op.qty} | "
                        f"depth {op.depth:.3f}\" | "
                        f"t/hole {op.time_per_hole:.2f} min | "
                        f"group {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            # Build the report
            report = []
            report.append("MACHINE HOURS ESTIMATION - DETAILED HOLE TABLE BREAKDOWN")
            report.append("=" * 74)
            report.append(f"Material: {quote_data.material_info.material_name}")
            report.append(f"Thickness: {quote_data.part_dimensions.thickness:.3f}\"")
            report.append(f"Hole entries: {len(machine_hours.drill_operations + machine_hours.tap_operations + machine_hours.cbore_operations)}")
            report.append("")

            # TIME PER HOLE - DRILL GROUPS
            if machine_hours.drill_operations:
                report.append("TIME PER HOLE - DRILL GROUPS")
                report.append("-" * 74)
                for op in machine_hours.drill_operations:
                    report.append(format_drill_group(op))
                report.append(f"\nTotal Drilling Time: {machine_hours.total_drill_minutes:.2f} minutes")
                report.append("")

            # TIME PER HOLE - JIG GRIND
            if machine_hours.jig_grind_operations:
                report.append("TIME PER HOLE - JIG GRIND")
                report.append("-" * 74)
                for op in machine_hours.jig_grind_operations:
                    report.append(format_jig_grind_group(op))
                report.append(f"\nTotal Jig Grind Time: {machine_hours.total_jig_grind_minutes:.2f} minutes")
                report.append("")

            # TIME PER HOLE - TAP
            if machine_hours.tap_operations:
                report.append("TIME PER HOLE - TAP")
                report.append("-" * 74)
                for op in machine_hours.tap_operations:
                    report.append(format_tap_group(op))
                report.append(f"\nTotal Tapping Time: {machine_hours.total_tap_minutes:.2f} minutes")
                report.append("")

            # TIME PER HOLE - C'BORE
            if machine_hours.cbore_operations:
                report.append("TIME PER HOLE - C'BORE")
                report.append("-" * 74)
                for op in machine_hours.cbore_operations:
                    report.append(format_cbore_group(op))
                report.append(f"\nTotal Counterbore Time: {machine_hours.total_cbore_minutes:.2f} minutes")
                report.append("")

            # TIME PER HOLE - CDRILL
            if machine_hours.cdrill_operations:
                report.append("TIME PER HOLE - CDRILL")
                report.append("-" * 74)
                for op in machine_hours.cdrill_operations:
                    report.append(format_cdrill_group(op))
                report.append(f"\nTotal Center Drill Time: {machine_hours.total_cdrill_minutes:.2f} minutes")
                report.append("")

            # Summary
            report.append("=" * 74)
            report.append(f"TOTAL MACHINE TIME: {machine_hours.total_minutes:.2f} minutes ({machine_hours.total_hours:.2f} hours)")
            report.append(f"TOTAL MACHINE COST: {machine_hours.machine_cost:.2f}")
            report.append("=" * 74)
            report.append("")

            self.machine_cost_total = machine_hours.machine_cost

            return "\n".join(report)

        except Exception as e:
            self.machine_cost_total = None
            import traceback
            return f"Error generating machine hours report:\n{str(e)}\n\n{traceback.format_exc()}"

    def _generate_labor_hours_report(self) -> str:
        """Generate formatted labor hours report using QuoteData."""
        self.labor_cost_total = None
        if not self.cad_file_path:
            return "No CAD file loaded. Please load a CAD file first."

        try:
            # Get cached QuoteData (avoids redundant ODA/OCR)
            quote_data = self._get_or_create_quote_data()

            labor_hours = quote_data.labor_hours

            # Format the report
            report = []
            report.append("LABOR HOURS ESTIMATION")
            report.append("=" * 74)
            report.append("")

            # Summary table
            report.append("LABOR BREAKDOWN BY CATEGORY")
            report.append("-" * 74)
            report.append(f"  Setup / Prep:                    {labor_hours.setup_minutes:>10.2f} minutes")
            report.append(f"  Programming / Prove-out:         {labor_hours.programming_minutes:>10.2f} minutes")
            report.append(f"  Machining Steps:                 {labor_hours.machining_steps_minutes:>10.2f} minutes")
            report.append(f"  Inspection:                      {labor_hours.inspection_minutes:>10.2f} minutes")
            report.append(f"  Finishing / Deburr:              {labor_hours.finishing_minutes:>10.2f} minutes")
            report.append("-" * 74)
            report.append(f"  TOTAL LABOR TIME:                {labor_hours.total_minutes:>10.2f} minutes")
            report.append(f"                                   {labor_hours.total_hours:>10.2f} hours")
            report.append(f"  TOTAL LABOR COST:                {labor_hours.labor_cost:>10.2f}")
            report.append("")
            report.append("=" * 74)
            report.append("")

            self.labor_cost_total = labor_hours.labor_cost

            return "\n".join(report)

        except Exception as e:
            self.labor_cost_total = None
            import traceback
            return f"Error generating labor hours report:\n{str(e)}\n\n{traceback.format_exc()}"
