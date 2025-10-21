# NRE / Setup Cost Sources

The quote rendering code builds the **Programming & Engineering** and **Fixturing** rows in the
NRE section directly from the structured data returned by `compute_quote_from_df`.

* Programming hours are driven by the `_compute_programming_detail_minutes(...)` helper instead of
  the retired programming-hour estimators. The routine inspects the geometry context for a
  `hole_count` (or falls back to counting `hole_diams_mm`) and counts one minute per hole. It then
  walks the process plan and adds another minute for each CNC or Wire EDM operation that appears.
  The total detail minutes are converted to hours (with a minimum of one minute) and stored as the
  automatic programming estimate. User supplied Programming Overrides, either from the spreadsheet
  or stored user preferences, continue to take precedence over the auto estimate.
* Engineering time is still read from rows such as *Fixture Design*, *Process Sheet*, *Traveler*,
  and *Documentation* because those activities are typically managed manually.
* Fixturing hours continue to come from rows labeled *Fixture Build* or *Custom Fixture Build*.
* The aggregated hours are multiplied by the configured shop rates, which default to $90/hour for
  both programmers and fixture builders in `cad_quoter.resources.rates_v1`.

The resulting `nre_detail` dictionary is what renders the quote output that shows entries such as
`Programmer: 0.85 hr @ $90.00/hr` and `Build Labor: 12.00 hr @ $90.00/hr`, with the programming
line automatically adjusting as the part geometry or process mix changes.
