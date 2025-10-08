# NRE / Setup Cost Sources

The quote rendering code builds the **Programming & Engineering** and **Fixturing** rows in the NRE section
directly from the structured data returned by `compute_quote_from_df`.

* Programming time is accumulated from every spreadsheet row whose label contains
  *Programming*, *2D CAM*, *3D CAM*, *Simulation*, *Verification*, *DFM*, or *Setup Sheets* (with
  CMM / override rows filtered out). The helper converts anything typed as
  hours or minutes into a numeric hour total. That logic lives in
  `sum_time(...)` and the programming block at the top of the costing pipeline.
* Fixturing hours are pulled in the same pass by summing rows labeled *Fixture Build* or
  *Custom Fixture Build*.
* The aggregated hours are multiplied by the configured shop rates, which default to
  $90/hour for both programmers and fixture builders in `cad_quoter/resources/rates_v1.json`.

The resulting `nre_detail` dictionary is what renders the quote output that shows entries such as
`Programmer: 11.75 hr @ $90.00/hr` and `Build Labor: 12.00 hr @ $90.00/hr` when the input sheet
contains that much time.
