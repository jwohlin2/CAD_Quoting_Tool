# NRE / Setup Cost Sources

The quote rendering code builds the **Programming & Engineering** and **Fixturing** rows in the
NRE section directly from the structured data returned by `compute_quote_from_df`.

* Programming hours are now derived by the `_estimate_programming_hours_auto(...)` helper instead
  of summing the spreadsheet. The heuristic looks at:
  * inferred setup count (sheet hint if present, otherwise face/normal counts from the geometry),
  * part complexity metrics (`GEO_Complexity_0to100`, thin wall flags, max dimension),
  * available process hours (milling, turning, EDM, drilling, finishing, inspection) to gauge the
    machining load, and
  * feature counts such as total faces, hole count, pockets, and slots.
  The contributions are summed, clamped by the simple-part cap (`ProgCapHr`) and the
  milling-ratio guard (`ProgMaxToMillingRatio`), and floored at 0.35 hr by default.  Entering a
  **Programming Override / Manual Hours** value on the sheet still forces a custom total.
* Engineering time is still read from rows such as *Fixture Design*, *Process Sheet*, *Traveler*,
  and *Documentation* because those activities are typically managed manually.
* Fixturing hours continue to come from rows labeled *Fixture Build* or *Custom Fixture Build*.
* The aggregated hours are multiplied by the configured shop rates, which default to $90/hour for
  both programmers and fixture builders in `cad_quoter/resources/rates_v1.json`.

The resulting `nre_detail` dictionary is what renders the quote output that shows entries such as
`Programmer: 0.85 hr @ $90.00/hr` and `Build Labor: 12.00 hr @ $90.00/hr`, with the programming
line automatically adjusting as the part geometry or process mix changes.
