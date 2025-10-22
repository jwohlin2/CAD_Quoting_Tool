# Consolidation opportunities

This repository still carries several legacy entry points and compatibility layers that duplicate newer, package‑scoped helpers. Consolidating these hotspots would simplify the import graph and reduce the risk of behaviour drift between the GUI (`appV5.py`) and the `cad_quoter` package.

## Completed cleanups

- Removed the redundant `dxf_text_extract.py` shim; callers should rely on `cad_quoter.geometry.dxf_text.extract_text_lines_from_dxf` exposed by the geometry package.
- Deleted the unused `time_models.py` module and updated migration docs to point to the actively maintained estimator in `cad_quoter.pricing.time_estimator`.

## Process-cost bucketisation

The standalone bucketiser has been folded into the pricing package so that both the GUI and the renderer share a single set of heuristics. Planner minutes now flow through helpers in `cad_quoter.pricing.process_buckets`, which define the canonical planner bucket order, inspection fallbacks, and cost aggregation logic.

## Planner pricing duplication

Planner cost conversion lives alongside the rate metadata in `cad_quoter.pricing.planner`. A thin compatibility shim `planner_pricing.py` re‑exports `price_with_planner`, allowing existing imports to continue working while the heavy lifting happens inside the package. The migration guide documents the new import path.

## Drilling estimator location

- The public drilling estimator shim still delegates to a legacy helper inside `appV5.py`, making it difficult to reuse or test the implementation without the Tkinter app.

Migrating the legacy helper into `cad_quoter.estimators` (or refactoring it to use `cad_quoter.pricing.time_estimator`) would decouple estimators from the GUI monolith and make it easier to test or reuse outside the Tkinter app.

