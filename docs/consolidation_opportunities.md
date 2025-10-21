# Consolidation opportunities

This repository still carries several legacy entry points and compatibility layers that duplicate newer, package-scoped helpers. Consolidating these hotspots would simplify the import graph and reduce the risk of behaviour drift between the GUI (`appV5.py`) and the `cad_quoter` package.

## Completed cleanups

* Removed the redundant `dxf_text_extract.py` shim so callers rely directly on `cad_quoter.geometry.dxf_text.extract_text_lines_from_dxf`, retaining compatibility through the geometry package re-export. 【F:cad_quoter/geometry/__init__.py†L215-L226】【F:cad_quoter/geometry/dxf_text.py†L1-L90】
* Deleted the unused `time_models.py` module and updated migration docs to point at the actively maintained estimator in `cad_quoter/pricing/time_estimator.py`. 【F:docs/planner_pricing_migration.md†L25-L35】【F:cad_quoter/pricing/time_estimator.py†L1-L200】

## Process-cost bucketisation

The standalone bucketiser has been folded into the pricing package so that both the GUI and the renderer share a single set of heuristics. Planner minutes now flow through the helpers in `cad_quoter/pricing/process_buckets.py`, which expose the canonical planner bucket order, inspection fallbacks, and cost aggregation logic. 【F:cad_quoter/pricing/process_buckets.py†L1-L410】

## Planner pricing duplication

Planner cost conversion lives alongside the rate metadata in `cad_quoter/pricing/planner.py`. The thin compatibility shim `planner_pricing.py` simply re-exports `price_with_planner`, allowing existing imports to continue working while the heavy lifting happens inside the package. The migration guide has been updated to point at the new module path. 【F:cad_quoter/pricing/planner.py†L1-L520】【F:planner_pricing.py†L1-L5】【F:docs/planner_pricing_migration.md†L1-L60】

## Drilling estimator location

* The public drilling estimator shim still delegates to `_legacy_estimate_drilling_hours` housed inside `appV5.py`, making it difficult to reuse or test the implementation without the Tkinter app. 【F:cad_quoter/estimators/drilling.py†L1-L45】【F:appV5.py†L13610-L13790】

Migrating the legacy helper into `cad_quoter/estimators` (or refactoring it to use `cad_quoter/pricing/time_estimator.py`) would decouple estimators from the GUI monolith and make it easier to test or reuse outside the Tkinter app.
