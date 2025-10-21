# Consolidation opportunities

This repository still carries several legacy entry points and compatibility layers that duplicate newer, package-scoped helpers. Consolidating these hotspots would simplify the import graph and reduce the risk of behaviour drift between the GUI (`appV5.py`) and the `cad_quoter` package.

## Completed cleanups

* Removed the redundant `dxf_text_extract.py` shim so callers rely directly on `cad_quoter.geometry.dxf_text.extract_text_lines_from_dxf`, retaining compatibility through the geometry package re-export. 【F:cad_quoter/geometry/__init__.py†L215-L226】【F:cad_quoter/geometry/dxf_text.py†L1-L90】
* Deleted the unused `time_models.py` module and updated migration docs to point at the actively maintained estimator in `cad_quoter/pricing/time_estimator.py`. 【F:docs/planner_pricing_migration.md†L25-L35】【F:cad_quoter/pricing/time_estimator.py†L1-L200】

## Process-cost bucketisation

* `bucketizer.py` computes presentation buckets from planner line items, including hard-coded bucket names, heuristics, and inspection fallbacks. 【F:bucketizer.py†L1-L200】【F:bucketizer.py†L200-L338】
* `cad_quoter/pricing/process_cost_renderer.py` performs the same normalisation and rendering with its own bucket ordering, alias table, and rate resolution logic. 【F:cad_quoter/pricing/process_cost_renderer.py†L1-L200】【F:cad_quoter/pricing/process_cost_renderer.py†L303-L382】

Keeping two parallel implementations makes it easy for naming or rate rules to diverge (for example, when introducing new planner operations). Extracting the shared mapping/rate logic into a single module and having both the GUI and pricing engine depend on it would prevent the drift that currently requires mirrored test fixtures (`tests/pricing/test_bucketizer_drilling.py` vs. `tests/pricing/test_process_cost_renderer.py`). 【F:tests/pricing/test_bucketizer_drilling.py†L1-L95】【F:tests/pricing/test_process_cost_renderer.py†L1-L160】

## Planner pricing duplication

* `planner_pricing.py` both interprets the process-planner output and applies its own bucket-to-rate mapping, duplicating the rate lookups that also exist in the pricing package. 【F:planner_pricing.py†L437-L520】
* The migration guide encourages new integrations to call `price_with_planner` directly, so the overlap between this script and `cad_quoter/pricing/process_cost_renderer.py` will persist until the rate logic is centralised. 【F:docs/planner_pricing_migration.md†L1-L48】

Extracting the shared rate/bucket mapping into a reusable helper (and letting `planner_pricing` focus solely on translating planner minutes) would make the CLI and pricing engine share one codepath for dollars-per-minute conversions.

## Drilling estimator location

* The public drilling estimator shim still delegates to `_legacy_estimate_drilling_hours` housed inside `appV5.py`, making it difficult to reuse or test the implementation without the Tkinter app. 【F:cad_quoter/estimators/drilling.py†L1-L45】【F:appV5.py†L13610-L13790】

Migrating the legacy helper into `cad_quoter/estimators` (or refactoring it to use `cad_quoter/pricing/time_estimator.py`) would decouple estimators from the GUI monolith and make it easier to test or reuse outside the Tkinter app.
