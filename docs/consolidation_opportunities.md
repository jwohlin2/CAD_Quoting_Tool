# Consolidation opportunities

> **Status: ALL CONSOLIDATIONS COMPLETE**

This document tracks legacy entry points and compatibility layers that have been consolidated into the modern `cad_quoter` package structure.

## Completed cleanups

All consolidation work has been completed:

- [x] Removed the redundant `dxf_text_extract.py` shim; callers now use `cad_quoter.geometry.dxf_text.extract_text_lines_from_dxf`
- [x] Deleted the unused `time_models.py` module; functionality in `cad_quoter.pricing.time_estimator`
- [x] Removed `planner_pricing.py` shim; callers import `cad_quoter.pricing.planner` directly
- [x] Consolidated drilling estimators into `cad_quoter.estimators` and `cad_quoter.pricing.time_estimator`
- [x] Removed entire `appkit/` directory and migrated all functionality to `cad_quoter/`
- [x] Removed `appkit/env_utils.py` (retired environment toggles)
- [x] Migrated `appkit/llm_adapter.py` to `cad_quoter/llm/__init__.py`
- [x] Migrated `appkit/llm_converters.py` to `cad_quoter/llm/converters.py`
- [x] Migrated `appkit/merge_utils.py` to `cad_quoter/app/merge_utils.py`
- [x] Migrated `appkit/guardrails.py` to `cad_quoter/app/guardrails.py`
- [x] Removed `appkit/ui/*` subtree (Tk UI removed from repository)

## Process-cost bucketisation

The standalone bucketiser has been folded into the pricing package so that both the GUI and the renderer share a single set of heuristics. Planner minutes now flow through helpers in `cad_quoter.pricing.process_buckets`, which define the canonical planner bucket order, inspection fallbacks, and cost aggregation logic.

## Planner pricing duplication

Planner cost conversion lives in `cad_quoter.pricing.planner`. The thin compatibility shim `planner_pricing.py` has been removed; callers now import directly from `cad_quoter.pricing.planner`.

## Drilling estimator location

All drilling estimator functionality has been migrated into `cad_quoter.estimators` and `cad_quoter.pricing.time_estimator`, decoupling estimators from the GUI and making them easier to test and reuse.

## appkit package removal

The entire `appkit/` package has been removed. See `docs/appkit_inventory.md` for the complete migration map showing where each module's functionality now resides in the `cad_quoter/` package.

