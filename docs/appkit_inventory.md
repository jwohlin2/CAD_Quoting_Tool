# appkit module inventory and production usage

The legacy `appkit/` package has been evaluated module-by-module to document how each
piece of functionality is used in production code (primarily `appV5.py` and the
`cad_quoter/` package). The table below captures the primary responsibilities and
callers for every module that currently ships under `appkit/`.

| Module | Key responsibilities | Primary production consumers |
| --- | --- | --- |
| `appkit/data/__init__.py` | Package resource helpers (`load_text`, `load_json`). | `cad_quoter/domain_models/materials.py`, `cad_quoter/estimators/drilling.py`, `cad_quoter/material_density.py`, `cad_quoter/llm/__init__.py`, `appV5.py` (via indirect imports). |
| `appkit/debug/debug_tables.py` | JSON-friendly debug serialization helpers and drilling debug table formatter. | `appkit/utils.__init__` (re-export), `appV5.py` (debug table rendering). |
| `appkit/effective.py` | LLM suggestion acceptance flags, guardrail-aware merge, and conversion to overrides. | `appV5.py`, `cad_quoter/domain.py`, `tests/domain/test_effective_state.py`. |
| `appkit/env_utils.py` | Lazy boolean flags backed by environment variables (`FORCE_ESTIMATOR`, `FORCE_PLANNER`). | `appV5.py`, `appkit/planner_adapter.py`. |
| `appkit/guardrails.py` | Guardrail calculations for drilling/tapping floors, setup minimums, and finish pass cost enforcement. | `appkit/effective.py`, `appkit/merge_utils.py`, `appV5.py`. |
| `appkit/llm_adapter.py` | Integration surface for LLM-based hour inference, normalization/clamping helpers. | `appV5.py` (UI hook-up), `cad_quoter/app/llm_helpers.py` (delegate configuration). |
| `appkit/llm_converters.py` | Utilities for converting quote state data into LLM payloads and interpreting results. | `cad_quoter/domain.py` (lazy imports), `appV5.py`. |
| `appkit/merge_utils.py` | Core merge algorithm for baseline vs LLM suggestions vs overrides plus helper constants. | `appkit/effective.py`, `appkit/ui/suggestions.py`, `appV5.py`, `tests/domain/test_effective_state.py`. |
| `appkit/occ_compat.py` | Thin compatibility layer for OCC/trimesh geometry operations used by the viewer. | `cad_quoter/geometry/__init__.py`. |
| `appkit/planner_adapter.py` | Glue code for planner and pricing source resolution against modern container APIs. | `cad_quoter/app/quote_doc.py`, `appV5.py`, `tests/unit/test_pricing_source_resolution.py`. |
| `appkit/planner_helpers.py` | Planner-related orchestration helpers (process plan job execution). | `appV5.py`. |
| `appkit/ui/editor_controls.py` | Tk widget metadata derivation for editor panes. | `appV5.py`, `tests/test_editor_controls.py`. |
| `appkit/ui/llm_panel.py` | Tk UI components for LLM configuration/status. | `appV5.py`. |
| `appkit/ui/planner_render.py` | Rendering helpers for planner/process tables. | `cad_quoter/pricing/process_view.py`, `appV5.py`, `tests/app/test_planner_render.py`. |
| `appkit/ui/services.py` | Quote configuration data class and helpers used by UI and quote document. | `cad_quoter/app/quote_doc.py`, `appV5.py`. |
| `appkit/ui/session_io.py` | Serialize/deserialize quote sessions and file dialogs. | `appV5.py`, `tests/integration/test_app_smoke.py`. |
| `appkit/ui/suggestions.py` | Build/iterate UI suggestion rows. | `appV5.py`, `tests/unit/test_suggestion_rows.py`. |
| `appkit/ui/tk_compat.py` | Tk compatibility imports/shims for UI modules. | `appkit/ui/*`, `appV5.py`. |
| `appkit/ui/widgets.py` | Miscellaneous Tk widget helpers (scroll frames, etc.). | `appV5.py`. |
| `appkit/utils/__init__.py` | Machining math helpers (feeds/speeds), numeric parsing, debug JSON adapters. | `cad_quoter/app/chart_lines.py`, `cad_quoter/estimators/drilling.py`, `appV5.py`. |
| `appkit/utils/text_rules.py` | Text normalization rules for amortized cost labels. | `appkit/ui/planner_render.py`, `appV5.py`. |
| `appkit/vendor_utils.py` | Lead-time heuristics and partner metadata. | `tests/app/test_vendor_utils_lead_times.py`, `appV5.py`. |
| `appkit/graveyard.py` | Historical helpers kept for reference (unused in production). | (Not imported by production modules; safe to drop or archive.) |

This inventory informs the relocation of each module into the modern `cad_quoter/`
package so that the legacy `appkit/` directory can be removed.
