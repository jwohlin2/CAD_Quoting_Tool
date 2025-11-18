# appkit module inventory and production usage

> **Status: CONSOLIDATION COMPLETE**
>
> The legacy `appkit/` package has been fully consolidated into the `cad_quoter/` package
> and removed from the repository. This document is retained for historical reference only.

## Consolidation Summary

All functionality from the legacy `appkit/` package has been migrated to the modern
`cad_quoter/` package structure. The table below shows where each module's functionality
now resides.

| Original Module | New Location | Status |
| --- | --- | --- |
| `appkit/data/__init__.py` | `cad_quoter/resources/loading.py` | Migrated |
| `appkit/debug/debug_tables.py` | `cad_quoter/utils/debug_tables.py` | Migrated |
| `appkit/effective.py` | `cad_quoter/app/effective.py` | Migrated |
| `appkit/env_utils.py` | _(removed)_ | Retired |
| `appkit/guardrails.py` | `cad_quoter/app/guardrails.py` | Migrated |
| `appkit/llm_adapter.py` | `cad_quoter/llm/__init__.py` | Migrated |
| `appkit/llm_converters.py` | `cad_quoter/llm/converters.py` | Migrated |
| `appkit/merge_utils.py` | `cad_quoter/app/merge_utils.py` | Migrated |
| `appkit/occ_compat.py` | `cad_quoter/geometry/occ_compat.py` | Migrated |
| `appkit/planner_adapter.py` | _(removed)_ | Retired |
| `appkit/planner_helpers.py` | _(removed)_ | Retired |
| `appkit/ui/editor_controls.py` | _(removed)_ | Removed with Tk UI |
| `appkit/ui/llm_panel.py` | _(removed)_ | Removed with Tk UI |
| `appkit/ui/planner_render.py` | _(removed)_ | Retired |
| `appkit/ui/services.py` | _(removed)_ | Removed with Tk UI |
| `appkit/ui/session_io.py` | _(removed)_ | Removed with Tk UI |
| `appkit/ui/suggestions.py` | `cad_quoter/app/suggestions.py` | Migrated |
| `appkit/ui/tk_compat.py` | _(removed)_ | Removed with Tk UI |
| `appkit/ui/widgets.py` | _(removed)_ | Removed with Tk UI |
| `appkit/utils/__init__.py` | `cad_quoter/utils/machining.py` | Migrated |
| `appkit/utils/text_rules.py` | `cad_quoter/utils/text_rules.py` | Migrated |
| `appkit/vendor_utils.py` | `cad_quoter/pricing/vendor_utils.py` | Migrated |
| `appkit/graveyard.py` | _(removed)_ | Archived/Deleted |

The `appkit/` directory has been completely removed from the repository.
