# AppV5 Simplification Plan

## Current Pain Points
- **Single monolithic module** – `appV5.py` bundles environment bootstrapping, CAD parsing, quoting logic, LLM orchestration, pricing integrations, and the Tk UI into one 12k-line script. Examples include environment configuration (`AppEnvironment`) at the top of the file, pricing providers in the middle, and the GUI class at the bottom of the file.【F:appV5.py†L1-L160】【F:appV5.py†L4782-L4897】【F:appV5.py†L11249-L11409】
- **Global state that crosses concerns** – shared dictionaries like `RATES_DEFAULT` and `PARAMS_DEFAULT` are defined once and mutated throughout the UI, making it hard to understand where values originate and to reuse the logic in other contexts.【F:appV5.py†L4653-L4705】【F:appV5.py†L11291-L11320】
- **Complex business rules mixed with I/O** – quoting state management (`QuoteState`), data normalization helpers, and LLM prompt handling live alongside GUI callbacks, complicating testing and reuse.【F:appV5.py†L673-L760】【F:appV5.py†L3573-L3680】【F:appV5.py†L11291-L11409】

## Recommended Modular Structure
1. **`cad_quoter/config.py`** – isolate environment and settings management (e.g., `AppEnvironment`, `describe_runtime_environment`) so configuration can be loaded without pulling in Tkinter.【F:appV5.py†L24-L63】
2. **`cad_quoter/geometry/` package** – house CAD ingestion and enrichment utilities (`load_cad_any`, STEP/STL enrichment helpers). This keeps OCC/trimesh dependencies away from non-geometry code and allows targeted unit tests.【F:appV5.py†L3333-L3572】
3. **`cad_quoter/domain.py`** – keep data classes and calculations such as `QuoteState`, scrap calculations, and `build_suggest_payload` in a pure-Python module that can be exercised by tests or future services.【F:appV5.py†L673-L760】
4. **`cad_quoter/llm.py`** – wrap `_LocalLLM`, prompt templates, and parsing helpers so model integration can be swapped or mocked independently of the UI.【F:appV5.py†L148-L160】【F:appV5.py†L3573-L3680】
5. **`cad_quoter/pricing/` package** – move `PriceProvider`, `MetalsAPI`, `VendorCSV`, caching, and Wieland integration into a coherent subsystem with clear interfaces and dependency injection.【F:appV5.py†L4768-L4897】
6. **`cad_quoter/ui/` package** – keep Tkinter widgets (`ScrollableFrame`, `App`) in their own module, importing the domain and services they depend on rather than defining everything inline.【F:appV5.py†L11249-L11409】

Organising the project into a package makes it easier to expose a CLI entry point for automation and, later, to ship a lighter-weight API without the GUI.

## Stepwise Refactor Plan
1. **Establish package scaffolding** – create the `cad_quoter` package and move non-UI helpers first (configuration, domain types). Update imports in `appV5.py` to consume the new modules while keeping behaviour identical.
2. **Introduce dependency injection** – change the UI to accept service objects (geometry loader, quote calculator, LLM client, pricing provider registry) via constructor parameters. This reduces reliance on globals and allows alternate front-ends.
3. **Extract geometry and pricing subsystems** – once the UI is parameterised, relocate OCC/trimesh geometry functions and pricing providers into their packages. Add smoke tests that import these modules without starting Tkinter.
4. **Modularise LLM features** – move prompt text, `_LocalLLM`, and parsing utilities into the dedicated module so they can be reused by a CLI or batch mode. Provide thin adapters in the UI for logging and error handling.
5. **Trim `appV5.py`** – after extractions, convert `appV5.py` into a lightweight launcher that wires together modules and starts `App`. This improves readability and speeds up load time for non-GUI tooling.
6. **Add regression tests** – with pure modules extracted, create unit tests for quoting calculations, pricing conversions, and geometry enrichment using fixtures. This protects against regressions during further refactors.

## Additional Simplifications
- **Configuration files** – serialise `RATES_DEFAULT` and `PARAMS_DEFAULT` to JSON/YAML so non-developers can maintain defaults without editing Python.【F:appV5.py†L4653-L4705】
- **State serialisation** – formalise `QuoteState` persistence (e.g., `to_dict` / `from_dict`) to enable saving/loading sessions or serving the logic via API.【F:appV5.py†L673-L687】
- **Logging** – replace scattered `print` debugging with a central logger, simplifying observability once the modules are separated.【F:appV5.py†L3504-L3524】

Following these steps will reduce `appV5.py` to a manageable orchestrator, clarify ownership of each subsystem, and make it far easier to onboard contributors or build automated tests.
