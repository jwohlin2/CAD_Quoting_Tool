# Internal workflow reference

This note captures the smoke-test commands that used to live in ad-hoc scripts and points to the canonical tests or runbooks that now cover the behaviour.

## DWG import smoke test

```bash
python dwg_smoketest.py <dwg_path> <ODAFileConverter.exe>
```

The helper was only a thin wrapper around `cad_quoter.geometry.convert_dwg_to_dxf`, which remains the supported integration point. The deployment guide documents how to enable the ODA converter and inspect importer diagnostics without the standalone script. 【F:cad_quoter/geometry/__init__.py†L611-L706】【F:docs/deployment_guide.md†L60-L109】

## McMaster-Carr API smoke test

```bash
CAD_QUOTER_ALLOW_REQUESTS=1 python smoke_test_mcmaster.py --part 4936K451
```

The CLI flow is fully exercised by the package-level entry point (`python -m cad_quoter.vendors.mcmaster_stock`), which has direct pytest coverage. The public README already walks through the same setup steps, so the workflow stays discoverable through the maintained documentation. 【F:README.md†L141-L182】【F:tests/pricing/test_mcmaster_cli.py†L1-L43】

## Planner pricing shim

```python
from cad_quoter.pricing.planner import price_with_planner
```

Consumers now import the planner directly from the package; the end-to-end scenarios remain enforced by `tests/test_planner_pricing.py`. 【F:cad_quoter/pricing/planner.py†L1-L520】【F:tests/test_planner_pricing.py†L1-L87】
