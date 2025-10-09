# CAD Quoting Tool

The CAD Quoting Tool is a desktop workflow for estimating part pricing from
engineering data.  It loads customer CAD (STEP/IGES/STL/DXF), applies quoting
heuristics, and optionally calls a local Qwen vision model to propose override
values for the standard quoting variables.  The Tkinter UI also exposes
override, audit and debugging panels so estimators can review generated quotes
before export.

## Prerequisites

* Python 3.11 or newer
* Windows, macOS or Linux with OpenCascade libraries available (for STEP/IGES)
* Optional: locally downloaded Qwen2.5-VL GGUF weights for LLM-assisted quoting

> **Tip:** The runtime checks for a few third-party Python packages at startup
> (`requests`, `beautifulsoup4` and `lxml`).  Install them with the provided
> `requirements.txt` before launching the UI.

## Setup

1. Create and activate a virtual environment in the repository root:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install the runtime dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Place Qwen GGUF weights in one of the recognised locations:
   * Set `QWEN_GGUF_PATH` and, for vision models, `QWEN_VL_MMPROJ_PATH`.
   * Or copy the `.gguf` files into `models/` next to `appV5.py`.
   * Legacy installs under `D:\CAD_Quoting_Tool\models` continue to be
     auto-discovered.

## Running the application

Launch the Tkinter UI after activating the virtual environment:

```bash
python appV5.py
```

Useful command-line flags:

* `--print-env` – dump a JSON report of the active configuration and exit.
* `--no-gui` – initialise dependencies without opening the Tkinter window.

## Tests and linting

Automated checks live under `tests/`.  Run the suite with:

```bash
pytest
```

The repository also includes a `docs/` directory with deployment notes and
integration guides for downstream teams.

## Troubleshooting

* Missing packages: ensure `pip install -r requirements.txt` has been executed.
* Missing LLM weights: set the `QWEN_*` environment variables or drop the
  required `.gguf` files into a `models/` directory.
* DXF enrichment: install `ezdxf` and (optionally) ODA File Converter to enable
  the DXF parsing shortcuts used by the geometry helpers.

## Pass-through cost categories

The quote breakdown includes a "Pass Through" table that exposes supplemental
line items which are transferred directly to the customer.  Each entry is
defined in `appV5.py` alongside its descriptive "basis" label.  For example, the
"Outsourced Vendors" row is backed by the `outsourced_costs` aggregate which
sums any detected heat treat, plating/coating, or passivation values parsed from
the estimating worksheet.  The label shown in the UI ("Basis: Outside processing
vendors") comes from the `pass_meta` dictionary declared near line 7,600 of the
same file.

When a pass-through field is populated in the workbook, the raw monetary value
flows into the corresponding row unchanged.  That is why a small entry such as
`$0.10` on "Outsourced Vendors" reflects the exact total imported from the
source spreadsheet or manual override rather than a derived or marked-up number.

## Geometry metrics expected from CAD/worksheet imports

The quoting pipeline normalises raw CAD features and worksheet entries into a
`geom` dictionary so downstream pricing heuristics can read consistent keys
regardless of the import source.  Populate the following metrics whenever they
are available; missing values fall back to conservative defaults so estimates
still complete, but richer geometry produces noticeably better machining time
predictions.

### Die plates

```python
geom = {
    "wedm": {"perimeter_in": 240.0, "starts": 6, "tabs": 6, "passes": 2, "wire_in": 0.010},
    "sg":   {"area_sq_in": 18.0 * 8.0 * 2, "stock_in": 0.001},           # both faces
    "blanchard": {"area_sq_in": 18.0 * 8.0, "stock_in": 0.004},
    "milling": {"volume_cuin": 12.0},                                   # pocket volume
    "drill": [{"dia_in": 0.375, "depth_in": 1.25}] * 8 +
              [{"dia_in": 0.500, "depth_in": 1.0}] * 4,
    "tapped_count": 8,
    "bores": [
        {"method": "jig_grind", "tol": 0.0003},
        {"method": "jig_bore", "tol": 0.0005},
    ],
    "length_ft_edges": 12.0,
    "lap_area_sq_in": 0.0,
}
```

### Punches

```python
geom = {
    "wedm": {"perimeter_in": 6.5, "starts": 1, "tabs": 2, "passes": 3, "wire_in": 0.008},
    "sg":   {"area_sq_in": 2.0, "stock_in": 0.0008},
    "drill": [],
    "bores": [],
    "sinker": [],  # or [{"vol_cuin": 0.02, "finish": True}]
    "length_ft_edges": 0.5,
    "lap_area_sq_in": 0.15,
}
```

Extending the schema for other part families follows the same pattern: aggregate
volumes, perimeters, pass counts, and any special operations into nested
dictionaries so Codex and pricing estimators can consume a single geometry
payload.

## McMaster-Carr API client (official)

The repository includes `mcmaster_api.py`, a simple CLI that authenticates to
McMaster-Carr’s official API using mutual TLS (client PFX certificate) and
prints price tiers for a given part number.

The quoting helpers under `mcmaster_stock.py` consume the same API to resolve
sheet SKUs and price-per-unit estimates programmatically, so no legacy web
scrapers are required.

Setup:

- Install extra dependencies:
  - `pip install requests-pkcs12 truststore python-dotenv`
- Obtain your McMaster-Carr API credentials and a `.pfx` client certificate.
- Provide credentials via environment variables or a `.env` file in the repo
  root:

```
MCMASTER_USER=your_email@example.com
MCMASTER_PASS=your_password
MCMASTER_PFX_PATH=D:\\path\\to\\client.pfx
MCMASTER_PFX_PASS=optional_pfx_password
```

Run:

```
python mcmaster_api.py
```

You will be prompted for any missing values and for a part number (e.g.,
`4936K451`). The tool logs in, subscribes the product if required, and prints
the returned pricing tiers.
