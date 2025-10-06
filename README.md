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

## McMaster-Carr sheet scraper

For quick material cost lookups the repository ships with
`scrape_mcmaster.py`, a Playwright-based scraper that selects the smallest
stock sheet which meets or exceeds the requested dimensions.

Install the optional dependencies and Playwright browser binaries:

```bash
pip install playwright rapidfuzz
python -m playwright install
```

Example usage:

```bash
# 5083 cast tooling plate @ 0.75" thick, target 12" × 18"
python scrape_mcmaster.py tool_jig --material 5083 --thickness "3/4" --width 12 --length 18

# MIC6 cast plate @ 0.5" thick, target 10" × 10"
python scrape_mcmaster.py tool_jig --material mic6 --thickness 0.5 --width 10 --length 10

# A2 tool steel sheet (tight tolerance) @ 0.375" thick, target 6" × 18"
python scrape_mcmaster.py a2 --tolerance tight --thickness 3/8 --width 6 --length 18
```

The script waits for McMaster-Carr's pricing grid to load, scrapes all visible
sheet sizes for the selected thickness, and chooses the smallest option with
width ≥ `--width` and length ≥ `--length`.  If your region requires cookies or a
login to view prices you may need to run the script manually in a desktop
browser session first so Playwright inherits the necessary site state.

## Troubleshooting

* Missing packages: ensure `pip install -r requirements.txt` has been executed.
* Missing LLM weights: set the `QWEN_*` environment variables or drop the
  required `.gguf` files into a `models/` directory.
* DXF enrichment: install `ezdxf` and (optionally) ODA File Converter to enable
  the DXF parsing shortcuts used by the geometry helpers.
