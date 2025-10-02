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
