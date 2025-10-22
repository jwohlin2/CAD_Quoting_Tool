# Deployment Guide

This document walks through moving the CAD Quoting Tool (`appV5.py`) to another
workstation or air-gapped environment.

## 1. Collect the project assets

On the source machine:

1. Ensure the entire repository is up-to-date, including the `Cad Files/`
directory and supporting spreadsheets (for example
`dummy_quote_sheet.xlsx`).
2. If you use a local Qwen GGUF model, note its location so you can copy it to
the target host. The application automatically searches for the model in
`QWEN_GGUF_PATH`, `models/`, and the Windows-specific path documented in the
source file.【F:appV5.py†L6-L15】
3. Optional vendor data such as `materials_backup.csv` or `vendor_prices.csv`
should travel with the deployment if you rely on them for pricing fallbacks.

Create an archive (`zip`, `tar.gz`, etc.) containing the repository root, the
`models/` directory (if present), and any vendor CSVs.

## 2. Prepare the target machine

1. Install **Python 3.11+**.
2. Install system packages required by heavy wheels (`OCP`, `llama-cpp-python`,
   `ezdxf`) if they are not provided as prebuilt wheels for your platform. On
   Windows, the official Python installer plus the "Desktop development with C++"
   workload from Visual Studio Build Tools is typically sufficient to build
   `llama-cpp-python`.
3. Extract the archive from step 1 into a working directory such as
   `C:\CAD_Quoting_Tool` or `/opt/CAD_Quoting_Tool`.

## 3. Create an isolated Python environment

Inside the extracted directory run:

```bash
export PIP_EXTRA_INDEX_URL="https://<your-private-index>/simple/"
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
source .venv/bin/activate  # bash/zsh
pip install --upgrade pip
pip install -r requirements.txt
```

All runtime dependencies ship from PyPI and are captured in
`requirements.txt`. The list covers the UI helpers, pandas/openpyxl for
spreadsheet ingestion, OCC/trimesh/ezdxf for CAD handling, and
`llama-cpp-python` for the local LLM integration.【F:requirements.txt†L1-L21】

## 4. Configure runtime variables

The configuration helper exposes the following environment variables. Set them
per your deployment needs before launching the application.【F:cad_quoter/config.py†L15-L52】【F:cad_quoter/pricing/metals_api.py†L16-L32】

| Variable | Purpose | Typical value |
| --- | --- | --- |
| `LLM_DEBUG` | Enable (1) or disable (0) structured LLM debug dumps. | `0` in production |
| `LLM_DEBUG_DIR` | Directory that receives LLM JSON traces when debugging. | e.g. `C:\cad_quoter\llm_debug` |
| `QWEN_GGUF_PATH` | Explicit path to the Qwen GGUF model. Overrides auto-discovery. | `D:\models\qwen-7b.gguf` |
| `METALS_API_KEY` | Enables live Metals API pricing lookups. | API key string |
| `QWEN_N_THREADS`, `QWEN_N_GPU_LAYERS`, etc. | Tune llama-cpp runtime parameters when necessary. | Leave unset unless tuning |
| `ODA_CONVERTER_EXE`, `DWG2DXF_EXE` | Optional DWG converters used by the DXF importer. | Absolute paths to vendor tools |

Environment variables can be stored in a `.env` file and loaded with a launcher
script, or defined in the shell before starting the program.

## 5. First-run validation

Before handing the build over to end users, run the following checks inside the
virtual environment:

```bash
python appV5.py --print-env
python appV5.py --no-gui
```

The `--print-env` command prints a redacted JSON summary of the active
configuration, while `--no-gui` exercises pricing and geometry subsystems
without launching the Tkinter interface. Both options are built into the entry
point for headless smoke testing.【F:appV5.py†L13566-L13599】

## 6. Launch the application

Once validation passes, launch the GUI with:

```bash
python appV5.py
```

If required dependencies are missing, the start-up checks will raise descriptive
errors that point to the missing package, prompting you to adjust the
environment.【F:appV5.py†L62-L79】

## 7. Optional integrations

* **Metals API** – Provide the `METALS_API_KEY` environment variable to enable
  HTTPS price fetching. Without it, the registry falls back to offline CSV
  pricing.【F:cad_quoter/pricing/__init__.py†L62-L130】
* **DXF/DWG enrichment** – Install `ezdxf` and the ODA File Converter binaries if
  you need automated DWG to DXF conversion. The geometry module exposes helper
  diagnostics via `geometry.get_import_diagnostics_text()` to confirm availability.【F:cad_quoter/geometry/__init__.py†L19-L38】【F:cad_quoter/geometry/__init__.py†L462-L476】
* **LLM suggestions** – Place the Qwen GGUF model alongside the application or
  configure `QWEN_GGUF_PATH`. The llama-cpp wrapper validates the presence of the
  model file at startup.【F:cad_quoter/llm/__init__.py†L86-L129】

## 8. Packaging tips

* Ship the `.venv/` directory alongside the project for a fully offline bundle,
  or include a `requirements.txt` snapshot and this guide for reproducible setup.
* Document which optional integrations are enabled in your distribution so that
  operators know which environment variables or API keys must be provided.
* Capture the output of `python appV5.py --print-env` as part of your deployment
  verification record.
