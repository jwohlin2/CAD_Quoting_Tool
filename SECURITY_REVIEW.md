# CAD Quoting Tool v5 – Security & Deployment Notes

This document summarises the primary integration points and configuration
surfaces for `appV5.py`.  It is intended to help reviewers quickly understand
what the application touches and how to run it safely in a controlled
environment.

## High-level architecture

* **User interface** – A Tkinter desktop application that orchestrates CAD
  imports, quoting logic and optional LLM assistance.
* **CAD processing** – Uses `OCP` (OpenCascade) for STEP/IGES geometry and
  `trimesh` for STL handling.  Optional DXF features rely on `ezdxf`.
* **Data sources** – Spreadsheets and CSV files for pricing lookups; optional
  metals pricing via HTTP (see below).
* **LLM integration** – Loads a local GGUF model through `llama_cpp` and
  applies lightweight post-processing before touching quoting numbers.

## Runtime configuration

The entry point now exposes `--print-env` to display the environment settings
used by the application.  Sensitive values such as API keys are redacted.  Key
variables include:

| Variable | Purpose |
| --- | --- |
| `LLM_DEBUG` / `LLM_DEBUG_DIR` | Enables structured dumps of LLM payloads for troubleshooting. |
| `QWEN_GGUF_PATH` | Path to the local Qwen model file. |
| `ODA_CONVERTER_EXE` / `DWG2DXF_EXE` | Optional converters for DWG support. |
| `DXF_EXTRUDE_THK_MM` | Overrides the default DXF extrusion thickness. |
| `METALS_API_KEY` | Enables live metal price lookups via HTTPS. |

Running `python appV5.py --print-env` is the quickest way to confirm the current
configuration prior to review.

## External communications

* **HTTP** – Only performed when the optional `MetalsAPI` provider is enabled;
  requests target `https://api.metals-api.com/v1/latest` using the provided API
  key.
* **Local executables** – If configured, the DWG-to-DXF converters specified by
  `ODA_CONVERTER_EXE` or `DWG2DXF_EXE` are invoked.
* **File system** – The tool writes diagnostic JSON snapshots into the
  directory indicated by `LLM_DEBUG_DIR` when debugging is enabled.  No other
  directories are modified without user interaction.

## Distribution recommendations

1. Ship the application inside a virtual environment with pinned dependencies
   (`requirements.txt` or equivalent) to ensure deterministic behaviour.
2. Provide a pre-populated `models/` directory or document the expected GGUF
   placement for offline LLM usage.
3. Audit optional integrations (Metals API, DWG converters) and either supply
   vetted binaries/API credentials or disable the features before packaging.
4. Encourage operators to run `appV5.py --print-env` and capture the output as
   part of security sign-off.

## Change log

* Centralised environment handling via the new `AppEnvironment` helper.
* Added a CLI for environment inspection and headless initialisation.
* Redacted sensitive configuration when generating diagnostics.
* Documented network and process touch points to streamline security review.
