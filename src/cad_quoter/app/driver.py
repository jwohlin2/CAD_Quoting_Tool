from __future__ import annotations
from pathlib import Path
from typing import Iterable, Any
import logging

from . import io as app_io
from . import runtime

log = logging.getLogger(__name__)

try:  # pragma: no cover - defensive guard for optional dependency chain
    from appV5 import compute_quote_from_df, render_quote, read_variables_file
except Exception:  # pragma: no cover - runtime guard keeps lazy import
    compute_quote_from_df = None  # type: ignore[assignment]
    render_quote = None  # type: ignore[assignment]
    read_variables_file = None  # type: ignore[assignment]


class DriverError(RuntimeError):
    """Raised when the quoting driver cannot complete a request."""


def run_single_file(cfg: runtime.RuntimeConfig, spec: app_io.InputSpec) -> list[str]:
    """Process a single-file quote request and return rendered output lines."""

    if compute_quote_from_df is None or render_quote is None:
        raise DriverError("compute/render pipeline not available")
    if not spec.input_path:
        raise DriverError("no input file provided")

    state = _load_state_from_input(spec.input_path, qty=spec.qty, cfg=cfg)
    log.debug("Loaded input %s for qty=%s", spec.input_path, spec.qty)

    quote_result = compute_quote_from_df(  # type: ignore[misc]
        state,
        params={},
        rates={},
        llm_enabled=cfg.llm_enabled,
    )
    rendered = render_quote(quote_result)
    return rendered.splitlines()


def run_batch(cfg: runtime.RuntimeConfig, spec: app_io.InputSpec) -> list[str]:
    """Process a folder of inputs and return aggregated rendered output."""

    outputs: list[str] = []
    if not spec.batch_dir:
        raise DriverError("batch directory not provided")

    for input_path in _iter_inputs(spec.batch_dir):
        try:
            lines = run_single_file(
                cfg,
                app_io.InputSpec(
                    input_path=input_path,
                    batch_dir=None,
                    qty=spec.qty,
                    out_path=None,
                    stdout=True,
                ),
            )
            outputs.append("\n".join(lines))
        except Exception as exc:  # pragma: no cover - per-file robustness
            log.exception("Failed to process %s: %s", input_path, exc)
    return ["\n\n".join(outputs)] if outputs else []


def _iter_inputs(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.glob("*")):
        if p.suffix.lower() in {".dwg", ".dxf", ".pdf", ".csv", ".xlsx"}:
            yield p


def _load_state_from_input(path: Path, *, qty: int, cfg: runtime.RuntimeConfig) -> Any:
    """Load quote variables from ``path`` for ``qty`` parts."""

    if read_variables_file is None:
        raise DriverError("variables loader unavailable")
    if not path.exists():
        raise DriverError(f"input does not exist: {path}")

    if path.suffix.lower() not in {".csv", ".xlsx"}:
        raise DriverError(f"unsupported input type: {path.suffix or path}")

    try:
        df = read_variables_file(str(path))  # type: ignore[misc]
    except Exception as exc:
        raise DriverError(f"unable to load variables from {path}: {exc}") from exc

    try:
        df = df.copy()
        if "Item" in df.columns:
            qty_mask = df["Item"].astype(str).str.lower() == "qty"
            if qty_mask.any():
                df.loc[qty_mask, "Example Values / Options"] = qty
            else:
                df.loc[len(df)] = {  # type: ignore[call-arg]
                    "Item": "Qty",
                    "Example Values / Options": qty,
                    "Data Type / Input Method": "number",
                }
        return df
    except Exception as exc:  # pragma: no cover - DataFrame mutation guard
        log.debug("Failed to normalise qty for %s: %s", path, exc)
        return df
