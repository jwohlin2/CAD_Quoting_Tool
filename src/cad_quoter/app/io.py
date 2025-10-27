from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import sys


@dataclass(frozen=True)
class InputSpec:
    """Resolved inputs derived from parsed CLI arguments."""

    input_path: Path | None
    batch_dir: Path | None
    qty: int
    out_path: Path | None
    stdout: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the quoting CLI."""

    p = argparse.ArgumentParser(prog="cad-quoter")
    p.add_argument("input", nargs="?", help="DWG/DXF/PDF/CSV or folder (batch).")
    p.add_argument("--qty", type=int, default=1)
    p.add_argument("--out", type=str, help="Output path for rendered quote (txt/md).")
    p.add_argument("--stdout", action="store_true", help="Print quote to stdout.")
    # Placeholder for additional legacy flags from appV5
    return p.parse_args(argv if argv is not None else sys.argv[1:])


def resolve_input(ns: argparse.Namespace) -> InputSpec:
    """Resolve CLI namespace into a structured :class:`InputSpec`."""

    p = Path(ns.input).resolve() if getattr(ns, "input", None) else None
    if p and p.exists() and p.is_dir():
        return InputSpec(
            input_path=None,
            batch_dir=p,
            qty=getattr(ns, "qty", 1) or 1,
            out_path=_maybe_path(getattr(ns, "out", None)),
            stdout=bool(getattr(ns, "stdout", False)),
        )
    return InputSpec(
        input_path=p,
        batch_dir=None,
        qty=getattr(ns, "qty", 1) or 1,
        out_path=_maybe_path(getattr(ns, "out", None)),
        stdout=bool(getattr(ns, "stdout", False)),
    )


def _maybe_path(s: str | None) -> Path | None:
    return Path(s).resolve() if s else None


def emit_output(ns: argparse.Namespace, text_lines: list[str]) -> None:
    """Write rendered quote ``text_lines`` to the requested destination."""

    out = _maybe_path(getattr(ns, "out", None))
    content = "\n".join(text_lines)
    if getattr(ns, "stdout", False) or not out:
        print(content)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
