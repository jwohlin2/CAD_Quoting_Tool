from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none


@dataclass(frozen=True)
class EditorControlSpec:
    """Instruction for rendering a Quote Editor control."""

    control: str
    entry_value: str = ""
    options: tuple[str, ...] = ()
    base_text: str = ""
    checkbox_state: bool = False
    checkbox_label: str = "Enabled"
    display_label: str = ""
    guessed_dropdown: bool = False


_BOOL_PAIR_RE = re.compile(
    r"^\s*(true|false|yes|no|on|off)\s*(?:/|\||,|\s+or\s+)\s*(true|false|yes|no|on|off)\s*$",
    re.IGNORECASE,
)
_TRUTHY_TOKENS = {"true", "1", "yes", "y", "on"}
_FALSY_TOKENS = {"false", "0", "no", "n", "off"}


def _coerce_checkbox_state(value: Any, default: bool = False) -> bool:
    """Best-effort conversion from spreadsheet text to a boolean."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            if math.isnan(value):
                return default
        except Exception:
            pass
        return bool(value)

    text = str(value).strip().lower()
    if not text:
        return default
    if text in _TRUTHY_TOKENS or text.startswith("y"):
        return True
    if text in _FALSY_TOKENS or text.startswith("n"):
        return False

    for part in re.split(r"[/|,\s]+", text):
        if part in _TRUTHY_TOKENS or part.startswith("y"):
            return True
        if part in _FALSY_TOKENS or part.startswith("n"):
            return False

    return default


def _split_editor_options(text: str) -> list[str]:
    if not text:
        return []
    if _BOOL_PAIR_RE.match(text):
        parts = re.split(r"[/|,]|\s+or\s+", text, flags=re.IGNORECASE)
    else:
        parts = re.split(r"[,\n;|]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _looks_like_bool_options(options: Sequence[str]) -> bool:
    if not options or len(options) > 4:
        return False
    normalized = {opt.lower() for opt in options}
    return bool(normalized & _TRUTHY_TOKENS) and bool(normalized & _FALSY_TOKENS)


def _is_numeric_token(token: str) -> bool:
    token = str(token).strip()
    if not token:
        return False
    try:
        float(token.replace(",", ""))
        return True
    except Exception:
        return False


def _format_numeric_entry_value(raw: str) -> tuple[str, bool]:
    parsed = _coerce_float_or_none(raw)
    if parsed is None:
        return str(raw).strip(), False
    txt = f"{float(parsed):.6f}".rstrip("0").rstrip(".")
    return (txt if txt else "0", True)


def derive_editor_control_spec(dtype_source: str, example_value: Any) -> EditorControlSpec:
    """Classify a spreadsheet row into a UI control plan."""

    dtype_raw = re.sub(
        r"\s+",
        " ",
        str((dtype_source or "")).replace("\u00A0", " "),
    ).strip().lower()
    raw_value = ""
    if example_value is not None and not (
        isinstance(example_value, float) and math.isnan(example_value)
    ):
        raw_value = str(example_value)
    initial_value = raw_value.strip()

    options = _split_editor_options(initial_value)
    looks_like_bool = _looks_like_bool_options(options)
    is_checkbox = "checkbox" in dtype_raw or looks_like_bool
    declared_dropdown = "dropdown" in dtype_raw or "select" in dtype_raw
    is_formula_like = any(term in dtype_raw for term in ("lookup", "calculated")) or (
        "value" in dtype_raw and ("lookup" in dtype_raw or "calculated" in dtype_raw)
    )
    if not is_formula_like and "value" in dtype_raw and not declared_dropdown:
        is_formula_like = True
    is_numeric_dtype = any(
        term in dtype_raw for term in ("number", "numeric", "decimal", "integer", "float")
    )

    has_formula_chars = bool(re.search(r"[=*()+{}]", initial_value))
    non_numeric_options = [opt for opt in options if not _is_numeric_token(opt)]
    guessed_dropdown = False

    if is_checkbox:
        normalized = initial_value.lower()
        if normalized in _TRUTHY_TOKENS:
            state = True
        elif normalized in _FALSY_TOKENS:
            state = False
        elif options:
            first = options[0].lower()
            state = first in _TRUTHY_TOKENS or first.startswith("y")
        else:
            state = _coerce_checkbox_state(initial_value, False)

        truthy_label = next(
            (
                opt.strip()
                for opt in options
                if opt and (opt.lower() in _TRUTHY_TOKENS or opt.lower().startswith("y"))
            ),
            "",
        )
        checkbox_label = truthy_label.title() if truthy_label else "Enabled"
        if checkbox_label.lower() in {"true", "1"}:
            checkbox_label = "Enabled"

        display = dtype_source.strip() if isinstance(dtype_source, str) and dtype_source.strip() else "Checkbox"
        return EditorControlSpec(
            control="checkbox",
            entry_value="True" if state else "False",
            options=tuple(options),
            checkbox_state=state,
            checkbox_label=checkbox_label,
            display_label=display,
        )

    if (
        declared_dropdown
        or (
            not is_formula_like
            and not is_numeric_dtype
            and non_numeric_options
            and len(options) >= 2
            and not has_formula_chars
        )
    ):
        guessed_dropdown = not declared_dropdown
        display = dtype_source.strip() if isinstance(dtype_source, str) and dtype_source.strip() else "Dropdown"
        selected = initial_value or (options[0] if options else "")
        if options and selected not in options:
            selected = options[0]
        return EditorControlSpec(
            control="dropdown",
            entry_value=selected,
            options=tuple(options),
            display_label=(display + " (auto)" if guessed_dropdown else display),
            guessed_dropdown=guessed_dropdown,
        )

    if is_formula_like or has_formula_chars:
        display = dtype_source.strip() if isinstance(dtype_source, str) and dtype_source.strip() else "Lookup / Calculated"
        entry_value, parsed_ok = _format_numeric_entry_value(initial_value)
        if not parsed_ok:
            entry_value = ""
        base_text = initial_value if initial_value else ""
        return EditorControlSpec(
            control="formula",
            entry_value=entry_value,
            base_text=base_text,
            display_label=display,
        )

    if is_numeric_dtype or (_is_numeric_token(initial_value) and not non_numeric_options):
        display = dtype_source.strip() if isinstance(dtype_source, str) and dtype_source.strip() else "Number"
        entry_value = ""
        if initial_value:
            entry_value, parsed_ok = _format_numeric_entry_value(initial_value)
            if not parsed_ok:
                entry_value = initial_value
        return EditorControlSpec(
            control="number",
            entry_value=entry_value,
            display_label=display,
        )

    display = dtype_source.strip() if isinstance(dtype_source, str) and dtype_source.strip() else "Text"
    return EditorControlSpec(
        control="text",
        entry_value=initial_value,
        display_label=display,
        options=tuple(options),
    )


__all__ = ["EditorControlSpec", "derive_editor_control_spec"]
