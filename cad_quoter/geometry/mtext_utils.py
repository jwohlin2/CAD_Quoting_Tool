"""Helpers for working with MTEXT content."""

from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict

_FORMAT_CODE_PATTERN = re.compile(r"\\[AaCcFfHh][^;]*;")
_MULTISPACE_PATTERN = re.compile(r"[ \t]+")
_DIA_REPLACEMENTS: dict[str, str] = {
    "Ø": "DIA ",
    "ø": "DIA ",
    "∅": "DIA ",
}

_NORMALIZATION_EXAMPLES: OrderedDict[str, str] = OrderedDict()
_NORMALIZATION_LOGGED = False


def _strip_mtext_codes(text: str) -> str:
    cleaned = text.replace("\\P", "\n").replace("\\~", "~")
    return _FORMAT_CODE_PATTERN.sub("", cleaned)


def normalize_mtext_plain_text(raw_text: str) -> str:
    """Return a simplified representation of ``raw_text`` from an MTEXT entity.

    The DXF specification allows inline formatting codes for MTEXT strings. When
    ezdxf cannot provide :meth:`plain_text`, we fall back to sanitising the raw
    string by stripping these codes and converting control markers into their
    textual equivalents.
    """

    text = str(raw_text or "")
    if not text:
        return ""
    return _strip_mtext_codes(text)


def _collapse_spaces(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        collapsed = _MULTISPACE_PATTERN.sub(" ", line.strip())
        lines.append(collapsed)
    normalized = "\n".join(lines)
    return normalized.strip()


def _replace_diameter_symbols(text: str) -> str:
    if not text:
        return ""
    for source, replacement in _DIA_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    return text.replace("⌀", "DIA ")


def _record_normalization_example(original: str, normalized: str) -> None:
    if not original or not normalized:
        return
    key = original.strip()
    value = normalized.strip()
    if not key or not value:
        return
    if key.lower() == value.lower():
        return
    if key in _NORMALIZATION_EXAMPLES:
        return
    if len(_NORMALIZATION_EXAMPLES) >= 3:
        return
    _NORMALIZATION_EXAMPLES[key] = value


def compute_plain_text_norm(
    *, plain_text: str | None = None, raw_text: str | None = None
) -> tuple[str, str]:
    """Return (lower, upper) normalized variants for DXF text content."""

    base: str | None = None
    if plain_text not in (None, ""):
        base = str(plain_text)
    elif raw_text not in (None, ""):
        base = _strip_mtext_codes(str(raw_text))
    if not base:
        return ("", "")
    candidate = unicodedata.normalize("NFKC", str(base))
    candidate = candidate.replace("\r\n", "\n").replace("\r", "\n")
    candidate = _replace_diameter_symbols(candidate)
    candidate = _collapse_spaces(candidate)
    if not candidate:
        return ("", "")
    lower = candidate.lower()
    upper = candidate.upper()
    _record_normalization_example(base, lower)
    return (lower, upper)


def flush_plain_text_normalization_log() -> None:
    """Emit a summary of notable text normalizations once per session."""

    global _NORMALIZATION_LOGGED
    if _NORMALIZATION_LOGGED:
        return
    if not _NORMALIZATION_EXAMPLES:
        return
    sample = next(iter(_NORMALIZATION_EXAMPLES.items()))
    sample_display = f'"{sample[0]}" → "{sample[1]}"'
    print(
        "[NORMALIZE] applied unicode + mtext cleanup: examples={count} (e.g., {example})".format(
            count=len(_NORMALIZATION_EXAMPLES),
            example=sample_display,
        )
    )
    _NORMALIZATION_LOGGED = True


__all__ = [
    "normalize_mtext_plain_text",
    "compute_plain_text_norm",
    "flush_plain_text_normalization_log",
]

