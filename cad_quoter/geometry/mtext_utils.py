"""Helpers for working with MTEXT content."""

from __future__ import annotations

import re

_FORMAT_CODE_PATTERN = re.compile(r"\\[AaCcFfHh][^;]*;")


def normalize_mtext_plain_text(raw_text: str) -> str:
    """Return a simplified representation of ``raw_text`` from an MTEXT entity.

    The DXF specification allows inline formatting codes for MTEXT strings.  When
    ezdxf cannot provide :meth:`plain_text`, we fall back to sanitising the raw
    string by stripping these codes and converting control markers into their
    textual equivalents.
    """

    text = str(raw_text or "")
    if not text:
        return ""
    text = text.replace("\\P", "\n").replace("\\~", "~")
    return _FORMAT_CODE_PATTERN.sub("", text)


__all__ = ["normalize_mtext_plain_text"]

