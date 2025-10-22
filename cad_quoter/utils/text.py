"""Text processing helpers shared across modules."""

from __future__ import annotations

import re


def to_noncapturing(expr: str) -> str:
    """Convert every capturing ``(`` to a non-capturing ``(?:``."""

    out: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        prev = expr[i - 1] if i > 0 else ""
        nxt = expr[i + 1] if i + 1 < len(expr) else ""
        if ch == "(" and prev != "\\" and nxt != "?":
            out.append("(?:")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


_to_noncapturing = to_noncapturing


def _match_items_contains(items: "pd.Series", pattern: str) -> "pd.Series":
    """Case-insensitive regex match over Items, with safe fallback."""

    pat = to_noncapturing(pattern)
    try:
        return items.str.contains(pat, case=False, regex=True, na=False)
    except Exception:
        return items.str.contains(re.escape(pattern), case=False, regex=True, na=False)


__all__ = ["to_noncapturing", "_to_noncapturing", "_match_items_contains"]
