"""Helpers for summarizing hole chart rows by operation bucket."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Any, Dict, Tuple

from cad_quoter import geo_extractor

_BUCKET_KEYS = (
    "tap",
    "counterbore",
    "counterdrill",
    "csink",
    "drill",
    "jig_grind",
    "spot",
    "npt",
    "unknown",
)


def _coerce_positive_int(value: Any) -> int | None:
    """Return ``value`` coerced to a positive ``int`` when possible."""

    try:
        candidate = int(round(float(value)))
    except Exception:
        return None
    return candidate if candidate > 0 else None


def classify_chart_rows(
    rows: Iterable[Mapping[str, Any] | Any] | None,
) -> Tuple[Dict[str, int], int, int]:
    """Classify *rows* into operation buckets.

    Parameters
    ----------
    rows:
        Iterable of row-like mappings. Each mapping should provide a ``qty`` field
        describing the hole count and a ``desc``/``description``/``text`` field
        describing the operation.

    Returns
    -------
    tuple(dict, int, int)
        ``(buckets, row_count, qty_sum)`` where ``buckets`` maps operation bucket
        names (``tap``, ``drill``, etc.) to the summed quantities from ``rows``.
        ``row_count`` and ``qty_sum`` represent the number of rows contributing to
        the totals and the cumulative quantity respectively.
    """

    if not rows:
        return ({}, 0, 0)

    totals: defaultdict[str, int] = defaultdict(int)
    row_count = 0
    qty_sum = 0

    for row in rows:
        mapping: Mapping[str, Any] | None = None
        if isinstance(row, Mapping):
            mapping = row
        else:
            attrs = {name: getattr(row, name) for name in ("qty", "desc", "description", "text") if hasattr(row, name)}
            if attrs:
                mapping = attrs  # type: ignore[assignment]
        if not mapping:
            continue

        qty = _coerce_positive_int(mapping.get("qty"))
        if qty is None:
            continue

        desc = (
            mapping.get("desc")
            or mapping.get("description")
            or mapping.get("text")
        )

        row_count += 1
        qty_sum += qty

        operations = geo_extractor.classify_op_row(desc)
        if not operations:
            totals["unknown"] += qty
            continue

        for op in operations:
            kind = str(op.get("kind") or "unknown").strip().lower()
            if kind not in _BUCKET_KEYS:
                kind = "unknown"
            totals[kind] += qty

    buckets = {key: value for key, value in totals.items() if value > 0}
    return (buckets, row_count, qty_sum)
