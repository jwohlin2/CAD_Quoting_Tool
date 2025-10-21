"""Reusable CSV parsing helpers with legacy delimiter support."""
from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from typing import Any

DEFAULT_ENCODING = "utf-8-sig"
DEFAULT_DELIMITER = ","
LEGACY_DELIMITER_CANDIDATES: tuple[str, ...] = ("	", DEFAULT_DELIMITER)


def sniff_delimiter(
    sample: str | None,
    *,
    candidates: Sequence[str] = LEGACY_DELIMITER_CANDIDATES,
    default: str = DEFAULT_DELIMITER,
) -> str:
    """Return the most likely delimiter for ``sample``."""

    text = sample or ""
    for candidate in candidates:
        if candidate and candidate in text:
            return candidate
    return default if default else DEFAULT_DELIMITER


def sanitize_csv_cell(value: Any) -> str:
    """Coerce ``value`` to a CSV-safe string representation."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def sanitize_csv_row(
    row: Sequence[Any],
    *,
    header_len: int,
    delimiter: str = DEFAULT_DELIMITER,
) -> list[str]:
    """Return ``row`` padded or collapsed to match ``header_len``.

    Extra trailing cells are merged into the final column so that spill-over
    notes are preserved instead of raising parsing errors. Short rows are padded
    with empty strings to maintain consistent column counts.
    """

    sanitized = [sanitize_csv_cell(cell) for cell in row]
    if header_len <= 0:
        return list(sanitized)

    if len(sanitized) > header_len:
        keep = header_len - 1
        head = list(sanitized[:keep]) if keep > 0 else []
        merged_tail = delimiter.join(sanitized[keep:])
        sanitized = head + [merged_tail]
    elif len(sanitized) < header_len:
        sanitized = list(sanitized) + ["" for _ in range(header_len - len(sanitized))]
    else:
        sanitized = list(sanitized)
    return sanitized


def _normalize_header(header: Sequence[Any]) -> list[str]:
    return [sanitize_csv_cell(cell) for cell in header]


def rows_to_dicts(
    header: Sequence[Any],
    rows: Iterable[Sequence[Any]],
    *,
    delimiter: str = DEFAULT_DELIMITER,
) -> list[dict[str, str]]:
    """Convert ``rows`` into dictionaries keyed by the sanitized header."""

    columns = _normalize_header(header)
    if not columns:
        raise ValueError("CSV header is empty")

    normalized: list[dict[str, str]] = []
    header_len = len(columns)
    for row in rows:
        normalized.append(
            dict(zip(columns, sanitize_csv_row(row, header_len=header_len, delimiter=delimiter)))
        )
    return normalized


def read_csv_rows(
    path: str,
    *,
    encoding: str = DEFAULT_ENCODING,
    delimiter: str | None = None,
    candidates: Sequence[str] = LEGACY_DELIMITER_CANDIDATES,
) -> tuple[list[str], list[list[str]], str]:
    """Read ``path`` and return ``(header, rows, delimiter)``."""

    delim = delimiter
    if delim is None:
        try:
            with open(path, encoding=encoding) as sniff:
                sample = sniff.readline()
        except FileNotFoundError:
            raise
        except Exception:
            sample = ""
        delim = sniff_delimiter(sample, candidates=candidates, default=DEFAULT_DELIMITER)

    with open(path, encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=delim)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file has no rows")

    header = rows[0]
    if not header:
        raise ValueError("CSV header is empty")

    return list(header), [list(row) for row in rows[1:]], delim


def read_csv_as_dicts(
    path: str,
    *,
    encoding: str = DEFAULT_ENCODING,
    delimiter: str | None = None,
    candidates: Sequence[str] = LEGACY_DELIMITER_CANDIDATES,
) -> list[dict[str, str]]:
    """Return ``path`` as a list of dictionaries with sanitized cells."""

    header, rows, delim = read_csv_rows(
        path,
        encoding=encoding,
        delimiter=delimiter,
        candidates=candidates,
    )
    return rows_to_dicts(header, rows, delimiter=delim)


__all__ = [
    "DEFAULT_DELIMITER",
    "DEFAULT_ENCODING",
    "LEGACY_DELIMITER_CANDIDATES",
    "read_csv_as_dicts",
    "read_csv_rows",
    "rows_to_dicts",
    "sanitize_csv_cell",
    "sanitize_csv_row",
    "sniff_delimiter",
]
