"""Quote document rendering helpers shared across UI entrypoints."""

from __future__ import annotations

import re
import textwrap
import unicodedata
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from appkit.ui.services import QuoteConfiguration

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RENDER_ASCII_REPLACEMENTS: dict[str, str] = {
    "—": "-",
    "•": "-",
    "…": "...",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "µ": "u",
    "μ": "u",
    "±": "+/-",
    "°": " deg ",
    "¼": "1/4",
    "½": "1/2",
    "¾": "3/4",
    " ": " ",  # non-breaking space
    "⚠️": "⚠",
}

_RENDER_PASSTHROUGH: dict[str, str] = {
    "–": "__EN_DASH__",
    "×": "__MULTIPLY__",
    "≥": "__GEQ__",
    "≤": "__LEQ__",
    "⚠": "__WARN__",
}


def _sanitize_render_text(value: Any) -> str:
    """Return a sanitized ASCII-only string for quote document output."""

    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    for source, placeholder in _RENDER_PASSTHROUGH.items():
        if source in text:
            text = text.replace(source, placeholder)
    text = text.replace("\t", " ")
    text = text.replace("\r", "")
    text = _ANSI_ESCAPE_RE.sub("", text)
    for source, replacement in _RENDER_ASCII_REPLACEMENTS.items():
        if source in text:
            text = text.replace(source, replacement)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii", "ignore")
    text = _CONTROL_CHAR_RE.sub("", text)
    for source, placeholder in _RENDER_PASSTHROUGH.items():
        if placeholder in text:
            text = text.replace(placeholder, source)
    return text


def _wrap_header_text(text: Any, page_width: int, indent: str = "") -> list[str]:
    """Helper mirroring :func:`write_wrapped` for header content."""

    if text is None:
        return []
    txt = str(text).strip()
    if not txt:
        return []
    wrapper = textwrap.TextWrapper(width=max(10, page_width - len(indent)))
    return [f"{indent}{chunk}" for chunk in wrapper.wrap(txt)]


def _resolve_pricing_source_value(
    base_value: Any,
    *,
    used_planner: bool | None = None,
    process_meta: Mapping[str, Any] | None = None,
    process_meta_raw: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
    planner_process_minutes: Any = None,
    hour_summary_entries: Mapping[str, Any] | None = None,
    additional_sources: Sequence[Any] | None = None,
    cfg: QuoteConfiguration | None = None,
) -> str | None:
    """Return a normalized pricing source, honoring explicit selections."""

    fallback_text: str | None = None
    if base_value is not None:
        candidate_text = str(base_value).strip()
        if candidate_text:
            lowered = candidate_text.lower()
            if lowered == "planner":
                return "planner"
            if lowered not in {"legacy", "auto", "default", "fallback"}:
                return candidate_text
            fallback_text = candidate_text

    if used_planner:
        if fallback_text:
            return fallback_text
        return "planner"

    # Delegate planner signal detection to the adapter helper
    from appkit.planner_adapter import (
        _planner_signals_present as _planner_signals_present_helper,
    )

    if _planner_signals_present_helper(
        process_meta=process_meta,
        process_meta_raw=process_meta_raw,
        breakdown=breakdown,
        planner_process_minutes=planner_process_minutes,
        hour_summary_entries=hour_summary_entries,
        additional_sources=list(additional_sources) if additional_sources is not None else None,
    ):
        if fallback_text:
            return fallback_text
        return "planner"

    if fallback_text:
        return fallback_text

    return None


def _build_quote_header_lines(
    *,
    qty: int,
    result: Mapping[str, Any] | None,
    breakdown: Mapping[str, Any] | None,
    page_width: int,
    divider: str,
    process_meta: Mapping[str, Any] | None,
    process_meta_raw: Mapping[str, Any] | None,
    hour_summary_entries: Mapping[str, Any] | None,
    cfg: QuoteConfiguration | None = None,
) -> tuple[list[str], str | None]:
    """Construct the canonical QUOTE SUMMARY header lines."""

    header_lines: list[str] = [f"QUOTE SUMMARY - Qty {qty}", divider]
    header_lines.append("Quote Summary (structured data attached below)")

    speeds_feeds_value = None
    if isinstance(result, Mapping):
        speeds_feeds_value = result.get("speeds_feeds_path")
    if speeds_feeds_value in (None, "") and isinstance(breakdown, Mapping):
        speeds_feeds_value = breakdown.get("speeds_feeds_path")
    path_text = str(speeds_feeds_value).strip() if speeds_feeds_value else ""

    speeds_feeds_loaded_display: bool | None = None
    for source in (result, breakdown):
        if not isinstance(source, Mapping):
            continue
        if "speeds_feeds_loaded" in source:
            raw_flag = source.get("speeds_feeds_loaded")
            speeds_feeds_loaded_display = None if raw_flag is None else bool(raw_flag)
            break

    if speeds_feeds_loaded_display is True:
        status_suffix = " (loaded)"
    elif speeds_feeds_loaded_display is False:
        status_suffix = " (not loaded)"
    else:
        status_suffix = ""

    if path_text:
        header_lines.extend(
            _wrap_header_text(
                f"Speeds/Feeds CSV: {path_text}{status_suffix}",
                page_width,
            )
        )
    elif status_suffix:
        header_lines.extend(
            _wrap_header_text(
                f"Speeds/Feeds CSV: (not set){status_suffix}",
                page_width,
            )
        )
    else:
        header_lines.append("Speeds/Feeds CSV: (not set)")

    def _coerce_pricing_source(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        # normalize synonyms
        if lowered in {"legacy", "est", "estimate", "estimator"}:
            return "estimator"
        if lowered in {"plan", "planner"}:
            return "planner"
        return text

    raw_pricing_source = None
    pricing_source_display = None
    if isinstance(breakdown, Mapping):
        raw_pricing_source = _coerce_pricing_source(breakdown.get("pricing_source"))
        if raw_pricing_source:
            pricing_source_display = str(raw_pricing_source).title()

    used_planner_flag: bool | None = None
    for source in (result, breakdown):
        if not isinstance(source, Mapping):
            continue
        for meta_key in ("app_meta", "app"):
            candidate = source.get(meta_key)
            if not isinstance(candidate, Mapping):
                continue
            if "used_planner" in candidate:
                try:
                    used_planner_flag = bool(candidate.get("used_planner"))
                except Exception:
                    used_planner_flag = True if candidate.get("used_planner") else False
                break
        if used_planner_flag is not None:
            break

    pricing_source_value = _resolve_pricing_source_value(
        raw_pricing_source,
        used_planner=used_planner_flag,
        process_meta=process_meta if isinstance(process_meta, Mapping) else None,
        process_meta_raw=process_meta_raw if isinstance(process_meta_raw, Mapping) else None,
        breakdown=breakdown if isinstance(breakdown, Mapping) else None,
        hour_summary_entries=hour_summary_entries,
        cfg=cfg,
    )

    # === HEADER: PRICING SOURCE OVERRIDE ===
    if getattr(cfg, "prefer_removal_drilling_hours", False):
        normalized_value = (
            str(pricing_source_value).strip().lower()
            if pricing_source_value is not None
            else ""
        )
        if not normalized_value or normalized_value == "legacy":
            pricing_source_value = "Estimator"
            pricing_source_display = "Estimator"

    normalized_pricing_source: str | None = None
    if pricing_source_value is not None:
        normalized_pricing_source = str(pricing_source_value).strip()
        if not normalized_pricing_source:
            normalized_pricing_source = None

    if normalized_pricing_source:
        normalized_pricing_source_lower = normalized_pricing_source.lower()
        raw_pricing_source_lower = (
            str(raw_pricing_source).strip().lower() if raw_pricing_source is not None else None
        )

        if (
            isinstance(breakdown, MutableMapping)
            and raw_pricing_source_lower != normalized_pricing_source_lower
        ):
            breakdown["pricing_source"] = pricing_source_value

        pricing_source_display = normalized_pricing_source.title()

    if pricing_source_display:
        display_value = pricing_source_display
        header_lines.append(f"Pricing Source: {display_value}")

    return header_lines, pricing_source_value


__all__ = [
    "_sanitize_render_text",
    "_wrap_header_text",
    "_resolve_pricing_source_value",
    "_build_quote_header_lines",
]

