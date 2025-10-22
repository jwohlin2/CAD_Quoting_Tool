"""Audit logging helpers shared by the CAD quoting tool."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from cad_quoter.utils.render_utils import fmt_money, fmt_percent


LOGS_DIR = Path(r"D:\CAD_Quoting_Tool\Logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    """Return the current timestamp formatted for audit records."""

    return time.strftime("%Y-%m-%dT%H:%M:%S")


def coerce_num(value: Any) -> Any:
    """Convert *value* to ``float`` when possible for easier diffing."""

    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return value


def used_item_values(df: Any, used_items: Iterable[str]) -> dict[str, Any]:
    """Return the example values for the estimator *used_items* from *df*."""

    values: dict[str, Any] = {}
    try:
        items = df["Item"].astype(str)
    except Exception:
        return values
    for name in used_items:
        try:
            mask = items.str.fullmatch(name, case=False, na=False)
        except Exception:
            continue
        if getattr(mask, "any", lambda: False)():
            try:
                cell = df.loc[mask, "Example Values / Options"].iloc[0]
            except Exception:
                continue
            values[name] = coerce_num(cell)
    return values


def diff_map(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a structured diff between two mapping-like objects."""

    changes: list[dict[str, Any]] = []
    keys = set(before.keys()) | set(after.keys())
    for key in sorted(keys):
        before_value = before.get(key, None)
        after_value = after.get(key, None)
        if before_value != after_value:
            changes.append({
                "key": key,
                "before": before_value,
                "after": after_value,
            })
    return changes


def render_llm_log_text(log: Mapping[str, Any]) -> str:
    """Render a human-readable summary of an LLM decision log entry."""

    def _format_money(value: Any) -> str:
        if isinstance(value, (int, float)):
            return fmt_money(value, "$")
        return str(value)

    lines: list[str] = []
    timestamp = log.get("timestamp", now_iso())
    model_name = log.get("model", "?")
    prompt_hash = str(log.get("prompt_sha256", ""))[:12]
    lines.append(f"LLM DECISION LOG • {timestamp}")
    lines.append(f"Model: {model_name}  |  Prompt SHA256: {prompt_hash}…")
    error = log.get("llm_error")
    if error:
        lines.append(f"LLM error: {error}")
    allowed = log.get("allowed_items_count", 0)
    suggested = log.get("sheet_edits_suggested", 0)
    applied = log.get("sheet_edits_applied", 0)
    lines.append(
        "Allowed items: {allowed} | Sheet edits suggested/applied: "
        "{suggested}/{applied}".format(
            allowed=allowed,
            suggested=suggested,
            applied=applied,
        )
    )
    price_before = log.get("price_before", 0)
    price_after = log.get("price_after", 0)
    delta = (price_after or 0) - (price_before or 0)
    try:
        pct_delta = (delta / price_before) if price_before else 0.0
    except Exception:
        pct_delta = 0.0
    lines.append("")
    lines.append(
        "Price before: {before}   →   after: {after}   Δ: {delta} "
        "({pct})".format(
            before=_format_money(price_before),
            after=_format_money(price_after),
            delta=_format_money(delta),
            pct=fmt_percent(pct_delta),
        )
    )
    lines.append("")

    def _append_changes(title: str, changes: Iterable[Mapping[str, Any]]) -> None:
        changes_list = list(changes)
        if not changes_list:
            return
        lines.append(title)
        for change in changes_list:
            key = change.get("key")
            before_value = change.get("before")
            after_value = change.get("after")
            reason = change.get("why")
            why = f"  • why: {reason}" if reason else ""
            lines.append(
                f"  • {key}: {before_value} → {after_value}{why}"
            )
        lines.append("")

    _append_changes("Sheet changes:", log.get("sheet_changes", []))
    _append_changes("Param nudges:", log.get("param_changes", []))

    geo_summary = log.get("geo_summary", [])
    if geo_summary:
        lines.append("GEO summary:")
        for key, value in geo_summary:
            lines.append(f"  - {key}: {value}")
    return "\n".join(lines)


def save_llm_log_json(log: Mapping[str, Any]) -> Path:
    """Persist *log* to ``LOGS_DIR`` and return the created path."""

    filename = f"llm_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path = LOGS_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2)
    return path


__all__ = [
    "LOGS_DIR",
    "now_iso",
    "coerce_num",
    "used_item_values",
    "diff_map",
    "render_llm_log_text",
    "save_llm_log_json",
]
