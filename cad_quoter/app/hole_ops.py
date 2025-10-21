"""Hole table parsing and aggregation helpers."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from fractions import Fraction
from typing import Any


from .chart_lines import (
    _build_ops_rows_from_lines_fallback as _chart_build_ops_rows_from_lines_fallback,
)


RE_TAP = re.compile(
    r"(\(\d+\)\s*)?("
    r"#\s*\d{1,2}-\d+"  # #10-32
    r"|(?:\d+/\d+)\s*-\s*\d+"  # 5/8-11
    r"|(?:\d+(?:\.\d+)?)\s*-\s*\d+"  # 0.190-32
    r"|M\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?"  # M8x1.25
    r")\s*TAP",
    re.I,
)
RE_NPT = re.compile(r"(\d+/\d+)\s*-\s*N\.?P\.?T\.?", re.I)
RE_THRU = re.compile(r"\bTHRU\b", re.I)
RE_CBORE = re.compile(r"C[’']?BORE|CBORE|COUNTERBORE", re.I)
RE_CSK = re.compile(r"CSK|C'SINK|COUNTERSINK", re.I)
_RE_DEPTH_OR_THICK = re.compile(r"(\d+(?:\.\d+)?)\s*DEEP(?:\s+FROM\s+(FRONT|BACK))?", re.I)
RE_DEPTH = _RE_DEPTH_OR_THICK
RE_DIA = re.compile(r"[Ø⌀\u00D8]?\s*(\d+(?:\.\d+)?)", re.I)
RE_FRONT_BACK = re.compile(
    r"FRONT\s*&\s*BACK|FRONT\s+AND\s+BACK|BOTH\s+SIDES|TWO\s+SIDES|2\s+SIDES|OPPOSITE\s+SIDE",
    re.I,
)

WIRE_GAGE_MAJOR_DIA_IN = {
    0: 0.06,
    1: 0.073,
    2: 0.086,
    3: 0.099,
    4: 0.112,
    5: 0.125,
    6: 0.138,
    7: 0.151,
    8: 0.164,
    9: 0.177,
    10: 0.19,
    11: 0.203,
    12: 0.216,
}

TAP_MINUTES_BY_CLASS = {
    "small": 0.22,
    "medium": 0.3,
    "large": 0.38,
    "xl": 0.45,
    "pipe": 0.55,
}

CBORE_MIN_PER_SIDE_MIN = 0.15
CSK_MIN_PER_SIDE_MIN = 0.12


def _major_diameter_from_thread(spec: str) -> float | None:
    if not spec:
        return None
    spec_clean = spec.strip().upper().replace(" ", "")
    if "NPT" in spec_clean:
        parts = spec_clean.split("NPT", 1)[0]
        parts = parts.rstrip("-")
        if not parts:
            return None
        try:
            return float(Fraction(parts))
        except Exception:
            pass
        try:
            return float(parts)
        except Exception:
            return None
    if spec_clean.startswith("#"):
        try:
            gauge = int(re.sub(r"[^0-9]", "", spec_clean.split("-", 1)[0]))
        except Exception:
            return None
        return WIRE_GAGE_MAJOR_DIA_IN.get(gauge)
    if spec_clean.startswith("M"):
        try:
            mm_val = float(re.findall(r"M(\d+(?:\.\d+)?)", spec_clean)[0])
        except Exception:
            return None
        return mm_val / 25.4
    lead = spec_clean.split("-", 1)[0]
    try:
        if "/" in lead:
            return float(Fraction(lead))
        return float(lead)
    except Exception:
        return None


def _classify_thread_spec(spec: str) -> tuple[str, float, bool]:
    major = _major_diameter_from_thread(spec)
    if spec and "NPT" in spec.upper():
        return "pipe", TAP_MINUTES_BY_CLASS["pipe"], True
    if major is None:
        return "unknown", TAP_MINUTES_BY_CLASS["medium"], False
    if major <= 0.2:
        return "small", TAP_MINUTES_BY_CLASS["small"], False
    if major <= 0.3125:
        return "medium", TAP_MINUTES_BY_CLASS["medium"], False
    if major <= 0.5:
        return "large", TAP_MINUTES_BY_CLASS["large"], False
    return "xl", TAP_MINUTES_BY_CLASS["xl"], False


def _normalize_hole_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip().upper()
    cleaned = cleaned.replace("Ø", "").replace("⌀", "")
    return cleaned


def _dedupe_hole_entries(
    existing_entries: Iterable[dict[str, Any]],
    new_entries: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = {
        _normalize_hole_text(entry.get("raw"))
        for entry in existing_entries
        if isinstance(entry, dict)
    }
    unique_entries: list[dict[str, Any]] = []
    for entry in new_entries:
        if not isinstance(entry, dict):
            continue
        fingerprint = _normalize_hole_text(entry.get("raw"))
        if fingerprint and fingerprint in seen:
            continue
        if fingerprint:
            seen.add(fingerprint)
        unique_entries.append(entry)
    return unique_entries


def _parse_hole_line(line: str, to_in: float, *, source: str | None = None) -> dict[str, Any] | None:
    if not line:
        return None
    U = line.upper()
    if not any(k in U for k in ("HOLE", "TAP", "THRU", "CBORE", "C'BORE", "DRILL")):
        return None

    entry: dict[str, Any] = {
        "qty": None,
        "tap": None,
        "tap_class": None,
        "tap_minutes_per": None,
        "tap_is_npt": False,
        "thru": False,
        "cbore": False,
        "csk": False,
        "ref_dia_in": None,
        "depth_in": None,
        "side": None,
        "double_sided": False,
        "raw": line,
    }
    if source:
        entry["source"] = source

    m_qty = re.search(r"\bQTY\b[:\s]*(\d+)", U)
    if m_qty:
        entry["qty"] = int(m_qty.group(1))

    mt = RE_TAP.search(U)
    if mt:
        thread_spec = None
        if mt.lastindex and mt.lastindex >= 2:
            thread_spec = mt.group(2)
        else:
            thread_spec = mt.group(1)
        if thread_spec:
            cleaned = thread_spec.replace(" ", "")
        else:
            cleaned = None
        entry["tap"] = cleaned
        if cleaned:
            cls, minutes_per, is_npt = _classify_thread_spec(cleaned)
            entry["tap_class"] = cls
            entry["tap_minutes_per"] = minutes_per
            entry["tap_is_npt"] = is_npt
    else:
        m_npt = RE_NPT.search(U)
        if m_npt:
            cleaned = m_npt.group(0).replace(" ", "")
            entry["tap"] = cleaned
            cls, minutes_per, is_npt = _classify_thread_spec(cleaned)
            entry["tap_class"] = cls
            entry["tap_minutes_per"] = minutes_per
            entry["tap_is_npt"] = is_npt
        else:
            entry["tap"] = None
    entry["thru"] = bool(RE_THRU.search(U))
    entry["cbore"] = bool(RE_CBORE.search(U))
    entry["csk"] = bool(RE_CSK.search(U))
    if RE_FRONT_BACK.search(U):
        entry["double_sided"] = True

    md = RE_DEPTH.search(U)
    if md:
        try:
            depth = float(md.group(1)) * float(to_in)
        except Exception:
            depth = None
        side = (md.group(2) or "").upper() or None
        if depth is not None:
            entry["depth_in"] = depth
        if side:
            entry["side"] = side

    back_hint = bool(
        re.search(r"\((?:FROM\s+)?BACK\)", U)
        or re.search(r"\bFROM\s+BACK\b", U)
        or re.search(r"\bBACK\s*SIDE\b", U)
        or "BACKSIDE" in U
    )
    if back_hint and str(entry.get("side") or "").upper() != "BACK":
        entry["side"] = "BACK"
    if re.search(r"\b(FRONT\s*&\s*BACK|BOTH\s+SIDES)\b", U):
        entry["double_sided"] = True

    mref = re.search(r"REF\s*[Ø⌀]\s*(\d+(?:\.\d+)?)", U)
    if mref:
        try:
            entry["ref_dia_in"] = float(mref.group(1)) * float(to_in)
        except Exception:
            entry["ref_dia_in"] = None

    if entry.get("ref_dia_in") is None:
        mdia = RE_DIA.search(U)
        if mdia and ("Ø" in U or "⌀" in U or " REF" in U):
            try:
                entry["ref_dia_in"] = float(mdia.group(1)) * float(to_in)
            except Exception:
                entry["ref_dia_in"] = None

    return entry


def _aggregate_hole_entries(entries: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    hole_count = 0
    tap_qty = 0
    cbore_qty = 0
    csk_qty = 0
    max_depth_in = 0.0
    back_ops = False
    tap_details: dict[str, dict[str, Any]] = {}
    tap_minutes_total = 0.0
    tap_class_counter: Counter[str] = Counter()
    npt_qty = 0
    double_cbore = False
    double_csk = False
    cbore_minutes_total = 0.0
    csk_minutes_total = 0.0
    if entries:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            qty_val = entry.get("qty")
            if qty_val is None:
                qty = 0
            else:
                try:
                    qty = int(qty_val)
                except Exception:
                    try:
                        qty = int(round(float(qty_val)))
                    except Exception:
                        qty = 0
            qty = qty if qty > 0 else 1
            hole_count += qty
            if entry.get("tap"):
                tap_qty += qty
                spec = str(entry.get("tap") or "").strip()
                cls = entry.get("tap_class") or "unknown"
                minutes_per = entry.get("tap_minutes_per") or TAP_MINUTES_BY_CLASS.get("medium", 0.3)
                tap_minutes_total += qty * float(minutes_per)
                tap_class_counter[str(cls)] += qty
                detail = tap_details.setdefault(
                    spec,
                    {
                        "spec": spec,
                        "qty": 0,
                        "class": cls,
                        "minutes_per_hole": float(minutes_per),
                        "total_minutes": 0.0,
                        "is_npt": bool(entry.get("tap_is_npt")),
                    },
                )
                detail["qty"] += qty
                detail["total_minutes"] += qty * float(minutes_per)
                if entry.get("tap_is_npt"):
                    npt_qty += qty
            if entry.get("cbore"):
                ops_qty = qty * (2 if entry.get("double_sided") else 1)
                cbore_qty += ops_qty
                cbore_minutes_total += ops_qty * CBORE_MIN_PER_SIDE_MIN
                if entry.get("double_sided"):
                    double_cbore = True
            if entry.get("csk"):
                ops_qty = qty * (2 if entry.get("double_sided") else 1)
                csk_qty += ops_qty
                csk_minutes_total += ops_qty * CSK_MIN_PER_SIDE_MIN
                if entry.get("double_sided"):
                    double_csk = True
            depth = entry.get("depth_in")
            try:
                if depth and float(depth) > max_depth_in:
                    max_depth_in = float(depth)
            except Exception:
                continue
            side = str(entry.get("side") or "").upper()
            if (
                side == "BACK"
                or (entry.get("raw") and "BACK" in str(entry.get("raw")).upper())
                or entry.get("double_sided")
                and (entry.get("cbore") or entry.get("csk"))
            ):
                back_ops = True
    tap_details_list = []
    for spec, detail in tap_details.items():
        detail["qty"] = int(detail.get("qty", 0) or 0)
        detail["total_minutes"] = round(float(detail.get("total_minutes", 0.0)), 3)
        detail["minutes_per_hole"] = round(float(detail.get("minutes_per_hole", 0.0)), 3)
        tap_details_list.append(detail)
    tap_details_list.sort(key=lambda d: -d.get("qty", 0))
    tap_class_counts = {cls: int(qty) for cls, qty in tap_class_counter.items() if qty}
    return {
        "hole_count": hole_count if hole_count else None,
        "tap_qty": tap_qty,
        "cbore_qty": cbore_qty,
        "csk_qty": csk_qty,
        "deepest_hole_in": max_depth_in if max_depth_in > 0 else None,
        "provenance": "HOLE TABLE / NOTES" if hole_count else None,
        "from_back": back_ops,
        "tap_details": tap_details_list,
        "tap_minutes_hint": round(tap_minutes_total, 3) if tap_minutes_total else None,
        "tap_class_counts": tap_class_counts,
        "npt_qty": npt_qty,
        "cbore_minutes_hint": round(cbore_minutes_total, 3) if cbore_minutes_total else None,
        "csk_minutes_hint": round(csk_minutes_total, 3) if csk_minutes_total else None,
        "double_sided_cbore": double_cbore,
        "double_sided_csk": double_csk,
    }


def summarize_hole_chart_lines(lines: Iterable[str] | None) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for raw in lines or []:
        text = str(raw or "")
        if not text.strip():
            continue
        entry = _parse_hole_line(text, 1.0, source="CHART")
        if not entry:
            continue
        if not entry.get("qty"):
            mqty = re.search(r"\((\d+)\)", text)
            if mqty:
                try:
                    entry["qty"] = int(mqty.group(1))
                except Exception:
                    pass
        entries.append(entry)
    agg = _aggregate_hole_entries(entries)
    return {
        "tap_qty": int(agg.get("tap_qty") or 0),
        "cbore_qty": int(agg.get("cbore_qty") or 0),
        "csk_qty": int(agg.get("csk_qty") or 0),
        "deepest_hole_in": agg.get("deepest_hole_in"),
        "from_back": bool(agg.get("from_back")),
        "tap_details": agg.get("tap_details") or [],
        "tap_minutes_hint": agg.get("tap_minutes_hint"),
        "tap_class_counts": agg.get("tap_class_counts") or {},
        "npt_qty": int(agg.get("npt_qty") or 0),
        "cbore_minutes_hint": agg.get("cbore_minutes_hint"),
        "csk_minutes_hint": agg.get("csk_minutes_hint"),
        "double_sided_cbore": bool(agg.get("double_sided_cbore")),
        "double_sided_csk": bool(agg.get("double_sided_csk")),
    }


__all__ = [
    "RE_TAP",
    "RE_NPT",
    "RE_THRU",
    "RE_CBORE",
    "RE_CSK",
    "RE_DEPTH",
    "RE_DIA",
    "RE_FRONT_BACK",
    "TAP_MINUTES_BY_CLASS",
    "CBORE_MIN_PER_SIDE_MIN",
    "CSK_MIN_PER_SIDE_MIN",
    "_major_diameter_from_thread",
    "_classify_thread_spec",
    "_normalize_hole_text",
    "_dedupe_hole_entries",
    "_parse_hole_line",
    "_aggregate_hole_entries",
    "_build_ops_rows_from_lines_fallback",
    "build_ops_rows_from_lines_fallback",
    "summarize_hole_chart_lines",
]


def build_ops_rows_from_lines_fallback(lines: Iterable[str] | None) -> list[dict]:
    """Proxy to the chart-line fallback parser exposed for hole operations helpers."""

    seq = [str(s) for s in lines or [] if str(s)]
    if not seq:
        return []
    return _chart_build_ops_rows_from_lines_fallback(seq)


_build_ops_rows_from_lines_fallback = _chart_build_ops_rows_from_lines_fallback
