from __future__ import annotations

"""Fallback helpers for parsing hole chart text into conservative operations."""

import math
import re
from typing import Iterable, Mapping, Sequence

from cad_quoter.utils.machining import (
    _ipm_from_rpm_ipr,
    _lookup_sfm_ipr,
    _parse_thread_major_in,
    _parse_tpi,
    _rpm_from_sfm_diam,
)

__all__ = [
    "norm_line",
    "build_ops_rows_from_lines_fallback",
    "collect_chart_lines_context",
    "coalesce_rows",
    "RE_QTY_LEAD",
    "RE_FROM_SIDE",
    "RE_DEPTH",
    "RE_THRU",
    "RE_DIA",
    "RE_DIA_ANY",
    "RE_MM_IN_DIA",
    "RE_PAREN_DIA",
    "RE_TAP",
    "RE_CBORE",
    "RE_NPT",
    "RE_JIG_GRIND",
]


_TAPPING_INDEX_MIN = 0.08
_COUNTERBORE_INDEX_MIN = 0.06
_COUNTERBORE_RETRACT_FACTOR = 1.3
_COUNTERBORE_EXTRA_TRAVEL_IN = 0.05
_SPOT_INDEX_MIN = 0.05
_SPOT_DEFAULT_DEPTH_IN = 0.1
_JIG_GRIND_RATE_IPM = 0.02
_JIG_GRIND_INDEX_MIN = 0.25
_MIN_IPM_DENOM = 0.1


# --- Clean DXF MTEXT escapes (alignment + symbols) --------------------------
_MT_ALIGN_RE = re.compile(r"\\A\d;")
_MT_BREAK_RE = re.compile(r"\\P", re.I)
_MT_SYMS = {"%%C": "Ø", "%%c": "Ø", "%%D": "°", "%%d": "°", "%%P": "±", "%%p": "±"}


def _clean_mtext(s: str) -> str:
    if not isinstance(s, str):
        return ""
    for token, replacement in _MT_SYMS.items():
        s = s.replace(token, replacement)
    s = _MT_ALIGN_RE.sub("", s)
    s = _MT_BREAK_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# --- Row start tokens (incl. Ø and %%C) -------------------------------------
_JOIN_START_TOKENS = re.compile(
    r"(?:^\s*\(\d+\)\s*)"
    r"|(?:\bTAP\b|N\.?P\.?T\.?)"
    r"|(?:C[’']?\s*BORE|CBORE|COUNTER\s*BORE)"
    r"|(?:[Ø⌀\u00D8]|%%[Cc])"
    r"|(?:C[’']?\s*DRILL|CENTER\s*DRILL|SPOT\s*DRILL\b)",
    re.I,
)


def _join_wrapped_chart_lines(chart_lines: list[str]) -> list[str]:
    if not chart_lines:
        return []
    out: list[str] = []
    buf = ""

    def _flush() -> None:
        nonlocal buf
        if buf.strip():
            out.append(re.sub(r"\s+", " ", buf).strip())
        buf = ""

    for raw in chart_lines:
        s = _clean_mtext(str(raw or ""))
        if not s:
            continue
        if _JOIN_START_TOKENS.search(s):
            _flush()
            buf = s
        else:
            buf += " " + s
    _flush()
    return out


def _safe_float(value: object) -> float | None:
    """Return ``value`` coerced to ``float`` when possible."""

    try:
        number = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return float(number)


def _format_feed(ipr: float | None, rpm: float | None, ipm: float | None) -> str:
    """Return a consistent feed summary string with dash placeholders."""

    ipr_txt = "-" if ipr is None else f"{ipr:.4f}"
    rpm_txt = "-" if rpm is None else f"{int(round(rpm))}"
    ipm_txt = "-" if ipm is None else f"{ipm:.3f}"
    return f"{ipr_txt} ipr | {rpm_txt} rpm | {ipm_txt} ipm"


def _tapping_runtime(thread: str, depth_in: float | None, is_thru: bool) -> tuple[float | None, str]:
    """Return tapping cycle minutes-per-hole and formatted feed string."""

    tpi = _parse_tpi(thread)
    ipr = (1.0 / float(tpi)) if tpi else None
    major = _parse_thread_major_in(thread)
    sfm, _ = _lookup_sfm_ipr("tapping", major, None, None)
    rpm = _rpm_from_sfm_diam(sfm, major)
    ipm = _ipm_from_rpm_ipr(rpm, ipr)
    travel_in = depth_in
    if travel_in is None and is_thru:
        travel_in = None  # defer to upstream thickness context when available
    minutes: float | None = None
    if travel_in is not None and ipm is not None:
        cycle = max(0.0, float(travel_in))
        minutes = (cycle / max(_MIN_IPM_DENOM, ipm)) + _TAPPING_INDEX_MIN
    feed_fmt = _format_feed(ipr, rpm, ipm)
    return minutes, feed_fmt


def _counterbore_runtime(diam_in: float | None, depth_in: float | None) -> tuple[float | None, str]:
    """Return counterbore cycle minutes-per-hole and feed format string."""

    sfm, ipr = _lookup_sfm_ipr("counterbore", diam_in, None, None)
    rpm = _rpm_from_sfm_diam(sfm, diam_in)
    ipm = _ipm_from_rpm_ipr(rpm, ipr)
    travel_in = None if depth_in is None else max(0.0, float(depth_in))
    minutes: float | None = None
    if travel_in is not None and ipm is not None:
        cycle = (travel_in * _COUNTERBORE_RETRACT_FACTOR) + _COUNTERBORE_EXTRA_TRAVEL_IN
        minutes = (cycle / max(_MIN_IPM_DENOM, ipm)) + _COUNTERBORE_INDEX_MIN
    feed_fmt = _format_feed(ipr, rpm, ipm)
    return minutes, feed_fmt


def _spot_runtime(depth_in: float | None) -> tuple[float | None, str]:
    """Return spot drill minutes-per-hole and feed format string."""

    sfm, ipr = _lookup_sfm_ipr("spot", 0.1875, None, None)
    rpm = _rpm_from_sfm_diam(sfm, 0.1875)
    ipm = _ipm_from_rpm_ipr(rpm, ipr)
    travel = depth_in if depth_in is not None else _SPOT_DEFAULT_DEPTH_IN
    minutes: float | None = None
    if ipm is not None:
        cycle = max(0.0, float(travel))
        minutes = (cycle / max(_MIN_IPM_DENOM, ipm)) + _SPOT_INDEX_MIN
    feed_fmt = _format_feed(ipr, rpm, ipm)
    return minutes, feed_fmt


def _jig_grind_runtime(depth_in: float | None) -> tuple[float | None, str]:
    """Return jig-grind minutes-per-feature and feed format string."""

    travel = _safe_float(depth_in)
    ipm = _JIG_GRIND_RATE_IPM
    minutes: float | None = None
    if travel is not None:
        denom = ipm if ipm and ipm > 0 else _MIN_IPM_DENOM
        minutes = (max(0.0, travel) / denom) + _JIG_GRIND_INDEX_MIN
    feed_fmt = _format_feed(None, None, ipm)
    return minutes, feed_fmt


def _norm_line(s: str) -> str:
    """Collapse repeated whitespace in *s* and strip surrounding spaces."""

    return re.sub(r"\s+", " ", (s or "")).strip()


_RE_QTY_LEAD = re.compile(r"^\s*\((\d+)\)\s*")
_RE_FROM_SIDE = re.compile(r"\bFROM\s+(FRONT|BACK)\b", re.I)
_RE_DEPTH_MULT = re.compile(r"[×x]\s*([0-9.]+)\b", re.I)
_RE_DEPTH_DEEP = re.compile(r"(\d+(?:\.\d+)?)\s*DEEP(?:\s+FROM\s+(FRONT|BACK))?", re.I)
_RE_THRU = re.compile(r"\bTHRU\b", re.I)
_RE_DIA_ANY = re.compile(r"(?:%%[Cc]|Ø|⌀|DIA|O)\s*([0-9.]+)|\(([0-9.]+)\s*Ø?\)|\b([0-9.]+)\b")
_RE_TAP = re.compile(
    r"(\(\d+\)\s*)?(#\s*\d{1,2}-\d+|(?:\d+/\d+)\s*-\s*\d+|(?:\d+(?:\.\d+)?)\s*-\s*\d+|M\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?)\s*TAP",
    re.I,
)
_RE_CBORE = re.compile(r"\b(C['’]?\s*BORE|CBORE|COUNTER\s*BORE)\b", re.I)
_RE_NPT = re.compile(r"(\d+/\d+)\s*-\s*N\.?P\.?T\.?", re.I)
_RE_MM_IN_DIA = re.compile(r"(?:%%[Cc]|Ø|⌀|O|DIA|\b)\s*([0-9.]+)")
_RE_PAREN_DIA = re.compile(r"\(([0-9.]+)\s*Ø?\)")
_RE_DIA_SIMPLE = re.compile(r"(?:%%[Cc]\s*|[Ø⌀\u00D8]\s*)?(\d+(?:\.\d+)?)", re.I)
_RE_JIG_GRIND = re.compile(r"\bJIG\s*GRIND\b", re.I)
_RE_COUNTERDRILL = re.compile(
    r"\b(?:C[’']\s*DRILL|C\s*DRILL|COUNTER[-\s]*DRILL)\b",
    re.I,
)
_RE_CENTER_OR_SPOT = re.compile(r"\b(CENTER\s*DRILL|SPOT\s*DRILL|SPOT)\b", re.I)



def _coalesce_rows(rows: Sequence[Mapping[str, object]] | Sequence[dict]) -> list[dict]:
    """Merge rows that share the same description/reference pair."""

    agg: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for r in rows:
        if not isinstance(r, Mapping):  # tolerate bare dicts without Mapping mixin
            if isinstance(r, dict):
                candidate = r
            else:
                continue
        else:
            candidate = r
        desc = str(candidate.get("desc", "") or "")
        ref = str(candidate.get("ref", "") or "")
        try:
            qty_val = int(candidate.get("qty") or 0)
        except Exception:
            qty_val = 0
        if qty_val <= 0:
            continue
        key = (desc, ref)
        if key not in agg:
            agg[key] = dict(candidate)
            agg[key]["hole"] = ""
            agg[key]["ref"] = ref
            agg[key]["desc"] = desc
            agg[key]["qty"] = qty_val
            order.append(key)
        else:
            agg[key]["qty"] = int(agg[key].get("qty") or 0) + qty_val
    return [agg[key] for key in order]



def _build_ops_rows_from_lines_fallback(lines: list[str]) -> list[dict]:
    """Best-effort parser that extracts operations from raw hole chart text."""

    if not lines:
        return []
    L = [_norm_line(s) for s in lines if _norm_line(s)]
    out: list[dict] = []
    i = 0
    while i < len(L):
        ln = L[i]
        if any(k in ln.upper() for k in ("BREAK ALL", "SHARP CORNERS", "RADIUS", "CHAMFER", "AS SHOWN")):
            i += 1
            continue
        qty = 1
        mqty = _RE_QTY_LEAD.match(ln)
        if mqty:
            qty = int(mqty.group(1))
            ln = ln[mqty.end():].strip()
        mtap = _RE_TAP.search(ln)
        if mtap:
            thread = mtap.group(2).replace(" ", "")
            tail = " ".join([ln] + L[i + 1:i + 3])
            desc = f"{thread} TAP"
            has_thru = bool(_RE_THRU.search(tail))
            if has_thru:
                desc += " THRU"
            depth_source = tail
            depth_match = _RE_DEPTH_MULT.search(depth_source)
            depth_value = depth_match.group(1) if depth_match else None
            deep_match = _RE_DEPTH_DEEP.search(depth_source)
            side_from_depth: str | None = None
            if deep_match:
                depth_value = depth_value or deep_match.group(1)
                side_from_depth = (deep_match.group(2) or "").upper() or None
            depth_in = _safe_float(depth_value)
            if depth_value:
                desc += f' × {float(depth_value):.2f}"'
            ms = side_from_depth or None
            if not ms:
                m_side = _RE_FROM_SIDE.search(tail)
                if m_side:
                    ms = m_side.group(1).upper()
            if ms:
                desc += f" FROM {ms}"
            minutes, feed_fmt = _tapping_runtime(thread, depth_in, has_thru)
            row = {"hole": "", "ref": "", "qty": qty, "desc": desc}
            if minutes is not None:
                row["t_per_hole_min"] = round(minutes, 3)
            row["feed_fmt"] = feed_fmt
            out.append(row)
            i += 1
            continue
        if _RE_CBORE.search(ln):
            tail = " ".join([ln] + L[i + 1:i + 2])
            mda = _RE_PAREN_DIA.search(tail) or _RE_MM_IN_DIA.search(tail) or _RE_DIA_ANY.search(tail)
            dia = None
            if mda:
                for g in mda.groups():
                    if g:
                        dia = float(g)
                        break
            desc = (f"{dia:.4f} C’BORE" if dia is not None else "C’BORE")
            depth_source = " ".join([ln] + L[i + 1:i + 2])
            depth_match = _RE_DEPTH_MULT.search(depth_source)
            depth_value = depth_match.group(1) if depth_match else None
            deep_match = _RE_DEPTH_DEEP.search(depth_source)
            side_from_depth: str | None = None
            if deep_match:
                depth_value = depth_value or deep_match.group(1)
                side_from_depth = (deep_match.group(2) or "").upper() or None
            depth_in = _safe_float(depth_value)
            if depth_value:
                desc += f' × {float(depth_value):.2f}"'
            ms = side_from_depth or None
            if not ms:
                m_side = _RE_FROM_SIDE.search(" ".join([ln] + L[i + 1:i + 2]))
                if m_side:
                    ms = m_side.group(1).upper()
            if ms:
                desc += f" FROM {ms}"
            minutes, feed_fmt = _counterbore_runtime(dia, depth_in)
            row = {"hole": "", "ref": "", "qty": qty, "desc": desc}
            if minutes is not None:
                row["t_per_hole_min"] = round(minutes, 3)
            row["feed_fmt"] = feed_fmt
            out.append(row)
            i += 1
            continue
        if (
            _RE_COUNTERDRILL.search(ln)
            and not _RE_CENTER_OR_SPOT.search(ln)
            and "DRILL THRU" not in ln.upper()
        ):
            tail = " ".join([ln] + L[i + 1:i + 2])
            depth_match = _RE_DEPTH_MULT.search(tail)
            depth_value = depth_match.group(1) if depth_match else None
            if not depth_value:
                deep_match = _RE_DEPTH_DEEP.search(tail)
                depth_value = deep_match.group(1) if deep_match else None
            desc = "COUNTERDRILL"
            if depth_value:
                desc += f' × {float(depth_value):.2f}"'
            out.append({"hole": "", "ref": "", "qty": qty, "desc": desc})
            i += 1
            continue
        if any(k in ln.upper() for k in ("C' DRILL", "C’DRILL", "CENTER DRILL", "SPOT DRILL")):
            tail = " ".join([ln] + L[i + 1:i + 2])
            depth_match = _RE_DEPTH_MULT.search(tail)
            depth_value = depth_match.group(1) if depth_match else None
            if not depth_value:
                deep_match = _RE_DEPTH_DEEP.search(tail)
                depth_value = deep_match.group(1) if deep_match else None
            desc = "C’DRILL"
            if depth_value:
                desc += f' × {float(depth_value):.2f}"'
            depth_in = _safe_float(depth_value)
            minutes, feed_fmt = _spot_runtime(depth_in)
            row = {"hole": "", "ref": "", "qty": qty, "desc": desc}
            if minutes is not None:
                row["t_per_hole_min"] = round(minutes, 3)
            row["feed_fmt"] = feed_fmt
            out.append(row)
            i += 1
            continue
        if _RE_JIG_GRIND.search(ln):
            tail = " ".join([ln] + L[i + 1:i + 2])
            depth_match = _RE_DEPTH_MULT.search(tail)
            depth_value = depth_match.group(1) if depth_match else None
            if not depth_value:
                deep_match = _RE_DEPTH_DEEP.search(tail)
                depth_value = deep_match.group(1) if deep_match else None
            desc = _norm_line(ln)
            depth_in = _safe_float(depth_value)
            if depth_value and "×" not in desc:
                desc += f' × {float(depth_value):.2f}"'
            minutes, feed_fmt = _jig_grind_runtime(depth_in)
            row = {"hole": "", "ref": "", "qty": qty, "desc": desc}
            if minutes is not None:
                row["t_per_hole_min"] = round(minutes, 3)
            row["feed_fmt"] = feed_fmt
            out.append(row)
            i += 1
            continue
        if "DRILL" in ln.upper() and _RE_THRU.search(ln):
            mda = _RE_DIA_ANY.search(ln)
            ref = ""
            if mda:
                for g in mda.groups():
                    if g:
                        ref = g
                        break
            out.append({"hole": "", "ref": ref, "qty": qty, "desc": (f"{ref} THRU").strip()})
            i += 1
            continue
        if _RE_NPT.search(ln):
            out.append({"hole": "", "ref": "", "qty": qty, "desc": _norm_line(ln)})
            i += 1
            continue
        i += 1
    return _coalesce_rows(out)



def _collect_chart_lines_context(*containers: Mapping[str, object] | Sequence[str] | None) -> list[str]:
    """Merge chart line arrays from multiple geometry containers, deduping entries."""

    keys = ("chart_lines", "hole_table_lines", "chart_text_lines", "hole_chart_lines")
    merged: list[str] = []
    seen: set[str] = set()
    for d in containers:
        if not isinstance(d, Mapping):
            continue
        for k in keys:
            v = d.get(k)
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                for s in v:
                    if s not in seen:
                        seen.add(s)
                        merged.append(s)
    return merged


# -- Public wrappers ---------------------------------------------------------

def norm_line(value: str) -> str:
    """Public wrapper around :func:`_norm_line`."""

    return _norm_line(value)


def build_ops_rows_from_lines_fallback(lines: Iterable[str]) -> list[dict]:
    """Return conservative ops rows based on raw chart text."""

    seq = list(lines) if not isinstance(lines, list) else list(lines)
    cleaned: list[str] = []
    for raw in seq:
        cleaned_line = _clean_mtext(str(raw or ""))
        if cleaned_line:
            cleaned.append(cleaned_line)
    joined = _join_wrapped_chart_lines(cleaned)
    return _build_ops_rows_from_lines_fallback(joined)


def collect_chart_lines_context(*containers: Mapping[str, object] | None) -> list[str]:
    """Public wrapper around :func:`_collect_chart_lines_context`."""

    return _collect_chart_lines_context(*containers)


def coalesce_rows(rows: Sequence[Mapping[str, object]] | Sequence[dict]) -> list[dict]:
    """Public wrapper around :func:`_coalesce_rows`."""

    return _coalesce_rows(rows)


# -- Export commonly used regex patterns ------------------------------------

RE_QTY_LEAD = _RE_QTY_LEAD
RE_FROM_SIDE = _RE_FROM_SIDE
RE_DEPTH = _RE_DEPTH_DEEP
RE_THRU = _RE_THRU
RE_DIA = _RE_DIA_SIMPLE
RE_DIA_ANY = _RE_DIA_ANY
RE_MM_IN_DIA = _RE_MM_IN_DIA
RE_PAREN_DIA = _RE_PAREN_DIA
RE_TAP = _RE_TAP
RE_CBORE = _RE_CBORE
RE_NPT = _RE_NPT
