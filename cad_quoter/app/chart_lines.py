from __future__ import annotations

"""Fallback helpers for parsing hole chart text into conservative operations."""

import re
from typing import Iterable, Mapping, Sequence

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
]


def _norm_line(s: str) -> str:
    """Collapse repeated whitespace in *s* and strip surrounding spaces."""

    return re.sub(r"\s+", " ", (s or "")).strip()


_RE_QTY_LEAD = re.compile(r"^\s*\((\d+)\)\s*")
_RE_FROM_SIDE = re.compile(r"\bFROM\s+(FRONT|BACK)\b", re.I)
_RE_DEPTH_MULT = re.compile(r"[×x]\s*([0-9.]+)\b", re.I)
_RE_DEPTH_DEEP = re.compile(r"(\d+(?:\.\d+)?)\s*DEEP(?:\s+FROM\s+(FRONT|BACK))?", re.I)
_RE_THRU = re.compile(r"\bTHRU\b", re.I)
_RE_DIA_ANY = re.compile(r"(?:Ø|⌀|DIA|O)\s*([0-9.]+)|\(([0-9.]+)\s*Ø?\)|\b([0-9.]+)\b")
_RE_TAP = re.compile(
    r"(\(\d+\)\s*)?(#\s*\d{1,2}-\d+|(?:\d+/\d+)\s*-\s*\d+|(?:\d+(?:\.\d+)?)\s*-\s*\d+|M\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?)\s*TAP",
    re.I,
)
_RE_CBORE = re.compile(r"\b(C['’]?\s*BORE|CBORE|COUNTER\s*BORE)\b", re.I)
_RE_NPT = re.compile(r"(\d+/\d+)\s*-\s*N\.?P\.?T\.?", re.I)
_RE_MM_IN_DIA = re.compile(r"(?:Ø|⌀|O|DIA|\b)\s*([0-9.]+)")
_RE_PAREN_DIA = re.compile(r"\(([0-9.]+)\s*Ø?\)")
_RE_DIA_SIMPLE = re.compile(r"[Ø⌀\u00D8]?\s*(\d+(?:\.\d+)?)", re.I)



def _coalesce_rows(rows: Sequence[Mapping[str, object]] | Sequence[dict]) -> list[dict]:
    """Merge rows that share the same description/reference pair."""

    agg: dict[tuple[str, str], int] = {}
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
            agg[key] = qty_val
            order.append(key)
        else:
            agg[key] += qty_val
    return [{"hole": "", "ref": ref, "qty": agg[(desc, ref)], "desc": desc} for (desc, ref) in order]



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
            if _RE_THRU.search(tail):
                desc += " THRU"
            depth_source = tail
            depth_match = _RE_DEPTH_MULT.search(depth_source)
            depth_value = depth_match.group(1) if depth_match else None
            deep_match = _RE_DEPTH_DEEP.search(depth_source)
            side_from_depth: str | None = None
            if deep_match:
                depth_value = depth_value or deep_match.group(1)
                side_from_depth = (deep_match.group(2) or "").upper() or None
            if depth_value:
                desc += f' × {float(depth_value):.2f}"'
            ms = side_from_depth or None
            if not ms:
                m_side = _RE_FROM_SIDE.search(tail)
                if m_side:
                    ms = m_side.group(1).upper()
            if ms:
                desc += f" FROM {ms}"
            out.append({"hole": "", "ref": "", "qty": qty, "desc": desc})
            i += 1
            continue
        if _RE_CBORE.search(ln):
            tail = " ".join([ln] + L[max(0, i - 1):i] + L[i + 1:i + 2])
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
            if depth_value:
                desc += f' × {float(depth_value):.2f}"'
            ms = side_from_depth or None
            if not ms:
                m_side = _RE_FROM_SIDE.search(" ".join([ln] + L[i + 1:i + 2]))
                if m_side:
                    ms = m_side.group(1).upper()
            if ms:
                desc += f" FROM {ms}"
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
            out.append({"hole": "", "ref": "", "qty": qty, "desc": desc})
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

    seq = list(lines) if not isinstance(lines, list) else lines
    return _build_ops_rows_from_lines_fallback(seq)


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
