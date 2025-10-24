"""Hole table parsing and aggregation helpers."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, MutableMapping
from fractions import Fraction
from typing import Any, Callable


from .chart_lines import (
    _build_ops_rows_from_lines_fallback as _chart_build_ops_rows_from_lines_fallback,
    collect_chart_lines_context as _collect_chart_lines_context,
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
RE_DIA = re.compile(r"(?:%%[Cc]\s*|[Ø⌀\u00D8]\s*)?(\d+(?:\.\d+)?)", re.I)
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

COUNTERDRILL_MIN_PER_SIDE_MIN = 0.12
CBORE_MIN_PER_SIDE_MIN = 0.15
CSK_MIN_PER_SIDE_MIN = 0.12

_SPOT_TOKENS = re.compile(r"(?:C['’]?\s*DRILL|CENTER\s*DRILL|SPOT\s*DRILL|SPOT)", re.I)
_THREAD_WITH_TPI_RE = re.compile(
    r"((?:#\d+)|(?:\d+/\d+)|(?:\d+(?:\.\d+)?))\s*-\s*(\d+)",
    re.I,
)
_THREAD_WITH_NPT_RE = re.compile(
    r"((?:#\d+)|(?:\d+/\d+)|(?:\d+(?:\.\d+)?))\s*-\s*(N\.?P\.?T\.?)",
    re.I,
)
_CBORE_RE = re.compile(
    r"(?:^|[ ;])(?:%%[Cc]|Ø|⌀|DIA)?\s*((?:\d+\s*/\s*\d+)|(?:\d+(?:\.\d+)?))\s*(?:C['’]?\s*BORE|CBORE|COUNTER\s*BORE)",
    re.I,
)
_SIDE_BACK = re.compile(r"\b(?:FROM\s+)?BACK\b", re.I)
_SIDE_FRONT = re.compile(r"\b(?:FROM\s+)?FRONT\b", re.I)
_DEPTH_TOKEN = re.compile(r"[×xX]\s*([0-9.]+)\b")
_DIA_TOKEN = re.compile(
    r"(?:%%[Cc]|Ø|⌀|REF|DIA)[^0-9]*((?:\d+\s*/\s*\d+)|(?:\d+)?\.\d+|\d+(?:\.\d+)?)",
    re.I,
)


def _parse_ref_to_inch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            val = float(value)
        except Exception:
            return None
        return val if math.isfinite(val) and val > 0 else None
    text = str(value).strip()
    if not text:
        return None
    cleaned = (
        text.replace("%%C", "")
        .replace("%%c", "")
        .replace("\u00D8", "")
        .replace("Ø", "")
        .replace("⌀", "")
        .replace("IN", "")
        .replace("in", "")
        .strip("\"' ")
    )
    if not cleaned:
        return None
    try:
        if "/" in cleaned:
            return float(Fraction(cleaned))
        return float(cleaned)
    except Exception:
        try:
            return float(Fraction(cleaned))
        except Exception:
            return None


def _rows_from_ops_summary(
    geo: Mapping[str, Any] | None,
    *,
    result: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
) -> list[dict]:
    geo_map: Mapping[str, Any] = geo if isinstance(geo, _MappingABC) else {}
    ops = (geo_map or {}).get("ops_summary") or {}
    rows = ops.get("rows") if isinstance(ops, dict) else None
    if rows:
        return list(rows)
    if isinstance(ops, dict):
        detail = ops.get("rows_detail")
        if isinstance(detail, list):
            fallback: list[dict[str, Any]] = []
            for entry in detail:
                if not isinstance(entry, _MappingABC):
                    continue
                base = {
                    "hole": entry.get("hole", ""),
                    "ref": entry.get("ref", ""),
                    "qty": entry.get("qty", 0),
                    "desc": entry.get("desc", ""),
                }
                if entry.get("diameter_in") is not None:
                    base["diameter_in"] = entry.get("diameter_in")
                fallback.append(base)
            if fallback:
                return fallback
    _containers: list[dict] = []
    for candidate in (result, breakdown, geo_map):
        if isinstance(candidate, dict):
            _containers.append(candidate)
        elif isinstance(candidate, _MappingABC):
            try:
                _containers.append(dict(candidate))
            except Exception:
                continue
    chart_lines = _collect_chart_lines_context(*_containers)
    if chart_lines:
        fallback_rows = _chart_build_ops_rows_from_lines_fallback(chart_lines)
        if fallback_rows:
            return fallback_rows
    return []


def _side_of(desc: str) -> str:
    if _SIDE_BACK.search(desc or ""):
        return "BACK"
    if _SIDE_FRONT.search(desc or ""):
        return "FRONT"
    return "FRONT"


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
    cleaned = cleaned.replace("%%C", "").replace("%%c", "").replace("Ø", "").replace("⌀", "")
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


def build_ops_summary_rows_from_hole_rows(
    hole_rows: Iterable[Any] | None,
) -> list[dict[str, Any]]:
    """Convert parsed hole rows into ops-summary table rows."""

    summary_rows: list[dict[str, Any]] = []
    if not hole_rows:
        return summary_rows

    for row in hole_rows:
        if row is None:
            continue

        try:
            qty = int(getattr(row, "qty", 0) or 0)
        except Exception:
            qty = 0

        ref = (
            getattr(row, "ref", None)
            or getattr(row, "drill_ref", None)
            or getattr(row, "pilot", None)
            or ""
        )

        desc = (
            getattr(row, "description", None)
            or getattr(row, "desc", None)
            or ""
        )

        if not desc:
            parts: list[str] = []
            try:
                features = list(getattr(row, "features", []) or [])
            except Exception:
                features = []
            for feature in features:
                if not isinstance(feature, dict):
                    continue
                feature_type = str(feature.get("type", "")).lower()
                side = str(feature.get("side", "")).upper()
                if feature_type == "tap":
                    thread = feature.get("thread") or ""
                    depth = feature.get("depth_in")
                    tap_desc = f"{thread} TAP" if thread else "TAP"
                    if isinstance(depth, (int, float)):
                        tap_desc += f" × {depth:.2f}\""
                    if side:
                        tap_desc += f" FROM {side}"
                    parts.append(tap_desc)
                elif feature_type == "cbore":
                    dia = feature.get("dia_in")
                    depth = feature.get("depth_in")
                    cbore_desc = ""
                    if isinstance(dia, (int, float)):
                        cbore_desc += f"{dia:.4f} "
                    cbore_desc += "C’BORE"
                    if isinstance(depth, (int, float)):
                        cbore_desc += f" × {depth:.2f}\""
                    if side:
                        cbore_desc += f" FROM {side}"
                    parts.append(cbore_desc)
                elif feature_type in {"csk", "countersink"}:
                    dia = feature.get("dia_in")
                    depth = feature.get("depth_in")
                    csk_desc = ""
                    if isinstance(dia, (int, float)):
                        csk_desc += f"{dia:.4f} "
                    csk_desc += "C’SINK"
                    if isinstance(depth, (int, float)):
                        csk_desc += f" × {depth:.2f}\""
                    if side:
                        csk_desc += f" FROM {side}"
                    parts.append(csk_desc)
                elif feature_type == "drill":
                    ref_local = feature.get("ref") or ref or ""
                    thru = " THRU" if feature.get("thru", True) else ""
                    parts.append(f"{ref_local}{thru}".strip())
                elif feature_type == "spot":
                    depth = feature.get("depth_in")
                    spot_desc = "C’DRILL"
                    if isinstance(depth, (int, float)):
                        spot_desc += f" × {depth:.2f}\""
                    parts.append(spot_desc)
                elif feature_type == "jig":
                    parts.append("JIG GRIND")
            desc = "; ".join(part for part in parts if part)

        summary_rows.append(
            {
                "hole": str(
                    getattr(row, "hole_id", "")
                    or getattr(row, "letter", "")
                    or ""
                ),
                "ref": str(ref or ""),
                "qty": int(qty),
                "desc": str(desc or ""),
            }
        )

    return summary_rows


def update_geo_ops_summary_from_hole_rows(
    geo: MutableMapping[str, Any],
    *,
    hole_rows: Iterable[Any] | None = None,
    chart_lines: Iterable[str] | None = None,
    chart_source: str | None = None,
    chart_summary: Mapping[str, Any] | None = None,
    apply_built_rows: Callable[[
        MutableMapping[str, Any] | Mapping[str, Any] | None,
        Iterable[Mapping[str, Any]] | None,
    ], int]
    | None = None,
) -> dict[str, Any]:
    """Populate geo["ops_summary"] with rows and totals derived from hole data."""

    ops_rows = build_ops_summary_rows_from_hole_rows(hole_rows)
    if not ops_rows and chart_lines:
        ops_rows = _chart_build_ops_rows_from_lines_fallback(chart_lines)

    if not ops_rows:
        return {"rows": [], "totals": {}}

    ops_summary_map = geo.setdefault("ops_summary", {})
    ops_summary_map["rows"] = ops_rows
    ops_summary_map["source"] = chart_source or "chart_lines"

    totals_from_summary: dict[str, int] = {}

    if apply_built_rows:
        try:
            apply_built_rows(ops_summary_map, ops_rows)
        except Exception:
            pass

    if chart_summary and isinstance(chart_summary, Mapping):
        try:
            tap_qty = int(chart_summary.get("tap_qty") or 0)
            if tap_qty:
                totals_from_summary["tap_total"] = tap_qty
        except Exception:
            pass
        try:
            cbore_qty = int(chart_summary.get("cbore_qty") or 0)
            if cbore_qty:
                totals_from_summary["cbore_total"] = cbore_qty
        except Exception:
            pass
        try:
            csk_qty = int(chart_summary.get("csk_qty") or 0)
            if csk_qty:
                totals_from_summary["csk_total"] = csk_qty
        except Exception:
            pass

    if totals_from_summary:
        existing_totals = ops_summary_map.get("totals")
        if isinstance(existing_totals, Mapping):
            totals_map = dict(existing_totals)
            totals_map.update(totals_from_summary)
        else:
            totals_map = dict(totals_from_summary)
        ops_summary_map["totals"] = totals_map
        # Maintain legacy top-level keys for downstream compatibility.
        for key, value in totals_from_summary.items():
            ops_summary_map[key] = value

    return {
        "rows": ops_rows,
        "totals": ops_summary_map.get("totals", {}),
    }


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
        "from_back": False,
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
            if side == "BACK":
                entry["from_back"] = True

    back_hint = bool(
        re.search(r"\((?:FROM\s+)?BACK\)", U)
        or re.search(r"\bFROM\s+BACK\b", U)
        or re.search(r"\bBACK\s*SIDE\b", U)
        or "BACKSIDE" in U
    )
    if back_hint and str(entry.get("side") or "").upper() != "BACK":
        entry["side"] = "BACK"
        entry["from_back"] = True
    if re.search(r"\b(FRONT\s*&\s*BACK|BOTH\s+SIDES)\b", U):
        entry["double_sided"] = True

    mref = re.search(r"REF\s*(?:%%[Cc]|[Ø⌀])\s*(\d+(?:\.\d+)?)", U)
    if mref:
        try:
            entry["ref_dia_in"] = float(mref.group(1)) * float(to_in)
        except Exception:
            entry["ref_dia_in"] = None

    if entry.get("ref_dia_in") is None:
        mdia = RE_DIA.search(U)
        if mdia and ("Ø" in U or "⌀" in U or " REF" in U or "%%C" in U or "%%c" in U):
            try:
                entry["ref_dia_in"] = float(mdia.group(1)) * float(to_in)
            except Exception:
                entry["ref_dia_in"] = None

    def _int_or_zero(value: Any) -> int:
        if value in (None, ""):
            return 0
        try:
            return int(value)
        except Exception:
            try:
                return int(round(float(value)))
            except Exception:
                return 0

    def _base_op_payload() -> dict[str, Any]:
        qty = _int_or_zero(entry.get("qty"))
        side_raw = str(entry.get("side") or "").strip().upper()
        if not side_raw:
            side_raw = "BACK" if entry.get("from_back") else "FRONT"
        payload: dict[str, Any] = {
            "qty": qty,
            "side": side_raw,
            "double_sided": bool(entry.get("double_sided")),
            "from_back": bool(entry.get("from_back")),
            "depth_in": entry.get("depth_in"),
            "ref_dia_in": entry.get("ref_dia_in"),
            "thru": bool(entry.get("thru")),
            "source": entry.get("source"),
        }
        ref_val = entry.get("ref")
        if ref_val:
            payload["ref"] = ref_val
        return payload

    ops: list[dict[str, Any]] = []

    if entry.get("tap"):
        tap_payload = _base_op_payload()
        tap_payload.update(
            {
                "type": "tap",
                "thread": entry.get("tap"),
                "tap_class": entry.get("tap_class"),
                "tap_minutes_per": entry.get("tap_minutes_per"),
                "tap_is_npt": entry.get("tap_is_npt"),
            }
        )
        ops.append({k: v for k, v in tap_payload.items() if v not in (None, "")})

    if entry.get("cbore"):
        cbore_payload = _base_op_payload()
        cbore_payload["type"] = "cbore"
        ops.append({k: v for k, v in cbore_payload.items() if v not in (None, "")})

    if entry.get("csk"):
        csk_payload = _base_op_payload()
        csk_payload["type"] = "csk"
        ops.append({k: v for k, v in csk_payload.items() if v not in (None, "")})

    if entry.get("thru") or entry.get("ref_dia_in"):
        drill_payload = _base_op_payload()
        drill_payload["type"] = "drill"
        if entry.get("tap"):
            drill_payload["claimed_by_tap"] = True
            drill_payload["pilot_for_thread"] = entry.get("tap")
            if entry.get("tap_is_npt"):
                drill_payload["claimed_is_npt"] = True
        ops.append({k: v for k, v in drill_payload.items() if v not in (None, "")})

    if ops:
        entry["ops"] = ops
        if not entry.get("type"):
            entry["type"] = ops[0].get("type")

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


def summarize_hole_chart_agreement(
    entity_holes_mm: Iterable[Any] | None, chart_ops: Iterable[dict] | None
) -> dict[str, Any]:
    """Compare entity-detected hole sizes with chart-derived operations."""

    def _as_qty(value: Any) -> int:
        try:
            qty = int(round(float(value)))
        except Exception:
            qty = 0
        return qty if qty > 0 else 1

    bin_key = lambda dia: round(float(dia), 1)

    ent_bins: Counter[float] = Counter()
    if entity_holes_mm:
        for value in entity_holes_mm:
            try:
                dia = float(value)
            except Exception:
                continue
            if dia > 0:
                ent_bins[bin_key(dia)] += 1

    chart_bins: Counter[float] = Counter()
    tap_qty = 0
    cbore_qty = 0
    csk_qty = 0
    if chart_ops:
        for op in chart_ops:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("type") or "").lower()
            qty = _as_qty(op.get("qty"))
            if op_type == "tap":
                tap_qty += qty
            if op_type == "cbore":
                cbore_qty += qty
            if op_type in {"csk", "countersink"}:
                csk_qty += qty
            if op_type != "drill":
                continue
            dia_raw = op.get("dia_mm")
            if dia_raw is None:
                continue
            try:
                dia_val = float(dia_raw)
            except Exception:
                continue
            if dia_val <= 0:
                continue
            chart_bins[bin_key(dia_val)] += qty

    entity_total = sum(ent_bins.values())
    chart_total = sum(chart_bins.values())
    max_total = max(entity_total, chart_total)
    tolerance = max(5, 0.1 * max_total) if max_total else 5
    agreement = abs(entity_total - chart_total) <= tolerance

    return {
        "entity_bins": {float(k): int(v) for k, v in ent_bins.items()},
        "chart_bins": {float(k): int(v) for k, v in chart_bins.items()},
        "tap_qty": int(tap_qty),
        "cbore_qty": int(cbore_qty),
        "csk_qty": int(csk_qty),
        "agreement": bool(agreement),
        "entity_total": int(entity_total),
        "chart_total": int(chart_total),
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
    "COUNTERDRILL_MIN_PER_SIDE_MIN",
    "CBORE_MIN_PER_SIDE_MIN",
    "CSK_MIN_PER_SIDE_MIN",
    "_SPOT_TOKENS",
    "_THREAD_WITH_TPI_RE",
    "_THREAD_WITH_NPT_RE",
    "_CBORE_RE",
    "_SIDE_BACK",
    "_SIDE_FRONT",
    "_DEPTH_TOKEN",
    "_DIA_TOKEN",
    "_parse_ref_to_inch",
    "_rows_from_ops_summary",
    "_side_of",
    "_major_diameter_from_thread",
    "_classify_thread_spec",
    "_normalize_hole_text",
    "_dedupe_hole_entries",
    "build_ops_summary_rows_from_hole_rows",
    "update_geo_ops_summary_from_hole_rows",
    "_parse_hole_line",
    "_aggregate_hole_entries",
    "_build_ops_rows_from_lines_fallback",
    "build_ops_rows_from_lines_fallback",
    "summarize_hole_chart_lines",
    "summarize_hole_chart_agreement",
]


def build_ops_rows_from_lines_fallback(lines: Iterable[str] | None) -> list[dict]:
    """Proxy to the chart-line fallback parser exposed for hole operations helpers."""

    seq = [str(s) for s in lines or [] if str(s)]
    if not seq:
        return []
    return _chart_build_ops_rows_from_lines_fallback(seq)


_build_ops_rows_from_lines_fallback = _chart_build_ops_rows_from_lines_fallback
