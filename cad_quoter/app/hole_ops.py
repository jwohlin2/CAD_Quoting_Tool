"""Hole table parsing and aggregation helpers."""

from __future__ import annotations

import math
import re
import typing
from collections import Counter, defaultdict
from collections.abc import (
    Iterable,
    Mapping as _MappingABC,
    MutableMapping as _MutableMappingABC,
    Sequence,
)
from fractions import Fraction
from typing import Any, Mapping, MutableMapping

from appkit.utils import (
    _ipm_from_rpm_ipr,
    _lookup_sfm_ipr,
    _parse_thread_major_in,
    _parse_tpi,
    _rpm_from_sfm_diam,
)
from cad_quoter.domain_models.values import safe_float as _safe_float

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
    r"(?:^|[ ;])(?:Ø|⌀|DIA)?\s*((?:\d+\s*/\s*\d+)|(?:\d+(?:\.\d+)?))\s*(?:C['’]?\s*BORE|CBORE|COUNTER\s*BORE)",
    re.I,
)
_SIDE_BACK = re.compile(r"\b(?:FROM\s+)?BACK\b", re.I)
_SIDE_FRONT = re.compile(r"\b(?:FROM\s+)?FRONT\b", re.I)
_DEPTH_TOKEN = re.compile(r"[×xX]\s*([0-9.]+)\b")
_DIA_TOKEN = re.compile(
    r"(?:Ø|⌀|REF|DIA)[^0-9]*((?:\d+\s*/\s*\d+)|(?:\d+)?\.\d+|\d+(?:\.\d+)?)",
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
        text.replace("\u00D8", "")
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


def _emit_tapping_card(
    lines: list[str],
    *,
    geo: Mapping[str, Any] | None,
    material_group: str | None,
    speeds_csv: dict | None,
    result: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
) -> None:
    rows = _rows_from_ops_summary(geo, result=result, breakdown=breakdown)
    groups: list[dict[str, Any]] = []
    for r in rows:
        desc = str(r.get("desc", ""))
        desc_upper = desc.upper()
        desc_clean = desc_upper.replace(".", "")
        if "TAP" not in desc_upper and "NPT" not in desc_clean:
            continue
        qty = int(r.get("qty") or 0)
        if qty <= 0:
            continue
        side = _side_of(desc)
        match = _THREAD_WITH_TPI_RE.search(desc)
        major_token = None
        tpi_token = None
        if match:
            try:
                major_token = match.group(1).strip()
                tpi_token = match.group(2).strip()
            except IndexError:
                match = None
        if match and major_token and tpi_token:
            thread = f"{major_token}-{tpi_token}"
            tpi = _parse_tpi(thread)
            major = _parse_thread_major_in(thread)
        else:
            match = _THREAD_WITH_NPT_RE.search(desc)
            if not match:
                continue
            major_token = match.group(1).strip()
            thread = f"{major_token}-NPT"
            tpi = None
            major = _parse_thread_major_in(f"{major_token}-1")
            if major is None:
                major = _parse_ref_to_inch(major_token)
        depth_match = _DEPTH_TOKEN.search(desc)
        depth_in = float(depth_match.group(1)) if depth_match else None
        pilot = (r.get("ref") or "").strip()
        pitch = (1.0 / float(tpi)) if tpi else None
        sfm, _ = _lookup_sfm_ipr("tapping", major, material_group, speeds_csv)
        rpm = _rpm_from_sfm_diam(sfm, major)
        ipm = _ipm_from_rpm_ipr(rpm, pitch)
        groups.append(
            {
                "thread": thread,
                "side": side,
                "qty": qty,
                "depth_in": depth_in,
                "pilot": pilot,
                "pitch_ipr": None if pitch is None else round(pitch, 4),
                "rpm": None if rpm is None else int(round(rpm)),
                "ipm": None if ipm is None else round(ipm, 3),
            }
        )
    if not groups:
        return
    total = sum(g["qty"] for g in groups)
    front = sum(g["qty"] for g in groups if g["side"] == "FRONT")
    back = total - front
    lines += [
        "MATERIAL REMOVAL – TAPPING",
        "=" * 64,
        "Inputs",
        "  Ops ............... Tapping (front + back), pre-drill counted in drilling",
        f"  Taps .............. {total} total  → {front} front, {back} back",
        "  Threads ........... " + ", ".join(sorted({g["thread"] for g in groups})),
        "",
        "TIME PER HOLE – TAP GROUPS",
        "-" * 66,
    ]
    for g in groups:
        depth_txt = "THRU" if g["depth_in"] is None else f'{g["depth_in"]:.2f}"'
        lines.append(
            f'{g["thread"]} × {g["qty"]}  ({g["side"]})'
            f'{(" | pilot " + g["pilot"]) if g.get("pilot") else ""}'
            f" | depth {depth_txt} | {g['pitch_ipr'] if g['pitch_ipr'] is not None else '-'} ipr"
            f" | {g['rpm'] if g['rpm'] is not None else '-'} rpm"
            f" | {g['ipm'] if g['ipm'] is not None else '-'} ipm"
            f" | t/hole — | group — "
        )
    lines.append("")


def _emit_counterbore_card(
    lines: list[str],
    *,
    geo: Mapping[str, Any] | None,
    material_group: str | None,
    speeds_csv: dict | None,
    result: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
) -> None:
    rows = _rows_from_ops_summary(geo, result=result, breakdown=breakdown)
    groups: defaultdict[tuple[float, str, float | None], int] = defaultdict(int)
    order: list[tuple[float, str, float | None]] = []
    for r in rows:
        desc = str(r.get("desc", ""))
        if "BORE" not in desc.upper():
            continue
        match = _CBORE_RE.search(desc)
        if not match:
            continue
        diam_in = float(match.group(1))
        side = _side_of(desc)
        depth_match = _DEPTH_TOKEN.search(desc)
        depth_in = float(depth_match.group(1)) if depth_match else None
        key = (round(diam_in, 4), side, depth_in)
        if key not in groups:
            order.append(key)
        groups[key] += int(r.get("qty") or 0)
    if not groups:
        return
    items = [(key, groups[key]) for key in sorted(order, key=lambda key: (key[0], key[1]))]
    total = sum(qty for _, qty in items)
    front = sum(qty for (key, qty) in items if key[1] == "FRONT")
    back = total - front
    lines += [
        "MATERIAL REMOVAL – COUNTERBORE",
        "=" * 64,
        "Inputs",
        "  Ops ............... Counterbore (front + back)",
        f"  Counterbores ...... {total} total  → {front} front, {back} back",
        "",
        "TIME PER HOLE – C’BORE GROUPS",
        "-" * 66,
    ]
    for (diam_in, side, depth_in), qty in items:
        sfm, ipr = _lookup_sfm_ipr("counterbore", diam_in, material_group, speeds_csv)
        rpm = _rpm_from_sfm_diam(sfm, diam_in)
        ipm = _ipm_from_rpm_ipr(rpm, ipr)
        depth_txt = "—" if depth_in is None else f'{depth_in:.2f}"'
        rpm_txt = "-" if rpm is None else str(int(rpm))
        ipm_txt = "-" if ipm is None else f"{ipm:.3f}"
        lines.append(
            f'Ø{diam_in:.4f}" × {qty}  ({side}) | depth {depth_txt} | {rpm_txt} rpm | '
            f"{ipm_txt} ipm | t/hole — | group — "
        )
    lines.append("")


def _emit_spot_and_jig_cards(
    lines: list[str],
    *,
    geo: Mapping[str, Any] | None,
    material_group: str | None,
    speeds_csv: dict | None,
    result: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
) -> None:
    rows = _rows_from_ops_summary(geo, result=result, breakdown=breakdown)
    spot_qty = 0
    spot_depth: float | None = None
    for r in rows:
        desc_upper = str(r.get("desc", "")).upper()
        if ("DRILL" in desc_upper and "C" in desc_upper) and ("THRU" not in desc_upper) and (
            "TAP" not in desc_upper
        ):
            depth_match = _DEPTH_TOKEN.search(desc_upper)
            if depth_match:
                try:
                    spot_depth = float(depth_match.group(1))
                except Exception:
                    spot_depth = None
            spot_qty += int(r.get("qty") or 0)
    jig_qty = sum(int(r.get("qty") or 0) for r in rows if "JIG GRIND" in str(r.get("desc", "")).upper())
    if spot_qty > 0:
        sfm, ipr = _lookup_sfm_ipr("spot", 0.1875, material_group, speeds_csv)
        rpm = _rpm_from_sfm_diam(sfm, 0.1875)
        ipm = _ipm_from_rpm_ipr(rpm, ipr)
        depth_txt = "—" if spot_depth is None else f'{spot_depth:.2f}"'
        rpm_txt = "-" if rpm is None else str(int(round(rpm)))
        ipm_txt = "-" if ipm is None else f"{ipm:.3f}"
        lines += [
            "MATERIAL REMOVAL – SPOT (CENTER DRILL)",
            "=" * 64,
            f"Spots .............. {spot_qty} (front-side unless noted)",
            "TIME PER HOLE – SPOT GROUPS",
            "-" * 66,
            f"Spot drill × {spot_qty} | depth {depth_txt} | {rpm_txt} rpm | {ipm_txt} ipm | t/hole — | group — ",
            "",
        ]
    if jig_qty > 0:
        lines += [
            "MATERIAL REMOVAL – JIG GRIND",
            "=" * 64,
            f"Jig-grind features . {jig_qty}",
            "TIME PER FEATURE",
            "-" * 66,
            f"Jig grind × {jig_qty} | t/feat — | group — ",
            "",
        ]


def _hole_table_section_present(lines: Sequence[str], header: str) -> bool:
    if not header:
        return False
    header_norm = header.strip().upper()
    for existing in lines:
        if isinstance(existing, str) and existing.strip().upper() == header_norm:
            return True
    return False


def _add_bucket_minutes(
    bucket_view: Mapping[str, Any] | MutableMapping[str, Any] | None,
    bucket_key: str,
    minutes: float,
    *,
    machine_rate: float = 0.0,
    labor_rate: float = 0.0,
    name: str = "",
) -> None:
    try:
        minutes_val = float(minutes)
    except Exception:
        minutes_val = 0.0
    if minutes_val <= 0.0:
        return

    if isinstance(bucket_view, dict):
        target_view = bucket_view
    elif isinstance(bucket_view, _MutableMappingABC):
        target_view = typing.cast(MutableMapping[str, Any], bucket_view)
    else:
        return

    buckets = target_view.setdefault(
        "buckets",
        {},
    )
    if not isinstance(buckets, dict):
        try:
            buckets = dict(buckets)  # type: ignore[arg-type]
        except Exception:
            buckets = {}
        target_view["buckets"] = buckets

    entry = buckets.setdefault(
        bucket_key,
        {"minutes": 0.0, "labor$": 0.0, "machine$": 0.0, "total$": 0.0},
    )
    try:
        entry_minutes = float(entry.get("minutes", 0.0))
    except Exception:
        entry_minutes = 0.0
    entry["minutes"] = entry_minutes + minutes_val

    hours = minutes_val / 60.0
    mach_rate = _safe_float(machine_rate, 0.0)
    lab_rate = _safe_float(labor_rate, 0.0)
    entry["machine$"] = float(entry.get("machine$", 0.0)) + hours * mach_rate
    entry["labor$"] = float(entry.get("labor$", 0.0)) + hours * lab_rate
    entry["total$"] = round(float(entry.get("machine$", 0.0)) + float(entry.get("labor$", 0.0)), 2)

    ops_map = target_view.setdefault("bucket_ops", {})
    if not isinstance(ops_map, dict):
        try:
            ops_map = dict(ops_map)  # type: ignore[arg-type]
        except Exception:
            ops_map = {}
        target_view["bucket_ops"] = ops_map
    ops_list = ops_map.setdefault(bucket_key, [])
    if isinstance(ops_list, list) and name:
        ops_list.append({"name": name, "minutes": minutes_val})


def _emit_hole_table_ops_cards(
    lines: list[str],
    *,
    geo: Mapping[str, Any] | None,
    material_group: str | None,
    speeds_csv: dict | None,
    result: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
    rates: Mapping[str, Any] | None = None,
) -> None:
    from appkit.ui.planner_render import (
        _hole_table_minutes_from_geo,
        _lookup_bucket_rate,
    )

    tap_minutes_hint, cbore_minutes_hint, spot_minutes_hint, jig_minutes_hint = (
        _hole_table_minutes_from_geo(geo)
    )

    bucket_view_obj: Mapping[str, Any] | MutableMapping[str, Any] | None = None
    for candidate in (breakdown, result):
        if isinstance(candidate, dict):
            bucket_view_obj = candidate.setdefault("bucket_view", {})
            break
        if isinstance(candidate, _MutableMappingABC):
            view = candidate.get("bucket_view")
            if not isinstance(view, dict):
                view = {}
                try:
                    candidate["bucket_view"] = view  # type: ignore[index]
                except Exception:
                    pass
            bucket_view_obj = typing.cast(MutableMapping[str, Any], view)
            break

    rates_map: Mapping[str, Any]
    if isinstance(breakdown, _MappingABC):
        rates_candidate = breakdown.get("rates")
        if isinstance(rates_candidate, _MappingABC):
            rates_map = rates_candidate
        elif isinstance(rates_candidate, dict):
            rates_map = rates_candidate
        else:
            rates_map = {}
    else:
        rates_map = {}
    if (not rates_map) and isinstance(rates, _MappingABC):
        rates_map = rates
    elif (not rates_map) and isinstance(rates, dict):
        rates_map = rates

    if not _hole_table_section_present(lines, "MATERIAL REMOVAL – TAPPING"):
        _emit_tapping_card(
            lines,
            geo=geo,
            material_group=material_group,
            speeds_csv=speeds_csv,
            result=result,
            breakdown=breakdown,
        )
        tap_labor_rate = _lookup_bucket_rate("tapping_labor", rates_map) or _lookup_bucket_rate(
            "labor",
            rates_map,
        )
        tap_machine_rate = _lookup_bucket_rate("tapping", rates_map) or 0.0
        _add_bucket_minutes(
            bucket_view_obj,
            "tapping",
            tap_minutes_hint,
            machine_rate=tap_machine_rate,
            labor_rate=tap_labor_rate,
            name="Tapping ops",
        )
    if not _hole_table_section_present(lines, "MATERIAL REMOVAL – COUNTERBORE"):
        _emit_counterbore_card(
            lines,
            geo=geo,
            material_group=material_group,
            speeds_csv=speeds_csv,
            result=result,
            breakdown=breakdown,
        )
        cbore_labor_rate = _lookup_bucket_rate("counterbore_labor", rates_map) or _lookup_bucket_rate(
            "labor",
            rates_map,
        )
        cbore_machine_rate = _lookup_bucket_rate("counterbore", rates_map) or _lookup_bucket_rate(
            "drilling",
            rates_map,
        )
        _add_bucket_minutes(
            bucket_view_obj,
            "counterbore",
            cbore_minutes_hint,
            machine_rate=cbore_machine_rate,
            labor_rate=cbore_labor_rate,
            name="Counterbore ops",
        )
    if not _hole_table_section_present(lines, "MATERIAL REMOVAL – SPOT (CENTER DRILL)"):
        _emit_spot_and_jig_cards(
            lines,
            geo=geo,
            material_group=material_group,
            speeds_csv=speeds_csv,
            result=result,
            breakdown=breakdown,
        )
        drill_labor_rate = _lookup_bucket_rate("drilling_labor", rates_map) or _lookup_bucket_rate(
            "labor",
            rates_map,
        )
        drill_machine_rate = _lookup_bucket_rate("drilling", rates_map) or 0.0
        _add_bucket_minutes(
            bucket_view_obj,
            "drilling",
            spot_minutes_hint,
            machine_rate=drill_machine_rate,
            labor_rate=drill_labor_rate,
            name="Spot drill ops",
        )
        grind_labor_rate = _lookup_bucket_rate("grinding_labor", rates_map) or _lookup_bucket_rate(
            "labor",
            rates_map,
        )
        grind_machine_rate = _lookup_bucket_rate("grinding", rates_map) or 0.0
        _add_bucket_minutes(
            bucket_view_obj,
            "grinding",
            jig_minutes_hint,
            machine_rate=grind_machine_rate,
            labor_rate=grind_labor_rate,
            name="Jig grind ops",
        )


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
    "_parse_hole_line",
    "_aggregate_hole_entries",
    "_build_ops_rows_from_lines_fallback",
    "build_ops_rows_from_lines_fallback",
    "summarize_hole_chart_lines",
    "_emit_tapping_card",
    "_emit_counterbore_card",
    "_emit_spot_and_jig_cards",
    "_hole_table_section_present",
    "_add_bucket_minutes",
    "_emit_hole_table_ops_cards",
]


def build_ops_rows_from_lines_fallback(lines: Iterable[str] | None) -> list[dict]:
    """Proxy to the chart-line fallback parser exposed for hole operations helpers."""

    seq = [str(s) for s in lines or [] if str(s)]
    if not seq:
        return []
    return _chart_build_ops_rows_from_lines_fallback(seq)


_build_ops_rows_from_lines_fallback = _chart_build_ops_rows_from_lines_fallback
