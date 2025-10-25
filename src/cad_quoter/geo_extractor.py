"""Isolated GEO extraction helpers for DWG/DXF sources."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from fractions import Fraction
import inspect
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Callable

from cad_quoter import geometry
from cad_quoter.geometry import convert_dwg_to_dxf
from cad_quoter.vendors import ezdxf as _ezdxf_vendor

_HAS_ODAFC = bool(getattr(geometry, "HAS_ODAFC", False))


@lru_cache(maxsize=1)
def _load_app_module():
    import importlib

    return importlib.import_module("appV5")


@lru_cache(maxsize=None)
def _resolve_app_callable(name: str) -> Callable[..., Any] | None:
    try:
        module = _load_app_module()
    except Exception:
        return None
    return getattr(module, name, None)


def _describe_helper(helper: Any) -> str:
    if helper is None:
        return "None"
    name = getattr(helper, "__name__", None)
    if isinstance(name, str):
        return name
    return repr(helper)


def _print_helper_debug(tag: str, helper: Any) -> None:
    try:
        helper_desc = _describe_helper(helper)
    except Exception:
        helper_desc = repr(helper)
    print(f"[EXTRACT] {tag} helper: {helper_desc}")


_ROW_START_RE = re.compile(r"\(\s*\d+\s*\)")
_DIAMETER_PREFIX_RE = re.compile(
    r"(?:Ø|⌀|DIA(?:\.\b|\b))\s*(\d+\s*/\s*\d+|\d*\.\d+|\.\d+|\d+)",
    re.IGNORECASE,
)
_DIAMETER_SUFFIX_RE = re.compile(
    r"(\d+\s*/\s*\d+|\d*\.\d+|\.\d+|\d+)\s*(?:Ø|⌀|DIA(?:\.\b|\b))",
    re.IGNORECASE,
)
_MTEXT_ALIGN_RE = re.compile(r"\\A\d;", re.IGNORECASE)
_MTEXT_BREAK_RE = re.compile(r"\\P", re.IGNORECASE)


def _score_table(info: Mapping[str, Any] | None) -> tuple[int, int]:
    if not isinstance(info, Mapping):
        return (0, 0)
    rows = info.get("rows") or []
    return (_sum_qty(rows), len(rows))


def _sum_qty(rows: Iterable[Mapping[str, Any]] | None) -> int:
    total = 0
    if not rows:
        return total
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        qty_val = row.get("qty")
        try:
            total += int(float(qty_val or 0))
        except Exception:
            continue
    return total


def read_acad_table(doc) -> dict[str, Any]:
    helper = _resolve_app_callable("hole_count_from_acad_table")
    _print_helper_debug("acad", helper)
    if callable(helper):
        try:
            result = helper(doc) or {}
        except Exception as exc:
            print(f"[EXTRACT] acad helper error: {exc}")
            raise
        if isinstance(result, Mapping):
            return dict(result)
        return {}
    return {}


def _collect_table_text_lines(doc: Any) -> list[str]:
    lines: list[str] = []
    if doc is None:
        return lines

    spaces: list[Any] = []
    modelspace = getattr(doc, "modelspace", None)
    if callable(modelspace):
        try:
            space = modelspace()
        except Exception:
            space = None
        if space is not None:
            spaces.append(space)

    for space in spaces:
        query = getattr(space, "query", None)
        if not callable(query):
            continue
        try:
            entities = list(query("TEXT, MTEXT"))
        except Exception:
            continue
        for entity in entities:
            raw_text = ""
            dxftype = None
            try:
                dxftype = entity.dxftype()
            except Exception:
                dxftype = None
            if dxftype == "MTEXT":
                plain_text = getattr(entity, "plain_text", None)
                if callable(plain_text):
                    try:
                        raw_text = plain_text()
                    except Exception:
                        raw_text = ""
                if not raw_text:
                    raw_text = getattr(entity, "text", "")
            elif dxftype == "TEXT":
                dxf_obj = getattr(entity, "dxf", None)
                raw_text = getattr(dxf_obj, "text", "") if dxf_obj is not None else ""
            else:
                raw_text = getattr(entity, "text", "")

            if not raw_text:
                continue
            for line in str(raw_text).splitlines():
                normalized = _normalize_table_fragment(line)
                if normalized:
                    lines.append(normalized)
    return lines


def _normalize_table_fragment(fragment: str) -> str:
    if not isinstance(fragment, str):
        fragment = str(fragment)
    cleaned = fragment.replace("%%C", "Ø").replace("%%c", "Ø")
    cleaned = _MTEXT_ALIGN_RE.sub("", cleaned)
    cleaned = _MTEXT_BREAK_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("|", " |")
    cleaned = cleaned.replace("\\~", "~")
    cleaned = cleaned.replace("\\`", "`")
    cleaned = cleaned.replace("\\", " ")
    return " ".join(cleaned.split())


def _parse_number_token(token: str) -> float | None:
    text = (token or "").strip()
    if not text:
        return None
    if "/" in text:
        try:
            return float(Fraction(text))
        except Exception:
            return None
    if text.startswith("."):
        text = "0" + text
    try:
        return float(text)
    except Exception:
        return None


def _merge_table_lines(lines: Iterable[str]) -> list[str]:
    merged: list[str] = []
    current: list[str] | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        has_row_start = bool(_ROW_START_RE.search(line))
        if has_row_start:
            if current:
                merged.append(" ".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        merged.append(" ".join(current))
    return merged


def _extract_diameter(text: str) -> float | None:
    search_space = text or ""
    match = _DIAMETER_PREFIX_RE.search(search_space)
    if not match:
        match = _DIAMETER_SUFFIX_RE.search(search_space)
    if not match:
        return None
    return _parse_number_token(match.group(1))


def _fallback_text_table(lines: Iterable[str]) -> dict[str, Any]:
    merged = _merge_table_lines(lines)
    rows: list[dict[str, Any]] = []
    families: dict[str, int] = {}
    total_qty = 0

    for entry in merged:
        qty_match = _ROW_START_RE.search(entry)
        if not qty_match:
            continue
        qty_text = qty_match.group(0).strip("() ")
        try:
            qty = int(qty_text)
        except Exception:
            continue
        prefix = entry[: qty_match.start()].strip()
        suffix = entry[qty_match.end() :].strip()
        combined = " ".join(part for part in (prefix, suffix) if part)
        combined = combined.replace("|", " ")
        desc = " ".join(combined.split())
        if not desc:
            continue
        rows.append({"hole": "", "ref": "", "qty": qty, "desc": desc})
        total_qty += qty

        diameter = _extract_diameter(prefix + " " + suffix)
        if diameter is not None:
            key = f"{diameter:.4f}".rstrip("0").rstrip(".")
            families[key] = families.get(key, 0) + qty

    if not rows:
        return {}

    result: dict[str, Any] = {"rows": rows, "hole_count": total_qty}
    if families:
        result["hole_diam_families_in"] = families
    result["provenance_holes"] = "HOLE TABLE (TEXT_FALLBACK)"
    return result


def read_text_table(doc) -> dict[str, Any]:
    helper = _resolve_app_callable("extract_hole_table_from_text")
    _print_helper_debug("text", helper)
    table_lines: list[str] | None = None
    fallback_candidate: Mapping[str, Any] | None = None

    def ensure_lines() -> list[str]:
        nonlocal table_lines
        if table_lines is None:
            table_lines = _collect_table_text_lines(doc)
        return table_lines

    if callable(helper):
        try:
            result = helper(doc) or {}
        except Exception as exc:
            print(f"[EXTRACT] text helper error: {exc}")
            raise
        if isinstance(result, Mapping):
            result_map = dict(result)
            if result_map.get("rows"):
                return result_map
            fallback_candidate = result_map
        else:
            fallback_candidate = {}

    legacy_helper = _resolve_app_callable("hole_count_from_text_table")
    _print_helper_debug("text_alt", legacy_helper)
    if callable(legacy_helper):
        needs_lines = False
        try:
            signature = inspect.signature(legacy_helper)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            required = [
                param
                for param in signature.parameters.values()
                if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
                and param.default is param.empty
            ]
            needs_lines = len(required) >= 2

        try:
            if needs_lines:
                lines = ensure_lines()
                result = legacy_helper(doc, lines) or {}
            else:
                result = legacy_helper(doc) or {}
        except TypeError as exc:
            print(f"[EXTRACT] text helper error: {exc}")
            lines = ensure_lines()
            result = legacy_helper(doc, lines) or {}
        except Exception as exc:
            print(f"[EXTRACT] text helper error: {exc}")
            raise

        if isinstance(result, Mapping):
            result_map = dict(result)
            if result_map.get("rows"):
                return result_map
            if fallback_candidate is None:
                fallback_candidate = result_map
    else:
        if fallback_candidate is None:
            fallback_candidate = {}

    lines = ensure_lines()
    fallback = _fallback_text_table(lines)
    if fallback:
        return fallback

    if isinstance(fallback_candidate, Mapping):
        return dict(fallback_candidate)
    return {}


def choose_better_table(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> Mapping[str, Any]:
    helper = _resolve_app_callable("_choose_better")
    if callable(helper):
        try:
            chosen = helper(a, b)
        except Exception:
            chosen = None
        if isinstance(chosen, Mapping):
            return chosen
        if isinstance(chosen, list):
            return {"rows": list(chosen)}
    score_a = _score_table(a)
    score_b = _score_table(b)
    candidate = a if score_a >= score_b else b
    if isinstance(candidate, Mapping):
        return candidate
    return {}


def promote_table_to_geo(geo: dict[str, Any], table_info: Mapping[str, Any], source_tag: str) -> None:
    helper = _resolve_app_callable("_persist_rows_and_totals")
    if callable(helper):
        try:
            helper(geo, table_info, src=source_tag)
            return
        except Exception:
            pass
    if not isinstance(table_info, Mapping):
        return

    raw_rows = table_info.get("rows")
    if not isinstance(raw_rows, Iterable):
        return

    normalized_rows: list[dict[str, Any]] = []
    for entry in raw_rows:
        if not isinstance(entry, Mapping):
            continue
        normalized = dict(entry)
        normalized["hole"] = str(entry.get("hole") or "")
        normalized["ref"] = str(entry.get("ref") or "")
        normalized["desc"] = str(entry.get("desc") or "")
        try:
            normalized["qty"] = int(float(entry.get("qty") or 0))
        except Exception:
            normalized["qty"] = 0
        normalized_rows.append(normalized)

    normalized_rows = [row for row in normalized_rows if int(row.get("qty", 0) or 0) > 0]
    if not normalized_rows:
        return

    def _detect_sides(text: str) -> set[str]:
        if not text:
            return {"front"}
        upper = text.upper()
        if any(
            token in upper
            for token in (
                "FRONT & BACK",
                "FRONT AND BACK",
                "FRONT/BACK",
                "BOTH SIDES",
                "DOUBLE SIDED",
                "DOUBLE-SIDED",
            )
        ):
            return {"front", "back"}
        has_front = "FRONT" in upper
        has_back = any(
            marker in upper
            for marker in ("BACK", "BACKSIDE", "FROM BACK", "BACK SIDE", "BACK-", "(BACK)")
        )
        if has_front and has_back:
            return {"front", "back"}
        if has_back:
            return {"back"}
        return {"front"}

    def _increment_with_sides(base_key: str, qty: int, sides: set[str]) -> None:
        if qty <= 0:
            return
        totals[base_key] += qty
        if "front" in sides:
            totals[f"{base_key}_front"] += qty
        if "back" in sides:
            totals[f"{base_key}_back"] += qty

    totals = defaultdict(int)
    for row in normalized_rows:
        qty = int(row.get("qty") or 0)
        if qty <= 0:
            continue
        desc_upper = row.get("desc", "").upper()
        sides = _detect_sides(desc_upper)
        has_tap = "TAP" in desc_upper
        has_cbore = any(
            marker in desc_upper
            for marker in ("CBORE", "C'BORE", "C’BORE", "COUNTERBORE", "COUNTER BORE", "C-BORE", "C BORE")
        )
        has_csk = any(marker in desc_upper for marker in ("C'SINK", "C’SINK", "CSK", "COUNTERSINK"))
        has_counterdrill = any(
            marker in desc_upper
            for marker in (
                "COUNTERDRILL",
                "COUNTER DRILL",
                "C DRILL",
                "C-DRILL",
                "C'DRILL",
                "C’DRILL",
                "CENTER DRILL",
            )
        )
        has_spot = (
            "SPOT" in desc_upper
            and "SPOTFACE" not in desc_upper
            and "SPOT FACE" not in desc_upper
        ) or has_counterdrill
        has_drill = (
            (" DRILL" in desc_upper or desc_upper.startswith("DRILL"))
            and not has_counterdrill
        )
        has_thru = "THRU" in desc_upper or "THROUGH" in desc_upper

        if has_tap:
            _increment_with_sides("tap", qty, sides)
            totals["drill"] += qty
        elif has_drill or (has_thru and not has_counterdrill):
            totals["drill"] += qty

        if has_cbore:
            _increment_with_sides("cbore", qty, sides)

        if has_csk:
            _increment_with_sides("csk", qty, sides)

        if has_counterdrill:
            _increment_with_sides("counterdrill", qty, sides)

        if has_spot:
            _increment_with_sides("spot", qty, sides)

        if "JIG GRIND" in desc_upper:
            totals["jig_grind"] += qty

    if totals:
        totals["tap_total"] = totals.get("tap_front", 0) + totals.get("tap_back", 0)
        totals["cbore_total"] = totals.get("cbore_front", 0) + totals.get("cbore_back", 0)
        totals["csk_total"] = totals.get("csk_front", 0) + totals.get("csk_back", 0)
        totals["spot_total"] = totals.get("spot_front", 0) + totals.get("spot_back", 0)
        totals["counterdrill_total"] = (
            totals.get("counterdrill_front", 0) + totals.get("counterdrill_back", 0)
        )
        totals_dict = {key: value for key, value in totals.items() if value > 0}
    else:
        totals_dict = {}

    ops_summary = geo.setdefault("ops_summary", {})
    ops_summary["rows"] = normalized_rows
    source_normalized = "acad_table" if source_tag == "acad_table" else "text_table"
    ops_summary["source"] = source_normalized
    if totals_dict:
        ops_summary["totals"] = totals_dict

    hole_candidates = (
        table_info.get("table_total"),
        table_info.get("hole_count"),
        _sum_qty(normalized_rows),
    )
    for candidate in hole_candidates:
        try:
            hole_count = int(float(candidate))
        except Exception:
            continue
        if hole_count > 0:
            geo["hole_count"] = hole_count
            break

    provenance = geo.setdefault("provenance", {})
    provenance["holes"] = "HOLE TABLE"


def extract_geometry(doc) -> dict[str, Any]:
    helper = _resolve_app_callable("_build_geo_from_ezdxf_doc")
    if callable(helper):
        try:
            geo = helper(doc)
        except Exception:
            geo = None
        if isinstance(geo, Mapping):
            return dict(geo)
    return {}


def _load_doc_for_path(path: Path, *, use_oda: bool) -> Any:
    ezdxf_mod = geometry.require_ezdxf()
    readfile = getattr(ezdxf_mod, "readfile", None)
    if not callable(readfile):
        raise AttributeError("ezdxf module does not expose a callable readfile")
    lower_suffix = path.suffix.lower()
    if lower_suffix == ".dwg":
        if use_oda and _HAS_ODAFC:
            odafc_mod = None
            try:
                odafc_mod = _ezdxf_vendor.require_odafc()
            except Exception:
                odafc_mod = None
            if odafc_mod is not None:
                odaread = getattr(odafc_mod, "readfile", None)
                if callable(odaread):
                    return odaread(str(path))
        dxf_path = convert_dwg_to_dxf(str(path))
        return readfile(dxf_path)
    return readfile(str(path))


def _ensure_ops_summary_map(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    return {}


def _best_geo_hole_count(geo: Mapping[str, Any]) -> int | None:
    for key in ("hole_count", "hole_count_geom", "hole_count_geom_dedup", "hole_count_geom_raw"):
        value = geo.get(key) if isinstance(geo, Mapping) else None
        try:
            val_int = int(float(value))
        except Exception:
            val_int = 0
        if val_int > 0:
            return val_int
    return None


def extract_geo_from_path(
    path: str,
    *,
    prefer_table: bool = True,
    use_oda: bool = True,
    feature_flags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load DWG/DXF at ``path`` and return a GEO dictionary."""

    del feature_flags  # placeholder for future feature toggles
    path_obj = Path(path)
    try:
        doc = _load_doc_for_path(path_obj, use_oda=use_oda)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[EXTRACT] failed to load document: {exc}")
        return {"error": str(exc)}

    geo = extract_geometry(doc)
    if not isinstance(geo, dict):
        geo = {}

    existing_ops_summary = geo.get("ops_summary") if isinstance(geo, Mapping) else {}
    provenance = geo.get("provenance") if isinstance(geo, Mapping) else {}
    provenance_holes = None
    if isinstance(provenance, Mapping):
        provenance_holes = provenance.get("holes")
    existing_source = ""
    if isinstance(existing_ops_summary, Mapping):
        existing_source = str(existing_ops_summary.get("source") or "")
    existing_is_table = bool(
        (existing_source and "table" in existing_source.lower())
        or (isinstance(provenance_holes, str) and provenance_holes.upper() == "HOLE TABLE")
    )
    if existing_is_table and isinstance(existing_ops_summary, Mapping):
        current_table_info = dict(existing_ops_summary)
        rows = current_table_info.get("rows")
        if isinstance(rows, Iterable) and not isinstance(rows, list):
            current_table_info["rows"] = list(rows)
    else:
        current_table_info = {}

    try:
        acad_info = read_acad_table(doc) or {}
    except Exception:
        acad_info = {}
    try:
        text_info = read_text_table(doc) or {}
    except Exception:
        text_info = {}

    acad_rows = len((acad_info.get("rows") or [])) if isinstance(acad_info, Mapping) else 0
    text_rows = len((text_info.get("rows") or [])) if isinstance(text_info, Mapping) else 0
    print(f"[EXTRACT] acad_rows={acad_rows} text_rows={text_rows}")

    best_table = choose_better_table(acad_info, text_info)
    score_a = _score_table(acad_info)
    score_b = _score_table(text_info)
    table_used = False
    source_tag = None
    existing_score = _score_table(current_table_info)
    best_score = _score_table(best_table)
    if isinstance(best_table, Mapping) and best_table.get("rows") and best_score > existing_score:
        source_tag = "acad_table" if score_a >= score_b else "text_table"
        promote_table_to_geo(geo, best_table, source_tag)
        table_used = True

    ops_summary = _ensure_ops_summary_map(geo.get("ops_summary"))
    geo["ops_summary"] = ops_summary
    rows = ops_summary.get("rows")
    if not isinstance(rows, list):
        if isinstance(rows, Iterable):
            rows = list(rows)
        else:
            rows = []
    if not table_used and existing_is_table:
        table_used = bool(rows)
    if table_used:
        qty_sum = _sum_qty(rows)
    else:
        if rows:
            ops_summary.pop("rows", None)
            rows = []
        qty_sum = 0
        ops_summary["source"] = "geom"
    if not table_used:
        hole_count = _best_geo_hole_count(geo)
        if hole_count:
            geo["hole_count"] = hole_count

    if table_used and source_tag:
        ops_summary["source"] = source_tag
    totals = ops_summary.get("totals")
    if isinstance(totals, Mapping):
        ops_summary["totals"] = dict(totals)

    rows_for_log = rows
    if table_used:
        rows_for_log = ops_summary.get("rows") or []
        if not isinstance(rows_for_log, list) and isinstance(rows_for_log, Iterable):
            rows_for_log = list(rows_for_log)
        qty_sum = _sum_qty(rows_for_log)
    provenance_holes = None
    provenance = geo.get("provenance")
    if isinstance(provenance, Mapping):
        provenance_holes = provenance.get("holes")

    hole_count_value = geo.get("hole_count")
    hole_count_repr = ""
    if hole_count_value is not None:
        try:
            hole_count_repr = str(int(float(hole_count_value)))
        except Exception:
            hole_count_repr = str(hole_count_value)

    details = [
        f"rows={len(rows_for_log)}",
        f"qty_sum={qty_sum}",
        f"source={ops_summary.get('source')}",
    ]
    if hole_count_repr:
        details.append(f"hole_count={hole_count_repr}")
    if provenance_holes:
        details.append(f"provenance={provenance_holes}")
    print("[EXTRACT] published " + " ".join(details))

    return geo


__all__ = [
    "extract_geo_from_path",
    "read_acad_table",
    "read_text_table",
    "choose_better_table",
    "promote_table_to_geo",
    "extract_geometry",
]