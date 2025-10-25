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


def _safe_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _doc_auditor_status(doc: Any) -> str | None:
    audit = getattr(doc, "audit", None)
    if not callable(audit):
        return None
    try:
        auditor = audit()
    except Exception as exc:  # pragma: no cover - defensive logging
        return f"error={exc}"
    errors = len(getattr(auditor, "errors", []) or [])
    warnings = len(getattr(auditor, "warnings", []) or [])
    fixes = len(getattr(auditor, "fixes", []) or [])
    return f"errors={errors} warnings={warnings} fixes={fixes}"


def _layout_entity_counts(doc: Any) -> dict[str, int]:
    totals: dict[str, int] = {"TABLE": 0, "MTEXT": 0, "TEXT": 0, "INSERT": 0}
    layouts = getattr(doc, "layouts", None)
    if layouts is None:
        return totals
    try:
        layout_iterable = list(layouts)
    except Exception:  # pragma: no cover - defensive logging
        try:
            layout_iterable = list(iter(layouts))
        except Exception:
            return totals
    if not layout_iterable:
        return totals
    for layout in layout_iterable:
        layout_name = getattr(layout, "name", None)
        if not layout_name:
            dxf_obj = getattr(layout, "dxf", None)
            layout_name = getattr(dxf_obj, "name", None)
        if not layout_name:
            layout_name = str(layout)
        counts: list[str] = []
        for entity_type in ("TABLE", "MTEXT", "TEXT", "INSERT"):
            query = getattr(layout, "query", None)
            count = 0
            if callable(query):
                try:
                    results = query(entity_type)
                except Exception:
                    results = []
                try:
                    count = len(results)
                except TypeError:
                    try:
                        count = sum(1 for _ in results)
                    except Exception:
                        count = 0
            totals[entity_type] = totals.get(entity_type, 0) + count
            counts.append(f"{entity_type}={count}")
        print(f"[LAYOUT] {layout_name}: {' '.join(counts)}")
    return totals


def _print_doc_open_diagnostics(doc: Any, path: Path) -> dict[str, int]:
    size = _safe_file_size(path)
    line = f"[OPEN] file={path} size={size if size is not None else 'unknown'}"
    auditor_status = _doc_auditor_status(doc)
    if auditor_status:
        line += f" auditor={auditor_status}"
    print(line)
    return _layout_entity_counts(doc)


def _should_attempt_dwg_conversion(counts: Mapping[str, int] | None) -> bool:
    if not isinstance(counts, Mapping):
        return False
    return sum(int(counts.get(key, 0)) for key in ("TABLE", "MTEXT", "TEXT")) == 0


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
    rows = table_info.get("rows") or []
    if not rows:
        return
    ops_summary = geo.setdefault("ops_summary", {})
    ops_summary["rows"] = list(rows)
    ops_summary["source"] = source_tag
    totals = defaultdict(int)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            qty = int(float(row.get("qty") or 0))
        except Exception:
            qty = 0
        desc = str(row.get("desc") or "").upper()
        if qty <= 0:
            continue
        if "TAP" in desc:
            totals["tap"] += qty
            totals["drill"] += qty
            if "BACK" in desc and "FRONT" not in desc:
                totals["tap_back"] += qty
            elif "FRONT" in desc and "BACK" in desc:
                totals["tap_front"] += qty
                totals["tap_back"] += qty
            else:
                totals["tap_front"] += qty
        if any(marker in desc for marker in ("CBORE", "COUNTERBORE", "C'BORE")):
            totals["counterbore"] += qty
            if "BACK" in desc and "FRONT" not in desc:
                totals["counterbore_back"] += qty
            elif "FRONT" in desc and "BACK" in desc:
                totals["counterbore_front"] += qty
                totals["counterbore_back"] += qty
            else:
                totals["counterbore_front"] += qty
        if "JIG GRIND" in desc:
            totals["jig_grind"] += qty
        if (
            "SPOT" in desc
            or "CENTER DRILL" in desc
            or "C DRILL" in desc
            or "C’DRILL" in desc
        ) and "TAP" not in desc and "THRU" not in desc:
            totals["spot"] += qty
    if totals:
        ops_summary["totals"] = dict(totals)
    hole_count = table_info.get("hole_count")
    if hole_count is None:
        hole_count = _sum_qty(rows)
    try:
        geo["hole_count"] = int(hole_count)
    except Exception:
        pass
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
        doc = None
        counts: Mapping[str, int] | None = None
        if use_oda and _HAS_ODAFC:
            odafc_mod = None
            try:
                odafc_mod = _ezdxf_vendor.require_odafc()
            except Exception:
                odafc_mod = None
            if odafc_mod is not None:
                odaread = getattr(odafc_mod, "readfile", None)
                if callable(odaread):
                    doc = odaread(str(path))
                    counts = _print_doc_open_diagnostics(doc, path)
                    if _should_attempt_dwg_conversion(counts):
                        print("[OPEN] No entities in DWG via ezdxf; attempting DWG→DXF conversion")
                        doc = None
        if doc is None:
            dxf_path = Path(convert_dwg_to_dxf(str(path)))
            doc = readfile(str(dxf_path))
            _print_doc_open_diagnostics(doc, dxf_path)
        return doc
    doc = readfile(str(path))
    _print_doc_open_diagnostics(doc, path)
    return doc


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
    print(
        f"[EXTRACT] published rows={len(rows_for_log)} qty_sum={qty_sum} "
        f"source={ops_summary.get('source')}"
    )
    provenance_holes = None
    provenance = geo.get("provenance")
    if isinstance(provenance, Mapping):
        provenance_holes = provenance.get("holes")
    print(f"[EXTRACT] provenance={provenance_holes}")

    return geo


__all__ = [
    "extract_geo_from_path",
    "read_acad_table",
    "read_text_table",
    "choose_better_table",
    "promote_table_to_geo",
    "extract_geometry",
]