from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cad_quoter import geo_extractor
from cad_quoter.geo_extractor import extract_geo_from_path

DEFAULT_SAMPLE_PATH = REPO_ROOT / "Cad Files" / "301_redacted.dwg"


def _sum_qty(rows: list[Mapping[str, object]] | None) -> int:
    total = 0
    if not rows:
        return total
    for row in rows:
        try:
            total += int(float(row.get("qty", 0) or 0))  # type: ignore[arg-type]
        except Exception:
            continue
    return total


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump GEO operations summary from a DXF/DWG file")
    parser.add_argument("path", nargs="?", help="Path to the DXF or DWG file")
    parser.add_argument("--no-oda", dest="use_oda", action="store_false", help="Disable ODA fallback")
    parser.add_argument("--debug", action="store_true", help="Print the first 10 rows for inspection")
    parser.add_argument(
        "--show-helpers",
        action="store_true",
        help="Print helper resolution diagnostics",
    )
    parser.add_argument(
        "--debug-entities",
        action="store_true",
        help="Print layout entity counts and table/text scans",
    )
    args = parser.parse_args(argv)

    path = (args.path or os.environ.get("GEO_DUMP_PATH") or "").strip()
    if not path:
        path = str(DEFAULT_SAMPLE_PATH)
        print(f"[geo_dump] Using default sample: {path}")

    if args.show_helpers:
        try:
            app_module = importlib.import_module("appV5")
        except Exception as exc:
            print(f"[geo_dump] appV5 import failed: {exc}")
        else:
            module_path = getattr(app_module, "__file__", None)
            print(f"[geo_dump] appV5 import ok: {module_path or app_module}")
        acad_helper = geo_extractor._resolve_app_callable("hole_count_from_acad_table")
        text_helper = geo_extractor._resolve_app_callable("extract_hole_table_from_text")
        text_alt_helper = geo_extractor._resolve_app_callable("hole_count_from_text_table")
        print(
            "helpers: acad={acad} text={text} text_alt={text_alt}".format(
                acad=geo_extractor._describe_helper(acad_helper),
                text=geo_extractor._describe_helper(text_helper),
                text_alt=geo_extractor._describe_helper(text_alt_helper),
            )
        )

    layout_counts: list[dict[str, Any]] | None = None
    table_scan: list[dict[str, Any]] | None = None
    text_scan: dict[str, list[str]] | None = None
    if args.debug_entities:
        layout_counts, table_scan, text_scan = _collect_entity_debug(Path(path), use_oda=args.use_oda)

    geo = extract_geo_from_path(path, use_oda=args.use_oda)
    ops_summary = geo.get("ops_summary") if isinstance(geo, Mapping) else {}
    if not isinstance(ops_summary, Mapping):
        ops_summary = {}
    rows = ops_summary.get("rows") if isinstance(ops_summary, Mapping) else []
    if not isinstance(rows, list):
        rows = list(rows or [])  # type: ignore[arg-type]
    qty_sum = _sum_qty(rows)
    provenance = geo.get("provenance") if isinstance(geo, Mapping) else {}
    holes_source = None
    if isinstance(provenance, Mapping):
        holes_source = provenance.get("holes")
    print(
        "rows={rows} qty_sum={qty} source={src} hole_count={hole_count} provenance={prov}".format(
            rows=len(rows),
            qty=qty_sum,
            src=ops_summary.get("source"),
            hole_count=geo.get("hole_count"),
            prov=holes_source,
        )
    )
    if args.debug_entities:
        if layout_counts is not None:
            print(f"[ENTITIES] layout_counts={layout_counts}")
        if table_scan is not None:
            print(f"[ENTITIES] table_scan={table_scan}")
        if text_scan is not None:
            print(f"[ENTITIES] text_scan={text_scan}")
    if args.debug and rows:
        print("first_rows:")
        for idx, row in enumerate(rows[:10]):
            ref = row.get("ref") if isinstance(row, Mapping) else ""
            desc = row.get("desc") if isinstance(row, Mapping) else ""
            qty = row.get("qty") if isinstance(row, Mapping) else ""
            hole = row.get("hole") if isinstance(row, Mapping) else ""
            print(
                f"  [{idx:02d}] QTY={qty} REF={ref} SIDE={row.get('side') if isinstance(row, Mapping) else ''} DESC={desc} HOLE={hole}"
            )

    if args.debug_entities and not rows:
        _maybe_write_entities_report(Path(path), layout_counts, table_scan, text_scan)

    return 0


def _collect_entity_debug(
    path: Path, *, use_oda: bool
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None, dict[str, list[str]] | None]:
    try:
        loader = getattr(geo_extractor, "_load_doc_for_path")
    except AttributeError:
        return (None, None, None)

    try:
        doc = loader(path, use_oda=use_oda)
    except Exception as exc:
        print(f"[geo_dump] failed to load doc for entity debug: {exc}")
        return (None, None, None)

    layout_counts = _gather_layout_counts(doc)
    table_scan = _gather_table_scan(doc)
    text_scan = _gather_text_scan(doc)
    return (layout_counts, table_scan, text_scan)


def _gather_layout_counts(doc: Any) -> list[dict[str, Any]]:
    counts: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    def _process_space(space: Any, label: str) -> None:
        if space is None:
            return
        key = id(space)
        if key in seen_ids:
            return
        seen_ids.add(key)
        type_counts: Counter[str] = Counter()
        total = 0
        try:
            iterator = iter(space)
        except TypeError:
            iterator = iter(())
        for entity in iterator:
            total += 1
            kind = ""
            dxftype = getattr(entity, "dxftype", None)
            if callable(dxftype):
                try:
                    kind = str(dxftype())
                except Exception:
                    kind = ""
            if not kind:
                kind = entity.__class__.__name__
            type_counts[kind] += 1
        counts.append(
            {
                "layout": label,
                "total": total,
                "types": {name: type_counts[name] for name in sorted(type_counts)},
            }
        )

    modelspace = getattr(doc, "modelspace", None)
    if callable(modelspace):
        try:
            ms = modelspace()
        except Exception:
            ms = None
        _process_space(ms, "Model")

    layouts = getattr(doc, "layouts", None)
    layout_iter = None
    if layouts is not None:
        layout_iter = getattr(layouts, "names_in_taborder", None)
    layout_names: list[str] = []
    if callable(layout_iter):
        try:
            layout_names = list(layout_iter())
        except Exception:
            layout_names = []
    for name in layout_names:
        if name.lower() in {"model", "defpoints"}:
            continue
        layout_obj = None
        if layouts is not None:
            getter = getattr(layouts, "get", None)
            if callable(getter):
                try:
                    layout_obj = getter(name)
                except Exception:
                    layout_obj = None
        entity_space = getattr(layout_obj, "entity_space", None)
        _process_space(entity_space, name)
    return counts


def _gather_table_scan(doc: Any) -> list[dict[str, Any]] | None:
    try:
        from cad_quoter.geometry import iter_table_entities
    except Exception:
        return None

    if iter_table_entities is None:
        return None

    summary: list[dict[str, Any]] = []
    for entity in iter_table_entities(doc):
        handle = None
        rows = None
        cols = None
        try:
            dxf_obj = getattr(entity, "dxf", None)
            handle = getattr(dxf_obj, "handle", None)
            rows = getattr(dxf_obj, "n_rows", None)
            cols = getattr(dxf_obj, "n_cols", None)
        except Exception:
            handle = None
        summary.append({"handle": handle, "rows": rows, "cols": cols})
    return summary


def _gather_text_scan(doc: Any) -> dict[str, list[str]] | None:
    try:
        from cad_quoter.geometry import iter_table_text
    except Exception:
        iter_table_text = None  # type: ignore[assignment]

    table_text: list[str] = []
    if iter_table_text is not None:
        try:
            table_text = list(iter_table_text(doc))
        except Exception:
            table_text = []

    try:
        fallback_lines = geo_extractor._collect_table_text_lines(doc)
    except Exception:
        fallback_lines = []

    if not table_text and not fallback_lines:
        return None
    return {"table_text": table_text, "fallback_text": fallback_lines}


def _maybe_write_entities_report(
    path: Path,
    layout_counts: list[dict[str, Any]] | None,
    table_scan: list[dict[str, Any]] | None,
    text_scan: dict[str, list[str]] | None,
) -> None:
    report_path = path.with_name("entities_report.txt")
    try:
        with report_path.open("w", encoding="utf-8") as handle:
            handle.write(f"source={path}\n")
            handle.write(f"layout_counts={layout_counts}\n")
            handle.write(f"table_scan={table_scan}\n")
            handle.write(f"text_scan={text_scan}\n")
    except Exception as exc:
        print(f"[geo_dump] failed to write entities_report.txt: {exc}")
    else:
        print(f"[geo_dump] wrote entity debug report: {report_path}")

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())