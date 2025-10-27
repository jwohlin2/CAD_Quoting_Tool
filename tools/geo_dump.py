from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cad_quoter import geo_extractor
from cad_quoter.geo_extractor import read_geo

DEFAULT_SAMPLE_PATH = REPO_ROOT / "Cad Files" / "301_redacted.dwg"
ARTIFACT_DIR = REPO_ROOT / "out"


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


def _payload_has_rows(payload: Mapping[str, object] | None) -> bool:
    """Return ``True`` when the GEO payload already published any rows."""

    if not isinstance(payload, Mapping):
        return False

    def _extract_rows(container: Mapping[str, object], key: str) -> list[object]:
        value = container.get(key) if isinstance(container, Mapping) else None
        if isinstance(value, list):
            return value
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            rows_list = list(value)
            if isinstance(container, dict):
                container[key] = rows_list
            return rows_list
        return []

    direct_rows = _extract_rows(payload, "rows")
    if direct_rows:
        return True

    ops_summary = payload.get("ops_summary")
    if not isinstance(ops_summary, Mapping):
        geo = payload.get("geo")
        if isinstance(geo, Mapping):
            ops_summary = geo.get("ops_summary")
    if isinstance(ops_summary, Mapping):
        if not isinstance(ops_summary, dict):
            ops_summary = dict(ops_summary)
            if isinstance(payload, dict):
                payload["ops_summary"] = ops_summary
        if _extract_rows(ops_summary, "rows"):
            return True

    return False


def _ordered_hole_row(row: Mapping[str, object]) -> dict[str, object]:
    ordered: dict[str, object] = {}
    preferred_order = ("hole", "qty", "ref", "side", "desc")
    for key in preferred_order:
        if key in row:
            ordered[key] = row[key]
    for key in sorted(row.keys()):
        if key not in ordered:
            ordered[key] = row[key]
    return ordered


def _build_hole_rows_artifact(
    rows: Iterable[Mapping[str, object]] | None,
    *,
    qty_sum: int,
    hole_count: int | None,
    provenance: object,
    source: object,
) -> dict[str, object]:
    serialized_rows: list[dict[str, object]] = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        serialized_rows.append(_ordered_hole_row(row))

    artifact: dict[str, object] = {"rows": serialized_rows, "qty_sum": int(qty_sum)}
    if hole_count not in (None, ""):
        try:
            artifact["hole_count"] = int(float(hole_count))
        except Exception:
            artifact["hole_count"] = hole_count
    if provenance not in (None, ""):
        artifact["provenance"] = provenance
    if source not in (None, ""):
        artifact["source"] = source
    return artifact


def _ordered_totals_map(totals: Mapping[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    preferred = (
        "tap",
        "tap_front",
        "tap_back",
        "counterbore",
        "counterbore_front",
        "counterbore_back",
        "drill",
        "spot",
        "jig_grind",
    )
    for key in preferred:
        if key in totals:
            ordered[key] = totals[key]
    for key in sorted(totals.keys()):
        if key not in ordered:
            ordered[key] = totals[key]
    return ordered


def _build_ops_totals_artifact(ops_summary: Mapping[str, object] | None) -> dict[str, object] | None:
    if not isinstance(ops_summary, Mapping):
        return None

    totals_raw = ops_summary.get("totals") if isinstance(ops_summary, Mapping) else None
    totals_ordered = _ordered_totals_map(totals_raw) if isinstance(totals_raw, Mapping) else None

    artifact: dict[str, object] = {}
    if totals_ordered:
        artifact["totals"] = totals_ordered

    supplemental_keys = (
        "tap_total",
        "cbore_total",
        "csk_total",
        "actions_total",
        "back_ops_total",
        "flip_required",
    )
    for key in supplemental_keys:
        if key in ops_summary:
            artifact[key] = ops_summary[key]

    if "source" in ops_summary:
        artifact["source"] = ops_summary["source"]

    return artifact or None


def _write_artifact(path: Path, payload: Mapping[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    except OSError as exc:  # pragma: no cover - filesystem issues
        print(f"[geo_dump] failed to write artifact {path}: {exc}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump GEO operations summary from a DXF/DWG file")
    parser.add_argument("path", nargs="?", help="Path to the DXF or DWG file")
    parser.add_argument("--no-oda", dest="use_oda", action="store_false", help="Disable ODA fallback")
    parser.add_argument("--debug", action="store_true", help="Print the first 10 rows for inspection")
    parser.add_argument(
        "--debug-entities",
        action="store_true",
        help="Print raw text table candidates from the DXF/DWG",
    )
    parser.add_argument(
        "--show-helpers",
        action="store_true",
        help="Print helper resolution diagnostics",
    )
    parser.add_argument(
        "--dump-lines",
        help="Write text-line debug dump to this path (banded cells use *_bands.tsv)",
    )
    parser.add_argument(
        "--dump-bands",
        action="store_true",
        help="Print reconstructed [TABLE-X] band previews (first 30)",
    )
    parser.add_argument(
        "--all-layouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process text tables from all layouts (use --no-all-layouts to restrict)",
    )
    parser.add_argument(
        "--layouts",
        dest="layouts",
        action="append",
        help="Regex pattern to match layout names (repeatable; case-insensitive)",
    )
    parser.add_argument(
        "--layer-allow",
        dest="layer_allow",
        action="append",
        help="Restrict table/text scans to the specified layer (repeatable; use ALL to disable filtering)",
    )
    parser.add_argument(
        "--allow-layers",
        dest="allow_layers",
        help="Comma-separated glob patterns of layers to allow (use ALL to disable filtering)",
    )
    parser.add_argument(
        "--block-allow",
        dest="block_allow",
        action="append",
        help="Treat INSERTs with these block names as preferred ROI seeds (repeatable)",
    )
    parser.add_argument(
        "--block-regex",
        dest="block_regex",
        action="append",
        help="Regex pattern to match INSERT block names for ROI seeding (repeatable)",
    )
    parser.add_argument(
        "--include-layer",
        dest="include_layer",
        action="append",
        help="Regex pattern for layers to include when scanning text (repeatable)",
    )
    parser.add_argument(
        "--exclude-layer",
        dest="exclude_layer",
        action="append",
        help="Regex pattern for layers to exclude when scanning text (repeatable)",
    )
    parser.add_argument(
        "--scan-acad-tables",
        action="store_true",
        help="Print ACAD_TABLE inventory details",
    )
    parser.add_argument(
        "--trace-acad",
        action="store_true",
        help="Enable detailed AutoCAD table tracing diagnostics",
    )
    parser.add_argument(
        "--depth-max",
        type=int,
        metavar="N",
        help="Override block INSERT recursion depth for ACAD tables",
    )
    parser.add_argument(
        "--show-rows",
        type=int,
        metavar="N",
        help="Print the first N rows as qty | ref | side | desc",
    )
    parser.add_argument(
        "--force-text",
        action="store_true",
        help="Force publishing text fallback rows when available",
    )
    parser.add_argument(
        "--debug-layouts",
        action="store_true",
        help="Print layout and layer summaries after extraction",
    )
    parser.add_argument(
        "--dump-rows-csv",
        nargs="?",
        const="debug/rows.csv",
        default=None,
        help="Write extracted rows to CSV (optional custom path; default debug/rows.csv)",
    )
    args = parser.parse_args(argv)

    if args.show_rows is not None:
        os.environ["CAD_QUOTER_SHOW_ROWS"] = str(args.show_rows)

    path = (args.path or os.environ.get("GEO_DUMP_PATH") or "").strip()
    if not path:
        path = str(DEFAULT_SAMPLE_PATH)
        print(f"[geo_dump] Using default sample: {path}")

    if args.depth_max is not None:
        os.environ["CAD_QUOTER_ACAD_DEPTH_MAX"] = str(args.depth_max)

    geo_extractor.set_trace_acad(bool(args.trace_acad))

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

    if args.debug_entities:
        os.environ["CAD_QUOTER_DEBUG_ENTITIES"] = "1"

    try:
        doc = geo_extractor._load_doc_for_path(Path(path), use_oda=args.use_oda)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[geo_dump] failed to load document: {exc}")
        return 1

    read_kwargs: dict[str, object] = {}
    layer_allow_args = list(args.layer_allow or [])
    allow_layers_arg = getattr(args, "allow_layers", None)
    if allow_layers_arg:
        layer_allow_args.extend(part.strip() for part in allow_layers_arg.split(","))
    if layer_allow_args:
        os.environ["CAD_QUOTER_ACAD_ALLOW_LAYERS"] = ",".join(layer_allow_args)
    else:
        os.environ.pop("CAD_QUOTER_ACAD_ALLOW_LAYERS", None)
    if layer_allow_args:
        normalized_layers: list[str] = []
        allow_all = False
        for value in layer_allow_args:
            if value is None:
                continue
            text = value.strip()
            if not text:
                continue
            upper = text.upper()
            if upper in {"ALL", "*", "<ALL>"}:
                allow_all = True
                break
            normalized_layers.append(text)
        if allow_all:
            read_kwargs["layer_allowlist"] = None
            print("[geo_dump] layer_allow=ALL")
        else:
            allowlist_set = {layer.strip() for layer in normalized_layers if layer.strip()}
            if allowlist_set:
                read_kwargs["layer_allowlist"] = allowlist_set
                print(f"[geo_dump] layer_allow={sorted(allowlist_set)}")
    block_allow_args = args.block_allow or []
    if block_allow_args:
        normalized_blocks = [
            value.strip()
            for value in block_allow_args
            if isinstance(value, str) and value.strip()
        ]
        if normalized_blocks:
            read_kwargs["block_name_allowlist"] = normalized_blocks
            print(f"[geo_dump] block_allow={normalized_blocks}")
    block_regex_args = args.block_regex or []
    if block_regex_args:
        normalized_patterns = [
            value.strip()
            for value in block_regex_args
            if isinstance(value, str) and value.strip()
        ]
        if normalized_patterns:
            read_kwargs["block_name_regex"] = normalized_patterns
            print(f"[geo_dump] block_regex={normalized_patterns}")
    include_layer_patterns = args.include_layer or []
    if include_layer_patterns:
        read_kwargs["layer_include_regex"] = list(include_layer_patterns)
        print(f"[geo_dump] include_layer={include_layer_patterns}")
    exclude_layer_patterns = args.exclude_layer or []
    if exclude_layer_patterns:
        read_kwargs["layer_exclude_regex"] = list(exclude_layer_patterns)
        print(f"[geo_dump] exclude_layer={exclude_layer_patterns}")
    if args.debug_layouts:
        read_kwargs["debug_layouts"] = True
    if args.force_text:
        read_kwargs["force_text"] = True
    payload = read_geo(doc, **read_kwargs)
    if isinstance(payload, Mapping):
        payload = dict(payload)
    else:
        payload = {}
    published = _payload_has_rows(payload)
    scan_info = geo_extractor.get_last_acad_table_scan() or {}
    tables_found = 0
    try:
        tables_found = int(scan_info.get("tables_found", 0))  # type: ignore[arg-type]
    except Exception:
        tables_found = 0
    geo_extractor.log_last_dxf_fallback(tables_found)
    if (
        tables_found == 0
        and not published
        and Path(path).suffix.lower() == ".dwg"
    ):
        fallback_versions = [
            "ACAD2000",
            "ACAD2004",
            "ACAD2007",
            "ACAD2013",
            "ACAD2018",
        ]
        for version in fallback_versions:
            normalized_version = geo_extractor._normalize_oda_version(version) or version
            print(f"[ACAD-TABLE] trying DXF fallback version={normalized_version}")
            try:
                fallback_doc = geo_extractor._load_doc_for_path(
                    Path(path), use_oda=args.use_oda, out_ver=normalized_version
                )
            except Exception as exc:
                print(f"[ACAD-TABLE] DXF fallback {normalized_version} failed: {exc}")
                continue
            payload = read_geo(fallback_doc, **read_kwargs)
            if isinstance(payload, Mapping):
                payload = dict(payload)
            else:
                payload = {}
            published = _payload_has_rows(payload)
            scan_info = geo_extractor.get_last_acad_table_scan() or {}
            try:
                tables_found = int(scan_info.get("tables_found", 0))  # type: ignore[arg-type]
            except Exception:
                tables_found = 0
            geo_extractor.log_last_dxf_fallback(tables_found)
            if tables_found or published:
                break
    if not isinstance(payload, Mapping):
        payload = {}
    else:
        payload = dict(payload)

    final_scan = geo_extractor.get_last_acad_table_scan() or scan_info
    if args.scan_acad_tables:
        tables: list[Mapping[str, object]] = []
        tables_found_display = 0
        if isinstance(final_scan, Mapping):
            try:
                tables_found_display = int(final_scan.get("tables_found", 0))
            except Exception:
                tables_found_display = 0
            raw_tables = final_scan.get("tables")
            if isinstance(raw_tables, list):
                tables = [entry for entry in raw_tables if isinstance(entry, Mapping)]
        print(f"[ACAD-TABLE] tables_found={tables_found_display}")
        if tables:
            for entry in tables:
                owner = str(entry.get("owner") or "-")
                layer = str(entry.get("layer") or "-")
                handle = str(entry.get("handle") or "-")
                dxftype = str(entry.get("type") or entry.get("dxftype") or "-")
                try:
                    rows_val = int(entry.get("rows") or entry.get("row_count") or 0)
                except Exception:
                    rows_val = 0
                try:
                    cols_val = int(entry.get("cols") or entry.get("n_cols") or 0)
                except Exception:
                    cols_val = 0
                print(
                    "[ACAD-TABLE] hit owner={owner} layer={layer} handle={handle} rows={rows} cols={cols} type={typ}".format(
                        owner=owner,
                        layer=layer,
                        handle=handle,
                        rows=rows_val,
                        cols=cols_val,
                        typ=dxftype,
                    )
                )
        else:
            print("[ACAD-TABLE] hit owner=<none> layer=- handle=- rows=0 cols=0 type=-")

    geo = payload.get("geo")
    if not isinstance(geo, Mapping):
        geo = {}
    ops_summary = payload.get("ops_summary")
    if not isinstance(ops_summary, Mapping):
        ops_summary = geo.get("ops_summary") if isinstance(geo, Mapping) else {}
    if not isinstance(ops_summary, Mapping):
        ops_summary = {}

    rows = payload.get("rows")
    if not isinstance(rows, list):
        rows = list(rows or [])  # type: ignore[arg-type]
    if not rows:
        ops_rows = ops_summary.get("rows") if isinstance(ops_summary, Mapping) else None
        if isinstance(ops_rows, list):
            rows = ops_rows
        else:
            rows = list(ops_rows or [])  # type: ignore[arg-type]

    qty_sum = payload.get("qty_sum")
    if isinstance(qty_sum, (int, float)):
        qty_sum = int(float(qty_sum))
    else:
        qty_sum = _sum_qty(rows)

    holes_source = payload.get("provenance_holes")
    if holes_source is None:
        provenance = geo.get("provenance") if isinstance(geo, Mapping) else {}
        if isinstance(provenance, Mapping):
            holes_source = provenance.get("holes")

    hole_count = payload.get("hole_count")
    if hole_count in (None, ""):
        hole_count = geo.get("hole_count") if isinstance(geo, Mapping) else None
    try:
        if hole_count not in (None, ""):
            hole_count = int(float(hole_count))
    except Exception:
        pass

    source = payload.get("source")
    if source is None and isinstance(ops_summary, Mapping):
        source = ops_summary.get("source")
    print(
        "rows={rows} qty_sum={qty} source={src} hole_count={hole_count} provenance={prov}".format(
            rows=len(rows),
            qty=qty_sum,
            src=source,
            hole_count=hole_count,
            prov=holes_source,
        )
    )

    hole_rows_artifact = _build_hole_rows_artifact(
        rows,
        qty_sum=qty_sum,
        hole_count=hole_count,
        provenance=holes_source,
        source=source,
    )
    _write_artifact(ARTIFACT_DIR / "hole_rows.json", hole_rows_artifact)

    ops_totals_artifact = _build_ops_totals_artifact(ops_summary)
    if ops_totals_artifact:
        _write_artifact(ARTIFACT_DIR / "op_totals.json", ops_totals_artifact)
    if args.show_rows and rows:
        limit = max(args.show_rows, 0)
        if limit > 0:
            count = min(limit, len(rows))
            print(f"[ROWS] preview_count={count}")
            for idx, row in enumerate(rows[:count]):
                qty_display = row.get("qty") if isinstance(row, Mapping) else None
                ref_display = row.get("ref") if isinstance(row, Mapping) else None
                side_display = row.get("side") if isinstance(row, Mapping) else None
                desc_display = row.get("desc") if isinstance(row, Mapping) else None
                print(
                    "[ROW {idx:02d}] {qty} | {ref} | {side} | {desc}".format(
                        idx=idx,
                        qty=qty_display if qty_display not in (None, "") else "-",
                        ref=ref_display if ref_display not in (None, "") else "-",
                        side=side_display if side_display not in (None, "") else "-",
                        desc=desc_display if desc_display not in (None, "") else "",
                    )
                )
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

    debug_payload = payload.get("debug_payload")
    if isinstance(debug_payload, Mapping):
        debug_payload = dict(debug_payload)
    else:
        debug_payload = geo_extractor.get_last_text_table_debug() or {}

    if args.dump_bands:
        row_debug = debug_payload.get("row_debug") or []
        columns = debug_payload.get("columns") or []
        if isinstance(row_debug, Mapping):
            row_debug = [row_debug]
        if isinstance(columns, Mapping):
            columns = list(columns.values())
        if isinstance(row_debug, list) and row_debug:
            print(
                f"[TABLE-Y] dump rows_total={len(row_debug)} columns={len(columns)}"
            )
            for entry in row_debug[:30]:
                if not isinstance(entry, Mapping):
                    continue
                row_idx = entry.get("index")
                try:
                    row_idx_int = int(row_idx)
                except Exception:
                    row_idx_int = -1
                cells = entry.get("cells") or []
                parts: list[str] = []
                for col_idx, cell_text in enumerate(cells):
                    preview = geo_extractor._truncate_cell_preview(str(cell_text or ""))
                    parts.append(f'C{col_idx}="{preview}"')
                preview_body = " | ".join(parts)
                label = row_idx_int if row_idx_int >= 0 else row_idx
                print(
                    f"[TABLE-X] row#{label} cols={len(cells)} | {preview_body}"
                )
        else:
            print("[TABLE-X] dump rows: no row_debug in debug payload")

    dump_base = args.dump_lines
    if args.debug_entities and len(rows) < 8 and not dump_base:
        base_name = Path(path).stem or "geo_dump"
        dump_base = str(Path.cwd() / f"{base_name}_lines.tsv")

    if dump_base:
        lines_path = Path(dump_base)
        bands_path = lines_path.with_name(f"{lines_path.stem}_bands.tsv")
        raw_lines = debug_payload.get("raw_lines") or []
        candidates = raw_lines or debug_payload.get("candidates", [])
        row_debug = debug_payload.get("row_debug", [])

        def _format_float(value: object) -> str:
            if isinstance(value, (int, float)):
                return f"{float(value):.3f}"
            try:
                return f"{float(value):.3f}"
            except Exception:
                return ""

        def _write_lines(path_obj: Path, header: str, rows_iter: list[dict[str, object]]) -> None:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with path_obj.open("w", encoding="utf-8") as handle:
                handle.write(header + "\n")
                for item in rows_iter:
                    if header.startswith("layout\t"):
                        layout = str(item.get("layout") or "")
                        in_block = "1" if item.get("in_block") else "0"
                        fields = [
                            layout,
                            in_block,
                            _format_float(item.get("x")),
                            _format_float(item.get("y")),
                            str(item.get("text") or ""),
                        ]
                    elif header.startswith("row\t"):
                        fields = [
                            str(item.get("row", "")),
                            str(item.get("col", "")),
                            _format_float(item.get("y_center")),
                            str(item.get("text") or ""),
                        ]
                    else:
                        fields = [
                            str(item.get("band", "")),
                            str(item.get("col", "")),
                            _format_float(item.get("x_center")),
                            _format_float(item.get("y_center")),
                            str(item.get("text") or ""),
                        ]
                    handle.write("\t".join(fields) + "\n")

        try:
            _write_lines(lines_path, "layout\tin_block\tx\ty\ttext", list(candidates))
            if row_debug:
                row_dump: list[dict[str, object]] = []
                for entry in row_debug:
                    if not isinstance(entry, Mapping):
                        continue
                    row_idx = entry.get("index")
                    y_center = entry.get("y")
                    cells = entry.get("cells") or []
                    for col_idx, cell_text in enumerate(cells):
                        row_dump.append(
                            {
                                "row": row_idx,
                                "col": col_idx,
                                "y_center": y_center,
                                "text": cell_text,
                            }
                        )
                if row_dump:
                    _write_lines(
                        bands_path,
                        "row\tcol\ty_center\ttext",
                        row_dump,
                    )
        except OSError as exc:  # pragma: no cover - filesystem issues
            print(f"[geo_dump] failed to write dumps: {exc}")
        else:
            print(f"[geo_dump] wrote debug dumps to {lines_path} and {bands_path}")

    rows_csv_path: Path | None = None
    if args.dump_rows_csv:
        csv_target = Path(args.dump_rows_csv)
        try:
            csv_target.parent.mkdir(parents=True, exist_ok=True)
            with csv_target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["qty", "ref", "side", "desc", "hole"])
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    qty_val = row.get("qty")
                    writer.writerow(
                        [
                            "" if qty_val in (None, "") else str(qty_val),
                            str(row.get("ref") or ""),
                            str(row.get("side") or ""),
                            str(row.get("desc") or ""),
                            str(row.get("hole") or ""),
                        ]
                    )
        except OSError as exc:  # pragma: no cover - filesystem issues
            print(f"[geo_dump] failed to write rows CSV: {exc}")
        else:
            rows_csv_path = csv_target
            print(f"[geo_dump] wrote rows CSV to {csv_target}")

    debug_info = geo_extractor.get_last_text_table_debug() or {}

    def _format_counts(counts: Mapping[str, int] | None) -> str:
        if not counts:
            return "{}"
        items = sorted(counts.items(), key=lambda item: (-item[1], item[0] or ""))
        top = ", ".join(f"{name or '-'}:{count}" for name, count in items[:5])
        if len(items) > 5:
            top += ", …"
        return "{" + top + "}"

    scanned_layouts = list(dict.fromkeys(debug_info.get("scanned_layouts") or []))
    scanned_layers = list(dict.fromkeys(debug_info.get("scanned_layers") or []))
    layout_summary = ",".join(scanned_layouts) if scanned_layouts else "-"
    layer_summary = ",".join(scanned_layers) if scanned_layers else "-"
    csv_display = str(rows_csv_path) if rows_csv_path else "-"
    print(
        "[geo_dump] summary layouts={layouts} layers={layers} rows={rows} csv={csv}".format(
            layouts=layout_summary,
            layers=layer_summary,
            rows=len(rows),
            csv=csv_display,
        )
    )

    if args.debug_layouts:
        layer_pre = debug_info.get("layer_counts_pre")
        layer_regex = debug_info.get("layer_counts_post_regex")
        layer_post = debug_info.get("layer_counts_post_allow")
        layout_pre = debug_info.get("layout_counts_pre")
        layout_regex = debug_info.get("layout_counts_post_regex")
        layout_post = debug_info.get("layout_counts_post_allow")
        include_patterns = debug_info.get("layer_regex_include") or []
        exclude_patterns = debug_info.get("layer_regex_exclude") or []
        if include_patterns or exclude_patterns:
            print(
                "[geo_dump] layer_regex include={incl} exclude={excl}".format(
                    incl=include_patterns or "-",
                    excl=exclude_patterns or "-",
                )
            )
        print(f"[geo_dump] layer_counts_pre={_format_counts(layer_pre)}")
        if layer_regex is not None:
            print(f"[geo_dump] layer_counts_regex={_format_counts(layer_regex)}")
        print(f"[geo_dump] layer_counts_post={_format_counts(layer_post)}")
        print(f"[geo_dump] layout_counts_pre={_format_counts(layout_pre)}")
        if layout_regex is not None:
            print(f"[geo_dump] layout_counts_regex={_format_counts(layout_regex)}")
        print(f"[geo_dump] layout_counts_post={_format_counts(layout_post)}")

    return 0

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())