from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cad_quoter import geo_extractor
from cad_quoter.geo_extractor import read_geo

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
        "--layer-allow",
        dest="layer_allow",
        action="append",
        help="Restrict table/text scans to the specified layer (repeatable; use ALL to disable filtering)",
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

    if args.debug_entities:
        os.environ["CAD_QUOTER_DEBUG_ENTITIES"] = "1"

    try:
        doc = geo_extractor._load_doc_for_path(Path(path), use_oda=args.use_oda)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[geo_dump] failed to load document: {exc}")
        return 1

    read_kwargs: dict[str, object] = {}
    layer_allow_args = args.layer_allow or []
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
    payload = read_geo(doc, **read_kwargs)
    if not isinstance(payload, Mapping):
        payload = {}
    else:
        payload = dict(payload)

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
        band_cells = debug_payload.get("band_cells") or []
        band_map: dict[int, dict[int, list[str]]] = {}
        if isinstance(band_cells, list):
            for cell in band_cells:
                if not isinstance(cell, Mapping):
                    continue
                band_raw = cell.get("band")
                col_raw = cell.get("col")
                try:
                    band_idx = int(band_raw)
                except Exception:
                    continue
                try:
                    col_idx = int(col_raw)
                except Exception:
                    continue
                text_val = str(cell.get("text") or "")
                column_map = band_map.setdefault(band_idx, {})
                column_map.setdefault(col_idx, []).append(text_val)
        band_indices = sorted(band_map.keys())
        if band_indices:
            print(f"[TABLE-Y] dump bands_total={len(band_indices)}")
            for band_idx in band_indices[:30]:
                column_map = band_map.get(band_idx, {})
                parts = []
                for col_idx in sorted(column_map.keys()):
                    cell_text = " ".join(part.strip() for part in column_map[col_idx] if part).strip()
                    preview = geo_extractor._truncate_cell_preview(cell_text)
                    parts.append(f'C{col_idx}="{preview}"')
                preview_body = " | ".join(parts)
                print(
                    f"[TABLE-X] band#{band_idx} cols={len(column_map)} | {preview_body}"
                )
        else:
            print("[TABLE-X] dump bands: no band_cells in debug payload")

    dump_base = args.dump_lines
    if args.debug_entities and len(rows) < 8 and not dump_base:
        base_name = Path(path).stem or "geo_dump"
        dump_base = str(Path.cwd() / f"{base_name}_lines.tsv")

    if dump_base:
        lines_path = Path(dump_base)
        bands_path = lines_path.with_name(f"{lines_path.stem}_bands.tsv")
        raw_lines = debug_payload.get("raw_lines") or []
        candidates = raw_lines or debug_payload.get("candidates", [])
        band_cells = debug_payload.get("band_cells", [])

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
            _write_lines(bands_path, "band\tcol\tx_center\ty_center\ttext", list(band_cells))
        except OSError as exc:  # pragma: no cover - filesystem issues
            print(f"[geo_dump] failed to write dumps: {exc}")
        else:
            print(f"[geo_dump] wrote debug dumps to {lines_path} and {bands_path}")

    return 0

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())