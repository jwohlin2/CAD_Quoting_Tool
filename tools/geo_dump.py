from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from typing import Sequence

from cad_quoter.geo_extractor import extract_geo_from_path


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
    args = parser.parse_args(argv)

    path = (args.path or os.environ.get("GEO_DUMP_PATH") or "").strip()
    if not path:
        try:
            path = input("Enter path to the DXF or DWG file: ").strip()
        except EOFError:
            path = ""

    if not path:
        parser.print_help()
        print("\nProvide a DXF/DWG path as an argument, set GEO_DUMP_PATH, or answer the prompt.")
        return 1

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
    if args.debug and rows:
        print("first_rows:")
        for idx, row in enumerate(rows[:10]):
            ref = row.get("ref") if isinstance(row, Mapping) else ""
            desc = row.get("desc") if isinstance(row, Mapping) else ""
            qty = row.get("qty") if isinstance(row, Mapping) else ""
            hole = row.get("hole") if isinstance(row, Mapping) else ""
            side = row.get("side") if isinstance(row, Mapping) else ""
            print(f"  [{idx:02d}] QTY={qty} REF={ref} SIDE={side} DESC={desc} HOLE={hole}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
