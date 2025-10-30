"""Temporary CLI that will evolve to produce hole table operations from a DXF text dump."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

from .hole_ops import explode_rows_to_operations


OPS_FIELDNAMES = ("HOLE", "REF_DIAM", "QTY", "DESCRIPTION/DEPTH")


def explode_hole_table(text_rows: Iterable[str]) -> List[List[str]]:
    """Explode raw DXF text rows into HOLE TABLE operations."""

    return explode_rows_to_operations(text_rows)


def _read_text_rows(dxf_csv: Path) -> list[str]:
    rows: list[str] = []
    with dxf_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for record in reader:
            if len(record) >= 4:
                rows.append(record[3])
    return rows


def _write_ops_csv(dxf_csv: Path, operations: Iterable[Iterable[str]]) -> Path:
    out_path = dxf_csv.with_name("hole_table_ops.csv")
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(OPS_FIELDNAMES)
        for op in operations:
            writer.writerow(list(op))
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explode hole table rows from a DXF CSV dump.")
    parser.add_argument("--dxf-csv", required=True, help="Path to the dxf_text_dump.csv file")
    args = parser.parse_args(argv)

    dxf_csv = Path(args.dxf_csv)
    text_rows = _read_text_rows(dxf_csv)
    operations = explode_hole_table(text_rows)
    _write_ops_csv(dxf_csv, operations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
