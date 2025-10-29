import csv
import json
import pathlib
import subprocess
import sys
from typing import Iterable

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLD = ROOT / "tests" / "gold"


with (GOLD / "hole_table_sample.jsonl").open(encoding="utf-8") as _fh:
    _CASES = [json.loads(line) for line in _fh if line.strip()]


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["case"])
def test_hole_table_ops_from_csv(tmp_path: pathlib.Path, case: dict[str, object]) -> None:
    raw_lines: Iterable[str] = case["raw_lines"]  # type: ignore[assignment]
    expected_ops: list[list[str]] = case["expected_ops"]  # type: ignore[assignment]

    sample_csv = tmp_path / "dxf_text_dump.csv"
    with sample_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for line in raw_lines:
            writer.writerow([
                "Model",
                "BALLOON",
                "PROXYTEXT",
                line,
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
            ])

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "tools" / "hole_table_from_csv.py"),
            "--dxf-csv",
            str(sample_csv),
        ]
    )

    out_csv = sample_csv.with_name("hole_table_ops.csv")
    got_rows: list[list[str]] = []
    with out_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            got_rows.append([
                row["HOLE"],
                row["REF_DIAM"],
                row["QTY"],
                row["DESCRIPTION/DEPTH"],
            ])

    assert got_rows == expected_ops
