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


# Tests for thread validation functions
from cad_quoter.geometry.hole_table_parser import (
    validate_and_correct_thread,
    is_valid_thread_spec,
    STANDARD_THREADS,
    THREAD_MAJOR_DIAMETERS,
)


class TestThreadValidation:
    """Tests for thread parsing sanity checks."""

    def test_valid_thread_unchanged(self):
        """Valid standard threads should not be corrected."""
        # 5/16-18 is valid coarse thread
        major, tpi, corrected = validate_and_correct_thread("5/16", 18)
        assert major == "5/16"
        assert tpi == 18
        assert corrected is False

    def test_valid_fine_thread_unchanged(self):
        """Valid fine threads should not be corrected."""
        # 1/4-28 is valid fine thread
        major, tpi, corrected = validate_and_correct_thread("1/4", 28)
        assert major == "1/4"
        assert tpi == 28
        assert corrected is False

    def test_invalid_tpi_corrected(self):
        """Invalid TPI should be corrected to nearest standard."""
        # 1/4-80 is invalid - should correct to 1/4-28 (nearest to 80 from [20, 28, 32])
        major, tpi, corrected = validate_and_correct_thread("1/4", 80)
        assert major == "1/4"
        assert tpi == 32  # 32 is closest to 80 in [20, 28, 32]
        assert corrected is True

    def test_decimal_major_normalized(self):
        """Decimal major diameter should be normalized to standard nominal."""
        # 0.2500 should map to 1/4
        major, tpi, corrected = validate_and_correct_thread("0.2500", 20)
        assert major == "1/4"
        assert tpi == 20
        assert corrected is False  # TPI was already valid for 1/4

    def test_decimal_with_invalid_tpi_corrected(self):
        """Decimal major with invalid TPI should be fully corrected."""
        # 0.3125 = 5/16, with TPI 80 (invalid)
        major, tpi, corrected = validate_and_correct_thread("0.3125", 80)
        assert major == "5/16"
        assert tpi == 32  # 32 is closest to 80 in [18, 24, 32]
        assert corrected is True

    def test_number_thread_valid(self):
        """Number threads should validate correctly."""
        # #10-32 is valid
        major, tpi, corrected = validate_and_correct_thread("#10", 32)
        assert major == "#10"
        assert tpi == 32
        assert corrected is False

    def test_number_thread_invalid_tpi(self):
        """Number thread with invalid TPI should be corrected."""
        # #10-48 is invalid - should correct to #10-32 (closest to 48 from [24, 32])
        major, tpi, corrected = validate_and_correct_thread("#10", 48)
        assert major == "#10"
        assert tpi == 32  # 32 is closest to 48 in [24, 32]
        assert corrected is True

    def test_is_valid_thread_spec_true(self):
        """is_valid_thread_spec should return True for valid threads."""
        assert is_valid_thread_spec("5/16", 18) is True
        assert is_valid_thread_spec("1/4", 20) is True
        assert is_valid_thread_spec("#10", 32) is True

    def test_is_valid_thread_spec_false(self):
        """is_valid_thread_spec should return False for invalid threads."""
        assert is_valid_thread_spec("1/4", 80) is False
        assert is_valid_thread_spec("5/16", 80) is False
        assert is_valid_thread_spec("#10", 80) is False

    def test_nearby_dimensions_helps_correction(self):
        """Nearby dimensions should help identify correct thread size."""
        # If we have a boss diameter around 0.35", that suggests 5/16 thread
        # (5/16 = 0.3125, boss clearance ~0.35")
        nearby = [0.35]
        major, tpi, corrected = validate_and_correct_thread("0.2500", 80, nearby_diameters=nearby)
        # Should correct to 5/16-18 based on nearby boss diameter
        assert major == "5/16"
        assert tpi == 18  # Coarse thread default
        assert corrected is True

    def test_standard_threads_coverage(self):
        """Ensure standard thread table has expected entries."""
        assert "1/4" in STANDARD_THREADS
        assert "5/16" in STANDARD_THREADS
        assert "3/8" in STANDARD_THREADS
        assert "#10" in STANDARD_THREADS

        # Check coarse threads are first
        assert STANDARD_THREADS["1/4"][0] == 20  # UNC
        assert STANDARD_THREADS["5/16"][0] == 18  # UNC
        assert STANDARD_THREADS["3/8"][0] == 16  # UNC

    def test_thread_major_diameters_accuracy(self):
        """Ensure thread major diameters are accurate."""
        assert THREAD_MAJOR_DIAMETERS["1/4"] == 0.2500
        assert THREAD_MAJOR_DIAMETERS["5/16"] == 0.3125
        assert THREAD_MAJOR_DIAMETERS["3/8"] == 0.3750
        assert THREAD_MAJOR_DIAMETERS["#10"] == 0.1900


class TestTPISanityCheck:
    """Tests for TPI sanity check (guard against bogus high TPI on larger taps)."""

    def test_high_tpi_on_large_tap_triggers_guard(self):
        """TPI > 40 on diameter > 0.19\" should trigger sanity check."""
        # 1/4-80 TAP is suspicious - 1/4" tap shouldn't have 80 TPI
        # Should be corrected by validate_and_correct_thread
        major, tpi, corrected = validate_and_correct_thread("1/4", 80)
        assert major == "1/4"
        assert tpi == 32  # Corrected to finest standard (closest to 80 from [20, 28, 32])
        assert corrected is True

    def test_high_tpi_on_small_tap_allowed(self):
        """TPI > 40 on small diameter (<=0.19\") should be allowed (e.g., #0-80)."""
        # #0-80 is valid (diameter = 0.0600", TPI = 80)
        major, tpi, corrected = validate_and_correct_thread("#0", 80)
        assert major == "#0"
        assert tpi == 80  # Should NOT be corrected - this is valid
        assert corrected is False

    def test_reasonable_tpi_on_large_tap_allowed(self):
        """TPI <= 40 on large diameter should be allowed."""
        # 1/4-20 is valid coarse thread
        major, tpi, corrected = validate_and_correct_thread("1/4", 20)
        assert major == "1/4"
        assert tpi == 20
        assert corrected is False

        # 5/16-32 is valid extra-fine thread
        major, tpi, corrected = validate_and_correct_thread("5/16", 32)
        assert major == "5/16"
        assert tpi == 32
        assert corrected is False

    def test_tpi_sanity_examples(self):
        """Test specific examples of bogus vs valid TPI."""
        # BOGUS: 5/16-80 TAP (should be 18, 24, or 32)
        major, tpi, corrected = validate_and_correct_thread("5/16", 80)
        assert corrected is True
        assert tpi == 32  # Closest to 80 from [18, 24, 32]

        # VALID: #10-32 TAP (standard fine thread)
        major, tpi, corrected = validate_and_correct_thread("#10", 32)
        assert corrected is False
        assert tpi == 32

        # VALID: #2-64 TAP (standard fine thread for small diameter)
        major, tpi, corrected = validate_and_correct_thread("#2", 64)
        assert corrected is False
        assert tpi == 64
