import csv, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLD = ROOT / "tests" / "gold"

def run_geo_dump(sample_csv: pathlib.Path):
    # run your script exactly how you do it today; adjust path/module if needed
    subprocess.check_call([sys.executable, str(ROOT / "cad_quoter" / "geo_dump.py"),
                           "--dump-text-all", "--text-layout", "CHART",
                           "--text-min-height", "0.0",
                           "--text-include-layers", ".*",
                           "--text-block-depth", "2",
                           "--dxf-csv", str(sample_csv)])  # <-- add an arg if needed

def read_ops_csv(out_dir: pathlib.Path):
    p = out_dir / "hole_table_ops.csv"
    rows = []
    with p.open(newline='', encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append([r["HOLE"], r["REF_DIAM"], r["QTY"], r["DESCRIPTION/DEPTH"]])
    return rows

def test_sample1_ops(tmp_path):
    data = json.loads((GOLD / "hole_table_sample.jsonl").read_text().splitlines()[0])
    # Write raw_lines to a synthetic dxf_text_dump.csv for this test
    sample_csv = tmp_path / "dxf_text_dump.csv"
    with sample_csv.open("w", encoding="utf-8", newline="") as fh:
        for line in data["raw_lines"]:
            fh.write(f"Model,BALLOON,PROXYTEXT,{line},0,0,0,0,0,0\n")
    # Run
    run_geo_dump(sample_csv)
    # Read ops
    got = read_ops_csv(tmp_path)
    exp = data["expected_ops"]
    assert got == exp
