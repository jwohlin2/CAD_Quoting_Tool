import re
import csv
from fractions import Fraction
from typing import Iterable, List, Sequence, Tuple

_UHEX_RE = re.compile(r"\\U\+([0-9A-Fa-f]{4})")


def decode_uplus(s: str) -> str:
    """Decode ``\\U+####`` sequences to their unicode characters."""

    return _UHEX_RE.sub(lambda m: chr(int(m.group(1), 16)), s)


def read_dump(csv_path: str) -> List[dict]:
    """Read ``dxf_text_dump.csv`` output into a list of dictionaries."""

    rows: List[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            row["text"] = decode_uplus(row.get("text", ""))
            rows.append(row)
    return rows


def _is_text_row(row: dict) -> bool:
    return row.get("etype") in {"PROXYTEXT", "MTEXT", "TEXT"}


def find_hole_table_chunks(rows: Sequence[dict]) -> Tuple[List[str], List[str]]:
    """Return the header/body string chunks for the first HOLE TABLE found."""

    header_chunks: List[str] = []
    body_chunks: List[str] = []

    i = 0
    while i < len(rows):
        if _is_text_row(rows[i]) and "HOLE TABLE" in rows[i].get("text", "").upper():
            break
        i += 1
    else:
        return header_chunks, body_chunks

    j = i
    saw_desc = False
    while j < len(rows) and _is_text_row(rows[j]):
        header_chunks.append(rows[j]["text"])
        if "DESCRIPTION" in rows[j]["text"].upper():
            saw_desc = True
            j += 1
            break
        j += 1

    if not saw_desc:
        for k in range(j, min(j + 5, len(rows))):
            if _is_text_row(rows[k]):
                header_chunks.append(rows[k]["text"])
                if "DESCRIPTION" in rows[k]["text"].upper():
                    j = k + 1
                    break

    while j < len(rows) and _is_text_row(rows[j]):
        body_chunks.append(rows[j]["text"])
        j += 1

    return header_chunks, body_chunks


def parse_header(header_chunks: Iterable[str]) -> Tuple[List[str], List[str], List[int]]:
    """Parse the header row into hole ids, ref diameters, and quantities."""

    header_text = re.sub(r"\s+", " ", " ".join(header_chunks)).strip().replace(",", " ")
    tokens = header_text.split()

    def find_token(key: str) -> int:
        target = key.upper()
        for idx, tok in enumerate(tokens):
            if tok.upper() == target:
                return idx
        return -1

    i_hole = find_token("HOLE")
    i_ref = find_token("REF")
    i_qty = find_token("QTY")
    i_desc = find_token("DESCRIPTION")
    if i_ref == -1:
        i_ref = find_token("Ø")

    if min(i_hole, i_ref, i_qty, i_desc) == -1:
        raise ValueError(f"Unexpected HOLE TABLE header: {header_text}")

    hole_letters = tokens[i_hole + 1 : i_ref]
    diam_tokens = tokens[i_ref + 1 : i_qty]
    qty_tokens = tokens[i_qty + 1 : i_desc]

    qty_list: List[int] = []
    for qty in qty_tokens:
        try:
            qty_list.append(int(qty))
        except ValueError:
            continue

    n = min(len(hole_letters), len(diam_tokens), len(qty_list))
    return hole_letters[:n], diam_tokens[:n], qty_list[:n]


def diameter_aliases(diameter: str) -> List[str]:
    """Create search aliases for a diameter token."""

    aliases = [diameter]
    if diameter.startswith(("Ø", "∅")):
        payload = diameter[1:]
        try:
            value = float(payload)
            fraction = Fraction(value).limit_denominator(64)
            if abs(float(fraction) - value) < 1e-4:
                aliases.extend(
                    [
                        f"Ø{fraction.numerator}/{fraction.denominator}",
                        f"∅{fraction.numerator}/{fraction.denominator}",
                    ]
                )
            compact = f"{value:.3f}".rstrip("0").rstrip(".")
            aliases.extend(
                [
                    f"Ø{compact}",
                    f"∅{compact}",
                    f"(Ø{compact})",
                    f"(∅{compact})",
                ]
            )
        except ValueError:
            pass
    return aliases


def split_descriptions(body_chunks: Iterable[str], diam_list: Sequence[str]) -> List[str]:
    """Split the table body into description strings aligned with diameters."""

    blob = re.sub(r"\s+", " ", " ".join(body_chunks)).strip()
    needles = [diameter_aliases(d) for d in diam_list]

    positions: List[int] = []
    cursor = 0
    for aliases in needles:
        position = None
        for alias in aliases:
            idx = blob.find(alias, cursor)
            if idx != -1:
                position = idx
                break
        if position is None:
            numeric_payload = re.sub(r"^[Ø∅]\s*", "", aliases[0])
            idx = blob.find(numeric_payload, cursor)
            position = idx if idx != -1 else len(blob)
        positions.append(position)
        cursor = max(position + 1, cursor)

    segments: List[str] = []
    for index, start in enumerate(positions):
        end = positions[index + 1] if index + 1 < len(positions) else len(blob)
        segment = blob[start:end].strip()
        segment = re.sub(r"^[\(\s]*[Ø∅][0-9.]+[\)]?\s*", "", segment)
        segment = re.sub(r"^[\(\s]*[Ø∅][0-9]+/[0-9]+[\)]?\s*", "", segment)
        segments.append(segment)
    return segments


def extract_hole_table(dump_csv: str, out_csv: str) -> None:
    """Extract the hole table rows from ``dump_csv`` into ``out_csv``."""

    rows = read_dump(dump_csv)
    header, body = find_hole_table_chunks(rows)
    if not header:
        raise RuntimeError("HOLE TABLE not found")

    holes, diameters, quantities = parse_header(header)
    descriptions = split_descriptions(body, diameters)

    table_rows = []
    for index, hole in enumerate(holes):
        table_rows.append(
            {
                "HOLE": hole,
                "REF_DIAM": diameters[index],
                "QTY": quantities[index],
                "DESCRIPTION": descriptions[index].strip() if index < len(descriptions) else "",
            }
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["HOLE", "REF_DIAM", "QTY", "DESCRIPTION"])
        writer.writeheader()
        writer.writerows(table_rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract HOLE TABLE from DXF dump CSV.")
    parser.add_argument("dump_csv", help="Path to dxf_text_dump.csv")
    parser.add_argument("out_csv", help="Path to write the structured hole table CSV")

    args = parser.parse_args()
    extract_hole_table(args.dump_csv, args.out_csv)
