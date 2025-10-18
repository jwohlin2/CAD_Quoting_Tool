"""Shared helpers for locating McMaster plate SKUs from the catalog CSV."""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from fractions import Fraction
from typing import Any, Mapping, Sequence

from cad_quoter.resources import default_catalog_csv
from cad_quoter.vendors.mcmaster_stock import parse_inches as _parse_inches


@lru_cache(maxsize=1)
def load_mcmaster_catalog_rows(path: str | None = None) -> list[dict[str, Any]]:
    """Return rows from the McMaster stock catalog CSV."""

    csv_path = path or os.getenv("CATALOG_CSV_PATH") or str(default_catalog_csv())
    if not csv_path:
        return []
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader if row]
    except FileNotFoundError:
        return []
    except Exception:
        return []


def _coerce_inches_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace("\u00a0", " ")
    if not text:
        return None

    try:
        parsed = float(_parse_inches(text))
    except Exception:
        parsed = None
    if parsed is not None and parsed > 0:
        return parsed

    try:
        return float(text)
    except Exception:
        pass

    try:
        return float(Fraction(text))
    except Exception:
        return None


def _pick_mcmaster_plate_sku_impl(
    need_L_in: float,
    need_W_in: float,
    need_T_in: float,
    *,
    material_key: str = "MIC6",
    catalog_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return the smallest-area McMaster plate covering the requested envelope."""

    import math as _math

    if not all(val > 0 for val in (need_L_in, need_W_in, need_T_in)):
        return None

    rows = list(catalog_rows) if catalog_rows is not None else load_mcmaster_catalog_rows()
    if not rows:
        return None

    target_key = str(material_key or "").strip().lower()
    if not target_key:
        return None

    tolerance = 0.02
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        material_text = str(
            (row.get("material") or row.get("Material") or "")
        ).strip().lower()
        if not material_text:
            continue
        variants = {target_key}
        if "_" in target_key:
            variants.add(target_key.replace("_", " "))
        if " " in target_key:
            variants.add(target_key.replace(" ", ""))
        normalised_material = material_text.replace("_", " ")
        if not any(variant and variant in normalised_material for variant in variants):
            continue

        length = _coerce_inches_value(
            row.get("length_in")
            or row.get("L_in")
            or row.get("len_in")
            or row.get("length")
        )
        width = _coerce_inches_value(
            row.get("width_in")
            or row.get("W_in")
            or row.get("wid_in")
            or row.get("width")
        )
        thickness = _coerce_inches_value(
            row.get("thickness_in")
            or row.get("T_in")
            or row.get("thk_in")
            or row.get("thickness")
        )
        if (
            length is None
            or width is None
            or thickness is None
            or length <= 0
            or width <= 0
            or thickness <= 0
        ):
            continue
        if abs(thickness - need_T_in) > tolerance:
            continue
        part_no = str(
            row.get("mcmaster_part")
            or row.get("part")
            or row.get("sku")
            or ""
        ).strip()
        if not part_no:
            continue

        def _covers(a: float, b: float, A: float, B: float) -> bool:
            return (A >= a) and (B >= b)

        ok1 = _covers(need_L_in, need_W_in, length, width)
        ok2 = _covers(need_L_in, need_W_in, width, length)
        if not (ok1 or ok2):
            continue

        area = length * width
        overL1 = (length - need_L_in) if ok1 else _math.inf
        overW1 = (width - need_W_in) if ok1 else _math.inf
        overL2 = (width - need_L_in) if ok2 else _math.inf
        overW2 = (length - need_W_in) if ok2 else _math.inf
        over_L = min(overL1, overL2)
        over_W = min(overW1, overW2)

        candidates.append(
            {
                "len_in": float(length),
                "wid_in": float(width),
                "thk_in": float(thickness),
                "mcmaster_part": part_no,
                "area": float(area),
                "overL": float(over_L),
                "overW": float(over_W),
                "source": row.get("source") or "mcmaster-catalog",
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c["area"], max(c["overL"], c["overW"])))
    best = candidates[0]
    return {
        "len_in": float(best["len_in"]),
        "wid_in": float(best["wid_in"]),
        "thk_in": float(best["thk_in"]),
        "mcmaster_part": best["mcmaster_part"],
        "source": best.get("source") or "mcmaster-catalog",
    }


def pick_mcmaster_plate_sku(
    need_L_in: float,
    need_W_in: float,
    need_T_in: float,
    *,
    material_key: str = "MIC6",
    catalog_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return the smallest-area McMaster plate covering the requested envelope."""

    return _pick_mcmaster_plate_sku_impl(
        need_L_in,
        need_W_in,
        need_T_in,
        material_key=material_key,
        catalog_rows=catalog_rows,
    )


def resolve_mcmaster_plate_for_quote(
    need_L_in: float | None,
    need_W_in: float | None,
    need_T_in: float | None,
    *,
    material_key: str,
    stock_L_in: float | None = None,
    stock_W_in: float | None = None,
    stock_T_in: float | None = None,
    catalog_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return a McMaster plate candidate using quote needs and existing stock sizing."""

    candidate: dict[str, Any] | None = None

    if need_L_in and need_W_in and need_T_in:
        try:
            candidate = pick_mcmaster_plate_sku(
                float(need_L_in),
                float(need_W_in),
                float(need_T_in),
                material_key=material_key,
                catalog_rows=catalog_rows,
            )
        except Exception:
            candidate = None

    if candidate:
        return candidate

    if stock_L_in and stock_W_in and stock_T_in:
        try:
            return pick_mcmaster_plate_sku(
                float(stock_L_in),
                float(stock_W_in),
                float(stock_T_in),
                material_key=material_key,
                catalog_rows=catalog_rows,
            )
        except Exception:
            return None

    return None


__all__ = [
    "load_mcmaster_catalog_rows",
    "pick_mcmaster_plate_sku",
    "resolve_mcmaster_plate_for_quote",
]
