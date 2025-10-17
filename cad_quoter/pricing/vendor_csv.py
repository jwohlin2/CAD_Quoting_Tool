"""Provider helpers for vendor-supplied pricing data.

This module primarily exposes :class:`VendorCSV`, a simple CSV-backed
price provider used by the legacy material pricing engine.  The quoting
flow now also needs a light-weight stock selector that can source blank
sizes (and optional pricing metadata) from the bundled McMaster-Carr
catalog CSV.  Those helpers live alongside the provider so both the UI
and pricing code can share a single implementation.
"""
from __future__ import annotations

import csv
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

from cad_quoter.resources import default_catalog_csv
from cad_quoter.vendors.mcmaster_stock import norm_material, parse_inches

from .base import PriceProvider


_SCRAP_FRACTION = 0.05


def _coerce_float(value: Any) -> float | None:
    """Best-effort conversion to ``float`` with ``None`` on failure."""

    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _normalise_material_label(label: str | None) -> str:
    return norm_material(label or "") if label else ""


def _catalog_header(reader: csv.DictReader) -> Mapping[str, str]:
    raw_headers = reader.fieldnames or []
    normalised: dict[str, str] = {}
    for header in raw_headers:
        key = str(header or "").strip().lower()
        if key and key not in normalised:
            normalised[key] = header
    return normalised


def _iter_catalog_rows(
    handle: Iterable[Mapping[str, str]], headers: Mapping[str, str]
) -> Iterator[dict[str, Any]]:
    material_header = headers.get("material")
    thickness_header = headers.get("thickness_in")
    length_header = headers.get("length_in")
    width_header = headers.get("width_in")
    part_header = headers.get("part")
    price_header = headers.get("price_usd")
    min_header = headers.get("min_charge_usd") or headers.get("minimum_charge_usd")
    vendor_header = headers.get("vendor")

    for row in handle:
        if not row:
            continue
        material_raw = row.get(material_header or "") if isinstance(row, Mapping) else None
        if not material_raw:
            continue
        material = _normalise_material_label(str(material_raw))
        thickness = (
            parse_inches(str(row.get(thickness_header or "", "")))
            if thickness_header
            else 0.0
        )
        length = (
            parse_inches(str(row.get(length_header or "", "")))
            if length_header
            else 0.0
        )
        width = (
            parse_inches(str(row.get(width_header or "", "")))
            if width_header
            else 0.0
        )
        if thickness <= 0 or length <= 0 or width <= 0:
            continue
        vendor = str(row.get(vendor_header or "") or "McMaster").strip() or "McMaster"
        price_usd = _coerce_float(row.get(price_header)) if price_header else None
        min_charge = _coerce_float(row.get(min_header)) if min_header else None
        part_no = str(row.get(part_header or "") or "").strip() or None
        yield {
            "material": material,
            "thk_in": float(thickness),
            "len_in": float(length),
            "wid_in": float(width),
            "vendor": vendor,
            "part_no": part_no,
            "price_usd": price_usd if price_usd and price_usd > 0 else None,
            "min_charge_usd": min_charge if min_charge and min_charge > 0 else None,
        }


@lru_cache(maxsize=4)
def _load_catalog_rows(csv_path: str | None = None) -> list[dict[str, Any]]:
    path = Path(csv_path or default_catalog_csv())
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = _catalog_header(reader)
        required = {"material", "thickness_in", "length_in", "width_in"}
        if not required.issubset(headers):
            return []
        return list(_iter_catalog_rows(reader, headers))


def _fits_blank(row: Mapping[str, Any], need_len: float, need_wid: float) -> bool:
    length = float(row.get("len_in") or 0.0)
    width = float(row.get("wid_in") or 0.0)
    if length <= 0 or width <= 0:
        return False
    return (length >= need_len and width >= need_wid) or (
        length >= need_wid and width >= need_len
    )


def pick_plate_from_mcmaster(
    material: str,
    need_len: float,
    need_wid: float,
    need_thk: float,
    *,
    scrap_fraction: float = _SCRAP_FRACTION,
    catalog_path: str | None = None,
    allow_thickness_upsize: bool = True,
    thickness_tolerance: float = 0.02,
) -> dict[str, Any] | None:
    """Return the smallest McMaster plate that covers the requested envelope."""

    if not need_len or not need_wid or not need_thk:
        return None

    rows = _load_catalog_rows(catalog_path)
    if not rows:
        return None

    norm_mat = _normalise_material_label(material)
    need_len_adj = float(max(need_len, need_wid)) * (1.0 + float(scrap_fraction))
    need_wid_adj = float(min(need_len, need_wid)) * (1.0 + float(scrap_fraction))
    need_thk_val = float(need_thk)
    allow_upsize = bool(allow_thickness_upsize)
    try:
        tol = float(thickness_tolerance)
    except Exception:
        tol = 0.02
    tol = max(0.0, tol)

    candidates: list[tuple[float, float, float, float, float, dict[str, Any]]] = []
    for row in rows:
        row_mat = str(row.get("material") or "")
        if norm_mat and row_mat:
            if row_mat != norm_mat and norm_mat not in row_mat and row_mat not in norm_mat:
                continue
        row_thk = float(row.get("thk_in") or 0.0)
        if row_thk <= 0:
            continue
        if row_thk + 1e-4 < need_thk_val:
            continue
        thickness_diff = abs(row_thk - need_thk_val)
        if not allow_upsize and thickness_diff > tol + 1e-9:
            continue
        if not _fits_blank(row, need_len_adj, need_wid_adj):
            continue
        length_raw = float(row.get("len_in") or 0.0)
        width_raw = float(row.get("wid_in") or 0.0)
        area = length_raw * width_raw if length_raw and width_raw else float("inf")
        candidates.append(
            (
                area,
                thickness_diff,
                row_thk,
                length_raw,
                width_raw,
                dict(row),
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda c: (
            c[0],
            c[1],
            c[2],
            max(c[3], c[4]),
            min(c[3], c[4]),
        )
    )
    area, thickness_diff, best_thk, length_raw, width_raw, best_row = candidates[0]
    length = float(length_raw or need_len)
    width = float(width_raw or need_wid)
    if length < width:
        length, width = width, length
    return {
        "vendor": best_row.get("vendor") or "McMaster",
        "part_no": best_row.get("part_no"),
        "len_in": length,
        "wid_in": width,
        "thk_in": float(best_row.get("thk_in") or need_thk_val),
        "thickness_diff_in": float(thickness_diff),
        "price_usd": best_row.get("price_usd"),
        "min_charge_usd": best_row.get("min_charge_usd"),
    }


def pick_from_stdgrid(
    need_len: float,
    need_wid: float,
    need_thk: float,
    *,
    scrap_fraction: float = _SCRAP_FRACTION,
) -> dict[str, Any]:
    """Fallback selector that approximates a standard plate grid."""

    stock_lengths: Sequence[float] = (6, 8, 10, 12, 18, 24, 36, 48)
    stock_widths: Sequence[float] = (6, 8, 10, 12, 18, 24, 36)
    stock_thicknesses: Sequence[float] = (
        0.125,
        0.1875,
        0.25,
        0.375,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
    )

    def pick_size(value: float, options: Sequence[float]) -> float:
        for opt in options:
            if value <= opt:
                return float(opt)
        return float(math.ceil(value))

    need_len_adj = float(max(need_len, need_wid)) * (1.0 + float(scrap_fraction))
    need_wid_adj = float(min(need_len, need_wid)) * (1.0 + float(scrap_fraction))
    stock_len = pick_size(need_len_adj, stock_lengths)
    stock_wid = pick_size(need_wid_adj, stock_widths)
    if stock_len < stock_wid:
        stock_len, stock_wid = stock_wid, stock_len
    stock_thk = pick_size(float(need_thk), stock_thicknesses)
    return {
        "vendor": "StdGrid",
        "part_no": None,
        "len_in": float(stock_len),
        "wid_in": float(stock_wid),
        "thk_in": float(stock_thk),
        "price_usd": None,
        "min_charge_usd": None,
    }


class VendorCSV(PriceProvider):
    name = "vendor_csv"
    quote_basis = "usd_per_kg"

    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError("VendorCSV requires a file path")
        self.path = path
        self._rows: Dict[str, float] | None = None

    def _load(self) -> None:
        if self._rows is not None:
            return
        self._rows = {}
        if os.path.isfile(self.path):
            with open(self.path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if not row:
                        continue
                    symbol, usd_per_kg, *_ = row
                    try:
                        price = float(usd_per_kg)
                    except ValueError:
                        continue
                    self._rows[symbol.strip().upper()] = price

    def get(self, symbol: str) -> tuple[float, str]:
        self._load()
        assert self._rows is not None
        price = self._rows.get(symbol.upper())
        if price is None:
            raise KeyError(symbol)
        return float(price), os.path.basename(self.path)


__all__ = [
    "VendorCSV",
    "pick_plate_from_mcmaster",
    "pick_from_stdgrid",
]
