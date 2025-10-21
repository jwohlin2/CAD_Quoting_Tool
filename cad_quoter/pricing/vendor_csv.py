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
from typing import Any, Dict, Iterable, Mapping, Sequence

from cad_quoter.pricing.mcmaster_helpers import (
    load_mcmaster_catalog_rows,
    pick_mcmaster_plate_sku,
    _coerce_inches_value as _coerce_inches_value,
)
from cad_quoter.vendors.mcmaster_stock import norm_material

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


def _extract_price_metadata(row: Mapping[str, Any]) -> tuple[str, str | None, float | None, float | None]:
    vendor = str(row.get("vendor") or row.get("Vendor") or "McMaster").strip() or "McMaster"
    part_no = str(
        row.get("mcmaster_part")
        or row.get("part")
        or row.get("sku")
        or ""
    ).strip() or None
    price = _coerce_float(
        row.get("price_usd")
        or row.get("Price_usd")
        or row.get("price")
        or row.get("Price")
    )
    min_charge = _coerce_float(
        row.get("min_charge_usd")
        or row.get("minimum_charge_usd")
        or row.get("min_charge")
        or row.get("MinimumCharge")
    )
    return vendor, part_no, price, min_charge


def _iter_candidate_thicknesses(
    rows: Iterable[Mapping[str, Any]],
    need_thk: float,
    *,
    allow_thickness_upsize: bool,
    thickness_tolerance: float,
) -> Iterable[float]:
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        thickness = _coerce_inches_value(
            row.get("thickness_in")
            or row.get("T_in")
            or row.get("thk_in")
            or row.get("thickness")
        )
        if thickness is None or thickness <= 0:
            continue
        if thickness + 1e-4 < need_thk:
            continue
        if not allow_thickness_upsize and abs(thickness - need_thk) > thickness_tolerance + 1e-9:
            continue
        yield float(thickness)


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

    try:
        scrap_fraction_val = float(scrap_fraction)
    except Exception:
        scrap_fraction_val = _SCRAP_FRACTION
    scrap_fraction_val = max(0.0, scrap_fraction_val)

    try:
        tol = float(thickness_tolerance)
    except Exception:
        tol = 0.02
    tol = max(0.0, tol)

    rows = load_mcmaster_catalog_rows(catalog_path)
    if not rows:
        return None

    norm_mat = norm_material(material or "")
    need_thk_val = float(need_thk)
    allow_upsize = bool(allow_thickness_upsize)

    def _pick_with_thickness(
        target_len: float, target_wid: float, target_thk: float
    ) -> dict[str, Any] | None:
        return pick_mcmaster_plate_sku(
            target_len,
            target_wid,
            target_thk,
            material_key=norm_mat or material,
            catalog_rows=rows,
        )

    scrap_options = [scrap_fraction_val]
    if scrap_fraction_val > 0:
        scrap_options.append(0.0)

    for scrap_opt in scrap_options:
        need_len_adj = float(max(need_len, need_wid)) * (1.0 + float(scrap_opt))
        need_wid_adj = float(min(need_len, need_wid)) * (1.0 + float(scrap_opt))

        thickness_values: list[float] = []
        seen: set[float] = set()

        if need_thk_val not in seen:
            thickness_values.append(need_thk_val)
            seen.add(need_thk_val)

        for row_thk in sorted(
            _iter_candidate_thicknesses(
                rows,
                need_thk_val,
                allow_thickness_upsize=allow_upsize,
                thickness_tolerance=tol,
            ),
            key=lambda value: (abs(value - need_thk_val), value),
        ):
            if row_thk in seen:
                continue
            thickness_values.append(row_thk)
            seen.add(row_thk)

        for target_thk in thickness_values:
            candidate = _pick_with_thickness(need_len_adj, need_wid_adj, target_thk)
            if not candidate:
                continue

            thickness = float(candidate.get("thk_in") or target_thk)
            thickness_diff = abs(thickness - need_thk_val)
            if not allow_upsize and thickness_diff > tol + 1e-9:
                continue

            part_no = candidate.get("mcmaster_part")
            matched_row: Mapping[str, Any] | None = None
            if part_no:
                for row in rows:
                    row_part = str(
                        row.get("mcmaster_part")
                        or row.get("part")
                        or row.get("sku")
                        or ""
                    ).strip()
                    if row_part and row_part == part_no:
                        matched_row = row
                        break

            vendor, part, price, min_charge = _extract_price_metadata(matched_row or {})
            if part_no and not part:
                part = part_no
            length = float(candidate.get("len_in") or need_len_adj)
            width = float(candidate.get("wid_in") or need_wid_adj)
            if length < width:
                length, width = width, length
            return {
                "vendor": vendor,
                "part_no": part,
                "len_in": float(length),
                "wid_in": float(width),
                "thk_in": thickness,
                "thickness_diff_in": float(thickness_diff),
                "price_usd": price if price and price > 0 else None,
                "min_charge_usd": min_charge if min_charge and min_charge > 0 else None,
            }

    return None


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
