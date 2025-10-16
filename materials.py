"""Material helpers extracted from appV5."""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import sys
from collections.abc import Mapping as _MappingABC, Sequence
from typing import Any, Literal, Mapping, overload

from appkit.scrap_helpers import normalize_scrap_pct
from cad_quoter.domain_models import (
    DEFAULT_MATERIAL_KEY,
    MATERIAL_DENSITY_G_CC_BY_KEY,
    MATERIAL_DENSITY_G_CC_BY_KEYWORD,
    MATERIAL_OTHER_KEY,
    coerce_float_or_none as _coerce_float_or_none,
    normalize_material_key,
)
from cad_quoter.llm_overrides import _plate_mass_properties, _plate_mass_from_dims
from cad_quoter.pricing import LB_PER_KG
from cad_quoter.pricing import resolve_material_unit_price as _resolve_material_unit_price_raw
from cad_quoter.pricing.vendor_csv import (
    pick_from_stdgrid as _pick_from_stdgrid,
    pick_plate_from_mcmaster as _pick_plate_from_mcmaster,
)
from cad_quoter.resources import default_catalog_csv as _default_catalog_csv

try:  # Optional dependency: McMaster mutual-TLS API client
    from mcmaster_api import McMasterAPI, load_env as _mcm_load_env  # type: ignore
except Exception:  # pragma: no cover - optional dependency / environment specific
    McMasterAPI = None  # type: ignore[assignment]
    _mcm_load_env = None  # type: ignore[assignment]

try:  # Optional dependency: McMaster catalog helpers
    import cad_quoter.vendors.mcmaster_stock as _mc  # type: ignore
except Exception:  # pragma: no cover - optional dependency / environment specific
    try:  # pragma: no cover - optional dependency / environment specific
        import mcmaster_stock as _mc  # type: ignore
    except Exception:  # pragma: no cover - optional dependency / environment specific
        _mc = None  # type: ignore[assignment]

_DEFAULT_MATERIAL_DENSITY_G_CC = MATERIAL_DENSITY_G_CC_BY_KEY.get(
    DEFAULT_MATERIAL_KEY, 7.85
)
_normalize_lookup_key = normalize_material_key


_log = logging.getLogger(__name__)


STANDARD_PLATE_SIDES_IN = [3, 6, 12, 18, 24, 36, 48, 60]
_MC_CATALOG_CACHE: dict[str, Any] = {}
_STOCK_SCRAP_FRACTION = 0.05


def _nearest_std_side(x_in: float) -> float:
    for s in STANDARD_PLATE_SIDES_IN:
        if s >= max(1.0, x_in):
            return float(s)
    return float(STANDARD_PLATE_SIDES_IN[-1])


def infer_plate_lw_in(geo: Mapping[str, Any] | None) -> tuple[float, float] | None:
    """Infer plate length/width in inches from geometry hints."""

    if not isinstance(geo, _MappingABC):
        return None

    outline = geo.get("outline_bbox")
    if isinstance(outline, _MappingABC):
        L = outline.get("plate_len_in")
        W = outline.get("plate_wid_in")
        if L and W:
            try:
                return float(L), float(W)
            except Exception:
                pass

    area_mm2 = geo.get("plate_bbox_area_mm2")
    if area_mm2:
        try:
            area_in2 = float(area_mm2) / 645.16
            if area_in2 > 0:
                side = math.sqrt(area_in2)
                return (float(side), float(side))
        except Exception:
            return None

    return None


def _resolve_price_per_lb(material_key: str, display_name: str | None = None) -> tuple[float, str]:
    """Return a USD/lb price using metals API or McMaster fallback."""

    key = str(material_key or "").strip()
    display = str(display_name or "").strip() or key or "aluminum"

    price = 0.0
    source = ""

    try:  # pragma: no cover - optional dependency / network access
        from metals_api import price_per_lb_for_material  # type: ignore

        candidate = price_per_lb_for_material(key or display)
        if candidate:
            price = float(candidate)
            if price > 0:
                source = "metals_api"
    except Exception:
        pass

    if price <= 0:
        try:
            from cad_quoter.pricing.materials import get_mcmaster_unit_price
        except Exception:  # pragma: no cover - optional dependency
            get_mcmaster_unit_price = None  # type: ignore[assignment]
        if get_mcmaster_unit_price is not None:
            try:
                mcm_price, mcm_source = get_mcmaster_unit_price(display, unit="lb")
            except Exception:
                mcm_price, mcm_source = None, ""
            if mcm_price and float(mcm_price) > 0:
                price = float(mcm_price)
                source = mcm_source or "mcmaster"

    if price <= 0:
        try:
            resolved_price, resolved_source = _resolve_material_unit_price(display, unit="lb")
        except Exception:
            resolved_price, resolved_source = None, ""
        if resolved_price and float(resolved_price) > 0:
            price = float(resolved_price)
            source = resolved_source or source

    return max(0.0, float(price)), source


def _resolve_material_unit_price(
    material_name: str,
    *,
    unit: str = "kg",
) -> tuple[float, str]:
    """Proxy that respects runtime monkeypatching in the app module."""

    resolver = _resolve_material_unit_price_raw
    try:
        app_module = sys.modules.get("appV5")
        patched = getattr(app_module, "_resolve_material_unit_price", None) if app_module else None
        if callable(patched):
            resolver = patched  # type: ignore[assignment]
    except Exception:
        pass
    return resolver(material_name, unit=unit)


def _mcm_price_for_part(part_number: str) -> float | None:
    """Return the qty=1 unit price for a McMaster part via the mutual-TLS API."""

    part = str(part_number or "").strip()
    if not part:
        return None
    if McMasterAPI is None or _mcm_load_env is None:  # pragma: no cover - optional dependency
        return None
    try:  # pragma: no cover - network call
        env = _mcm_load_env()
        username = env.get("MCMASTER_USER")
        password = env.get("MCMASTER_PASS")
        pfx_path = env.get("MCMASTER_PFX_PATH")
        pfx_password = env.get("MCMASTER_PFX_PASS")
        if not all([username, password, pfx_path]):
            return None
        api = McMasterAPI(
            username=username,
            password=password,
            pfx_path=pfx_path,
            pfx_password=pfx_password or "",
        )
        api.login()
        tiers = api.get_price_tiers(part)
        if not tiers:
            return None
        tier = next(
            (
                t
                for t in tiers
                if (t.get("MinimumQuantity") or 0) <= 1
            ),
            tiers[0],
        )
        amount = tier.get("Amount")
        if isinstance(amount, (int, float)):
            return float(amount)
        try:
            return float(amount)
        except Exception:
            return None
    except Exception as exc:  # pragma: no cover - logging side-effect only
        _log.warning("[mcmaster] price lookup failed for %s: %s", part, exc)
        return None


def _catalog_path_from_env() -> str | None:
    """Return the catalog CSV path from env or the packaged default."""

    path = os.getenv("CATALOG_CSV_PATH")
    if path:
        return path
    try:
        return str(_default_catalog_csv())
    except Exception:
        return None


def _load_mcmaster_catalog(csv_path: str) -> Any:
    """Load and cache the McMaster catalog when helpers are available."""

    if _mc is None:
        return None
    try:
        key = os.path.abspath(csv_path)
    except Exception:
        key = csv_path
    cached = _MC_CATALOG_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        catalog = _mc.load_catalog(csv_path)
    except Exception:
        return None
    _MC_CATALOG_CACHE[key] = catalog
    return catalog


def _fallback_catalog_lookup(
    csv_path: str,
    material_display: str,
    need_L: float,
    need_W: float,
    thk_in: float,
) -> dict[str, Any] | None:
    """Scan the catalog CSV for the smallest plate covering the requirement."""

    def inch_to_float(s: str | None) -> float | None:
        text = (s or "").strip().lower()
        text = (
            text.replace("in.", "")
            .replace("in", "")
            .replace('"', "")
            .replace("″", "")
        )
        match = re.match(r"^(\d+)\s+(\d+)/(\d+)$", text) or re.match(
            r"^(\d+)/(\d+)$",
            text,
        )
        if match and len(match.groups()) == 3:
            return float(match.group(1)) + float(match.group(2)) / float(match.group(3))
        if match and len(match.groups()) == 2:
            return float(match.group(1)) / float(match.group(2))
        try:
            return float(text)
        except Exception:
            return None

    try:
        L_need = max(float(need_L), float(need_W))
        W_need = min(float(need_L), float(need_W))
        thk_val = float(thk_in)
        material_norm = str(material_display or "").strip().lower()
    except Exception:
        return None

    best: dict[str, Any] | None = None
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                mat = (row.get("material") or "").strip().lower()
                if material_norm and material_norm not in mat:
                    continue
                t_val = inch_to_float(row.get("thickness_in"))
                L_val = inch_to_float(row.get("length_in"))
                W_val = inch_to_float(row.get("width_in"))
                if None in (t_val, L_val, W_val):
                    continue
                if abs(float(t_val) - thk_val) > 0.02:
                    continue
                covers = (L_val >= L_need and W_val >= W_need) or (
                    L_val >= W_need and W_val >= L_need
                )
                if not covers:
                    continue
                area = float(L_val) * float(W_val)
                if not best or area < best["area"]:
                    best = {
                        "len_in": float(max(L_val, W_val)),
                        "wid_in": float(min(L_val, W_val)),
                        "thk_in": float(t_val),
                        "part": row.get("part"),
                        "area": area,
                    }
    except Exception:
        return None
    return best


def pick_stock_from_mcmaster(
    material_display: str,
    need_L: float,
    need_W: float,
    thickness_in: float,
    *,
    scrap_fraction: float = _STOCK_SCRAP_FRACTION,
) -> dict[str, Any] | None:
    """Return McMaster stock details with CSV fallback if helpers are missing."""

    try:
        L_val = float(need_L)
        W_val = float(need_W)
        thk_val = float(thickness_in)
    except Exception:
        return None
    if L_val <= 0 or W_val <= 0 or thk_val <= 0:
        return None

    scrap = max(0.0, float(scrap_fraction))
    L_need = max(L_val, W_val) * (1.0 + scrap)
    W_need = min(L_val, W_val) * (1.0 + scrap)
    material_label = str(material_display or "")
    catalog_path = _catalog_path_from_env()

    result: dict[str, Any] = {}
    if _mc is not None and catalog_path:
        catalog = _load_mcmaster_catalog(catalog_path)
        if catalog:
            try:
                item = _mc.choose_item(catalog, material_label, L_need, W_need, thk_val)
            except Exception:
                item = None
            if item:
                fallback_used = False
                length = float(max(item.length, item.width))
                width = float(min(item.length, item.width))
                chosen_thk = float(item.thickness)
                if abs(chosen_thk - thk_val) > 0.05:
                    forced_item = None
                    original_allow = getattr(_mc, "ALLOW_NEXT_THICKER", None)
                    try:
                        if original_allow is not None:
                            setattr(_mc, "ALLOW_NEXT_THICKER", False)
                        forced_item = _mc.choose_item(
                            catalog, material_label, L_need, W_need, thk_val
                        )
                    except Exception:
                        forced_item = None
                    finally:
                        if original_allow is not None:
                            try:
                                setattr(_mc, "ALLOW_NEXT_THICKER", original_allow)
                            except Exception:
                                pass
                    if forced_item and abs(float(forced_item.thickness) - thk_val) <= 0.05:
                        item = forced_item
                        length = float(max(item.length, item.width))
                        width = float(min(item.length, item.width))
                        chosen_thk = float(item.thickness)
                    else:
                        _log.warning(
                            "[stock] Thickness upsize blocked: need %.2f in, got %.2f in; "
                            "falling back to exact-thickness CSV lookup",
                            thk_val,
                            chosen_thk,
                        )
                        csv_hint = os.getenv("CATALOG_CSV_PATH") or catalog_path or ""
                        fallback = _fallback_catalog_lookup(
                            csv_hint, material_label, L_need, W_need, thk_val
                        )
                        if fallback:
                            fallback_used = True
                            part_number = str(fallback.get("part") or "").strip()
                            updates = {
                                "vendor": "McMaster",
                                "len_in": float(fallback["len_in"]),
                                "wid_in": float(fallback["wid_in"]),
                                "thk_in": float(fallback["thk_in"]),
                                "source": "mcmaster-catalog-csv",
                            }
                            if part_number:
                                updates["mcmaster_part"] = part_number
                                updates["part_no"] = part_number
                            result.update(updates)
                        else:
                            result["thickness_upsize_reason"] = (
                                "catalog exact thickness not found; "
                                f"used {chosen_thk:.2f} in"
                            )
                if not fallback_used:
                    result.update(
                        {
                            "vendor": "McMaster",
                            "len_in": length,
                            "wid_in": width,
                            "thk_in": float(item.thickness),
                            "mcmaster_part": item.part,
                            "part_no": item.part,
                            "source": "mcmaster-catalog",
                        }
                    )

    if not result.get("mcmaster_part"):
        csv_path = catalog_path or ""
        if csv_path and os.path.exists(csv_path):
            fallback = _fallback_catalog_lookup(
                csv_path, material_label, L_need, W_need, thk_val
            )
            if fallback:
                part_number = str(fallback.get("part") or "").strip()
                updates = {
                    "len_in": float(fallback["len_in"]),
                    "wid_in": float(fallback["wid_in"]),
                    "thk_in": float(fallback["thk_in"]),
                    "vendor": "McMaster",
                    "source": "mcmaster-catalog-csv",
                }
                if part_number:
                    updates["mcmaster_part"] = part_number
                    updates["part_no"] = part_number
                result.update(updates)
                price = None
                if part_number:
                    try:
                        price = _mcm_price_for_part(part_number)
                    except Exception:
                        price = None
                if price:
                    price_val = float(price)
                    result["price$"] = price_val
                    result["price_usd"] = price_val
                    result["stock_piece_api_price"] = price_val
                    result["stock_piece_api_source"] = "mcmaster_api"

    if (
        result
        and result.get("vendor") == "McMaster"
        and _mc is not None
        and not result.get("stock_piece_api_price")
    ):
        try:
            length_in = float(result.get("len_in") or 0.0)
            width_in = float(result.get("wid_in") or 0.0)
            thk_in = float(result.get("thk_in") or 0.0)
        except Exception:
            length_in = width_in = thk_in = 0.0
        if length_in > 0 and width_in > 0 and thk_in > 0:
            try:
                sku, price_each, _, _ = _mc.lookup_sku_and_price_for_mm(
                    material_label,
                    length_in * 25.4,
                    width_in * 25.4,
                    thk_in * 25.4,
                    qty=1,
                )
            except Exception:
                sku, price_each = None, None
            if sku:
                part_str = str(sku)
                if part_str:
                    result.setdefault("mcmaster_part", part_str)
                    result.setdefault("part_no", part_str)
            if price_each:
                try:
                    price_val = float(price_each)
                except Exception:
                    price_val = None
                if price_val and price_val > 0:
                    result["price$"] = price_val
                    result["price_usd"] = price_val
                    result["stock_piece_api_price"] = price_val
                    result["stock_piece_api_source"] = "mcmaster_api"

    return result or None


def _material_cost_components(
    material_block: Mapping[str, Any] | None,
    *,
    overrides: Mapping[str, Any] | None = None,
    cfg: Any | None = None,
) -> dict[str, Any]:
    """Return normalized material cost components for UI breakdowns."""

    block = material_block or {}
    overrides = overrides or {}

    start_g = (
        block.get("starting_mass_g")
        or block.get("start_mass_g")
        or block.get("start_g")
        or 0.0
    )
    scrap_g = block.get("scrap_mass_g") or 0.0
    try:
        start_lb = float(start_g) * 0.00220462262
    except Exception:
        start_lb = 0.0
    try:
        scrap_lb = float(scrap_g) * 0.00220462262
    except Exception:
        scrap_lb = 0.0

    per_lb = block.get("unit_price_per_lb_usd")
    per_lb_src = block.get("unit_price_per_lb_source") or block.get("unit_price_source")

    stock_piece_usd_raw = block.get("stock_piece_price_usd")
    stock_source = block.get("stock_piece_source")
    stock_piece_usd = None
    try:
        if isinstance(stock_piece_usd_raw, (int, float)) and float(stock_piece_usd_raw) > 0:
            stock_piece_usd = float(stock_piece_usd_raw)
    except Exception:
        stock_piece_usd = None

    if stock_piece_usd is not None:
        base_usd = float(stock_piece_usd)
        base_src = stock_source or "Stock piece"
    else:
        try:
            price_per_lb = float(per_lb or 0.0)
        except Exception:
            price_per_lb = 0.0
        base_usd = float(start_lb) * price_per_lb
        base_src = f"{per_lb_src or 'per-lb'} @ ${price_per_lb:.2f}/lb"

    try:
        tax_usd = float(block.get("material_tax_usd") or block.get("material_tax") or 0.0)
    except Exception:
        tax_usd = 0.0

    scrap_usd_lb = block.get("scrap_price_usd_per_lb")
    if scrap_usd_lb is None:
        try:
            scrap_usd_lb = float(overrides.get("scrap_usd_per_lb"))
        except Exception:
            scrap_usd_lb = None
    recovery = overrides.get("scrap_recovery_pct", overrides.get("scrap_recovery_fraction"))
    try:
        recovery_val = float(recovery) if recovery is not None else 0.85
    except Exception:
        recovery_val = 0.85
    if recovery_val > 1.0 + 1e-6:
        recovery_val = recovery_val / 100.0
    recovery_val = max(0.0, min(1.0, recovery_val))

    try:
        scrap_unit_price = float(scrap_usd_lb or 0.0)
    except Exception:
        scrap_unit_price = 0.0

    scrap_credit = float(scrap_lb) * scrap_unit_price * recovery_val
    explicit_credit = block.get("material_scrap_credit") or block.get("scrap_credit_usd")
    if explicit_credit not in (None, ""):
        try:
            scrap_credit = max(0.0, float(explicit_credit))
        except Exception:
            pass
    scrap_rate_text = None
    if scrap_unit_price > 0 and scrap_lb > 0:
        scrap_rate_text = f"${scrap_unit_price:.2f}/lb × {recovery_val:.0%}"

    explicit_base_pre_credit = None
    for key in (
        "material_cost_before_credit",
        "material_cost_pre_credit",
        "material_cost_pre_scrap",
        "material_cost_before_scrap",
        "material_base_cost",
    ):
        explicit_base_pre_credit = _coerce_float_or_none(block.get(key))
        if explicit_base_pre_credit is not None:
            break

    net_candidate = None
    if explicit_base_pre_credit is None:
        for key in (
            "total_material_cost",
            "material_cost",
            "material_direct_cost",
            "material_total_cost",
            "total_cost",
        ):
            net_candidate = _coerce_float_or_none(block.get(key))
            if net_candidate is not None:
                break
        if net_candidate is not None:
            explicit_base_pre_credit = float(net_candidate) + float(scrap_credit)

    if explicit_base_pre_credit is not None:
        try:
            base_usd = float(explicit_base_pre_credit)
        except Exception:
            base_usd = float(base_usd)
        base_src_candidates = (
            block.get("material_cost_source"),
            block.get("material_cost_basis"),
            block.get("material_cost_label"),
            block.get("material_source"),
        )
        for candidate in base_src_candidates:
            if candidate in (None, ""):
                continue
            try:
                text = str(candidate).strip()
            except Exception:
                text = ""
            if text:
                base_src = text
                break

    net_usd = float(base_usd) - float(scrap_credit)
    if net_usd < 0:
        net_usd = 0.0
    net_usd = round(net_usd, 2)

    total = float(base_usd) + float(tax_usd) - float(scrap_credit)
    if total < 0:
        total = 0.0
    total = round(total, 2)

    return {
        "base_usd": round(base_usd, 2),
        "base_source": base_src,
        "tax_usd": round(tax_usd, 2),
        "scrap_credit_usd": round(scrap_credit, 2) if scrap_credit else 0.0,
        "scrap_rate_text": scrap_rate_text,
        "stock_piece_usd": round(stock_piece_usd, 2) if stock_piece_usd else None,
        "stock_source": stock_source,
        "net_usd": net_usd,
        "total_usd": total,
    }


def _compute_material_block(
    geo_ctx: dict,
    material_key: str,
    density_g_cc: float | None,
    scrap_pct: float,
    *,
    stock_price_source: str | None = None,
):
    """Produce a normalized material record including weights and cost."""

    t_in = float(geo_ctx.get("thickness_in") or 0.0)
    if not t_in:
        t_in = float(geo_ctx.get("thickness_mm", 0) / 25.4 or 0.0)
    if not t_in:
        t_in = float(geo_ctx.get("thickness_in_guess") or 0.0)

    dims = infer_plate_lw_in(geo_ctx)
    if not dims or not t_in:
        return {
            "material": geo_ctx.get("material_display") or material_key,
            "stock_L_in": None,
            "stock_W_in": None,
            "stock_T_in": t_in or None,
            "start_lb": 0.0,
            "net_lb": 0.0,
            "scrap_lb": 0.0,
            "scrap_pct": scrap_pct,
            "source": "insufficient-geometry",
            "stock_dims_in": None,
            "supplier_min$": 0.0,
            "stock_price$": None,
            "total_material_cost": 0.0,
        }

    L_in, W_in = dims
    material_label = str(geo_ctx.get("material_display") or material_key or "")
    try:
        stock_info = pick_stock_from_mcmaster(
            material_label,
            float(L_in),
            float(W_in),
            float(t_in),
        )
    except Exception:
        stock_info = None
    if not isinstance(stock_info, dict) or not stock_info:
        try:
            stock_info = _pick_plate_from_mcmaster(
                material_label,
                float(L_in),
                float(W_in),
                float(t_in),
            )
        except Exception:
            stock_info = None
    if not isinstance(stock_info, dict) or not stock_info:
        stock_info = _pick_from_stdgrid(float(L_in), float(W_in), float(t_in))

    stock_L_in = float(stock_info.get("len_in") or L_in)
    stock_W_in = float(stock_info.get("wid_in") or W_in)
    stock_T_in = float(stock_info.get("thk_in") or t_in)
    vendor_label = str(stock_info.get("vendor") or "StdGrid")
    part_no = stock_info.get("part_no")
    stock_price_val = _coerce_float_or_none(stock_info.get("price_usd"))
    stock_price = float(stock_price_val) if stock_price_val and stock_price_val > 0 else None
    stock_supplier_min_val = _coerce_float_or_none(stock_info.get("min_charge_usd"))
    stock_supplier_min = float(stock_supplier_min_val) if stock_supplier_min_val else 0.0
    stock_piece_api_price = _coerce_float_or_none(stock_info.get("stock_piece_api_price"))
    if stock_piece_api_price is not None and stock_piece_api_price <= 0:
        stock_piece_api_price = None
    stock_piece_api_source = stock_info.get("stock_piece_api_source")

    rho = float(density_g_cc or 2.70)
    vol_net_in3 = float(L_in) * float(W_in) * float(t_in)
    vol_start_in3 = float(stock_L_in) * float(stock_W_in) * float(stock_T_in)
    g_cc_to_lb_in3 = 0.0361273
    net_lb = vol_net_in3 * rho * g_cc_to_lb_in3
    start_lb = vol_start_in3 * rho * g_cc_to_lb_in3
    scrap_lb_geom = max(0.0, start_lb - net_lb)
    min_scrap_lb = max(0.0, start_lb * float(scrap_pct))
    scrap_lb = max(scrap_lb_geom, min_scrap_lb)
    net_after_scrap = max(0.0, start_lb - scrap_lb)

    provenance: list[str] = []
    price_source = ""
    price_per_lb = 0.0

    unit_price_each = stock_price if stock_price and stock_price > 0 else None

    unit_price_usd: float | None = None
    api_price = None
    api_price_source: str | None = None
    if part_no and str(stock_price_source or "").strip().lower() == "mcmaster_api":
        api_price = _mcm_price_for_part(str(part_no))
    if api_price and api_price > 0:
        unit_price_each = float(api_price)
        stock_price = float(api_price)
        unit_price_usd = float(api_price)
        if start_lb > 0:
            price_per_lb = float(unit_price_each) / float(start_lb)
        api_price_source = f"McMaster API (qty=1, part={part_no})"
        price_source = api_price_source
        provenance.append(price_source)

    if price_per_lb <= 0:
        resolved_per_lb, resolved_source = _resolve_price_per_lb(material_key, material_label)
        try:
            price_per_lb = float(resolved_per_lb)
        except (TypeError, ValueError):
            price_per_lb = 0.0
        if not price_source:
            price_source = resolved_source or ""

    if (unit_price_each is None or unit_price_each <= 0) and price_per_lb > 0:
        unit_price_each = float(net_after_scrap) * float(price_per_lb)

    if unit_price_each is not None and unit_price_each > 0:
        stock_price = float(unit_price_each)
        if not price_source:
            price_source = vendor_label or "stock"

    supplier_min_candidates = (
        geo_ctx.get("supplier_min$"),
        geo_ctx.get("supplier_min"),
        geo_ctx.get("supplier_min_charge"),
        geo_ctx.get("supplier_minimum_charge"),
        geo_ctx.get("minimum_order$"),
        geo_ctx.get("minimum_order"),
    )
    supplier_min = 0.0
    for candidate in supplier_min_candidates:
        val = _coerce_float_or_none(candidate)
        if val is not None and math.isfinite(val):
            supplier_min = max(supplier_min, float(val))
    supplier_min = max(0.0, max(supplier_min, stock_supplier_min))

    total_mat_cost = max(float(unit_price_each or 0.0), float(supplier_min))

    source_note = vendor_label or "stock"
    if price_source:
        source_note = price_source

    result = {
        "material": geo_ctx.get("material_display") or material_key,
        "stock_L_in": float(stock_L_in),
        "stock_W_in": float(stock_W_in),
        "stock_T_in": float(stock_T_in),
        "stock_dims_in": (float(stock_L_in), float(stock_W_in), float(stock_T_in)),
        "start_lb": float(start_lb),
        "starting_weight_lb": float(start_lb),
        "net_lb": float(net_after_scrap),
        "net_weight_lb": float(net_after_scrap),
        "scrap_lb": float(scrap_lb),
        "scrap_weight_lb": float(scrap_lb),
        "scrap_pct": float(scrap_pct),
        "price_per_lb": float(price_per_lb),
        "price_per_lb$": float(price_per_lb),
        "price_source": price_source,
        "supplier_min": float(supplier_min),
        "supplier_min$": float(supplier_min),
        "source": source_note,
        "stock_vendor": vendor_label,
        "part_no": part_no,
        "stock_price$": float(stock_price) if stock_price is not None else None,
        "unit_price_each$": float(stock_price) if stock_price is not None else None,
        "unit_price$": float(stock_price) if stock_price is not None else None,
        "supplier_min_charge$": float(supplier_min),
        "total_material_cost": float(total_mat_cost),
    }

    if api_price and api_price > 0 and api_price_source:
        result["stock_piece_price_usd"] = float(api_price)
        result["stock_piece_source"] = api_price_source

    if price_source:
        result["unit_price_source"] = price_source
        if price_source.startswith("McMaster API"):
            result["unit_price_confidence"] = "high"
    if unit_price_usd is not None:
        result["unit_price_usd"] = float(unit_price_usd)
    if provenance:
        result["provenance"] = provenance
    if stock_piece_api_price is not None and stock_piece_api_price > 0:
        result["stock_piece_price_usd"] = float(stock_piece_api_price)
        result["stock_piece_source"] = stock_piece_api_source or price_source

    return result


def _compute_scrap_mass_g(
    *,
    removal_mass_g_est: float | str | None,
    scrap_pct_raw: float | str | None,
    effective_mass_g: float | str | None,
    net_mass_g: float | str | None,
    prefer_pct: bool = False,
) -> float | None:
    """Return the scrap mass for display/credit in grams.

    Rendering pulls data from several sources (UI state, planner output,
    persisted quotes) so the inputs might be strings, numbers or ``None``.  We
    therefore coerce values defensively and pick the most reliable signal in
    order: an explicit removal estimate, the scrap percentage applied to the
    best mass reference, and finally the difference between effective and net
    masses when all else fails.
    """

    removal_mass = _coerce_float_or_none(removal_mass_g_est)
    if removal_mass is not None:
        removal_mass = max(0.0, float(removal_mass))
        if removal_mass > 0:
            return removal_mass

    effective_mass = _coerce_float_or_none(effective_mass_g)
    net_mass = _coerce_float_or_none(net_mass_g)

    if (
        not prefer_pct
        and effective_mass is not None
        and net_mass is not None
        and float(effective_mass) > float(net_mass)
    ):
        return float(effective_mass) - float(net_mass)

    scrap_frac = normalize_scrap_pct(scrap_pct_raw)
    if scrap_frac is not None and scrap_frac > 0:
        base_mass = _coerce_float_or_none(effective_mass_g)
        if base_mass is None or base_mass <= 0:
            base_mass = _coerce_float_or_none(net_mass_g)
        if base_mass is not None and base_mass > 0:
            return max(0.0, float(base_mass)) * float(scrap_frac)

    if (
        effective_mass is not None
        and net_mass is not None
        and float(effective_mass) > float(net_mass)
    ):
        return float(effective_mass) - float(net_mass)

    if removal_mass is not None:
        return removal_mass

    return None


def _material_price_per_g_from_choice(
    choice: str, material_lookup: dict[str, float]
) -> tuple[float | None, str]:
    """Resolve a material price per-gram along with its source label."""

    choice = str(choice or "").strip()
    if not choice:
        return None, ""

    norm_choice = _normalize_lookup_key(choice)
    if norm_choice == MATERIAL_OTHER_KEY:
        return None, ""

    price = material_lookup.get(norm_choice)
    if price is not None:
        return float(price), "material_lookup"

    try:
        price_per_kg, source_label = _resolve_material_unit_price(choice, unit="kg")
    except Exception:
        return None, ""
    if not price_per_kg:
        return None, source_label or ""
    return float(price_per_kg) / 1000.0, source_label or ""


def _material_price_from_choice(choice: str, material_lookup: dict[str, float]) -> float | None:
    """Resolve a material price per-gram for the editor helpers."""

    price_per_g, _source = _material_price_per_g_from_choice(choice, material_lookup)
    return price_per_g


def _material_family(material: object | None) -> str:
    name = _normalize_lookup_key(str(material or ""))
    if not name:
        return "steel"
    if any(tag in name for tag in ("alum", "6061", "7075", "2024", "5052", "5083")):
        return "alum"
    if any(tag in name for tag in ("stainless", "17 4", "316", "304", "ss")):
        return "stainless"
    if any(tag in name for tag in ("titanium", "ti-6al-4v", "ti64", "grade 5")):
        return "titanium"
    if any(tag in name for tag in ("copper", "c110", "cu")):
        return "copper"
    if any(tag in name for tag in ("brass", "c360", "c260")):
        return "brass"
    if any(
        tag in name
        for tag in ("plastic", "uhmw", "delrin", "acetal", "peek", "abs", "nylon")
    ):
        return "plastic"

    return "steel"


def _density_for_material(
    material: object | None, default: float = _DEFAULT_MATERIAL_DENSITY_G_CC
) -> float:
    """Return a rough density guess (g/cc) for the requested material."""

    raw = str(material or "").strip()
    if not raw:
        return default

    normalized = _normalize_lookup_key(raw)
    collapsed = normalized.replace(" ", "")

    for token in (normalized, collapsed):
        if token and token in MATERIAL_DENSITY_G_CC_BY_KEYWORD:
            return MATERIAL_DENSITY_G_CC_BY_KEYWORD[token]

    for token, density in MATERIAL_DENSITY_G_CC_BY_KEYWORD.items():
        if not token:
            continue
        if token in normalized or token in collapsed:
            return density

    lower = raw.lower()
    if any(tag in lower for tag in ("plastic", "uhmw", "delrin", "acetal", "peek", "abs", "nylon")):
        return 1.45
    if any(tag in lower for tag in ("foam", "poly", "composite")):
        return 1.10
    if any(tag in lower for tag in ("magnesium", "az31", "az61")):
        return 1.80
    if "graphite" in lower:
        return 1.85

    return default


@overload

def net_mass_kg(
    plate_L_in,
    plate_W_in,
    t_in,
    hole_d_mm,
    density_g_cc: float = 7.85,
    *,
    return_removed_mass: Literal[True],
) -> tuple[float, float] | tuple[None, None]:
    ...


@overload

def net_mass_kg(
    plate_L_in,
    plate_W_in,
    t_in,
    hole_d_mm,
    density_g_cc: float = 7.85,
    *,
    return_removed_mass: Literal[False] = False,
) -> float | None:
    ...


def net_mass_kg(
    plate_L_in,
    plate_W_in,
    t_in,
    hole_d_mm,
    density_g_cc: float = 7.85,
    *,
    return_removed_mass: bool = False,
):
    """Estimate the net mass of a rectangular plate and optional removed material."""

    try:
        plate_mass_fn = _plate_mass_properties  # type: ignore[name-defined]
    except NameError:
        plate_mass_fn = None  # type: ignore[assignment]

    if not callable(plate_mass_fn):

        def _plate_mass_properties_fallback(
            plate_L_in_fallback: Any,
            plate_W_in_fallback: Any,
            t_in_fallback: Any,
            density_g_cc_fallback: Any,
            hole_d_mm_fallback: Any,
        ) -> tuple[float | None, float | None]:
            length_in = _coerce_float_or_none(plate_L_in_fallback)
            width_in = _coerce_float_or_none(plate_W_in_fallback)
            thickness_in = _coerce_float_or_none(t_in_fallback)
            density = _coerce_float_or_none(density_g_cc_fallback)
            if (
                length_in is None
                or width_in is None
                or thickness_in is None
                or density is None
                or density <= 0
            ):
                return (None, None)
            try:
                removed_mass_g = 0.0
                for dia_mm, qty in hole_d_mm_fallback or []:
                    dia_mm_val = _coerce_float_or_none(dia_mm)
                    qty_val = _coerce_float_or_none(qty)
                    if dia_mm_val and qty_val:
                        removed_volume_cm3 = (
                            math.pi * (float(dia_mm_val) / 2.0) ** 2 * qty_val * float(t_in_fallback) * 0.0163871
                        )
                        removed_mass_g += removed_volume_cm3 * float(density)
            except Exception:
                removed_mass_g = None
            volume_cm3 = float(length_in) * float(width_in) * float(thickness_in) * 0.0163871
            net_mass_g = volume_cm3 * float(density)
            if removed_mass_g is not None:
                net_mass_g = max(0.0, net_mass_g - removed_mass_g)
            return (net_mass_g / 1000.0, removed_mass_g / 1000.0 if removed_mass_g is not None else None)

        plate_mass_fn = _plate_mass_properties_fallback

    net_mass, removed_mass = plate_mass_fn(
        plate_L_in,
        plate_W_in,
        t_in,
        density_g_cc,
        hole_d_mm,
    )
    if return_removed_mass:
        if net_mass is None:
            return (None, None)
        return net_mass, removed_mass
    return net_mass


def plan_stock_blank(
    plate_len_in: float | None,
    plate_wid_in: float | None,
    thickness_in: float | None,
    density_g_cc: float | None,
    hole_families: dict | None,
) -> dict[str, Any]:
    if not plate_len_in or not plate_wid_in or not thickness_in:
        return {}
    stock_lengths = [6, 8, 10, 12, 18, 24, 36, 48]
    stock_widths = [6, 8, 10, 12, 18, 24, 36]
    stock_thicknesses = [0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    def pick_size(value: float, options: Sequence[float]) -> float:
        for opt in options:
            if value <= opt:
                return opt
        return math.ceil(value)

    stock_len = pick_size(float(plate_len_in) * 1.05, stock_lengths)
    stock_wid = pick_size(float(plate_wid_in) * 1.05, stock_widths)
    stock_thk = pick_size(float(thickness_in) * 1.05, stock_thicknesses)
    volume_in3 = float(plate_len_in) * float(plate_wid_in) * float(thickness_in)
    hole_area = 0.0
    holes_mm: list[float] = []
    if hole_families:
        for dia_in, qty in hole_families.items():
            try:
                d = float(dia_in)
                q = float(qty)
            except Exception:
                continue
            hole_area += math.pi * (d / 2.0) ** 2 * q
            hole_count = int(round(q)) if q and q > 0 else 0
            if hole_count > 0:
                holes_mm.extend([d * 25.4] * hole_count)
    net_volume_in3 = max(volume_in3 - hole_area * float(thickness_in), 0.0)

    density_val = _coerce_float_or_none(density_g_cc)
    if density_val and density_val > 0:
        part_dims_in = (plate_len_in, plate_wid_in, thickness_in)
        part_mass_kg, _ = _plate_mass_from_dims(
            float(plate_len_in) * 25.4,
            float(plate_wid_in) * 25.4,
            float(thickness_in) * 25.4,
            density_val,
            dims_in=part_dims_in,
            hole_d_mm=holes_mm,
        )
        stock_dims_in = (stock_len, stock_wid, stock_thk)
        stock_mass_kg, _ = _plate_mass_from_dims(
            float(stock_len) * 25.4,
            float(stock_wid) * 25.4,
            float(stock_thk) * 25.4,
            density_val,
            dims_in=stock_dims_in,
            hole_d_mm=(),
        )
    else:
        part_mass_kg = None
        stock_mass_kg = None

    part_mass_lb = (part_mass_kg * LB_PER_KG) if part_mass_kg is not None else None
    stock_volume_in3 = stock_len * stock_wid * stock_thk
    stock_mass_lb = (stock_mass_kg * LB_PER_KG) if stock_mass_kg is not None else None
    return {
        "stock_len_in": round(stock_len, 3),
        "stock_wid_in": round(stock_wid, 3),
        "stock_thk_in": round(stock_thk, 3),
        "part_volume_in3": round(volume_in3, 3),
        "stock_volume_in3": round(stock_volume_in3, 3),
        "net_volume_in3": round(net_volume_in3, 3),
        "part_mass_lb": round(part_mass_lb, 3) if part_mass_lb else None,
        "stock_mass_lb": round(stock_mass_lb, 3) if stock_mass_lb else None,
    }
