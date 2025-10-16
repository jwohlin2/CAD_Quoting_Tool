"""Material helpers extracted from appV5."""

from __future__ import annotations

import logging
import math
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

try:  # Optional dependency: McMaster mutual-TLS API client
    from mcmaster_api import McMasterAPI, load_env as _mcm_load_env  # type: ignore
except Exception:  # pragma: no cover - optional dependency / environment specific
    McMasterAPI = None  # type: ignore[assignment]
    _mcm_load_env = None  # type: ignore[assignment]

_DEFAULT_MATERIAL_DENSITY_G_CC = MATERIAL_DENSITY_G_CC_BY_KEY.get(
    DEFAULT_MATERIAL_KEY, 7.85
)
_normalize_lookup_key = normalize_material_key


_log = logging.getLogger(__name__)


STANDARD_PLATE_SIDES_IN = [3, 6, 12, 18, 24, 36, 48, 60]


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
    if part_no and str(stock_price_source or "").strip().lower() == "mcmaster_api":
        api_price = _mcm_price_for_part(str(part_no))
    if api_price and api_price > 0:
        unit_price_each = float(api_price)
        stock_price = float(api_price)
        unit_price_usd = float(api_price)
        if start_lb > 0:
            price_per_lb = float(unit_price_each) / float(start_lb)
        price_source = f"McMaster API (qty=1, part={part_no})"
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

    if price_source:
        result["unit_price_source"] = price_source
        if price_source.startswith("McMaster API"):
            result["unit_price_confidence"] = "high"
    if unit_price_usd is not None:
        result["unit_price_usd"] = float(unit_price_usd)
    if provenance:
        result["provenance"] = provenance

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
