"""Helpers for handling scrap estimates and stock plans."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Iterable

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.material_density import (
    MATERIAL_DENSITY_G_CC_BY_KEY,
    MATERIAL_DENSITY_G_CC_BY_KEYWORD,
    normalize_material_key as _normalize_lookup_key,
)
from cad_quoter.utils.numeric import coerce_positive_float as _coerce_positive_float
from cad_quoter.llm_overrides import _plate_mass_properties

SCRAP_DEFAULT_GUESS = 0.15
HOLE_SCRAP_MULT = 1.0  # tune 0.5–1.5 if you want holes to “count” more/less
HOLE_SCRAP_CAP = 0.25
OPS_TOTAL_OVERRIDES_DEFAULT: Mapping[str, int] = {}


@dataclass
class ScrapCreditDecision:
    """Details about an applied scrap credit."""

    mass_lb: float | None = None
    amount_usd: float | None = None
    unit_price_usd_per_lb: float | None = None
    recovery_fraction: float | None = None
    source: str | None = None
    wieland_price_usd_per_lb: float | None = None
    override_amount_usd: float | None = None
    override_unit_price_usd_per_lb: float | None = None
    override_recovery_fraction: float | None = None


@dataclass
class MaterialScrapResolution:
    """Structured result describing material scrap resolution."""

    scrap_fraction: float
    scrap_source: str
    scrap_source_label: str
    hole_reason_applied: bool = False
    density_g_cc: float | None = None
    net_volume_cm3: float | None = None
    removal_mass_g: float | None = None
    scrap_credit: ScrapCreditDecision = field(default_factory=ScrapCreditDecision)

    def finalize_credit(
        self,
        *,
        material_block: Mapping[str, Any] | None,
        overrides: Mapping[str, Any] | None,
        geo_context: Mapping[str, Any] | None,
        material_display: str | None,
        material_group_display: str | None,
        wieland_lookup: Callable[[str | None], float | None] | None,
        default_recovery_fraction: float,
        default_scrap_price_usd_per_lb: float,
    ) -> "MaterialScrapResolution":
        """Populate scrap credit fields using ``material_block`` details."""

        scrap_mass_lb_val: float | None = None
        if isinstance(material_block, Mapping):
            scrap_mass_lb_val = _coerce_float_or_none(
                material_block.get("scrap_weight_lb") or material_block.get("scrap_lb")
            )
        scrap_mass_lb = (
            float(scrap_mass_lb_val) if scrap_mass_lb_val is not None and scrap_mass_lb_val > 0 else None
        )

        credit_override_amount: float | None = None
        unit_price_override: float | None = None
        recovery_override: float | None = None

        if isinstance(overrides, Mapping):
            for key in (
                "Material Scrap / Remnant Value",
                "material_scrap_credit",
                "material_scrap_credit_usd",
                "scrap_credit_usd",
            ):
                override_val = overrides.get(key)
                if override_val in (None, ""):
                    continue
                coerced_override = _coerce_float_or_none(override_val)
                if coerced_override is not None:
                    credit_override_amount = abs(float(coerced_override))
                    break

            for key in (
                "scrap_credit_unit_price_usd_per_lb",
                "scrap_price_usd_per_lb",
                "scrap_usd_per_lb",
                "Scrap Price ($/lb)",
                "Scrap Credit ($/lb)",
                "Scrap Unit Price ($/lb)",
            ):
                price_val = overrides.get(key)
                if price_val in (None, ""):
                    continue
                coerced_price = _coerce_float_or_none(price_val)
                if coerced_price is not None and coerced_price >= 0:
                    unit_price_override = float(coerced_price)
                    break

            for key in (
                "scrap_recovery_pct",
                "Scrap Recovery (%)",
                "scrap_recovery_fraction",
            ):
                recovery_val = overrides.get(key)
                if recovery_val in (None, ""):
                    continue
                coerced_recovery = _coerce_float_or_none(recovery_val)
                if coerced_recovery is None:
                    continue
                recovery_fraction = float(coerced_recovery)
                if recovery_fraction > 1.0 + 1e-6:
                    recovery_fraction = recovery_fraction / 100.0
                recovery_override = max(0.0, min(1.0, recovery_fraction))
                break

        scrap_credit_amount: float | None = None
        scrap_price_used: float | None = None
        scrap_recovery_used: float | None = None
        scrap_credit_source: str | None = None
        wieland_scrap_price: float | None = None

        if credit_override_amount is not None:
            scrap_credit_amount = max(0.0, float(credit_override_amount))
            scrap_credit_source = "override_amount"
        elif scrap_mass_lb is not None:
            recovery = recovery_override if recovery_override is not None else default_recovery_fraction
            scrap_recovery_used = recovery
            if unit_price_override is not None:
                scrap_price_used = max(0.0, float(unit_price_override))
                scrap_credit_source = "override_unit_price"
            else:
                family_hint: str | None = None
                if isinstance(geo_context, Mapping):
                    family_hint = (
                        geo_context.get("material_family")
                        or geo_context.get("material_group")
                    )
                family_hint = family_hint or material_group_display or material_display
                if family_hint is not None and not isinstance(family_hint, str):
                    family_hint = str(family_hint)
                if isinstance(family_hint, str):
                    family_hint = family_hint.strip() or None

                price_candidate: float | None = None
                if wieland_lookup is not None:
                    try:
                        price_candidate = wieland_lookup(family_hint)
                    except Exception:
                        price_candidate = None
                if price_candidate is not None:
                    wieland_scrap_price = float(price_candidate)
                    scrap_price_used = float(price_candidate)
                    scrap_credit_source = "wieland"
                else:
                    scrap_price_used = float(default_scrap_price_usd_per_lb)
                    scrap_credit_source = "default"

            scrap_credit_amount = float(scrap_mass_lb) * float(scrap_price_used or 0.0) * float(
                scrap_recovery_used or 0.0
            )

        self.scrap_credit = ScrapCreditDecision(
            mass_lb=scrap_mass_lb,
            amount_usd=(float(scrap_credit_amount) if scrap_credit_amount is not None else None),
            unit_price_usd_per_lb=(
                float(scrap_price_used) if scrap_price_used is not None else None
            ),
            recovery_fraction=(
                float(scrap_recovery_used) if scrap_recovery_used is not None else None
            ),
            source=scrap_credit_source,
            wieland_price_usd_per_lb=(
                float(wieland_scrap_price) if wieland_scrap_price is not None else None
            ),
            override_amount_usd=(
                float(credit_override_amount) if credit_override_amount is not None else None
            ),
            override_unit_price_usd_per_lb=(
                float(unit_price_override) if unit_price_override is not None else None
            ),
            override_recovery_fraction=(
                float(recovery_override) if recovery_override is not None else None
            ),
        )

        return self
def _holes_removed_mass_g(geo: Mapping[str, Any] | None) -> float | None:
    """Estimate mass removed by holes using geometry context."""

    if not isinstance(geo, Mapping):
        return None

    t_in = _coerce_float_or_none(geo.get("thickness_in"))
    if t_in is None:
        t_mm = _coerce_positive_float(geo.get("thickness_mm"))
        if t_mm is not None:
            t_in = float(t_mm) / 25.4
    if t_in is None or t_in <= 0:
        return None

    hole_diams_mm: list[float] = []
    raw_list = geo.get("hole_diams_mm")
    if isinstance(raw_list, Sequence) and not isinstance(raw_list, (str, bytes, bytearray)):
        for v in raw_list:
            val = _coerce_positive_float(v)
            if val is not None:
                hole_diams_mm.append(val)
    if not hole_diams_mm:
        families = geo.get("hole_diam_families_in")
        if isinstance(families, Mapping):
            for dia_in, qty in families.items():
                d_in = _coerce_positive_float(dia_in)
                q = _coerce_float_or_none(qty)
                if d_in and q and q > 0:
                    hole_diams_mm.extend([d_in * 25.4] * int(round(q)))
    if not hole_diams_mm:
        return None

    density_g_cc = _coerce_float_or_none(geo.get("density_g_cc"))
    if density_g_cc in (None, 0.0):
        material_text = geo.get("material") or geo.get("material_name")
        if isinstance(material_text, str) and material_text.strip():
            norm_key = _normalize_lookup_key(material_text)
            collapsed = norm_key.replace(" ", "")
            density_g_cc = (
                MATERIAL_DENSITY_G_CC_BY_KEYWORD.get(norm_key)
                or MATERIAL_DENSITY_G_CC_BY_KEYWORD.get(collapsed)
                or MATERIAL_DENSITY_G_CC_BY_KEY.get(norm_key)
            )
            if not density_g_cc:
                for token, density in MATERIAL_DENSITY_G_CC_BY_KEYWORD.items():
                    if token and (token in norm_key or token in collapsed):
                        density_g_cc = density
                        break
    if density_g_cc is None or density_g_cc <= 0:
        return None

    plate_len_in = _coerce_float_or_none(geo.get("plate_len_in"))
    plate_wid_in = _coerce_float_or_none(geo.get("plate_wid_in"))
    if plate_len_in is None:
        plate_len_mm = _coerce_positive_float(geo.get("plate_len_mm"))
        if plate_len_mm is not None:
            plate_len_in = float(plate_len_mm) / 25.4
    if plate_wid_in is None:
        plate_wid_mm = _coerce_positive_float(geo.get("plate_wid_mm"))
        if plate_wid_mm is not None:
            plate_wid_in = float(plate_wid_mm) / 25.4

    length_in = plate_len_in if plate_len_in and plate_len_in > 0 else 1.0
    width_in = plate_wid_in if plate_wid_in and plate_wid_in > 0 else 1.0

    _, removed_mass_g = _plate_mass_properties(
        length_in,
        width_in,
        t_in,
        density_g_cc,
        hole_diams_mm,
    )

    if removed_mass_g is None or removed_mass_g <= 0:
        return None

    return removed_mass_g


def build_drill_groups_from_geometry(
    hole_diams_mm: Sequence[Any] | None,
    thickness_in: Any | None,
    ops_claims: Mapping[str, Any] | None = None,
    geo_map: Mapping[str, Any] | None = None,
    *,
    drop_large_holes: bool = True,
) -> list[dict[str, Any]]:
    """Create simple drill groups from hole diameters and plate thickness."""

    try:
        t_in = float(thickness_in) if thickness_in is not None else None
    except Exception:
        t_in = None
    if t_in is not None and (not math.isfinite(t_in) or t_in <= 0):
        t_in = None

    groups: dict[float, dict[str, Any]] = {}
    counts_by_diam_raw: Counter[float] = Counter()
    if isinstance(hole_diams_mm, Sequence) and not isinstance(hole_diams_mm, (str, bytes, bytearray)):
        for raw in hole_diams_mm:
            d_mm = _coerce_positive_float(raw)
            if d_mm is None:
                continue
            d_in = float(d_mm) / 25.4
            if not (d_in > 0 and math.isfinite(d_in)):
                continue
            key = round(d_in, 4)
            bucket = groups.setdefault(
                key,
                {
                    "diameter_in": float(key),
                    "qty": 0,
                    "depth_in": t_in,
                    "op": "deep_drill" if (t_in is not None and t_in >= 3.0 * float(key)) else "drill",
                },
            )
            bucket["qty"] += 1
            counts_by_diam_raw[key] += 1

    counts_by_diam: dict[float, int] = {
        float(dia): int(qty)
        for dia, qty in counts_by_diam_raw.items()
    }

    if counts_by_diam:

        def _nearest_bin(val: float, bins: Sequence[float]) -> float:
            return min(bins, key=lambda b: abs(b - val))

        bins = sorted(float(d) for d in counts_by_diam.keys())

        if isinstance(ops_claims, Mapping):
            claimed = (ops_claims or {}).get("claimed_pilot_diams")
            if claimed:
                claimed_ctr = Counter(
                    round(float(d), 4) for d in claimed if d is not None
                )
                for val, qty in claimed_ctr.items():
                    if not bins:
                        break
                    target = _nearest_bin(val, bins)
                    if abs(target - val) <= 0.015:
                        counts_by_diam[target] = max(
                            0,
                            int(counts_by_diam.get(target, 0)) - int(qty),
                        )

            cb_face_counts: Counter[float] = Counter()
            for (diam, _side, _depth), qty in (ops_claims or {}).get("cb_groups", {}).items():
                if diam is None:
                    continue
                try:
                    cb_face_counts[round(float(diam), 4)] += int(qty)
                except Exception:
                    continue

            for face_diam, face_qty in cb_face_counts.items():
                if not bins:
                    break
                target = _nearest_bin(face_diam, bins)
                if abs(target - face_diam) <= 0.02:
                    counts_by_diam[target] = max(
                        0,
                        int(counts_by_diam.get(target, 0)) - int(face_qty),
                    )

        for diameter in list(counts_by_diam.keys()):
            if drop_large_holes and diameter >= 1.0:
                counts_by_diam[diameter] = 0

        ops_hint: Mapping[str, Any] = {}
        try:
            ops_hint = (
                (geo_map.get("ops_totals_hint") or {})
                if isinstance(geo_map, Mapping)
                else {}
            )
        except Exception:
            ops_hint = {}
        ops_hint = dict(OPS_TOTAL_OVERRIDES_DEFAULT) | dict(ops_hint or {})

        if "drill" in ops_hint:
            try:
                target = int(ops_hint["drill"])
            except Exception:
                target = None
            if target is not None:
                cur = sum(max(0, int(v)) for v in counts_by_diam.values())
                if cur > target:
                    over = cur - target
                    for diameter in sorted(counts_by_diam.keys(), reverse=True):
                        if over <= 0:
                            break
                        take = min(over, counts_by_diam[diameter])
                        counts_by_diam[diameter] -= take
                        over -= take

        for key in list(groups.keys()):
            qty_adj = int(max(0, counts_by_diam.get(key, 0)))
            if qty_adj <= 0:
                groups.pop(key, None)
            else:
                groups[key]["qty"] = qty_adj

    ordered = [groups[k] for k in sorted(groups.keys())]
    return ordered


def _coerce_scrap_fraction(val: Any, cap: float = HOLE_SCRAP_CAP) -> float:
    """Return a scrap fraction clamped within ``[0, cap]``.

    The helper accepts common UI inputs such as ``15`` (percent),
    ``0.15`` (fraction), values with a trailing ``%`` and gracefully handles
    ``None``/empty strings by falling back to ``0``.
    """

    cap_val = _coerce_cap(cap, default=HOLE_SCRAP_CAP)

    if val is None:
        raw = 0.0
    elif isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            raw = 0.0
        elif stripped.endswith("%"):
            raw = _float_or_default(stripped.rstrip("%"), default=0.0) / 100.0
        else:
            raw = _float_or_default(stripped, default=0.0)
    else:
        raw = _float_or_default(val, default=0.0)

    if not math.isfinite(raw):
        raw = 0.0
    if raw > 1.0:
        raw = raw / 100.0
    if raw < 0.0:
        raw = 0.0
    return min(cap_val, raw)


def normalize_scrap_pct(val: Any, cap: float = HOLE_SCRAP_CAP) -> float:
    """Backwards-compatible alias for :func:`_coerce_scrap_fraction`."""

    return _coerce_scrap_fraction(val, cap)


def resolve_material_scrap_state(
    *,
    value_map: Mapping[str, Any] | None,
    geo_payload: Mapping[str, Any] | None,
    material_text: str | None,
    scrap_value: Any,
    material_group_display: str | None = None,
    density_lookup: Callable[[str], float | None] | None = None,
    scrap_default_guess: float = SCRAP_DEFAULT_GUESS,
    hole_scrap_cap: float = HOLE_SCRAP_CAP,
) -> MaterialScrapResolution:
    """Return a :class:`MaterialScrapResolution` for the provided inputs."""

    normalized_scrap = normalize_scrap_pct(scrap_value, cap=hole_scrap_cap)
    scrap_frac: float | None = normalized_scrap if normalized_scrap > 0 else None
    scrap_source = "ui" if scrap_frac is not None else "default_guess"
    scrap_source_label = scrap_source
    hole_reason_applied = False

    if scrap_frac is None:
        stock_plan = geo_payload.get("stock_plan_guess") if isinstance(geo_payload, Mapping) else None
        if isinstance(stock_plan, Mapping):
            net = _coerce_float_or_none(stock_plan.get("net_volume_in3"))
            stock = _coerce_float_or_none(stock_plan.get("stock_volume_in3"))
            if net and stock and net > 0 and stock >= net:
                scrap_frac = max(0.0, min(hole_scrap_cap, (stock - net) / net))
                scrap_source = "stock_plan_guess"
                scrap_source_label = scrap_source
        if scrap_frac is None:
            scrap_frac = float(scrap_default_guess)
            scrap_source = "default_guess"
            scrap_source_label = scrap_source

    assert scrap_frac is not None
    scrap_frac = float(scrap_frac)

    hole_scrap_frac_est = _holes_scrap_fraction(geo_payload, cap=hole_scrap_cap)
    if hole_scrap_frac_est > 0:
        hole_scrap_frac_clamped = max(0.0, min(hole_scrap_cap, float(hole_scrap_frac_est)))
        ui_scrap_frac = normalize_scrap_pct(scrap_value, cap=hole_scrap_cap)
        scrap_candidate = max(ui_scrap_frac, hole_scrap_frac_clamped)
        if scrap_candidate > scrap_frac + 1e-9:
            scrap_frac = scrap_candidate
            scrap_source_label = f"{scrap_source}+holes"
            hole_reason_applied = True

    density_g_cc = None
    if isinstance(value_map, Mapping):
        density_g_cc = _coerce_float_or_none(value_map.get("Material Density"))
    if (density_g_cc in (None, 0.0)) and material_text and density_lookup is not None:
        density_hint = density_lookup(material_text)
        if density_hint:
            density_g_cc = density_hint

    net_volume_cm3 = None
    if isinstance(value_map, Mapping):
        net_volume_cm3 = _coerce_float_or_none(value_map.get("Net Volume (cm^3)"))

    length_in = _coerce_float_or_none(value_map.get("Plate Length (in)")) if isinstance(value_map, Mapping) else None
    width_in = _coerce_float_or_none(value_map.get("Plate Width (in)")) if isinstance(value_map, Mapping) else None
    thickness_in_val = _coerce_float_or_none(value_map.get("Thickness (in)")) if isinstance(value_map, Mapping) else None

    if isinstance(geo_payload, Mapping):
        if length_in is None:
            length_in = _coerce_float_or_none(
                geo_payload.get("plate_len_in") or geo_payload.get("plate_length_in")
            )
        if width_in is None:
            width_in = _coerce_float_or_none(
                geo_payload.get("plate_wid_in") or geo_payload.get("plate_width_in")
            )

        if length_in is None:
            length_mm = _coerce_positive_float(
                geo_payload.get("plate_len_mm") or geo_payload.get("plate_length_mm")
            )
            if length_mm:
                length_in = float(length_mm) / 25.4
        if width_in is None:
            width_mm = _coerce_positive_float(
                geo_payload.get("plate_wid_mm") or geo_payload.get("plate_width_mm")
            )
            if width_mm:
                width_in = float(width_mm) / 25.4

        if thickness_in_val is None:
            thickness_in_val = _coerce_float_or_none(geo_payload.get("thickness_in"))
            if thickness_in_val is None:
                thickness_mm_geo = _coerce_positive_float(geo_payload.get("thickness_mm"))
                if thickness_mm_geo:
                    thickness_in_val = float(thickness_mm_geo) / 25.4

        if length_in is None or width_in is None:
            derived_ctx = geo_payload.get("derived")
            if isinstance(derived_ctx, Mapping):
                bbox_mm = derived_ctx.get("bbox_mm")
                if (
                    isinstance(bbox_mm, (list, tuple))
                    and len(bbox_mm) == 2
                ):
                    bbox_L_mm = _coerce_positive_float(bbox_mm[0])
                    bbox_W_mm = _coerce_positive_float(bbox_mm[1])
                    if length_in is None and bbox_L_mm:
                        length_in = float(bbox_L_mm) / 25.4
                    if width_in is None and bbox_W_mm:
                        width_in = float(bbox_W_mm) / 25.4

    if net_volume_cm3 is None and length_in and width_in and thickness_in_val:
        volume_in3 = float(length_in) * float(width_in) * float(thickness_in_val)
        net_volume_cm3 = volume_in3 * 16.387064

    removal_mass_g = _holes_removed_mass_g(geo_payload)
    if net_volume_cm3 and density_g_cc and removal_mass_g:
        net_mass_g = float(net_volume_cm3) * float(density_g_cc)
        base_for_removal = float(scrap_frac or 0.0) if scrap_source != "default_guess" else 0.0

        base_net = max(0.0, float(net_mass_g))
        removal = max(0.0, float(removal_mass_g))
        base_scrap = normalize_scrap_pct(base_for_removal, cap=hole_scrap_cap)

        if base_net > 0 and removal > 0:
            removal = min(removal, base_net)
            net_after = max(1e-6, base_net - removal)
            scrap_mass = base_net * base_scrap + removal
            scrap_after = scrap_mass / net_after if net_after > 0 else hole_scrap_cap
            scrap_after = max(0.0, min(hole_scrap_cap, scrap_after))
            scrap_frac = scrap_after
            scrap_source = "geometry"
            if hole_reason_applied:
                scrap_source_label = f"{scrap_source_label}+geometry"
            else:
                scrap_source_label = scrap_source

    resolution = MaterialScrapResolution(
        scrap_fraction=float(scrap_frac),
        scrap_source=scrap_source,
        scrap_source_label=scrap_source_label,
        hole_reason_applied=hole_reason_applied,
        density_g_cc=(float(density_g_cc) if density_g_cc not in (None, 0.0) else None),
        net_volume_cm3=(float(net_volume_cm3) if net_volume_cm3 else None),
        removal_mass_g=(float(removal_mass_g) if removal_mass_g else None),
    )

    return resolution


def _iter_hole_diams_mm(geo_ctx: Mapping[str, Any] | None) -> list[float]:
    """Return all positive hole diameters from a geometry context."""

    if not isinstance(geo_ctx, Mapping):
        return []

    geo_map = geo_ctx
    derived_obj = geo_map.get("derived")
    derived: Mapping[str, Any] = derived_obj if isinstance(derived_obj, Mapping) else {}

    out: list[float] = []
    dxf_diams = derived.get("hole_diams_mm")
    if isinstance(dxf_diams, Iterable) and not isinstance(dxf_diams, (str, bytes)):
        for d in dxf_diams:
            v = _float_or_default(d)
            if v is not None and v > 0:
                out.append(v)

    step_holes = derived.get("holes")
    if isinstance(step_holes, Iterable) and not isinstance(step_holes, (str, bytes)):
        for h in step_holes:
            if isinstance(h, Mapping):
                v = _float_or_default(h.get("dia_mm"))
            else:
                v = _float_or_default(getattr(h, "dia_mm", None))
            if v is not None and v > 0:
                out.append(v)

    return out


def _plate_bbox_mm2(geo_ctx: Mapping[str, Any] | None) -> float:
    """Return the estimated bounding-box area of a plate in square millimetres."""

    if not isinstance(geo_ctx, Mapping):
        return 0.0

    geo_map = geo_ctx
    derived_obj = geo_map.get("derived")
    derived: Mapping[str, Any] = derived_obj if isinstance(derived_obj, Mapping) else {}

    bbox_mm = derived.get("bbox_mm")
    if (
        isinstance(bbox_mm, (list, tuple))
        and len(bbox_mm) == 2
        and all(isinstance(x, (int, float)) and x > 0 for x in bbox_mm)
    ):
        return float(bbox_mm[0]) * float(bbox_mm[1])

    def _coerce_in(val: Any) -> float | None:
        v = _float_or_default(val)
        if v is not None and v > 0:
            return float(v)
        return None

    try:
        L_in = float(geo_map.get("plate_length_mm")) / 25.4
    except Exception:
        L_in = _coerce_in(geo_map.get("plate_length_in"))
    try:
        W_in = float(geo_map.get("plate_width_mm")) / 25.4
    except Exception:
        W_in = _coerce_in(geo_map.get("plate_width_in"))

    if L_in and W_in:
        return float(L_in * 25.4) * float(W_in * 25.4)
    return 0.0


def _holes_scrap_fraction(
    geo_ctx: Mapping[str, Any] | None,
    *,
    cap: float = HOLE_SCRAP_CAP,
) -> float:
    """Estimate scrap fraction implied by drilled holes in *geo_ctx*."""

    diams = _iter_hole_diams_mm(geo_ctx)
    if not diams:
        return 0.0

    plate_area_mm2 = _plate_bbox_mm2(geo_ctx)
    if plate_area_mm2 <= 0:
        return 0.0

    holes_area_mm2 = 0.0
    for d in diams:
        r = 0.5 * float(d)
        holes_area_mm2 += math.pi * r * r

    frac = HOLE_SCRAP_MULT * (holes_area_mm2 / plate_area_mm2)
    if not math.isfinite(frac) or frac < 0:
        return 0.0

    cap_val = _coerce_cap(cap, default=HOLE_SCRAP_CAP)
    return min(cap_val, float(frac))


def _estimate_scrap_from_stock_plan(
    geo_ctx: Mapping[str, Any] | None,
) -> tuple[float | None, str | None]:
    """Attempt to infer a scrap fraction from stock planning hints."""

    contexts: list[Mapping[str, Any]] = []
    if isinstance(geo_ctx, Mapping):
        contexts.append(geo_ctx)
        inner = geo_ctx.get("geo")
        if isinstance(inner, Mapping):
            contexts.append(inner)

    for ctx in contexts:
        plan = ctx.get("stock_plan_guess") or ctx.get("stock_plan")
        if not isinstance(plan, Mapping):
            continue
        net_vol = _coerce_float_or_none(plan.get("net_volume_in3"))
        stock_vol = _coerce_float_or_none(plan.get("stock_volume_in3"))
        scrap: float | None = None
        if net_vol and net_vol > 0 and stock_vol and stock_vol > 0:
            scrap = max(0.0, (stock_vol - net_vol) / net_vol)
        else:
            part_mass_lb = _coerce_float_or_none(plan.get("part_mass_lb"))
            stock_mass_lb = _coerce_float_or_none(plan.get("stock_mass_lb"))
            if part_mass_lb and part_mass_lb > 0 and stock_mass_lb and stock_mass_lb > 0:
                scrap = max(0.0, (stock_mass_lb - part_mass_lb) / part_mass_lb)
        if scrap is not None:
            return min(HOLE_SCRAP_CAP, float(scrap)), "stock_plan_guess"
    return None, None


def _coerce_cap(cap: Any, *, default: float) -> float:
    """Normalize *cap* to a usable float value."""

    cap_val = _float_or_default(cap, default=default)
    if cap_val is None or not math.isfinite(cap_val):
        return default
    return max(0.0, float(cap_val))


def _float_or_default(val: Any, default: float | None = None) -> float | None:
    """Return ``float(val)`` or ``default`` when conversion fails."""

    try:
        result = float(val)  # type: ignore[arg-type]
    except Exception:
        return default
    return result


__all__ = [
    "SCRAP_DEFAULT_GUESS",
    "HOLE_SCRAP_MULT",
    "HOLE_SCRAP_CAP",
    "_holes_removed_mass_g",
    "build_drill_groups_from_geometry",
    "_coerce_scrap_fraction",
    "normalize_scrap_pct",
    "_iter_hole_diams_mm",
    "_plate_bbox_mm2",
    "_holes_scrap_fraction",
    "_estimate_scrap_from_stock_plan",
]
