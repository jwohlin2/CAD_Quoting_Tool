# test_dim_price.py
# Given L, W, T, material -> pick the smallest >= size (never smaller) and price it via the McMaster API.
import math
from typing import Dict, List, Tuple, Optional
from mcmaster_api import McMasterAPI, load_env

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- seed catalog ----
# material -> thickness -> list[(L, W, part)]
CATALOG: Dict[str, Dict[float, List[Tuple[float, float, str]]]] = {
    "aluminum": {
        2.0: [
            (12.0, 14.0, "86825K954"),  # known-good: MIC-6, 2" x 12" x 14"
        ],
        # add 1.0" entries here as you discover them, e.g.:
        # 1.0: [(6.0, 6.0, "XXXXX"), (12.0, 12.0, "YYYYY"), ...]
    },
}

def _get_env_or_prompt():
    env = load_env()
    return (
        env.get("MCMASTER_USER", ""),
        env.get("MCMASTER_PASS", ""),
        env.get("MCMASTER_PFX_PATH", ""),
        env.get("MCMASTER_PFX_PASS", ""),
    )

def norm_material(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"al", "alum", "aluminum", "aluminium", "mic-6", "mic6", "tool & jig plate"}:
        return "aluminum"
    if s in {"tool steel", "a2", "o1", "d2"}:
        return "tool_steel"
    if s in {"tungsten carbide", "carbide"}:
        return "tungsten_carbide"
    return s

def choose_size(material: str, L: float, W: float, T: float) -> Optional[Tuple[float, float, str]]:
    fam = CATALOG.get(material, {})
    options = fam.get(T, [])
    best = None
    best_area = math.inf
    for (L0, W0, part) in options:
        # allow rotation
        if (L0 >= L and W0 >= W) or (L0 >= W and W0 >= L):
            area = L0 * W0
            if area < best_area:
                best_area = area
                best = (L0, W0, part)
    return best

def best_unit_price_tier(tiers: List[dict], qty: int = 1):
    tiers_sorted = sorted(tiers, key=lambda t: t.get("MinimumQuantity", 10**9))
    eligible = [t for t in tiers_sorted if t.get("MinimumQuantity", 10**9) <= qty]
    return eligible[-1] if eligible else (tiers_sorted[0] if tiers_sorted else None)

def run_once():
    # --- get inputs ---
    try:
        L = float(input('Length (in): ').strip().replace('"', ""))
        W = float(input('Width  (in): ').strip().replace('"', ""))
        T = float(input('Thick  (in): ').strip().replace('"', ""))
    except Exception:
        print("Please enter numeric inches for L/W/T.")
        return
    material = norm_material(input('Material (e.g., aluminum): ').strip())

    # --- pick a catalog SKU, or fall back to manual entry ---
    picked = choose_size(material, L, W, T)
    if picked:
        (L0, W0, part) = picked
        print(f'Chosen SKU: {part}  ({L0:.3f}" × {W0:.3f}" × {T:.3f}")')
    else:
        print(f'No catalog entry found ≥ {L}×{W}×{T} in {material}.')
        part = input("Enter McMaster part number to price anyway (or leave blank to cancel): ").strip()
        if not part:
            return

    # --- auth + pricing ---
    user, pw, pfx, pfxp = _get_env_or_prompt()
    api = McMasterAPI(username=user, password=pw, pfx_path=pfx, pfx_password=pfxp)
    api.login()
    tiers = api.get_price_tiers(part)
    if not tiers:
        print("No price tiers returned.")
        return
    one = best_unit_price_tier(tiers, qty=1)
    if one:
        amt = one["Amount"]
        uom = one["UnitOfMeasure"]
        print(f"Price @ qty=1: ${amt:.2f} {uom}")
    else:
        print("No eligible tier for qty=1.")

if __name__ == "__main__":
    run_once()
