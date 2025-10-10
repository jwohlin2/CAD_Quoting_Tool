
# test_dim_price_csv.py
# Load a CSV (material, thickness_in, length_in, width_in, part),
# pick the smallest stock size that is >= requested LxW (rotation allowed) at thickness T
# (optionally allow the next thicker), then price via McMaster API.
#
# Requirements:
#   pip install python-dotenv requests requests-pkcs12
#   (plus whatever mcmaster_api.py requires in your environment)
#
# Usage:
#   python test_dim_price_csv.py
#   -> enter path to your catalog.csv (e.g., D:\CAD_Quoting_Tool\catalog.csv)
#   -> enter L/W/T/material and get price @ qty=1
#
# Environment (recommended):
#   setx MCMASTER_USER "you@example.com"
#   setx MCMASTER_PASS "YOURPASSWORD"
#   setx MCMASTER_PFX_PATH "D:\Composidie.pfx"
#   setx MCMASTER_PFX_PASS "YOURPFXPASSWORD"

import os, csv, re
from dataclasses import dataclass
from typing import Dict, List, Optional

# If your mcmaster_api.py lives alongside this file, this import will work.
# Otherwise adjust sys.path or place this script next to mcmaster_api.py.
from mcmaster_api import McMasterAPI, load_env

# --- Config ---
ALLOW_NEXT_THICKER = False  # set True to allow selecting the next thicker plate if exact T not found
# Hard-coded catalog CSV path (user requested no prompt). You can still override
# with the CATALOG_CSV_PATH environment variable if needed.
CATALOG_CSV_PATH = os.getenv("CATALOG_CSV_PATH", r"D:\\CAD_Quoting_Tool\\catalog.csv")

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional: trust OS cert store if you still see TLS errors (Windows/enterprise proxies)
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    try:
        import pip_system_certs.wrapt_requests  # noqa: F401
    except Exception:
        pass

@dataclass(frozen=True)
class StockItem:
    material: str
    thickness: float  # inches
    length: float     # inches
    width: float      # inches
    part: str

def norm_material(s: str) -> str:
    s = (s or "").strip().lower()
    # Light normalization; keep distinct alloys if present
    s = re.sub(r"\s+", " ", s)
    replacements = {
        "aluminium": "aluminum",
        "alum": "aluminum",
        "mic 6": "mic6",
        "mic-6": "mic6",
        "tool & jig plate": "aluminum mic6",
    }
    for k,v in replacements.items():
        s = s.replace(k, v)
    return s

def _parse_fraction_token(tok: str) -> float:
    """Parse a single token that may be a fraction like '3/8' or a decimal '1.25'."""
    tok = tok.strip()
    if not tok:
        return 0.0
    if '/' in tok:
        num, den = tok.split('/', 1)
        return float(num) / float(den)
    return float(tok)

def _normalize_inch_string(s: str) -> str:
    """Normalize Unicode quotes, primes, hyphens, NBSP and fraction slash."""
    if not s:
        return s
    s = (s
         .replace('\u201d', '"')  # right smart quote ”
         .replace('\u201c', '"')  # left smart quote “
         .replace('\u2033', '"')  # double prime ″
         .replace('\uff02', '"')  # fullwidth quotation mark ＂
         .replace('\u2044', '/')   # fraction slash ⁄ -> /
         .replace('\xa0', ' ')     # NBSP -> space
         .replace('\u2011', '-')   # non-breaking hyphen
         .replace('\u2013', '-')   # en dash
         .replace('\u2014', '-')   # em dash
    )
    return s

def parse_inches(val: str) -> float:
    """
    Parse strings like: 1/4", 3/8, 1 1/2", 2", 2.25, 2 1/4, 1-1/2"
    Returns float inches.
    """
    s = str(val or "").strip().lower()
    s = _normalize_inch_string(s)
    s = s.replace('"', '').replace("in", '').replace("inch", '').replace("in.", '').strip()
    s = s.replace("–", "-").replace("—", "-")
    # Normalize common "1-1/2" to "1 1/2"
    s = re.sub(r'(?<=\d)-(?=\d)', ' ', s)
    # Split by whitespace, sum tokens
    total = 0.0
    for tok in s.split():
        total += _parse_fraction_token(tok)
    # If we failed to split (e.g., "3/8"), handle it
    if total == 0.0 and s:
        total = _parse_fraction_token(s)
    return total

def load_catalog(csv_path: str) -> Dict[str, Dict[float, List[StockItem]]]:
    """
    Returns a nested dict: catalog[material][thickness] -> [StockItem, ...]
    Expects headers: material, thickness_in, length_in, width_in, part (case-insensitive)
    """
    # Determine actual column names case-insensitively
    # Use utf-8-sig to transparently strip BOM (\ufeff) from the first header
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        hdrs = { (h or "").strip().lower(): h for h in (reader.fieldnames or []) }
        required = ["material","thickness_in","length_in","width_in","part"]
        missing = [x for x in required if x not in hdrs]
        if missing:
            raise ValueError(f"CSV missing required headers: {missing}. Found: {list(hdrs.keys())}")

    # Second pass: parse rows
    catalog: Dict[str, Dict[float, List[StockItem]]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mat = norm_material(row[hdrs["material"]])
            t = parse_inches(row[hdrs["thickness_in"]])
            L = parse_inches(row[hdrs["length_in"]])
            W = parse_inches(row[hdrs["width_in"]])
            part = str(row[hdrs["part"]]).strip()
            if not part or t <= 0 or L <= 0 or W <= 0:
                continue
            item = StockItem(mat, t, L, W, part)
            catalog.setdefault(mat, {}).setdefault(t, []).append(item)
    return catalog

def _fits(item: StockItem, L: float, W: float) -> bool:
    # rotation allowed
    return (item.length >= L and item.width >= W) or (item.length >= W and item.width >= L)

def _area(item: StockItem) -> float:
    return item.length * item.width

def _best_fit(items: List[StockItem], L: float, W: float) -> Optional[StockItem]:
    candidates = [it for it in items if _fits(it, L, W)]
    if not candidates:
        return None
    # Choose minimal area; tie-breaker by smaller min edge then part
    candidates.sort(key=lambda it: (_area(it), min(it.length, it.width), it.part))
    return candidates[0]

def choose_item(catalog: Dict[str, Dict[float, List[StockItem]]],
                material: str, L: float, W: float, T: float) -> Optional[StockItem]:
    mat = norm_material(material)
    by_t = catalog.get(mat, {})
    # 1) exact thickness
    items = by_t.get(T, [])
    best = _best_fit(items, L, W)
    if best:
        return best
    # 2) next thicker (optional)
    if ALLOW_NEXT_THICKER:
        thicker_ts = sorted([t for t in by_t.keys() if t > T])
        for t2 in thicker_ts:
            best = _best_fit(by_t[t2], L, W)
            if best:
                return best
    return None

def best_unit_price_tier(tiers: List[dict], qty: int = 1):
    tiers_sorted = sorted(tiers, key=lambda t: t.get("MinimumQuantity", 10**9))
    eligible = [t for t in tiers_sorted if t.get("MinimumQuantity", 10**9) <= qty]
    return eligible[-1] if eligible else (tiers_sorted[0] if tiers_sorted else None)

def _get_env_or_prompt():
    env = load_env()
    return (
        env.get("MCMASTER_USER", ""),
        env.get("MCMASTER_PASS", ""),
        env.get("MCMASTER_PFX_PATH", ""),
        env.get("MCMASTER_PFX_PASS", ""),
    )

def main():
    # 1) CSV path (hard-coded, no prompt)
    csv_path = CATALOG_CSV_PATH
    print(f"Using catalog CSV: {csv_path}")
    if not csv_path:
        print("No CSV provided.")
        return
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    # 2) Load catalog
    try:
        catalog = load_catalog(csv_path)
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return
    if not catalog:
        print("Catalog is empty after load.")
        return

    # 3) Get requested dims/material
    try:
        L = float(input('Length (in, numeric): ').strip().replace('"', ""))
        W = float(input('Width  (in, numeric): ').strip().replace('"', ""))
        T = float(input('Thick  (in, numeric): ').strip().replace('"', ""))
    except Exception:
        print("Please enter numeric inches for L/W/T (e.g., 12, 14, 2).")
        return
    material = input('Material (e.g., "aluminum mic6", "aluminum 5083", "tool steel a2"): ').strip()

    # 4) Choose item
    item = choose_item(catalog, material, L, W, T)
    if not item:
        print(f'No match found ≥ {L}×{W}×{T} in "{norm_material(material)}".')
        if not ALLOW_NEXT_THICKER:
            print("Tip: set ALLOW_NEXT_THICKER=True to allow thicker stock.")
        return

    print(f'Chosen SKU: {item.part}  ({item.length:.3f}" × {item.width:.3f}" × {item.thickness:.3f}"  {item.material})')

    # 5) Price via API
    user, pw, pfx, pfxp = _get_env_or_prompt()
    api = McMasterAPI(username=user, password=pw, pfx_path=pfx, pfx_password=pfxp)
    api.login()
    tiers = api.get_price_tiers(item.part)
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
    main()
