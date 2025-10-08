# cad_quoter/vendors/mcmaster_stock.py

from __future__ import annotations
import os, csv, math, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Reuse your working API client + env loader
# (the class and endpoints are already defined in mcmaster_api.py)
from mcmaster_api import McMasterAPI, load_env  # :contentReference[oaicite:0]{index=0}

# -------- CSV → in-memory catalog --------

@dataclass(frozen=True)
class StockItem:
    material: str
    thickness: float  # inches
    length: float     # inches
    width: float      # inches
    part: str

ALLOW_NEXT_THICKER = False  # flip if you want “next thicker” allowed by default
_CATALOG_CACHE: Optional[Dict[str, Dict[float, List[StockItem]]]] = None
_PRICE_CACHE: dict[Tuple[str,int], Tuple[float,str]] = {}  # (sku, qty) -> (price_each, uom)

def _normalize_inch_string(s: str) -> str:
    return (s or "").replace("\u201d", '"').replace("\u201c", '"').replace("\u2033", '"') \
        .replace("\uff02", '"').replace("\u2044", '/').replace("\xa0", ' ') \
        .replace("\u2011", '-').replace("\u2013", '-').replace("\u2014", '-')

def _parse_fraction_token(tok: str) -> float:
    tok = tok.strip()
    if not tok: return 0.0
    if "/" in tok:
        a,b = tok.split("/",1)
        return float(a)/float(b)
    return float(tok)

def parse_inches(val: str) -> float:
    s = _normalize_inch_string(str(val or "").strip().lower())
    s = s.replace('"','').replace("in.","").replace("inch","").replace("in","").strip()
    s = re.sub(r"(?<=\d)-(?=\d)", " ", s)
    total = 0.0
    for tok in s.split():
        total += _parse_fraction_token(tok)
    if total == 0.0 and s:
        total = _parse_fraction_token(s)
    return total

def norm_material(s: str) -> str:
    s = re.sub(r"\s+"," ", (s or "").strip().lower())
    repl = {"aluminium":"aluminum","alum":"aluminum","mic 6":"mic6","mic-6":"mic6","tool & jig plate":"aluminum mic6"}
    for k,v in repl.items():
        s = s.replace(k,v)
    return s

def load_catalog(csv_path: str) -> Dict[str, Dict[float, List[StockItem]]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        hdrs = { (h or "").strip().lower(): h for h in (reader.fieldnames or []) }
        required = ["material","thickness_in","length_in","width_in","part"]
        missing = [x for x in required if x not in hdrs]
        if missing:
            raise ValueError(f"CSV missing required headers: {missing}. Found: {list(hdrs.keys())}")
    out: Dict[str, Dict[float, List[StockItem]]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mat = norm_material(row[hdrs["material"]])
            t   = parse_inches(row[hdrs["thickness_in"]])
            L   = parse_inches(row[hdrs["length_in"]])
            W   = parse_inches(row[hdrs["width_in"]])
            part= str(row[hdrs["part"]]).strip()
            if not part or t <= 0 or L <= 0 or W <= 0:
                continue
            out.setdefault(mat, {}).setdefault(t, []).append(StockItem(mat,t,L,W,part))
    return out

def _fits(it: StockItem, L: float, W: float) -> bool:
    return (it.length>=L and it.width>=W) or (it.length>=W and it.width>=L)

def _area(it: StockItem) -> float:
    return it.length * it.width

def _best_fit(items: List[StockItem], L: float, W: float) -> Optional[StockItem]:
    cands = [it for it in items if _fits(it, L, W)]
    if not cands: return None
    cands.sort(key=lambda it: (_area(it), min(it.length,it.width), it.part))
    return cands[0]

def choose_item(catalog, material: str, L: float, W: float, T: float) -> Optional[StockItem]:
    mat = norm_material(material)
    by_t = catalog.get(mat, {})
    best = _best_fit(by_t.get(T, []), L, W)
    if best: return best
    if ALLOW_NEXT_THICKER:
        for t2 in sorted([t for t in by_t.keys() if t > T]):
            best = _best_fit(by_t[t2], L, W)
            if best: return best
    return None

def _in_mm_to_in_inches(mm: float) -> float:
    return float(mm)/25.4 if mm is not None else 0.0

def _best_unit_price_tier(tiers: List[dict], qty: int=1) -> Optional[dict]:
    if not tiers: return None
    tiers_sorted = sorted(tiers, key=lambda t: t.get("MinimumQuantity", 10**9))
    eligible = [t for t in tiers_sorted if t.get("MinimumQuantity", 10**9) <= qty]
    return eligible[-1] if eligible else tiers_sorted[0]

def _get_catalog() -> Dict[str, Dict[float, List[StockItem]]]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        csv_path = os.getenv("CATALOG_CSV_PATH", r"D:\CAD_Quoting_Tool\catalog.csv")
        _CATALOG_CACHE = load_catalog(csv_path)
    return _CATALOG_CACHE

def lookup_sku_and_price_for_mm(material: str, L_mm: float, W_mm: float, T_mm: float, qty: int=1) -> tuple[str,float,str,tuple[float,float,float]] | tuple[None,None,None,tuple[float,float,float]]:
    """
    Returns (sku, price_each, uom, (L_in, W_in, T_in)) or (None, None, None, dims)
    """
    # 1) pick SKU from CSV
    catalog = _get_catalog()
    L_in, W_in, T_in = map(_in_mm_to_in_inches, (L_mm, W_mm, T_mm))
    item = choose_item(catalog, material, L_in, W_in, T_in)
    if not item:
        return None, None, None, (L_in, W_in, T_in)

    # 2) price via McMaster API (cache per sku/qty)
    cache_key = (item.part, int(qty))
    if cache_key in _PRICE_CACHE:
        price_each, uom = _PRICE_CACHE[cache_key]
        return item.part, price_each, uom, (item.length, item.width, item.thickness)

    env = load_env()  # reads env / optionally prompts
    api = McMasterAPI(env["MCMASTER_USER"], env["MCMASTER_PASS"], env["MCMASTER_PFX_PATH"], env["MCMASTER_PFX_PASS"])  # :contentReference[oaicite:1]{index=1}
    api.login()
    tiers = api.get_price_tiers(item.part)  # :contentReference[oaicite:2]{index=2}
    tier = _best_unit_price_tier(tiers, qty=qty)
    if not tier:
        return item.part, None, None, (item.length, item.width, item.thickness)
    price_each = float(tier["Amount"])
    uom = str(tier["UnitOfMeasure"])
    _PRICE_CACHE[cache_key] = (price_each, uom)
    return item.part, price_each, uom, (item.length, item.width, item.thickness)
