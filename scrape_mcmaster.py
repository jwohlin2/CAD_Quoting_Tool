
#!/usr/bin/env python3
# onlinemetals_scraper.py
#
# Zero-login, Requests-only scraper for OnlineMetals listing/product pages.
# Supports:
#   - MIC-6 aluminum plate category (width x length options with prices)
#   - A2 tool steel rectangle bar category (thickness x width with length/price options)
#   - Tungsten carbide rectangle bar (C2 or Standard Micrograin) via product pages
#
# Usage examples:
#   python onlinemetals_scraper.py mic6 --width 12 --length 12 --thickness 0.5
#   python onlinemetals_scraper.py a2 --thickness 0.0625 --width 1 --length 36
#   python onlinemetals_scraper.py carbide --thickness 0.25 --width 1
#
# Notes:
# - Be polite: low request rate, cache results if you can.
# - This script relies primarily on text parsing to avoid brittle CSS class coupling.
#
import argparse, json, math, re, sys, time, atexit
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import requests
from bs4 import BeautifulSoup
try:
    # Playwright is optional; used when requests gets 403 or when --browser is set
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except Exception:
    _PLAYWRIGHT_AVAILABLE = False

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Dest": "document",
}

# Global fetch mode and browser config (set from CLI in main())
GET_MODE = "auto"  # one of: "requests", "browser", "auto"
_BROWSER_ENGINE = "chromium"  # chromium|msedge|chrome|firefox|webkit
_HEADFUL = False
_SLOWMO = 0
_USER_DATA_DIR = None
_DISABLE_JS = False
DEBUG_DIR = None

_PW = None
_BROWSER = None
_CONTEXT = None
_PAGE = None

def _close_browser():
    global _PW, _BROWSER, _CONTEXT, _PAGE
    try:
        if _PAGE:
            _PAGE.close()
    except Exception:
        pass
    try:
        if _CONTEXT:
            _CONTEXT.close()
    except Exception:
        pass
    try:
        if _BROWSER:
            _BROWSER.close()
    except Exception:
        pass
    try:
        if _PW:
            _PW.stop()
    except Exception:
        pass

atexit.register(_close_browser)

MIC6_URL = "https://www.onlinemetals.com/en/buy/aluminum-sheet-plate-mic-6"
A2_URL = "https://www.onlinemetals.com/en/buy/tool-steel-rectangle-bar-a2"
CARBIDE_CAT_URL = "https://www.onlinemetals.com/en/buy/tungsten-carbide-rectangle-bar"

# ---------- utilities ----------

def frac_to_float(txt: str) -> float:
    """Parse '1 1/2', '3/8', '0.5', '12', '12″' to float inches."""
    s = (txt or "").strip().lower().replace('″','"').replace('in.','').replace('in','').replace('”','"')
    s = s.replace('\xa0',' ').strip()
    # feet not expected; ignore
    # mixed number: 1 1/2
    m = re.match(r'^\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*"?\s*$', s)
    if m:
        return float(m.group(1)) + float(m.group(2))/float(m.group(3))
    # fraction only: 3/8
    m = re.match(r'^\s*(\d+)\s*/\s*(\d+)\s*"?\s*$', s)
    if m:
        return float(m.group(1))/float(m.group(2))
    # decimal/integer: 0.5 or 12
    m = re.match(r'^\s*(\d+(?:\.\d+)?)\s*"?\s*$', s)
    if m:
        return float(m.group(1))
    # last resort: keep digits & dot
    m = re.findall(r'[\d.]+', s)
    return float(m[0]) if m else 0.0

def price_to_float(text: str) -> Optional[float]:
    s = str(text or "")
    m = re.search(r'\$\s*([0-9][0-9,]*(?:\.\d{2})?)', s)
    if m:
        return float(m.group(1).replace(',', ''))
    m2 = re.findall(r'([0-9][0-9,]*(?:\.\d{2})?)', s)
    return float(m2[-1].replace(',', '')) if m2 else None

def _ensure_browser():
    global _PW, _BROWSER, _CONTEXT, _PAGE
    if _PW is not None and _PAGE is not None:
        return
    if not _PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright not installed. Install with: pip install playwright && playwright install")
    _PW = sync_playwright().start()
    engine = (_BROWSER_ENGINE or "chromium").lower()
    headless = not _HEADFUL
    slow_mo = int(_SLOWMO or 0)
    if engine in ("msedge", "edge"):
        _BROWSER = _PW.chromium.launch(channel="msedge", headless=headless, slow_mo=slow_mo)
    elif engine in ("chrome", "google-chrome"):
        _BROWSER = _PW.chromium.launch(channel="chrome", headless=headless, slow_mo=slow_mo)
    elif engine == "chromium":
        _BROWSER = _PW.chromium.launch(headless=headless, slow_mo=slow_mo)
    elif engine == "firefox":
        _BROWSER = _PW.firefox.launch(headless=headless, slow_mo=slow_mo)
    elif engine == "webkit":
        _BROWSER = _PW.webkit.launch(headless=headless, slow_mo=slow_mo)
    else:
        _BROWSER = _PW.chromium.launch(headless=headless, slow_mo=slow_mo)
    ctx_kwargs = {
        "java_script_enabled": not _DISABLE_JS,
        "extra_http_headers": HEADERS,
    }
    if _USER_DATA_DIR and engine in ("chromium", "msedge", "edge", "chrome", "google-chrome"):
        # Use a persistent context to keep cookies, if requested
        _CONTEXT = _PW.chromium.launch_persistent_context(_USER_DATA_DIR, channel=("msedge" if engine in ("msedge","edge") else ("chrome" if engine in ("chrome","google-chrome") else None)), headless=headless, slow_mo=slow_mo, **ctx_kwargs)
    else:
        _CONTEXT = _BROWSER.new_context(**ctx_kwargs)
    _PAGE = _CONTEXT.new_page()

def browser_get(url: str, timeout=30000) -> str:
    _ensure_browser()
    _PAGE.goto(url, wait_until="load", timeout=timeout)
    # Try to allow client-side rendering to finish
    try:
        _PAGE.wait_for_load_state("networkidle", timeout=timeout)
    except Exception:
        pass
    # Give a short settle time and try to wait for any price text
    try:
        _PAGE.locator("text=/\\$\\s*[0-9]/").first.wait_for(timeout=2000)
    except Exception:
        time.sleep(0.5)
    return _PAGE.content()

def requests_get(url: str, timeout=20) -> str:
    # Use a Session with retries & headers
    s = requests.Session()
    s.headers.update(HEADERS)
    r = s.get(url, timeout=timeout)
    if r.status_code == 403:
        # propagate; caller will decide fallback
        r.raise_for_status()
    r.raise_for_status()
    return r.text

def get(url: str, timeout=20) -> str:
    mode = (GET_MODE or "auto").lower()
    if mode == "requests":
        return requests_get(url, timeout=timeout)
    if mode == "browser":
        return browser_get(url)
    # auto: try requests, if 403 then browser
    try:
        return requests_get(url, timeout=timeout)
    except requests.HTTPError as e:
        if getattr(e.response, "status_code", None) == 403:
            return browser_get(url)
        raise

def textify(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text("\n", strip=True)

def lines(html: str) -> List[str]:
    txt = textify(html)
    return [ln for ln in (ln.strip() for ln in txt.splitlines()) if ln]

# ---------- MIC-6 (sheet/plate) ----------

MIC6_SIZE_PRICE_RE = re.compile(
    r'(?P<w>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*["″]\s*[×xX]\s*'
    r'(?P<l>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*["″].{0,80}?\$'
    r'(?P<p>[0-9][0-9,]*\.\d{2})\s*ea\.', re.IGNORECASE
)

MIC6_THICK_TITLE_RE = re.compile(
    r'(?P<t>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*["″]\s*Aluminum\s+Plate\s+MIC-6', re.IGNORECASE
)

def scrape_mic6(url: str = MIC6_URL) -> List[Dict]:
    html = get(url)
    raw = textify(html)
    out: List[Dict] = []
    # find all size/price matches and assign to nearest previous thickness heading
    # build list of (pos, thickness_in) markers
    thickness_markers: List[Tuple[int, float]] = []
    for m in MIC6_THICK_TITLE_RE.finditer(raw):
        thickness_markers.append((m.start(), frac_to_float(m.group('t'))))
    # ensure sorted
    thickness_markers.sort()
    def nearest_thickness(pos: int) -> Optional[float]:
        prev = None
        for p,t in thickness_markers:
            if p <= pos:
                prev = (p,t)
            else:
                break
        return prev[1] if prev else None

    for m in MIC6_SIZE_PRICE_RE.finditer(raw):
        w = frac_to_float(m.group('w'))
        l = frac_to_float(m.group('l'))
        p = price_to_float(m.group('p'))
        t = nearest_thickness(m.start())
        if t is None or p is None:
            continue
        out.append({
            "material": "MIC6",
            "thickness_in": t,
            "width_in": w,
            "length_in": l,
            "price_each": p,
            "source_url": url
        })
    # Fallback extraction if nothing matched: be more permissive around price and dims
    if not out:
        dim_re = re.compile(
            r'(?P<w>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*(?:["”]?|in\.?)?\s*(?:[xX×])\s*'
            r'(?P<l>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*(?:["”]?|in\.?)?',
            re.IGNORECASE,
        )
        price_re = re.compile(r'\$\s*([0-9][0-9,]*(?:\.\d{2})?)\s*(?:ea\.?|each)?', re.IGNORECASE)
        # rebuild thickness markers using a looser title pattern
        thick_re = re.compile(
            r'(?P<t>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*(?:["”]?|in\.?)?\s+(?:Aluminum|Cast)\b.*?MIC[\-\s]?6',
            re.IGNORECASE,
        )
        thickness_markers = []
        for m in thick_re.finditer(raw):
            thickness_markers.append((m.start(), frac_to_float(m.group('t'))))
        thickness_markers.sort()
        def nearest_thickness2(pos: int) -> Optional[float]:
            prev = None
            for p,t in thickness_markers:
                if p <= pos:
                    prev = (p,t)
                else:
                    break
            return prev[1] if prev else None
        for mp in price_re.finditer(raw):
            p = float(mp.group(1).replace(',',''))
            start = max(0, mp.start() - 300)
            window = raw[start:mp.start()]
            dm = None
            for _dm in dim_re.finditer(window):
                dm = _dm
            if not dm:
                continue
            w = frac_to_float(dm.group('w'))
            l = frac_to_float(dm.group('l'))
            t = nearest_thickness2(mp.start())
            if t is None:
                continue
            out.append({
                "material": "MIC6",
                "thickness_in": t,
                "width_in": w,
                "length_in": l,
                "price_each": p,
                "source_url": url
            })
    return out

# ---------- A2 tool steel rectangle bar ----------

A2_TITLE_RE = re.compile(
    r'(?P<t>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*["″]\s*x\s*'
    r'(?P<w>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*["″]\s*Tool\s*Steel\s*Rectangle\s*Bar\s*A2',
    re.IGNORECASE
)
A2_LENGTH_PRICE_RE = re.compile(
    r'(?P<len>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*["′ftin\)\( ]*-\s*\$'
    r'(?P<p>[0-9][0-9,]*\.\d{2})\s*ea\.', re.IGNORECASE
)

def scrape_a2(url: str = A2_URL) -> List[Dict]:
    html = get(url)
    raw = textify(html)
    out: List[Dict] = []
    # We scan forward: whenever a title match appears, capture subsequent length/price lines until next title.
    spans = list(A2_TITLE_RE.finditer(raw))
    spans.append(re.compile(r'$').search(raw))  # sentinel end
    for i in range(len(spans)-1):
        m = spans[i]
        t = frac_to_float(m.group('t')); w = frac_to_float(m.group('w'))
        block = raw[m.end():spans[i+1].start()]
        for mp in A2_LENGTH_PRICE_RE.finditer(block):
            L = frac_to_float(mp.group('len'))
            p = price_to_float(mp.group('p'))
            if p is None:
                continue
            out.append({
                "material": "A2",
                "thickness_in": t,
                "width_in": w,
                "length_in": L,
                "price_each": p,
                "source_url": url
            })
    # Fallback: permissive scan if no matches
    if not out:
        # Find dimension titles with A2 and dimensions
        title_re = re.compile(
            r'(?:A2\s*Tool\s*Steel).*?(?:Rectangle\s*Bar)?[^\n]{0,120}?'
            r'(?P<t>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*(?:["”]?|in\.?)?\s*(?:[xX×])\s*'
            r'(?P<w>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))',
            re.IGNORECASE,
        )
        lp_re = re.compile(
            r'(?P<len>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*(?:["”]?|in\.?|ft)?\s*(?:-|–|—)?\s*\$\s*'
            r'(?P<p>[0-9][0-9,]*(?:\.\d{2})?)\s*(?:ea\.?|each)?',
            re.IGNORECASE,
        )
        titles = list(title_re.finditer(raw))
        titles.append(re.compile(r'$').search(raw))
        for i in range(len(titles)-1):
            m = titles[i]
            t = frac_to_float(m.group('t')); w = frac_to_float(m.group('w'))
            block = raw[m.end():titles[i+1].start()]
            for mp in lp_re.finditer(block):
                L = frac_to_float(mp.group('len'))
                p = float(mp.group('p').replace(',',''))
                out.append({
                    "material": "A2",
                    "thickness_in": t,
                    "width_in": w,
                    "length_in": L,
                    "price_each": p,
                    "source_url": url
                })
    return out

# ---------- Tungsten Carbide rectangle bar ----------

HREF_RE = re.compile(r'href=["\'](?P<h>[^"\']+)["\']')

def absolute(url: str, href: str) -> str:
    if href.startswith('http'):
        return href
    base = "https://www.onlinemetals.com"
    if href.startswith('/'):
        return base + href
    # category pages use absolute paths; fallback:
    return base + '/' + href

def scrape_carbide(category_url: str = CARBIDE_CAT_URL, max_items: int = 100) -> List[Dict]:
    """Collect product links from the carbide rectangle bar category (both C2 & micrograin),
    then open product pages to read price and dimensions."""
    out: List[Dict] = []
    html = get(category_url)
    # collect product hrefs
    hrefs = set()
    for h in HREF_RE.findall(html):
        if '/en/buy/tungsten-carbide/' in h and ('rectangle-bar' in h or 'rectangle' in h) and '/pid/' in h:
            hrefs.add(absolute(category_url, h))
    href_list = list(hrefs)[:max_items]
    for idx, h in enumerate(href_list):
        time.sleep(0.6)  # be polite
        try:
            ph = get(h)
        except Exception:
            continue
        txt = textify(ph)
        # price near "ea."
        pm = re.search(r'\$[ \t]*([0-9][0-9,]*\.\d{2})\s*ea\.', txt, re.IGNORECASE)
        price = float(pm.group(1).replace(',', '')) if pm else None
        # dimensions list (Thickness, Width) often shown
        tm = re.search(r'Thickness:\s*([0-9/.\s″"]+)', txt, re.IGNORECASE)
        wm = re.search(r'Width:\s*([0-9/.\s″"]+)', txt, re.IGNORECASE)
        t = frac_to_float(tm.group(1)) if tm else None
        w = frac_to_float(wm.group(1)) if wm else None
        # some pages include length; grab if present
        lm = re.search(r'Length:\s*([0-9/.\s″"]+)', txt, re.IGNORECASE)
        L = frac_to_float(lm.group(1)) if lm else None
        if price is None or t is None or w is None:
            continue
        out.append({
            "material": "Carbide",
            "grade": "C2_or_StandardMicrograin",
            "thickness_in": t,
            "width_in": w,
            "length_in": L,
            "price_each": price,
            "source_url": h
        })
    return out

# ---------- selection helpers ----------

def closest_bigger_sheet(rows: List[Dict], t_in: float, w_in: float, l_in: float) -> Optional[Dict]:
    cands = [r for r in rows if abs(r['thickness_in'] - t_in) < 1e-6 and r['width_in'] >= w_in and r['length_in'] >= l_in]
    if not cands:
        # allow rotation
        cands = [r for r in rows if abs(r['thickness_in'] - t_in) < 1e-6 and r['width_in'] >= l_in and r['length_in'] >= w_in]
        rotated = True
    else:
        rotated = False
    if not cands:
        return None
    cands.sort(key=lambda r: (r['width_in']*r['length_in'], r['price_each']))
    sel = dict(cands[0])
    sel['rotated'] = rotated
    return sel

def closest_a2_bar(rows: List[Dict], t_in: float, w_in: float, L_in: float) -> Optional[Dict]:
    # choose smallest width >= requested and length >= requested at exact thickness
    cands = [r for r in rows if abs(r['thickness_in'] - t_in) < 1e-6 and r['width_in'] >= w_in and r['length_in'] >= L_in]
    if not cands:
        # allow taller bar rotated (swap w with t) -> generally not applicable; skip
        return None
    cands.sort(key=lambda r: (r['width_in'], r['length_in'], r['price_each']))
    return cands[0]

def closest_carbide_bar(rows: List[Dict], t_in: float, w_in: float) -> Optional[Dict]:
    cands = [r for r in rows if abs(r['thickness_in'] - t_in) < 1e-6 and r['width_in'] >= w_in]
    if not cands:
        cands = [r for r in rows if r['thickness_in'] >= t_in and r['width_in'] >= w_in]
    if not cands:
        return None
    cands.sort(key=lambda r: (r['thickness_in'], r['width_in'], r['price_each']))
    return cands[0]

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="OnlineMetals scraper (requests + browser fallback)")
    sub = ap.add_subparsers(dest="mode", required=True)

    mic6 = sub.add_parser("mic6", help="Scrape MIC-6 category and pick a size")
    mic6.add_argument("--width", type=float, required=True)
    mic6.add_argument("--length", type=float, required=True)
    mic6.add_argument("--thickness", required=True, help='inches, e.g. 0.5 or "1/2"')
    mic6.add_argument("--url", default=MIC6_URL)
    mic6.add_argument("--dump", action="store_true", help="Dump parsed rows as JSON instead of selecting")

    a2 = sub.add_parser("a2", help="Scrape A2 rectangle bar and pick a bar")
    a2.add_argument("--thickness", required=True, help='inches, e.g. 0.0625 or "1/16"')
    a2.add_argument("--width", type=float, required=True, help="bar width in inches")
    a2.add_argument("--length", type=float, required=True, help="requested length, inches (choose 18 or 36 typically)")
    a2.add_argument("--url", default=A2_URL)
    a2.add_argument("--dump", action="store_true", help="Dump parsed rows as JSON instead of selecting")

    carb = sub.add_parser("carbide", help="Scrape carbide rectangle bar product pages and pick a bar")
    carb.add_argument("--thickness", required=True, help='inches, e.g. 0.25')
    carb.add_argument("--width", type=float, required=True)
    carb.add_argument("--category-url", default=CARBIDE_CAT_URL)
    carb.add_argument("--dump", action="store_true", help="Dump parsed rows as JSON instead of selecting")

    # global fetch controls
    ap.add_argument("--dump", action="store_true", help="Dump parsed rows as JSON instead of selecting")
    ap.add_argument("--sleep", type=float, default=0.8, help="delay between pages/requests (sec)")
    ap.add_argument("--requests-only", action="store_true", help="Force Requests only (no browser)")
    ap.add_argument("--browser", action="store_true", help="Force browser fetch (Playwright)")
    ap.add_argument("--engine", choices=["chromium","msedge","chrome","firefox","webkit"], default="chromium")
    ap.add_argument("--headful", action="store_true", help="Show the browser window when using Playwright")
    ap.add_argument("--user-data-dir", default=None, help="Persistent profile dir for Playwright (keeps cookies)")
    ap.add_argument("--disable-js", action="store_true", help="Disable JavaScript in Playwright context")
    ap.add_argument("--slowmo", type=int, default=0, help="Playwright slow_mo in ms")
    ap.add_argument("--debug-dir", default=None, help="If set, dump rendered text/HTML for debugging to this directory")

    args = ap.parse_args()

    # Set globals for fetch mode
    global GET_MODE, _BROWSER_ENGINE, _HEADFUL, _USER_DATA_DIR, _DISABLE_JS, _SLOWMO
    if args.requests_only:
        GET_MODE = "requests"
    elif args.browser:
        GET_MODE = "browser"
    else:
        GET_MODE = "auto"
    _BROWSER_ENGINE = args.engine
    _HEADFUL = bool(args.headful)
    _USER_DATA_DIR = args.user_data_dir
    _DISABLE_JS = bool(args.disable_js)
    _SLOWMO = int(args.slowmo or 0)
    global DEBUG_DIR
    DEBUG_DIR = args.debug_dir
    # Initialize debug directory early if requested
    if DEBUG_DIR:
        try:
            import os, json as _json
            os.makedirs(DEBUG_DIR, exist_ok=True)
            with open(os.path.join(DEBUG_DIR, "_debug_enabled.txt"), "w", encoding="utf-8") as f:
                f.write("debug enabled\n")
            with open(os.path.join(DEBUG_DIR, "_args.json"), "w", encoding="utf-8") as f:
                f.write(_json.dumps(vars(args), indent=2, default=str))
        except Exception:
            pass

    if args.mode == "mic6":
        t = frac_to_float(args.thickness)
        rows = scrape_mic6(args.url)
        if args.dump:
            print(json.dumps(rows, indent=2)); return
        sel = closest_bigger_sheet(rows, t, args.width, args.length)
        out = {
            "ok": bool(sel),
            "requested": {"material": "MIC6", "thickness_in": t, "width_in": args.width, "length_in": args.length},
            "selected": sel,
            "source": {"url": args.url}
        }
        print(json.dumps(out, indent=2)); return

    if args.mode == "a2":
        t = frac_to_float(args.thickness)
        rows = scrape_a2(args.url)
        if args.dump:
            print(json.dumps(rows, indent=2)); return
        sel = closest_a2_bar(rows, t, args.width, args.length)
        out = {
            "ok": bool(sel),
            "requested": {"material": "A2", "thickness_in": t, "width_in": args.width, "length_in": args.length},
            "selected": sel,
            "source": {"url": args.url}
        }
        print(json.dumps(out, indent=2)); return

    if args.mode == "carbide":
        t = frac_to_float(args.thickness)
        rows = scrape_carbide(args.category_url)
        if args.dump:
            print(json.dumps(rows, indent=2)); return
        sel = closest_carbide_bar(rows, t, args.width)
        out = {
            "ok": bool(sel),
            "requested": {"material": "Carbide", "thickness_in": t, "width_in": args.width},
            "selected": sel,
            "source": {"url": args.category_url}
        }
        print(json.dumps(out, indent=2)); return

## main() entry is moved to end of file so all helpers are defined first


# --- pagination helper ---
def collect_pages(start_url: str, max_pages: int = 20) -> List[str]:
    """Return a list of category page URLs to fetch (start + discovered)."""
    visited = set()
    queue = [start_url]
    out = []
    while queue and len(out) < max_pages:
        url = queue.pop(0)
        if url in visited: 
            continue
        visited.add(url)
        out.append(url)
        try:
            html = get(url)
        except Exception:
            continue
        # discover "next" style links
        for h in HREF_RE.findall(html):
            ah = absolute(url, h)
            if ('/en/buy/' in ah and ('?p=' in ah or '?page=' in ah)) and ah.startswith(start_url.split('?')[0]):
                if ah not in visited and ah not in queue and len(out)+len(queue) < max_pages:
                    queue.append(ah)
    return out

def scrape_a2(url: str = A2_URL) -> List[Dict]:
    pages = collect_pages(url, max_pages=12)  # first ~12 pages if present
    out: List[Dict] = []
    for page_index, u in enumerate(pages):
        html = get(u); raw = textify(html)
        if DEBUG_DIR:
            try:
                import os
                os.makedirs(DEBUG_DIR, exist_ok=True)
                with open(os.path.join(DEBUG_DIR, f"a2_cat_{page_index}.txt"), "w", encoding="utf-8") as f:
                    f.write(raw)
                with open(os.path.join(DEBUG_DIR, f"a2_cat_{page_index}.html"), "w", encoding="utf-8") as f:
                    f.write(html)
            except Exception:
                pass
        spans = list(A2_TITLE_RE.finditer(raw))
        spans.append(re.compile(r'$').search(raw))  # sentinel end
        for i in range(len(spans)-1):
            m = spans[i]
            t = frac_to_float(m.group('t')); w = frac_to_float(m.group('w'))
            block = raw[m.end():spans[i+1].start()]
            for mp in A2_LENGTH_PRICE_RE.finditer(block):
                L = frac_to_float(mp.group('len'))
                p = price_to_float(mp.group('p'))
                if p is None:
                    continue
                out.append({
                    "material": "A2",
                    "thickness_in": t,
                    "width_in": w,
                    "length_in": L,
                    "price_each": p,
                    "source_url": u
                })
        # Fallback on this page if nothing found
        if not out:
            title_re = re.compile(
                r'(?:A2\s*Tool\s*Steel).*?(?:Rectangle\s*Bar)?[^\n]{0,120}?'
                r'(?P<t>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*(?:["”]?|in\.?)?\s*(?:[xX×])\s*'
                r'(?P<w>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))',
                re.IGNORECASE,
            )
            lp_re = re.compile(
                r'(?P<len>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*(?:["”]?|in\.?|ft)?\s*(?:-|–|—)?\s*\$\s*'
                r'(?P<p>[0-9][0-9,]*(?:\.\d{2})?)\s*(?:ea\.?|each)?',
                re.IGNORECASE,
            )
            titles = list(title_re.finditer(raw))
            titles.append(re.compile(r'$').search(raw))
            for i in range(len(titles)-1):
                m = titles[i]
                t = frac_to_float(m.group('t')); w = frac_to_float(m.group('w'))
                block = raw[m.end():titles[i+1].start()]
                for mp in lp_re.finditer(block):
                    L = frac_to_float(mp.group('len'))
                    p = float(mp.group('p').replace(',',''))
                    out.append({
                        "material": "A2",
                        "thickness_in": t,
                        "width_in": w,
                        "length_in": L,
                        "price_each": p,
                        "source_url": u
                    })
    if out:
        return out
    # Product-page fallback: follow product links and parse lengths/prices
    def parse_a2_product(purl: str) -> List[Dict]:
        try:
            ph = get(purl)
        except Exception:
            return []
        soup = BeautifulSoup(ph, "lxml")
        txt = soup.get_text("\n", strip=True)
        if DEBUG_DIR:
            try:
                import os, hashlib
                os.makedirs(DEBUG_DIR, exist_ok=True)
                h = hashlib.sha1(purl.encode()).hexdigest()[:8]
                with open(os.path.join(DEBUG_DIR, f"a2_prod_{h}.txt"), "w", encoding="utf-8") as f:
                    f.write(txt)
            except Exception:
                pass
        # Try to find thickness/width
        tm = re.search(r'Thickness:\s*([0-9/.\s�?3"]+)', txt, re.IGNORECASE)
        wm = re.search(r'Width:\s*([0-9/.\s�?3"]+)', txt, re.IGNORECASE)
        if tm and wm:
            t = frac_to_float(tm.group(1)); w = frac_to_float(wm.group(1))
        else:
            # Try title form: A2 Tool Steel ... t x w
            m = re.search(r'A2[^\n]{0,80}?(?:Rectangle|Rectangular)?\s*Bar[^\n]{0,80}?'
                          r'(?P<t>[0-9/.\s]+)\s*(?:["”]?|in\.?)*\s*(?:x|×)\s*'
                          r'(?P<w>[0-9/.\s]+)', txt, re.IGNORECASE)
            if not m:
                return []
            t = frac_to_float(m.group('t')); w = frac_to_float(m.group('w'))
        # Length/price pairs anywhere in text (near each other)
        results: List[Dict] = []
        for mp in re.finditer(r'(?P<len>(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*(?:["”]?|in\.?|ft)\b.{0,40}?\$\s*(?P<p>[0-9][0-9,]*(?:\.\d{2})?)',
                               txt, re.IGNORECASE | re.DOTALL):
            L = frac_to_float(mp.group('len'))
            price = float(mp.group('p').replace(',', ''))
            results.append({
                "material": "A2",
                "thickness_in": t,
                "width_in": w,
                "length_in": L,
                "price_each": price,
                "source_url": purl
            })
        if results:
            return results
        # Fallback: a single price without explicit length
        sp = re.search(r'(?:Starting\s+at|From)\s*\$\s*([0-9][0-9,]*(?:\.\d{2})?)', txt, re.IGNORECASE)
        if sp:
            price = float(sp.group(1).replace(',', ''))
            return [{
                "material": "A2",
                "thickness_in": t,
                "width_in": w,
                "length_in": None,
                "price_each": price,
                "source_url": purl
            }]
        return []

    # Gather product links from category pages
    hrefs = set()
    for u in pages:
        try:
            html = get(u)
        except Exception:
            continue
        for h in HREF_RE.findall(html):
            ah = absolute(u, h)
            if ('/en/buy/' in ah and '/pid/' in ah and 'tool-steel' in ah and 
                (('rectangle' in ah) or ('rectangular' in ah) or ('bar' in ah))):
                hrefs.add(ah)
    href_list = list(hrefs)[:120]
    if DEBUG_DIR:
        try:
            import os
            os.makedirs(DEBUG_DIR, exist_ok=True)
            with open(os.path.join(DEBUG_DIR, "a2_hrefs.txt"), "w", encoding="utf-8") as f:
                for h in sorted(href_list):
                    f.write(h+"\n")
        except Exception:
            pass
    for h in href_list:
        out.extend(parse_a2_product(h))
    return out

if __name__ == "__main__":
    main()
