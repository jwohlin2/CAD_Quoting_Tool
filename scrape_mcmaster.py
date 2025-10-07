from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
import os
from math import gcd
from typing import Dict, List, Optional, Tuple

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright
from rapidfuzz import fuzz

SIZE_RE = re.compile(r'(\d+\s*\d*(?:/\d+)?)"\s*[×x]\s*(\d+\s*\d*(?:/\d+)?)"\s*')
NUM_RE = re.compile(r'^\s*(\d+)(?:\s+(\d+)/(\d+))?\s*$')
DIM_WITH_LABEL_RE = re.compile(
    r'(\d+(?:\s+\d+/\d+)?|\d+/\d+|\d*\.\d+)\s*"?\s*(thick(?:ness)?|width|wide|height|tall|length|long)',
    re.IGNORECASE,
)

IN3_TO_CC = 16.387064
LB_PER_KG = 2.2046226218


@dataclass
class SheetOption:
    size_label: str
    width_in: float
    length_in: float
    thickness_in: float
    price_text: str
    price_value: float
    row_text: str


def frac_to_float(txt: str) -> float:
    """Parse fractional strings such as '1 1/2' or '3/8' into floats."""
    txt = txt.replace('"', '').replace('in', '').strip()
    if 'ft' in txt:
        match = re.findall(r'(\d+)\s*ft', txt)
        return float(match[0]) * 12.0 if match else 0.0

    match = NUM_RE.match(txt)
    if not match:
        return float(re.sub(r"[^\d.]+", "", txt)) if re.search(r"\d", txt) else 0.0

    whole = float(match.group(1))
    if match.group(2) and match.group(3):
        whole += float(match.group(2)) / float(match.group(3))
    return whole


def to_frac_label(value: float) -> str:
    """Convert decimal inches to the label used in McMaster filters (nearest 1/16)."""
    sixteenths = round(value * 16)
    value = sixteenths / 16.0
    whole = int(value)
    remainder = sixteenths - whole * 16
    if remainder == 0:
        return f'{whole}"'
    reduced_gcd = gcd(remainder, 16)
    numerator = remainder // reduced_gcd
    denominator = 16 // reduced_gcd
    return f'{whole} {numerator}/{denominator}"' if whole else f'{numerator}/{denominator}"'


def parse_size_label(txt: str) -> Optional[Tuple[float, float]]:
    match = SIZE_RE.search(txt)
    if not match:
        return None
    width = frac_to_float(match.group(1))
    length = frac_to_float(match.group(2))
    return width, length


def parse_bar_dimensions(text: str) -> Optional[Tuple[float, float, float]]:
    """Extract thickness, width, and length measurements from a table row."""

    dims: dict[str, float] = {}
    for match in DIM_WITH_LABEL_RE.finditer(text):
        value = frac_to_float(match.group(1))
        label = match.group(2).lower()
        if "thick" in label or "tall" in label:
            dims.setdefault("thickness", value)
        elif "width" in label or "wide" in label:
            dims.setdefault("width", value)
        elif "length" in label or "long" in label:
            dims.setdefault("length", value)
        elif "height" in label:
            dims.setdefault("thickness", value)
    if {"thickness", "width", "length"}.issubset(dims.keys()):
        return dims["thickness"], dims["width"], dims["length"]
    return None


def price_to_float(text: str) -> Optional[float]:
    match = re.search(r'\$\s*([0-9][0-9,]*(?:\.\d{2})?)', text)
    if not match:
        return None
    return float(match.group(1).replace(',', ''))


def closest_bigger(target_w: float, target_l: float, candidates: List[SheetOption]) -> Optional[SheetOption]:
    fits = [candidate for candidate in candidates if candidate.width_in >= target_w and candidate.length_in >= target_l]
    if not fits:
        return None
    fits.sort(key=lambda option: (option.width_in * option.length_in, option.width_in, option.length_in))
    return fits[0]


def closest_bigger_bar(
    target_t: float, target_w: float, target_l: float, candidates: List[SheetOption]
) -> Optional[SheetOption]:
    fits = [
        candidate
        for candidate in candidates
        if candidate.thickness_in >= target_t and candidate.width_in >= target_w and candidate.length_in >= target_l
    ]
    if not fits:
        return None
    fits.sort(
        key=lambda option: (
            option.thickness_in * option.width_in * option.length_in,
            option.thickness_in,
            option.width_in,
            option.length_in,
        )
    )
    return fits[0]


def wait_for_prices(section, timeout: int = 15000) -> None:
    section.locator('text=Sheets').first.wait_for(timeout=timeout)
    table = section.locator('xpath=.//*[normalize-space(text())="Sheets"]/following::table[1]')
    table.wait_for(timeout=timeout)
    table.locator(':text-is("$")').first.wait_for(timeout=timeout)


def extract_sheet_table(section, thickness_label: str) -> List[SheetOption]:
    table = section.locator('xpath=.//*[normalize-space(text())="Sheets"]/following::table[1]')
    rows = table.locator('tr')

    options: List[SheetOption] = []
    current_label: Optional[str] = None
    current_dims: Optional[Tuple[float, float]] = None

    row_count = rows.count()
    for index in range(row_count):
        row = rows.nth(index)
        text = row.inner_text().strip()
        dimensions = parse_size_label(text)
        if dimensions:
            current_label = SIZE_RE.search(text).group(0)
            current_dims = dimensions
            continue
        if not current_label or not current_dims:
            continue
        if thickness_label not in text:
            continue

        cells = row.locator('td')

        try:
            last_cell_text = cells.nth(cells.count() - 1).inner_text().strip()
        except PWTimeout:
            last_cell_text = text

        price_value = price_to_float(last_cell_text) or price_to_float(text) or -1.0
        price_text = last_cell_text if '$' in last_cell_text else (re.search(r'\$.*', text).group(0) if re.search(r'\$.*', text) else last_cell_text)

        options.append(
            SheetOption(
                size_label=current_label,
                width_in=current_dims[0],
                length_in=current_dims[1],
                thickness_in=frac_to_float(thickness_label),
                price_text=price_text,
                price_value=price_value,
                row_text=text,
            )
        )

    return options


def extract_bar_table(section) -> List[SheetOption]:
    table = section.locator('table').first
    rows = table.locator('tr')
    options: List[SheetOption] = []
    row_count = rows.count()
    for index in range(row_count):
        row = rows.nth(index)
        text = row.inner_text().strip()
        dims = parse_bar_dimensions(text)
        if not dims:
            continue
        thickness, width, length = dims
        cells = row.locator('td')
        try:
            last_cell_text = cells.nth(cells.count() - 1).inner_text().strip()
        except PWTimeout:
            last_cell_text = text
        price_value = price_to_float(last_cell_text) or price_to_float(text) or -1.0
        price_text = last_cell_text if '$' in last_cell_text else (re.search(r'\$.*', text).group(0) if re.search(r'\$.*', text) else last_cell_text)
        options.append(
            SheetOption(
                size_label=f"{thickness}\" × {width}\" × {length}\"",
                width_in=width,
                length_in=length,
                thickness_in=thickness,
                price_text=price_text,
                price_value=price_value,
                row_text=text,
            )
        )
    return options


def _find_section_with_keywords(page, *keywords: str):
    lowered = [kw.lower() for kw in keywords]
    sections = page.locator('section')
    count = sections.count()
    for index in range(count):
        section = sections.nth(index)
        try:
            heading = section.locator('h3').first.inner_text().strip()
        except Exception:
            continue
        heading_lower = heading.lower()
        if all(keyword in heading_lower for keyword in lowered):
            return section
    return None


def _score_button(label: str, candidate: str) -> int:
    return fuzz.partial_ratio(label.lower(), candidate.lower())


def click_filter_buttons(page, labels: List[str]) -> None:
    for label in labels:
        # Try robust role-based matching first
        locator = page.get_by_role("button", name=label)
        if locator.count() == 0:
            # Fallback: :has-text with single quotes to allow inch (") characters
            locator = page.locator(f"button:has-text('{label}')")
        if locator.count() == 0:
            # Fallback: :has-text with escaped double quotes
            safe = label.replace('"', '\\"')
            locator = page.locator(f'button:has-text("{safe}")')
        if locator.count() == 0:
            # Last resort: fuzzy match over all buttons
            buttons = page.locator('button')
            best_index = -1
            best_score = 0
            for i in range(buttons.count()):
                candidate_text = buttons.nth(i).inner_text().strip()
                score = _score_button(label, candidate_text)
                if score > best_score:
                    best_score = score
                    best_index = i
            if best_index >= 0 and best_score >= 75:
                locator = buttons.nth(best_index)
        if locator.count() > 0:
            locator.first.click()
            page.wait_for_timeout(300)


def maybe_accept_cookies(page) -> None:
    """Best-effort acceptance of cookie/location banners to reveal prices."""
    try:
        candidates = [
            "Accept",
            "I Agree",
            "Agree",
            "OK",
            "Got it",
            "Allow",
            "Continue",
            "Close",
        ]
        for label in candidates:
            btn = page.get_by_role("button", name=label)
            if btn.count() > 0:
                try:
                    btn.first.click()
                    page.wait_for_timeout(500)
                except Exception:
                    pass
    except Exception:
        pass


def _raise_if_restricted(page, url: str) -> None:
    """Detect McMaster's access restriction wall and raise a helpful error."""

    if page.locator('text=Access has been restricted').count() > 0:
        raise RuntimeError(
            "McMaster has temporarily restricted automated access for this browser session. "
            "Open the scraper with --headful and --keep-open, log in manually, and allow the "
            "script to save the resulting cookies using --state-file so subsequent headless "
            "runs reuse that session. You may also need to wait before retrying."
            f" (while visiting {url})"
        )


def extract_candidates(page, section_selector: str, thickness_label: str) -> List[SheetOption]:
    section = page.locator(section_selector)
    wait_for_prices(section)
    candidates = extract_sheet_table(section, thickness_label)
    if not candidates:
        raise RuntimeError(
            f'No price rows found for thickness {thickness_label}. The site layout may have changed or prices are gated.'
        )
    return candidates


def parse_current_sheet_page(page, thickness_in: float, w_in: float, l_in: float) -> Dict:
    thickness_label = to_frac_label(thickness_in)
    section = _find_section_with_keywords(page, 'aluminum', 'tool', 'jig') or page.locator('section').first
    try:
        wait_for_prices(section, timeout=30000)
    except Exception:
        pass
    candidates = extract_sheet_table(section, thickness_label)
    best = closest_bigger(w_in, l_in, candidates) or closest_bigger(l_in, w_in, candidates)
    if not best:
        raise RuntimeError('No sheet found meeting/exceeding requested size on current page.')
    return {
        'product_family': 'Manual Page Parse',
        'url': page.url,
        'requested': {
            'width_in': w_in,
            'length_in': l_in,
            'thickness_in': thickness_in,
        },
        'selected': {
            'size_label': best.size_label,
            'width_in': best.width_in,
            'length_in': best.length_in,
            'thickness_in': best.thickness_in,
            'price_each_text': best.price_text,
            'price_each_value': None if best.price_value < 0 else best.price_value,
        },
    }


def scrape_tool_jig(page, material: str, thickness_in: float, w_in: float, l_in: float) -> Dict:
    url = 'https://www.mcmaster.com/products/tool-and-jig-plates/'
    page.goto(url, wait_until='networkidle')

    _raise_if_restricted(page, url)

    click_filter_buttons(page, ['Inch', 'Sheet'])
    maybe_accept_cookies(page)
    material = material.lower()
    material_label = 'MIC6 Aluminum' if material.startswith('mic') else '5083 Aluminum'
    click_filter_buttons(page, [material_label])

    thickness_label = to_frac_label(thickness_in)
    click_filter_buttons(page, [thickness_label])

    heading = 'Easy-to-Machine MIC6' if 'MIC6' in material_label else 'Easy-to-Machine Cast Aluminum for Tool and Jig Plates'
    section_selector = f'section:has(h3:has-text("{heading}"))'

    try:
        candidates = extract_candidates(page, section_selector, thickness_label)
    except Exception:
        # Fallback: try to locate a likely section by keywords if the exact heading changes
        section = _find_section_with_keywords(page, 'aluminum', 'tool', 'jig')
        if section is not None:
            try:
                wait_for_prices(section, timeout=30000)
            except Exception:
                pass
            candidates = extract_sheet_table(section, thickness_label)
        else:
            # Last resort: scan the first section for any table we can parse
            section = page.locator('section').first
            try:
                wait_for_prices(section, timeout=30000)
            except Exception:
                pass
            candidates = extract_sheet_table(section, thickness_label)

    best = closest_bigger(w_in, l_in, candidates) or closest_bigger(l_in, w_in, candidates)
    if not best:
        raise RuntimeError('No sheet found that meets or exceeds the requested dimensions.')

    return {
        'product_family': f'{material_label} Sheets',
        'url': url,
        'requested': {
            'width_in': w_in,
            'length_in': l_in,
            'thickness_in': thickness_in,
        },
        'selected': {
            'size_label': best.size_label,
            'width_in': best.width_in,
            'length_in': best.length_in,
            'thickness_in': best.thickness_in,
            'price_each_text': best.price_text,
            'price_each_value': None if best.price_value < 0 else best.price_value,
        },
    }


def scrape_a2(page, tolerance: str, thickness_in: float, w_in: float, l_in: float) -> Dict:
    url = 'https://www.mcmaster.com/products/steel/material~a2-tool-steel/'
    page.goto(url, wait_until='networkidle')

    _raise_if_restricted(page, url)

    click_filter_buttons(page, ['Inch', 'Sheet'])

    thickness_label = to_frac_label(thickness_in)
    click_filter_buttons(page, [thickness_label])

    tolerance = (tolerance or 'tight').lower()
    if 'tight' in tolerance:
        heading = 'Tight-Tolerance Multipurpose Air-Hardening A2 Tool Steel Sheets and Bars'
        family = 'A2 Tool Steel Sheets (Tight Tolerance)'
    else:
        heading = 'Oversized Multipurpose Air-Hardening A2 Tool Steel Sheets and Bars'
        family = 'A2 Tool Steel Sheets (Oversized)'
    section_selector = f'section:has(h3:has-text("{heading}"))'

    candidates = extract_candidates(page, section_selector, thickness_label)

    best = closest_bigger(w_in, l_in, candidates) or closest_bigger(l_in, w_in, candidates)
    if not best:
        raise RuntimeError('No sheet found that meets or exceeds the requested dimensions.')

    return {
        'product_family': family,
        'url': url,
        'requested': {
            'width_in': w_in,
            'length_in': l_in,
            'thickness_in': thickness_in,
        },
        'selected': {
            'size_label': best.size_label,
            'width_in': best.width_in,
            'length_in': best.length_in,
            'thickness_in': best.thickness_in,
            'price_each_text': best.price_text,
            'price_each_value': None if best.price_value < 0 else best.price_value,
        },
    }


def scrape_carbide_bar(page, thickness_in: float, w_in: float, l_in: float) -> Dict:
    url = 'https://www.mcmaster.com/products/~/material~tungsten-carbide/shape~bar-1/?s=carbide'
    page.goto(url, wait_until='networkidle')

    _raise_if_restricted(page, url)

    click_filter_buttons(page, ['Inch'])

    section = _find_section_with_keywords(page, 'carbide', 'bar')
    if section is None:
        raise RuntimeError('Unable to locate tungsten carbide bar pricing section.')

    candidates = extract_bar_table(section)
    if not candidates:
        raise RuntimeError('No tungsten carbide bar price rows found.')

    best = closest_bigger_bar(thickness_in, w_in, l_in, candidates)
    if not best:
        # fall back to the largest available option to avoid failing hard
        best = max(
            candidates,
            key=lambda option: option.thickness_in * option.width_in * option.length_in,
        )

    return {
        'product_family': 'Tungsten Carbide Rectangular Bars',
        'url': url,
        'requested': {
            'width_in': w_in,
            'length_in': l_in,
            'thickness_in': thickness_in,
        },
        'selected': {
            'size_label': best.size_label,
            'width_in': best.width_in,
            'length_in': best.length_in,
            'thickness_in': best.thickness_in,
            'price_each_text': best.price_text,
            'price_each_value': None if best.price_value < 0 else best.price_value,
        },
    }


def _run_scraper_job(job, engine: str = 'chromium', headful: bool = False, state_file: Optional[str] = None, slow_mo: int = 0):
    with sync_playwright() as pw:
        if engine in ('chrome', 'msedge'):
            browser = pw.chromium.launch(channel=engine, headless=not headful, slow_mo=slow_mo)
        elif engine == 'firefox':
            browser = pw.firefox.launch(headless=not headful, slow_mo=slow_mo)
        elif engine == 'webkit':
            browser = pw.webkit.launch(headless=not headful, slow_mo=slow_mo)
        else:
            browser = pw.chromium.launch(headless=not headful, slow_mo=slow_mo)

        storage_state = state_file if (state_file and os.path.isfile(state_file)) else None
        context = browser.new_context(
            viewport={'width': 1400, 'height': 1000},
            java_script_enabled=True,
            storage_state=storage_state,
        )
        page = context.new_page()
        try:
            return job(page)
        finally:
            context.close()
            browser.close()


def _mass_kg_from_dimensions(thickness_in: float, width_in: float, length_in: float, density_g_cc: float) -> float:
    volume_in3 = max(0.0, thickness_in) * max(0.0, width_in) * max(0.0, length_in)
    mass_g = volume_in3 * IN3_TO_CC * max(0.0, density_g_cc)
    return mass_g / 1000.0


def _compute_unit_prices_from_result(result: Dict, density_g_cc: float) -> Optional[Dict[str, float]]:
    selected = result.get('selected') or {}
    price_each = selected.get('price_each_value')
    if price_each in (None, -1.0):
        price_each = price_to_float(selected.get('price_each_text', ''))
    if price_each in (None, -1.0):
        return None

    try:
        thickness = float(selected.get('thickness_in'))
        width = float(selected.get('width_in'))
        length = float(selected.get('length_in'))
    except Exception:
        return None

    mass_kg = _mass_kg_from_dimensions(thickness, width, length, density_g_cc)
    if mass_kg <= 0:
        return None

    usd_per_kg = float(price_each) / mass_kg
    usd_per_lb = usd_per_kg / LB_PER_KG
    return {
        'usd_per_kg': usd_per_kg,
        'usd_per_lb': usd_per_lb,
        'price_each': float(price_each),
        'mass_kg': mass_kg,
    }


def get_standard_stock_unit_prices(material_name: str) -> Optional[Dict[str, float | str]]:
    """Scrape McMaster for representative stock and convert to unit pricing."""

    name = (material_name or '').lower()
    specs = [
        {
            'key': 'aluminum',
            'match': lambda n: any(token in n for token in ('alum', '5083', 'mic6', '6061')),
            'job': lambda page: scrape_tool_jig(page, '5083', 0.5, 12.0, 12.0),
            'density_g_cc': 2.66,
        },
        {
            'key': 'tool_steel',
            'match': lambda n: 'tool' in n and 'steel' in n,
            'job': lambda page: scrape_a2(page, 'tight', 0.5, 6.0, 18.0),
            'density_g_cc': 7.85,
        },
        {
            'key': 'carbide',
            'match': lambda n: 'carbide' in n,
            'job': lambda page: scrape_carbide_bar(page, 0.25, 1.0, 12.0),
            'density_g_cc': 15.5,
        },
    ]

    for spec in specs:
        try:
            if not spec['match'](name):
                continue
        except Exception:
            continue

        result = _run_scraper_job(spec['job'])
        unit_prices = _compute_unit_prices_from_result(result, spec['density_g_cc'])
        if not unit_prices:
            return None
        unit_prices['source'] = f"mcmaster:{result.get('product_family', spec['key'])}"
        if result.get('url'):
            unit_prices['url'] = result['url']
        unit_prices['product_family'] = result.get('product_family')
        unit_prices['requested'] = result.get('requested')
        return unit_prices

    return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Scrape McMaster-Carr for the smallest sheet larger than requested dimensions.'
    )
    subparsers = parser.add_subparsers(dest='mode', required=True)

    tooling = subparsers.add_parser('tool_jig', help='Cast tooling plate sheets (MIC6 or 5083).')
    tooling.add_argument('--material', choices=['5083', 'mic6'], default='5083')
    tooling.add_argument('--thickness', required=True, help='Thickness in inches (e.g., 0.5 or "1/2").')
    tooling.add_argument('--width', type=float, required=True, help='Target width in inches.')
    tooling.add_argument('--length', type=float, required=True, help='Target length in inches.')

    a2 = subparsers.add_parser('a2', help='A2 tool steel sheets.')
    a2.add_argument('--tolerance', choices=['tight', 'oversized'], default='tight')
    a2.add_argument('--thickness', required=True, help='Thickness in inches (e.g., 0.375 or "3/8").')
    a2.add_argument('--width', type=float, required=True, help='Target width in inches.')
    a2.add_argument('--length', type=float, required=True, help='Target length in inches.')

    carbide = subparsers.add_parser('carbide', help='Tungsten carbide rectangular bars.')
    carbide.add_argument('--thickness', required=True, help='Thickness in inches (e.g., 0.25 or "1/4").')
    carbide.add_argument('--width', type=float, required=True, help='Target width in inches.')
    carbide.add_argument('--length', type=float, required=True, help='Target length in inches.')

    parser.add_argument('--headful', action='store_true', help='Run visible browser (disable headless).')
    parser.add_argument('--keep-open', action='store_true', help='Keep the browser open until you press Enter (headful only).')
    parser.add_argument('--state-file', default='.playwright_state.json', help='Path to persist cookies/storage state for subsequent runs.')
    parser.add_argument('--slowmo', type=int, default=0, help='Slow down headful actions by N ms for debugging (headful only).')
    parser.add_argument(
        '--engine',
        choices=['chromium', 'chrome', 'msedge', 'firefox', 'webkit'],
        default='chromium',
        help='Browser engine/channel to use',
    )
    parser.add_argument('--user-data-dir', help='Path to a persistent Chrome/Edge/Chromium profile')
    parser.add_argument(
        '--no-click',
        action='store_true',
        help='Manual mode: you navigate & filter; script only reads the table on the current page',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    # Expose for helper functions that need CLI switches
    globals()['CLI_ARGS'] = args
    thickness_in = frac_to_float(str(args.thickness))

    with sync_playwright() as playwright:
        headful = bool(getattr(args, 'headful', False))
        slow_mo = int(getattr(args, 'slowmo', 0)) if headful else 0
        engine = getattr(args, 'engine', 'chromium')

        persistent = bool(getattr(args, 'user_data_dir', None)) and engine in ('chromium', 'chrome', 'msedge')
        browser = None
        context = None
        result = None

        try:
            if persistent:
                context = playwright.chromium.launch_persistent_context(
                    args.user_data_dir,
                    channel=engine if engine in ('chrome', 'msedge') else None,
                    headless=not headful,
                    slow_mo=slow_mo,
                    viewport={'width': 1400, 'height': 1000},
                )
            else:
                if engine in ('chrome', 'msedge'):
                    browser = playwright.chromium.launch(channel=engine, headless=not headful, slow_mo=slow_mo)
                elif engine == 'firefox':
                    browser = playwright.firefox.launch(headless=not headful, slow_mo=slow_mo)
                elif engine == 'webkit':
                    browser = playwright.webkit.launch(headless=not headful, slow_mo=slow_mo)
                else:
                    browser = playwright.chromium.launch(headless=not headful, slow_mo=slow_mo)

                storage_state = None
                try:
                    # If a previous state file exists, reuse it so prices are visible without prompting again
                    if args.state_file and os.path.isfile(args.state_file):
                        storage_state = args.state_file
                except Exception:
                    storage_state = None

                context = browser.new_context(
                    viewport={'width': 1400, 'height': 1000},
                    java_script_enabled=True,
                    storage_state=storage_state,
                )

            page = context.new_page()

            if headful and getattr(args, 'no_click', False):
                print(
                    'Headful manual mode: open the exact McMaster page and set the filters you want. '
                    'Press Enter once the table is visible…'
                )
                input()
                if args.mode == 'tool_jig':
                    result = parse_current_sheet_page(page, thickness_in, args.width, args.length)
                else:
                    raise RuntimeError('Manual mode parser currently only supports tool_jig scraping.')
            else:
                if args.mode == 'tool_jig':
                    result = scrape_tool_jig(page, args.material, thickness_in, args.width, args.length)
                elif args.mode == 'a2':
                    result = scrape_a2(page, args.tolerance, thickness_in, args.width, args.length)
                else:
                    result = scrape_carbide_bar(page, thickness_in, args.width, args.length)
        except PWTimeout as exc:
            print(
                'Timed out waiting for prices. If you see items but no prices, McMaster may require a location cookie or login.',
                file=sys.stderr,
            )
            raise exc
        finally:
            # Save storage state to reuse accepted cookies or login across runs
            if not persistent:
                try:
                    if getattr(args, 'state_file', None) and context is not None:
                        context.storage_state(path=args.state_file)
                except Exception:
                    pass
            if headful and getattr(args, 'keep_open', False):
                try:
                    input('Headful browser is open. Press Enter to close...')
                except Exception:
                    pass
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()

    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == '__main__':
    main()
