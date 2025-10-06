from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from math import gcd
from typing import Dict, List, Optional, Tuple

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright
from rapidfuzz import fuzz

SIZE_RE = re.compile(r'(\d+\s*\d*(?:/\d+)?)"\s*[×x]\s*(\d+\s*\d*(?:/\d+)?)"\s*')
NUM_RE = re.compile(r'^\s*(\d+)(?:\s+(\d+)/(\d+))?\s*$')


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


def _score_button(label: str, candidate: str) -> int:
    return fuzz.partial_ratio(label.lower(), candidate.lower())


def click_filter_buttons(page, labels: List[str]) -> None:
    for label in labels:
        locator = page.locator(f'button:has-text("{label}")')
        if locator.count() == 0:
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


def extract_candidates(page, section_selector: str, thickness_label: str) -> List[SheetOption]:
    section = page.locator(section_selector)
    wait_for_prices(section)
    candidates = extract_sheet_table(section, thickness_label)
    if not candidates:
        raise RuntimeError(
            f'No price rows found for thickness {thickness_label}. The site layout may have changed or prices are gated.'
        )
    return candidates


def scrape_tool_jig(page, material: str, thickness_in: float, w_in: float, l_in: float) -> Dict:
    url = 'https://www.mcmaster.com/products/tool-and-jig-plates/'
    page.goto(url, wait_until='networkidle')

    click_filter_buttons(page, ['Inch', 'Sheet'])
    material = material.lower()
    material_label = 'MIC6 Aluminum' if material.startswith('mic') else '5083 Aluminum'
    click_filter_buttons(page, [material_label])

    thickness_label = to_frac_label(thickness_in)
    click_filter_buttons(page, [thickness_label])

    heading = 'Easy-to-Machine MIC6' if 'MIC6' in material_label else 'Easy-to-Machine Cast Aluminum for Tool and Jig Plates'
    section_selector = f'section:has(h3:has-text("{heading}"))'

    candidates = extract_candidates(page, section_selector, thickness_label)

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

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    thickness_in = frac_to_float(str(args.thickness))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={'width': 1400, 'height': 1000},
            java_script_enabled=True,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118 Safari/537.36',
        )
        page = context.new_page()

        try:
            if args.mode == 'tool_jig':
                result = scrape_tool_jig(page, args.material, thickness_in, args.width, args.length)
            else:
                result = scrape_a2(page, args.tolerance, thickness_in, args.width, args.length)
        except PWTimeout as exc:
            print(
                'Timed out waiting for prices. If you see items but no prices, McMaster may require a location cookie or login.',
                file=sys.stderr,
            )
            raise exc
        finally:
            context.close()
            browser.close()

    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == '__main__':
    main()
