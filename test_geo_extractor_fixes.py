#!/usr/bin/env python3
"""Test script for geo_extractor GD&T font and radial dimension fixes."""

import sys
sys.path.insert(0, '.')

from cad_quoter.geo_extractor import _plain

def test_gdt_font_mapping():
    """Test GD&T font character mapping."""
    test_cases = [
        ('1.1933 {\\Famgdt|c0;q}', '1.1933 Ø'),
        ('{\\Famgdt|c0;q}.3834', 'Ø.3834'),
        ('TEXT {\\FAMGDT|c0;q}', 'TEXT Ø'),  # uppercase version
        ('%%c.5', 'Ø.5'),  # old style diameter
        ('.010 {\\Famgdt|c0;h}', '.010 ⌭'),  # perpendicularity
    ]

    print('Testing GD&T font character mapping:')
    print('=' * 60)
    passed = 0
    for input_text, expected in test_cases:
        result = _plain(input_text)
        status = '✓' if result == expected else '✗'
        if result == expected:
            passed += 1
        print(f'{status} Input:    {repr(input_text)}')
        print(f'  Expected: {repr(expected)}')
        print(f'  Got:      {repr(result)}')
        print()

    print(f'Passed {passed}/{len(test_cases)} tests')
    return passed == len(test_cases)

if __name__ == '__main__':
    success = test_gdt_font_mapping()
    sys.exit(0 if success else 1)
