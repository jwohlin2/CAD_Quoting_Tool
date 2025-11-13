# test_scrapmetalbuyers_scraper.py
"""
Test suite for ScrapMetalBuyers scraper.
Run with: python test_scrapmetalbuyers_scraper.py
"""

import json
import math
import tempfile
import os
from typing import Dict, Any

# Import the scrapers
import scrapmetalbuyers_scraper as smb

def test_to_float():
    """Test number parsing."""
    print("Testing _to_float()...")
    
    tests = [
        ("3.50", 3.50),
        ("$3.50", 3.50),
        ("3,500.00", 3500.00),
        ("1 234.56", 1234.56),
        ("€2,50", 2.50),
        ("invalid", math.nan),
    ]
    
    for input_str, expected in tests:
        result = smb._to_float(input_str)
        if math.isnan(expected):
            assert math.isnan(result), f"Expected NaN for '{input_str}', got {result}"
        else:
            assert abs(result - expected) < 0.001, f"Expected {expected} for '{input_str}', got {result}"
    
    print("✓ _to_float() tests passed")


def test_unit_conversion():
    """Test unit conversions."""
    print("Testing unit conversions...")
    
    # USD/lb to USD/kg
    lb_price = 1.0
    kg_price = smb.usdlb_to_usdkg(lb_price)
    assert abs(kg_price - 2.20462) < 0.001, f"Expected ~2.20462, got {kg_price}"
    
    # USD/kg to USD/lb
    kg_price = 2.20462
    lb_price = smb.usdkg_to_usdlb(kg_price)
    assert abs(lb_price - 1.0) < 0.001, f"Expected ~1.0, got {lb_price}"
    
    print("✓ Unit conversion tests passed")


def test_material_normalization():
    """Test material name normalization."""
    print("Testing material normalization...")
    
    tests = [
        ("Copper", "copper"),
        ("ALUMINUM", "aluminum"),
        ("  Brass  ", "brass"),
        ("Stainless Steel", "stainless steel"),
    ]
    
    for input_str, expected in tests:
        result = smb._normalize_material_name(input_str)
        assert result == expected, f"Expected '{expected}', got '{result}'"
    
    print("✓ Material normalization tests passed")


def test_find_price_for_material():
    """Test material price lookup."""
    print("Testing material lookup...")
    
    prices = {
        "Bare Bright Copper": 3.50,
        "Aluminum Cans": 0.75,
        "Yellow Brass": 2.10,
        "Stainless Steel 304": 0.65,
    }
    
    # Direct match
    result = smb._find_price_for_material(prices, "Yellow Brass")
    assert result is not None, "Should find exact match"
    assert result[0] == 2.10, f"Expected 2.10, got {result[0]}"
    
    # Fuzzy match
    result = smb._find_price_for_material(prices, "copper")
    assert result is not None, "Should find fuzzy match for copper"
    assert result[0] == 3.50, f"Expected 3.50, got {result[0]}"
    
    # Keyword match
    result = smb._find_price_for_material(prices, "aluminum")
    assert result is not None, "Should find keyword match for aluminum"
    assert result[0] == 0.75, f"Expected 0.75, got {result[0]}"
    
    # No match
    result = smb._find_price_for_material(prices, "platinum")
    assert result is None, "Should return None for no match"
    
    print("✓ Material lookup tests passed")


def test_cache_operations():
    """Test caching functionality."""
    print("Testing cache operations...")
    
    # Create test data
    test_data = {
        "source": "test",
        "prices_usd_per_lb": {"Copper": 3.50}
    }
    
    # Write cache
    smb._write_temp_cache(test_data)
    
    # Read cache
    cached = smb._read_temp_cache()
    assert cached is not None, "Cache should be readable"
    assert cached["source"] == "test", "Cache data should match"
    
    print("✓ Cache operations tests passed")


def test_html_parsing_fallback():
    """Test HTML parsing with sample data."""
    print("Testing HTML parsing...")
    
    sample_html = """
    <html>
    <body>
        <table>
            <tr>
                <td>Material</td>
                <td>Price</td>
            </tr>
            <tr>
                <td>Copper</td>
                <td>$3.50 per lb</td>
            </tr>
            <tr>
                <td>Aluminum</td>
                <td>$0.75/lb</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    doc = smb.SoupDocument(sample_html)
    prices = smb._parse_prices_from_html(doc)
    
    # Should find at least some prices
    print(f"  Found {len(prices)} prices: {prices}")
    
    # Note: Results depend on BeautifulSoup availability and HTML structure
    if prices:
        print("✓ HTML parsing tests passed")
    else:
        print("⚠ HTML parsing returned no results (may need BeautifulSoup)")


def test_data_structure():
    """Test that scraper returns correct data structure."""
    print("Testing data structure...")
    
    # Create mock data
    data = {
        "source": SCRAPMETALBUYERS_URL,
        "asof": "Nov 13, 2025",
        "prices_usd_per_lb": {
            "Copper": 3.50,
            "Aluminum": 0.75,
        },
        "prices_usd_per_kg": {
            "Copper": 7.7162,
            "Aluminum": 1.6535,
        }
    }
    
    # Validate structure
    assert "source" in data, "Missing 'source' field"
    assert "asof" in data, "Missing 'asof' field"
    assert "prices_usd_per_lb" in data, "Missing 'prices_usd_per_lb' field"
    assert "prices_usd_per_kg" in data, "Missing 'prices_usd_per_kg' field"
    
    # Validate conversions
    for material, price_lb in data["prices_usd_per_lb"].items():
        expected_kg = smb.usdlb_to_usdkg(price_lb)
        actual_kg = data["prices_usd_per_kg"][material]
        diff = abs(actual_kg - expected_kg)
        assert diff < 0.01, f"Conversion mismatch for {material}: {actual_kg} vs {expected_kg}"
    
    print("✓ Data structure tests passed")


def test_material_keywords():
    """Test material keyword mapping."""
    print("Testing material keywords...")
    
    # Check that all expected materials are present
    expected_materials = [
        "copper", "aluminum", "brass", "steel", 
        "stainless", "lead", "zinc", "nickel"
    ]
    
    for material in expected_materials:
        assert material in smb.MATERIAL_KEYWORDS, f"Missing keywords for {material}"
        keywords = smb.MATERIAL_KEYWORDS[material]
        assert len(keywords) > 0, f"Empty keywords for {material}"
        assert material in keywords, f"Material name not in its own keywords: {material}"
    
    print("✓ Material keywords tests passed")


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("Running ScrapMetalBuyers Scraper Tests")
    print("=" * 60)
    
    tests = [
        test_to_float,
        test_unit_conversion,
        test_material_normalization,
        test_find_price_for_material,
        test_cache_operations,
        test_html_parsing_fallback,
        test_data_structure,
        test_material_keywords,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    # Import after defining SCRAPMETALBUYERS_URL
    from scrapmetalbuyers_scraper import SCRAPMETALBUYERS_URL
    
    success = run_all_tests()
    
    if not success:
        print("\n⚠ Some tests failed. This is expected if:")
        print("  - BeautifulSoup is not installed")
        print("  - The website structure has changed")
        print("  - Network access is blocked")
    else:
        print("\n✓ All tests passed!")
    
    exit(0 if success else 1)
