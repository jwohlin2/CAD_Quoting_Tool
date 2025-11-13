# ScrapMetalBuyers Scraper

A web scraper for extracting scrap metal prices from https://scrapmetalbuyers.com/current-prices/

## Overview

This scraper follows the same architecture as `wieland_scraper.py` and provides:
- Price extraction in USD/lb and USD/kg
- Caching (in-memory and temp file)
- Material lookup with fuzzy matching
- CLI interface for testing and usage

## Installation

Requires Python 3.7+ and optionally BeautifulSoup4 for better HTML parsing:

```bash
pip install beautifulsoup4 lxml
```

## Usage

### Command Line

```bash
# Show all prices
python scrapmetalbuyers_scraper.py

# Get JSON output
python scrapmetalbuyers_scraper.py --json

# Look up specific material
python scrapmetalbuyers_scraper.py --material copper
python scrapmetalbuyers_scraper.py --material aluminum --unit kg
python scrapmetalbuyers_scraper.py --material brass --unit both

# Force refresh (bypass cache)
python scrapmetalbuyers_scraper.py --force

# Debug mode (saves HTML snapshot)
python scrapmetalbuyers_scraper.py --debug
```

### Python API

```python
from scrapmetalbuyers_scraper import (
    scrape_scrapmetalbuyers_prices,
    get_live_scrap_price_usd_per_lb,
    get_live_scrap_price_usd_per_kg
)

# Get all prices
data = scrape_scrapmetalbuyers_prices()
print(data['prices_usd_per_lb'])

# Look up specific material
price_lb, source = get_live_scrap_price_usd_per_lb('copper')
print(f"Copper: ${price_lb}/lb from {source}")

# Get price in kg
price_kg, source = get_live_scrap_price_usd_per_kg('aluminum')
print(f"Aluminum: ${price_kg}/kg from {source}")
```

## Data Structure

The scraper returns a dictionary with this structure:

```python
{
    "source": "https://scrapmetalbuyers.com/current-prices/",
    "asof": "Nov 13, 2025",  # or None if not found
    "prices_usd_per_lb": {
        "Copper": 3.50,
        "Aluminum": 0.75,
        "Brass": 2.10,
        # ... more materials
    },
    "prices_usd_per_kg": {
        "Copper": 7.7161,
        "Aluminum": 1.6535,
        "Brass": 4.6297,
        # ... more materials (auto-converted)
    }
}
```

## Material Mapping

The scraper includes fuzzy matching for common material names:

- **Copper**: copper, cu, bare bright, wire, #1 copper, #2 copper
- **Aluminum**: aluminum, aluminium, al, alum, 6061, extrusion
- **Brass**: brass, yellow brass, red brass
- **Steel**: steel, iron, ferrous, scrap steel, sheet iron
- **Stainless**: stainless, stainless steel, 304, 316, ss
- **Lead**: lead, pb, battery, wheel weight
- And more...

## Caching

- **In-memory cache**: 30 minutes (configurable via `SMB_CACHE_TTL_S`)
- **File cache**: Located in system temp directory
- Use `--force` flag to bypass cache

## Configuration

Environment variables:

- `SMB_CACHE_TTL_S`: Cache TTL in seconds (default: 1800)
- `SMB_REQ_TIMEOUT_S`: Request timeout (default: 30)
- `SMB_USER_AGENT`: Custom user agent string

## Notes on Website Structure

The ScrapMetalBuyers website may use:
1. Dynamic content loading (JavaScript)
2. CDN/WAF protection (Cloudflare, etc.)
3. Rate limiting

If you encounter 403 errors or empty results, you may need to:

1. **Use a headless browser** (Selenium/Playwright):
```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)
driver.get('https://scrapmetalbuyers.com/current-prices/')
html = driver.page_source
driver.quit()
```

2. **Add delays between requests**:
```python
import time
time.sleep(2)  # Wait 2 seconds between requests
```

3. **Use a proxy service** that handles JavaScript rendering

4. **Check for API endpoints**: Some sites have hidden JSON APIs that are easier to scrape

## Comparison with Wieland Scraper

| Feature | Wieland | ScrapMetalBuyers |
|---------|---------|------------------|
| Data source | https://www.wieland.com | https://scrapmetalbuyers.com |
| Prices | LME + vendor lists | Scrap buyer prices |
| FX rates | Yes (EUR/USD/GBP) | No (USD only) |
| Units | Multiple (EUR/100KG, GBP/t, USD/kg) | USD/lb primarily |
| Materials | Industrial metals, alloys | Scrap metals |
| JSON payloads | Yes (__NEXT_DATA__) | Possibly (not confirmed) |

## Troubleshooting

### 403 Forbidden Error
The website is blocking the request. Try:
- Using Selenium/Playwright
- Using a different IP/proxy
- Adding more realistic browser headers

### Empty Results
- Check if the HTML structure has changed
- Run with `--debug` to save HTML snapshot
- Inspect the saved HTML file for table structure

### Prices Not Matching
- Verify the table structure in debug HTML
- Check regex patterns in `_parse_prices_from_html()`
- Material names may have changed

## Integration with Existing Quoter System

To integrate with your existing `cad_quoter` system:

```python
# In your pricing module
from cad_quoter.pricing import scrapmetalbuyers_scraper

def get_scrap_price(material: str) -> float:
    """Get scrap price for a material."""
    price, source = scrapmetalbuyers_scraper.get_live_scrap_price_usd_per_lb(material)
    return price
```

## Future Improvements

1. **Selenium/Playwright support** for JavaScript-rendered content
2. **Retry logic** with exponential backoff
3. **Multiple scraper sources** (fallback to other scrap price sites)
4. **Historical price tracking**
5. **Alert system** for price changes
6. **API endpoint detection** for more reliable data

## License

Follow the same license as the parent `cad_quoter` project.
