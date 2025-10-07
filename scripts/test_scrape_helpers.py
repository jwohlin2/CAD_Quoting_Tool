import sys
from pathlib import Path

# Ensure repository root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scrape_mcmaster as m

def main():
    print('frac_to_float("1 1/2") =', m.frac_to_float('1 1/2'))
    print('frac_to_float("3/8") =', m.frac_to_float('3/8'))
    print('to_frac_label(0.5) =', m.to_frac_label(0.5))
    print('parse_bar_dimensions sample =', m.parse_bar_dimensions('0.25" thick, 1 1/2" wide, 12" long  $45.67'))
    print('price_to_float sample =', m.price_to_float('Price $123.45 ea'))

if __name__ == '__main__':
    main()
