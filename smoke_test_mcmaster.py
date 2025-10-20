"""
Quick smoke test for the McMaster-Carr API client.

Usage:
  - Ensure dependencies are installed:
      pip install requests-pkcs12 truststore python-dotenv
  - Set CAD_QUOTER_ALLOW_REQUESTS=1 so the real 'requests' module is used.
  - Provide MCMASTER_* env vars or a .env file as documented in README.

Run:
  CAD_QUOTER_ALLOW_REQUESTS=1 python smoke_test_mcmaster.py --part 4936K451
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

from mcmaster_api import McMasterAPI, load_env, print_tiers


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test McMaster API")
    parser.add_argument(
        "--part",
        default=os.environ.get("SMOKE_PART", "4936K451"),
        help="McMaster part number to query (default: 4936K451)",
    )
    args = parser.parse_args(argv)

    # Ensure we use the real requests library (not the local stub)
    if not os.environ.get("CAD_QUOTER_ALLOW_REQUESTS"):
        os.environ["CAD_QUOTER_ALLOW_REQUESTS"] = "1"

    env = load_env()

    print("Authenticating…", flush=True)
    api = McMasterAPI(
        username=env["MCMASTER_USER"],
        password=env["MCMASTER_PASS"],
        pfx_path=env["MCMASTER_PFX_PATH"],
        pfx_password=env["MCMASTER_PFX_PASS"],
    )

    try:
        api.login()
        print(f"Fetching price tiers for part {args.part}…", flush=True)
        tiers: List[Dict[str, Any]] = api.get_price_tiers(args.part)
    except Exception as e:
        print(f"Smoke test failed: {e}")
        return 2

    # Print a minimal summary and then the full table
    print(f"Received {len(tiers)} pricing tiers.")
    if tiers:
        first = tiers[0]
        print(
            "First tier: MinQty=%s UnitPrice=%s UoM=%s"
            % (first.get("MinimumQuantity"), first.get("Amount"), first.get("UnitOfMeasure"))
        )
    print()
    print_tiers(tiers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

