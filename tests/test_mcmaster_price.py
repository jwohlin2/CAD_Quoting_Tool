#!/usr/bin/env python
"""
Simple test script to fetch McMaster product info via API.
Usage: python test_mcmaster_product.py [part_number]
"""

import os
import sys
import json

os.environ['CAD_QUOTER_ALLOW_REQUESTS'] = '1'

from mcmaster_api import McMasterAPI, load_env

def main():
    # Get part number from command line or prompt
    if len(sys.argv) > 1:
        part_number = sys.argv[1]
    else:
        part_number = input("Enter McMaster part number: ").strip()

    if not part_number:
        print("No part number provided")
        sys.exit(1)

    # Load credentials and create API client
    env = load_env()
    api = McMasterAPI(
        username=env['MCMASTER_USER'],
        password=env['MCMASTER_PASS'],
        pfx_path=env['MCMASTER_PFX_PATH'],
        pfx_password=env['MCMASTER_PFX_PASS'],
    )

    # Login
    print(f"Logging in as {env['MCMASTER_USER']}...")
    api.login()
    print("Login successful\n")

    # Fetch price info
    print(f"Fetching price for {part_number}...")
    url = f"https://api.mcmaster.com/v1/products/{part_number}/price"

    r = api.session.get(url, headers={'Authorization': f'Bearer {api.token}'})

    print(f"Status: {r.status_code}")
    print(f"URL: {url}\n")

    if r.status_code == 200:
        data = r.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {r.text}")

if __name__ == "__main__":
    main()
