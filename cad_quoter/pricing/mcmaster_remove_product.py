#!/usr/bin/env python
"""
Simple test script to remove a product from McMaster subscribed products.
Usage: python test_mcmaster_remove_product.py [part_number]
"""

import os
import sys
import json

# Add project root to path so we can import mcmaster_api
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ['CAD_QUOTER_ALLOW_REQUESTS'] = '1'

from mcmaster_api import McMasterAPI, load_env

def main():
    # Get part number from command line or prompt
    if len(sys.argv) > 1:
        part_number = sys.argv[1]
    else:
        part_number = input("Enter McMaster part number to remove: ").strip()

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

    # Remove product from subscribed list
    print(f"Removing product {part_number} from subscribed list...")
    url = "https://api.mcmaster.com/v1/products"
    payload = {"URL": f"https://mcmaster.com/{part_number}"}

    r = api.session.delete(
        url,
        json=payload,
        headers={'Authorization': f'Bearer {api.token}'}
    )

    print(f"Status: {r.status_code}")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload)}\n")

    if r.status_code in [200, 204]:
        if r.text:
            data = r.json()
            print("Response:")
            print(json.dumps(data, indent=2))
        else:
            print("Product removed successfully")
    else:
        print(f"Error: {r.text}")

if __name__ == "__main__":
    main()
