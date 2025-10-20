# mcmaster_api.py
# Prompts for a McMaster-Carr part number and prints its price tiers.
# Uses mutual TLS with a .pfx client certificate.

import os
import sys
import time
from typing import List, Dict, Any, Optional
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    # Optional: fall back to certifi/requests default trust store
    pass

# Force the local requests stub to proxy to the real package when this module
# is imported, so callers (including appV5) do not need to set any env vars.
# This must be set BEFORE importing "requests" so the stub detects it.
os.environ.setdefault("CAD_QUOTER_ALLOW_REQUESTS", "1")

import requests
from cad_quoter.utils import jdump

API_BASE = "https://api.mcmaster.com"

class McMasterAPI:
    def __init__(
        self,
        username: str,
        password: str,
        pfx_path: str,
        pfx_password: str = "",
    ):
        self.username = username
        self.password = password
        self.token: Optional[str] = None
        self.token_expiry = 0  # epoch seconds
        self.session = requests.Session()
        try:
            from requests_pkcs12 import Pkcs12Adapter
        except ImportError:
            raise SystemExit(
                "Missing dependency 'requests-pkcs12'.\n"
                f"Interpreter: {sys.executable}\n"
                "Install into THIS interpreter with:\n"
                f"  \"{sys.executable}\" -m pip install requests-pkcs12\n"
                "If using conda, ensure the env is activated before installing."
            )
        if not os.path.exists(pfx_path):
            raise SystemExit(f"PFX not found at: {pfx_path}")
        self.session.mount("https://", Pkcs12Adapter(
            pkcs12_filename=pfx_path,
            pkcs12_password=pfx_password or ""
        ))

    # ---------- low-level helpers ----------
    def _auth_header(self) -> Dict[str, str]:
        if not self.token or time.time() > (self.token_expiry - 60):
            self.login()
        return {"Authorization": f"Bearer {self.token}"}

    def login(self) -> None:
        r = self.session.post(f"{API_BASE}/v1/login", json={
            "UserName": self.username,
            "Password": self.password,
        })
        if r.status_code == 401:
            raise SystemExit(
                "401 Unauthorized on /v1/login. "
                "Double-check MCMASTER_USER / MCMASTER_PASS and that your PFX is mounted."
            )
        r.raise_for_status()
        data = r.json()
        self.token = data.get("AuthToken")
        if not self.token:
            raise SystemExit(f"Unexpected login response: {data}")
        # Token is ~24h; refresh proactively.
        self.token_expiry = time.time() + 23 * 3600

    def add_product(self, part_number: str) -> Dict[str, Any]:
        payload = {"URL": f"https://mcmaster.com/{part_number}"}
        r = self.session.put(
            f"{API_BASE}/v1/products",
            headers=self._auth_header(),
            json=payload
        )
        if r.status_code == 403:
            # Common: not subscribed or token drift; relog + retry once.
            self.login()
            r = self.session.put(
                f"{API_BASE}/v1/products",
                headers=self._auth_header(),
                json=payload
            )
        r.raise_for_status()
        return r.json()

    def get_price_tiers(self, part_number: str) -> List[Dict[str, Any]]:
        r = self.session.get(
            f"{API_BASE}/v1/products/{part_number}/price",
            headers=self._auth_header()
        )
        if r.status_code == 403:
            # Not subscribed yet – subscribe and retry.
            self.add_product(part_number)
            r = self.session.get(
                f"{API_BASE}/v1/products/{part_number}/price",
                headers=self._auth_header()
            )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise SystemExit(f"Unexpected price payload: {jdump(data, default=None)}")
        # Normalize keys casing just in case.
        norm = []
        for t in data:
            norm.append({
                "MinimumQuantity": t.get("MinimumQuantity") or t.get("minimumQuantity"),
                "Amount": t.get("Amount") or t.get("amount"),
                "UnitOfMeasure": t.get("UnitOfMeasure") or t.get("unitOfMeasure"),
            })
        # Sort by MinimumQuantity ascending
        norm.sort(key=lambda x: (x["MinimumQuantity"] if x["MinimumQuantity"] is not None else 10**12))
        return norm

# ---------- CLI ----------
def load_env():
    """Load .env if present; return a dict of required values, prompting if missing."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        # dotenv is optional; ignore if not installed
        pass

    # Hard-coded defaults (override via environment variables if set)
    default_user = "jwohlin2@gmail.com"
    default_pass = "Fuckingpassword1!"
    default_pfx_path = r"D:\\Composidie.pfx"
    default_pfx_pass = "Lashfg32864!"

    vals = {
        "MCMASTER_USER": os.getenv("MCMASTER_USER", default_user).strip(),
        "MCMASTER_PASS": os.getenv("MCMASTER_PASS", default_pass).strip(),
        "MCMASTER_PFX_PATH": os.getenv("MCMASTER_PFX_PATH", default_pfx_path).strip(),
        "MCMASTER_PFX_PASS": os.getenv("MCMASTER_PFX_PASS", default_pfx_pass).strip(),
    }

    # Prompt only for what’s missing (keeps your creds out of source)
    if not vals["MCMASTER_USER"]:
        vals["MCMASTER_USER"] = input("MCMASTER_USER (email): ").strip()
    if not vals["MCMASTER_PASS"]:
        # don’t import getpass unless needed
        import getpass
        vals["MCMASTER_PASS"] = getpass.getpass("MCMASTER_PASS: ").strip()
    if not vals["MCMASTER_PFX_PATH"]:
        vals["MCMASTER_PFX_PATH"] = input("MCMASTER_PFX_PATH (e.g., D:\\Composidie.pfx): ").strip()
    if not vals["MCMASTER_PFX_PASS"]:
        import getpass
        vals["MCMASTER_PFX_PASS"] = getpass.getpass("MCMASTER_PFX_PASS (blank if none): ")

    return vals

def print_tiers(tiers: List[Dict[str, Any]]) -> None:
    if not tiers:
        print("No pricing tiers returned for this part.")
        return
    # Pretty print simple table
    width_qty = max(14, max(len(str(t["MinimumQuantity"])) for t in tiers) + 2)
    width_amt = max(10, max(len(f'{t["Amount"]:.2f}') if isinstance(t["Amount"], (int, float)) else len(str(t["Amount"])) for t in tiers) + 2)
    width_uom = max(6, max(len(str(t["UnitOfMeasure"])) for t in tiers) + 2)
    header = f'{"MinQty".ljust(width_qty)}{"UnitPrice".ljust(width_amt)}{"UoM".ljust(width_uom)}'
    print(header)
    print("-" * len(header))
    for t in tiers:
        qty = str(t["MinimumQuantity"])
        amt = t["Amount"]
        if isinstance(amt, (int, float)):
            amt = f"{amt:.4f}"
        uom = str(t["UnitOfMeasure"])
        print(qty.ljust(width_qty) + str(amt).ljust(width_amt) + uom.ljust(width_uom))

    # If there is a tier for qty=1, highlight it.
    one_tier = next((t for t in tiers if (t.get("MinimumQuantity") or 0) <= 1), None)
    if one_tier:
        amt = one_tier["Amount"]
        if isinstance(amt, (int, float)):
            amt = f"{amt:.4f}"
        print(f"\nUnit price at qty=1: {amt} {one_tier['UnitOfMeasure']}")

def main():
    env = load_env()

    api = McMasterAPI(
        username=env["MCMASTER_USER"],
        password=env["MCMASTER_PASS"],
        pfx_path=env["MCMASTER_PFX_PATH"],
        pfx_password=env["MCMASTER_PFX_PASS"],
    )

    # Prompt for part number
    part = input("Enter McMaster part number (e.g., 4936K451): ").strip()
    if not part:
        print("No part number entered. Exiting.")
        sys.exit(1)

    # Login + price fetch
    try:
        api.login()
        tiers = api.get_price_tiers(part)
        print_tiers(tiers)
    except requests.HTTPError as e:
        # Surface helpful server responses
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text if e.response is not None else str(e)
        print(f"HTTP error: {e}\nDetails: {detail}")
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
