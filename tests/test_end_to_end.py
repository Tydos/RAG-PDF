#!/usr/bin/env python3
"""
Smoke test: verify API is up and /documents responds.
(Search and retrieval tests were removed when hybrid search was dropped.)
"""

import sys
import requests

API_BASE = "http://localhost:8000"


def main() -> int:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
    except requests.RequestException as e:
        print(f"Cannot reach API: {e}")
        return 1
    if r.status_code != 200:
        print(f"Health failed: {r.status_code} {r.text}")
        return 1
    body = r.json()
    print("Health:", body)
    if body.get("status") != "ok":
        print("Status is not ok")
        return 1

    r = requests.get(f"{API_BASE}/documents", timeout=10)
    if r.status_code != 200:
        print(f"/documents failed: {r.status_code} {r.text}")
        return 1
    print("Documents:", len(r.json().get("documents", [])), "row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
