#!/usr/bin/env python3
"""
Download VerIH dataset files from GitHub.

Downloads train.json and test.json from the VerIH repository
into verih/data/.
"""

from __future__ import annotations

import json
import os
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/skai-research/VerIH/main/RLVR/dataset/verih"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download_file(url: str, dest: str) -> None:
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    # Verify it's valid JSON
    with open(dest, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Downloaded {len(data)} records")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for filename in ["train.json", "test.json"]:
        url = f"{BASE_URL}/{filename}"
        dest = os.path.join(DATA_DIR, filename)
        if os.path.exists(dest):
            print(f"  {dest} already exists, skipping.")
            continue
        download_file(url, dest)

    print("Done.")


if __name__ == "__main__":
    main()
