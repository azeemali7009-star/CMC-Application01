#!/usr/bin/env python3
"""Download the open tumour-stroma spheroid feature files from Figshare.

Dataset item: 26346454
DOI collection: 10.6084/m9.figshare.c.7357135

Public Figshare metadata and file downloads do not require authentication.
The script intentionally downloads only small tabular feature files unless
--all is supplied; the raw multi-TIFF collection is large.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import requests

ITEM_ID = 26346454
API = f"https://api.figshare.com/v2/articles/{ITEM_ID}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/experimental_figshare")
    parser.add_argument("--all", action="store_true", help="download every file in the item")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    response = requests.get(API, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    wanted = {"Features_pixel.csv", "Features_um.csv", "Features_all.xls"}
    for info in metadata.get("files", []):
        if not args.all and info["name"] not in wanted:
            continue
        target = out / info["name"]
        with requests.get(info["download_url"], stream=True, timeout=180) as r:
            r.raise_for_status()
            with target.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        print(f"downloaded {target}")


if __name__ == "__main__":
    main()
