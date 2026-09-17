#!/usr/bin/env python
"""Download Amazon 2014 per-category product metadata files.

The Amazon 2014 dataset (McAuley, https://jmcauley.ucsd.edu/data/amazon/index_2014.html)
ships product metadata as per-category gzip files named ``meta_<Category>.json.gz``,
hosted on Stanford SNAP:

    https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_<Category>.json.gz

This downloads those files into a folder using the exact ``meta_{category}.json.gz``
naming that ``data_module.build_amazon_2014_dataset`` expects, so you can then run with:

    dataset_version: 2014
    amazon_2014_dir: "<output dir>"

Note: a few large categories (e.g. Books) historically required emailing the author and
may return HTTP 403; the smaller categories download without auth.

Usage:
    python download_amazon_2014.py --out ./amazon2014_meta Beauty Office_Products
    python download_amazon_2014.py --out ./amazon2014_meta --all
    python download_amazon_2014.py --list
"""

import argparse
import os
import sys
import urllib.error
import urllib.request

BASE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
FILENAME_TEMPLATE = "meta_{category}.json.gz"

# Known Amazon 2014 category names (used by --all / --list). Not exhaustive; you can
# pass any valid category name as a positional argument.
KNOWN_CATEGORIES = [
    "Amazon_Instant_Video",
    "Apps_for_Android",
    "Automotive",
    "Baby",
    "Beauty",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Grocery_and_Gourmet_Food",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Kindle_Store",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Sports_and_Outdoors",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
]

# A browser-like UA avoids the 403 the server returns to some default clients.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; amazon2014-downloader/1.0)"}


def _human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def download_category(category: str, out_dir: str, overwrite: bool = False) -> bool:
    """Download one category's meta file. Returns True on success (or already present)."""
    filename = FILENAME_TEMPLATE.format(category=category)
    url = f"{BASE_URL}/{filename}"
    dest = os.path.join(out_dir, filename)

    if os.path.isfile(dest) and not overwrite:
        print(f"[skip] {filename} already exists ({_human(os.path.getsize(dest))})")
        return True

    tmp = dest + ".part"
    print(f"[get ] {url}")
    try:
        request = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(request) as response, open(tmp, "wb") as handle:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = response.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(
                        f"\r       {_human(downloaded)}/{_human(total)} ({pct}%)",
                        end="",
                        flush=True,
                    )
        if total:
            print()
        os.replace(tmp, dest)
        print(f"[done] {dest} ({_human(os.path.getsize(dest))})")
        return True
    except urllib.error.HTTPError as exc:
        print(f"[fail] {filename}: HTTP {exc.code} {exc.reason}", file=sys.stderr)
        if exc.code == 403:
            print(
                "       This category may require permission from the dataset author "
                "(see https://jmcauley.ucsd.edu/data/amazon/index_2014.html).",
                file=sys.stderr,
            )
    except (urllib.error.URLError, OSError) as exc:
        print(f"[fail] {filename}: {exc}", file=sys.stderr)
    if os.path.isfile(tmp):
        os.remove(tmp)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("categories", nargs="*", help="Category names, e.g. Beauty Office_Products")
    parser.add_argument("--out", default="./amazon2014_meta", help="Output directory (default: ./amazon2014_meta)")
    parser.add_argument("--all", action="store_true", help="Download all known categories")
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if the file exists")
    parser.add_argument("--list", action="store_true", help="Print known category names and exit")
    args = parser.parse_args()

    if args.list:
        print("\n".join(KNOWN_CATEGORIES))
        return 0

    categories = KNOWN_CATEGORIES if args.all else args.categories
    if not categories:
        parser.error("pass one or more category names, or --all (see --list)")

    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading {len(categories)} category file(s) into {os.path.abspath(args.out)}\n")

    failures = [c for c in categories if not download_category(c, args.out, args.overwrite)]
    print()
    if failures:
        print(f"Completed with {len(failures)} failure(s): {', '.join(failures)}", file=sys.stderr)
        return 1
    print("All requested category files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
