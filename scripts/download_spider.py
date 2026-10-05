#!/usr/bin/env python3
"""Download Spider 1.0 (or fall back to documenting manual steps)."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

# Yale Spider release (public academic download). If this URL breaks, see README.
SPIDER_ZIP_URLS = [
    "https://huggingface.co/datasets/xlangai/spider/resolve/main/spider_data.zip",
    "https://drive.usercontent.google.com/download?id=1TqleX89RFiZY_FETwWrhgd48uEEi0OnG&export=download&confirm=t",
]


def _try_download(url: str, dest: Path, timeout: int = 120) -> bool:
    print(f"Trying {url} ...")
    try:
        req = Request(url, headers={"User-Agent": "text-to-sql-grpo/0.1"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 10_000:
            print(f"  response too small ({len(data)} bytes), skipping")
            return False
        dest.write_bytes(data)
        print(f"  saved {dest} ({len(data):,} bytes)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  failed: {exc}")
        return False


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"Extracted to {out_dir}")


def write_manual_instructions(out_dir: Path) -> None:
    msg = {
        "status": "manual_download_required",
        "instructions": [
            "1. Download Spider 1.0 from https://yale-lily.github.io/spider",
            "2. Unzip so you have data/spider/train_spider.json, dev.json, tables.json,",
            "   and data/spider/database/<db_id>/<db_id>.sqlite",
            "3. Or place the zip at data/spider_data.zip and re-run this script with --zip",
            "4. Free-track smoke tests do NOT need full Spider — use data/sample/",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "DOWNLOAD.md").write_text(
        "# Spider download\n\n"
        + "\n".join(f"- {s}" for s in msg["instructions"])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(msg, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/spider"),
        help="Extract directory (default: data/spider)",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Use an already-downloaded zip instead of fetching",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only write manual instructions",
    )
    args = parser.parse_args()

    if args.skip_download:
        write_manual_instructions(args.out)
        return 0

    zip_path = args.zip or Path("data/spider_data.zip")
    if args.zip:
        if not zip_path.exists():
            print(f"Zip not found: {zip_path}", file=sys.stderr)
            return 1
    elif not zip_path.exists():
        ok = False
        for url in SPIDER_ZIP_URLS:
            if _try_download(url, zip_path):
                ok = True
                break
        if not ok:
            write_manual_instructions(args.out)
            print(
                "\nAutomatic download failed. Free-track smoke path uses data/sample/ "
                "and does not require Spider.",
                file=sys.stderr,
            )
            return 2

    extract_zip(zip_path, args.out.parent)
    # Normalize: some zips extract to spider/ or spider_data/
    candidates = [
        args.out,
        args.out.parent / "spider",
        args.out.parent / "Spider",
        args.out.parent / "spider_data",
    ]
    for c in candidates:
        if (c / "train_spider.json").exists() or (c / "database").exists():
            print(f"Spider root ready at: {c.resolve()}")
            return 0
    print("Zip extracted but train_spider.json not found — check layout.", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
