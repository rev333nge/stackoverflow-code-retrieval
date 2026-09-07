"""
Phase 1 - Step 1: download the Stack Overflow `posts` dump (parquet).

Source: ClickHouse's public parquet mirror of the official Stack Exchange
Data Dump, hosted on S3 with no authentication required:
    https://datasets-documentation.s3.eu-west-3.amazonaws.com/stackoverflow/parquet/posts/

We pull the `posts` table only (questions + answers + wikis), split by year,
2008-2024, ~38 GB total. The Comments / Votes / Users tables are not needed:
the relevance signal we care about (accepted answer, score) lives directly on
the post rows.

The download is resumable. Re-running the script:
  - skips files that are already complete,
  - continues partially downloaded files from where they stopped,
  - retries transient network errors a few times per file.

Each file's final size is checked against the size published in the bucket
listing, so a truncated download is caught rather than silently used.

Usage:
    python scripts/download_posts.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = (
    "https://datasets-documentation.s3.eu-west-3.amazonaws.com"
    "/stackoverflow/parquet/posts"
)

# (filename, expected size in bytes). Sizes come from the S3 bucket listing and
# are used to detect truncated or corrupted downloads.
FILES: list[tuple[str, int]] = [
    ("2008.parquet", 114_202_383),
    ("2009.parquet", 572_813_781),
    ("2010.parquet", 1_000_405_052),
    ("2011.parquet", 1_696_802_710),
    ("2012.parquet", 2_395_041_166),
    ("2013.parquet", 3_076_752_845),
    ("2014.parquet", 3_239_525_632),
    ("2015.parquet", 3_406_541_279),
    ("2016.parquet", 3_452_125_362),
    ("2017.parquet", 3_346_038_783),
    ("2018.parquet", 3_031_439_838),
    ("2019.parquet", 2_896_135_538),
    ("2020.parquet", 3_062_744_031),
    ("2021.parquet", 2_561_142_532),
    ("2022.parquet", 2_240_372_351),
    ("2023.parquet", 1_643_348_090),
    ("2024.parquet", 338_404_102),
]

DEST_DIR = Path("data/raw/posts")
CHUNK = 1 << 20           # 1 MiB read buffer
MAX_RETRIES = 3
TIMEOUT = 120             # seconds, per socket operation


def human(n: float) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def local_size(name: str) -> int:
    """Bytes already on disk for `name`, or 0 if the file is missing."""
    path = DEST_DIR / name
    return path.stat().st_size if path.exists() else 0


def download_one(name: str, expected: int) -> None:
    """Download a single file, resuming if a partial copy already exists."""
    dest = DEST_DIR / name
    have = local_size(name)

    if have == expected:
        print(f"  already complete ({human(expected)}) - skipping")
        return
    if have > expected:
        print("  local file is larger than expected - deleting and restarting")
        dest.unlink()
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    req = Request(f"{BASE_URL}/{name}", headers=headers)

    if have:
        print(f"  resuming at {human(have)} / {human(expected)}")
    else:
        print(f"  starting ({human(expected)})")

    start = time.time()
    downloaded = have
    with urlopen(req, timeout=TIMEOUT) as resp:
        # If we asked for a byte range but the server sent the whole file
        # (status 200, not 206), appending would corrupt the file. Bail out.
        if have and getattr(resp, "status", 200) != 206:
            raise RuntimeError(
                f"server did not honour the resume request "
                f"(HTTP {getattr(resp, 'status', '?')}); "
                f"delete {dest} and run again"
            )

        mode = "ab" if have else "wb"
        with open(dest, mode) as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = max(time.time() - start, 1e-6)
                speed = (downloaded - have) / elapsed
                pct = downloaded / expected * 100
                print(
                    f"\r  {pct:5.1f}%  {human(downloaded)} / {human(expected)}"
                    f"  ({human(speed)}/s)   ",
                    end="",
                    flush=True,
                )
    print()

    final = local_size(name)
    if final != expected:
        raise RuntimeError(
            f"size mismatch after download (got {final}, expected {expected}); "
            f"run again to resume"
        )


def check_disk_space(missing_bytes: int) -> None:
    """Warn (but do not abort) if free space looks too tight."""
    free = shutil.disk_usage(DEST_DIR).free
    if free < missing_bytes * 1.05:
        print(
            f"WARNING: ~{human(missing_bytes)} still to download but only "
            f"{human(free)} free on this drive.\n"
        )


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    total = sum(size for _, size in FILES)
    already = sum(min(local_size(name), size) for name, size in FILES)
    print(f"Destination : {DEST_DIR.resolve()}")
    print(f"Files       : {len(FILES)}")
    print(f"Total size  : {human(total)}")
    print(f"Already have : {human(already)}\n")
    check_disk_space(total - already)

    for i, (name, size) in enumerate(FILES, 1):
        print(f"[{i}/{len(FILES)}] {name}")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                download_one(name, size)
                break
            except (URLError, HTTPError, RuntimeError, OSError) as err:
                print(f"  error: {err}")
                if attempt < MAX_RETRIES:
                    print(f"  retry {attempt}/{MAX_RETRIES - 1} in 5s...")
                    time.sleep(5)
                else:
                    print("  giving up for now - re-run the script to continue")
                    sys.exit(1)

    print("\nAll 17 files present and size-verified.")
    print(f"Raw dump ready in {DEST_DIR.resolve()}")


if __name__ == "__main__":
    main()
