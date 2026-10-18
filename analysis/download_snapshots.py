"""Download yearly snapshots of the court register (ЄДРСР) from data.gov.ua.

Each archive holds documents.csv (one row per court document) and its dictionaries
(courts, categories, judgment forms, ...). Texts are not included: each row links to an RTF.

    uv run python analysis/download_snapshots.py --years 2021 2022 2023

Writes data/raw/edrsr_<year>.zip exactly as published (~0.3 GB each; documents.csv inside is
~1.8 GB, so readers stream it from the archive instead of extracting it) and
data/raw/edrsr_<year>.json (source URL, publication timestamp, sha256 of the archive).
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

API = "https://data.gov.ua/api/3/action/package_show?id={}"
# data.gov.ua answers 403 to urllib's default User-Agent
HEADERS = {"User-Agent": "ua-court-classifier/0.1 (KSE MLOps course project)"}
# data.gov.ua dataset ids of "Єдиний державний реєстр судових рішень за <year> рік"
DATASETS = {
    2019: "1c13f2f1-8bc4-41d4-81dc-ea4527fa73de",
    2020: "83f1292d-43c9-43b8-8f89-aa2f3d14fd3a",
    2021: "bcffb3ae-8e4f-4443-8058-d525257ab69e",
    2022: "42c195b3-b915-4000-934a-7741825d1812",
    2023: "e0f2c66a-f773-442e-afcd-569be6a36c82",
    2024: "1a6b5c08-0879-4387-8878-ac0027494274",
    2025: "2a95ae17-ed46-42ae-ad66-dd9b7a727bb7",
}


def open_url(url: str, timeout: int):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout)


def archive_resource(dataset_id: str) -> dict:
    with open_url(API.format(dataset_id), timeout=60) as resp:
        package = json.load(resp)["result"]
    return next(r for r in package["resources"] if r["url"].endswith(".zip"))


def download(url: str, dest: Path) -> str:
    """Stream url to dest and return the sha256 of what was written."""
    digest = hashlib.sha256()
    with open_url(url, timeout=120) as resp, dest.open("wb") as out:
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def fetch_year(year: int, raw_dir: Path, force: bool) -> None:
    archive = raw_dir / f"edrsr_{year}.zip"
    if archive.exists() and not force:
        print(f"{year}: {archive} already present, skipping (use --force to refresh)")
        return
    resource = archive_resource(DATASETS[year])
    print(f"{year}: downloading {resource['url']}")
    partial = archive.with_suffix(".zip.part")
    sha256 = download(resource["url"], partial)
    partial.replace(archive)  # only a complete download gets the final name
    manifest = {
        "year": year,
        "url": resource["url"],
        "last_modified": resource.get("last_modified") or resource.get("metadata_modified"),
        "sha256": sha256,
    }
    archive.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{year}: saved {archive} (sha256 {sha256[:12]}...)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023],
                        choices=sorted(DATASETS))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="re-download existing years")
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        fetch_year(year, args.raw_dir, args.force)


if __name__ == "__main__":
    main()
