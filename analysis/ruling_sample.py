"""Text statistics of first rulings behind docs/design.md (sections 5.2, 5.3, 6.3, 6.9).

Samples first rulings per registration year, downloads their RTF texts (cached), and measures
how often the claim section, the filing date, the category code and the judge appear in them.

    uv run python analysis/ruling_sample.py --per-year 100

Reads data/interim/first_docs.parquet (from case_stats.py), caches RTF files in
data/raw/rulings/, writes data/interim/ruling_sample.parquet.
"""

import argparse
import re
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

MONTHS = {"січня": 1, "лютого": 2, "березня": 3, "квітня": 4, "травня": 5, "червня": 6,
          "липня": 7, "серпня": 8, "вересня": 9, "жовтня": 10, "листопада": 11, "грудня": 12}
DATE = re.compile(r"(\d{1,2})[.\s](\d{2}|" + "|".join(MONTHS) + r")[.\s](\d{4})")
FILING_VERB = re.compile(r"надійш|звернув|звернул|подан|подав|подал|зареєстр", re.I)
CLAIM_HEADING = re.compile(r"[ву]\s*с\s*т\s*а\s*н\s*о\s*в\s*и\s*(?:в|л\s*а)", re.I)
OPERATIVE_HEADING = re.compile(r"у\s*х\s*в\s*а\s*л\s*и\s*в|п\s*о\s*с\s*т\s*а\s*н\s*о\s*в\s*и\s*в",
                               re.I)
CATEGORY_LEAK = re.compile(r"категорі[яї]\s*(?:справи\s*)?(?:№\s*)?\d{3,}", re.I)
JUDGE = re.compile(r"[Сс]удд[яі]\s+[А-ЯІЇЄҐ][а-яіїєґ'’]+")
HEADERS = {"User-Agent": "ua-court-classifier/0.1 (KSE MLOps course project)"}


def rtf_to_text(raw: str) -> str:
    """Minimal RTF -> text for the register's cp1251 files."""
    raw = re.sub(r"\{\\fonttbl.*?\}\}", "", raw, flags=re.S)
    raw = re.sub(r"\{\\\*[^{}]*\}", "", raw)
    raw = re.sub(r"\\par(?![a-z])", "\n", raw).replace("\\tab", " ").replace("\\~", " ")
    raw = re.sub(r"\\'([0-9a-fA-F]{2})",
                 lambda m: bytes([int(m.group(1), 16)]).decode("cp1251"), raw)
    raw = re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m.group(1)) % 65536), raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", raw)
    return re.sub(r"\s+", " ", raw.replace("{", "").replace("}", "")).strip()


def parse_date(match: re.Match) -> date | None:
    day, month, year = match.groups()
    month_num = int(month) if month.isdigit() else MONTHS[month]
    try:
        return date(int(year), month_num, int(day))
    except ValueError:
        return None


def filing_date(text: str, ruled: date) -> date | None:
    """Latest date within 120 characters of a filing verb, before the operative part,
    no later than the ruling and at most a year before it."""
    body = OPERATIVE_HEADING.split(text)[0]
    found = []
    for m in DATE.finditer(body):
        d = parse_date(m)
        window = body[max(0, m.start() - 120): m.end() + 120]
        if d and FILING_VERB.search(window) and 0 <= (ruled - d).days <= 365:
            found.append(d)
    return max(found) if found else None


def fetch(url: str, dest: Path, delay: float) -> bool:
    if dest.exists():
        return True
    request = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            dest.write_bytes(resp.read())
    except OSError as err:
        print(f"  skip {dest.stem}: {err}")
        return False
    time.sleep(delay)  # be polite to the court's file server
    return True


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first-docs", type=Path,
                        default=Path("data/interim/first_docs.parquet"))
    parser.add_argument("--rulings-dir", type=Path, default=Path("data/raw/rulings"))
    parser.add_argument("--out", type=Path, default=Path("data/interim/ruling_sample.parquet"))
    parser.add_argument("--per-year", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.3, help="seconds between downloads")
    args = parser.parse_args()

    first = pd.read_parquet(args.first_docs)
    candidates = first[(first.judgment_code == "5") & first.doc_url.notna()]
    sample = candidates.groupby("reg_year").sample(n=args.per_year, random_state=args.seed)
    args.rulings_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in sample.itertuples():
        path = args.rulings_dir / f"{r.doc_id}.rtf"
        if not fetch(r.doc_url, path, args.delay):
            continue
        text = rtf_to_text(path.read_text(encoding="latin-1"))
        filed = filing_date(text, r.adjudication_date.date())
        rows.append({
            "doc_id": r.doc_id,
            "reg_year": r.reg_year,
            "chars": len(text),
            "claim_heading": bool(CLAIM_HEADING.search(text)),
            "filing_date_found": filed is not None,
            "filing_gap_days": (r.adjudication_date.date() - filed).days if filed else None,
            "category_code_in_text": bool(CATEGORY_LEAK.search(text)),
            "judge_in_text": bool(JUDGE.search(text)),
        })
    res = pd.DataFrame(rows)

    print(f"rulings parsed: {len(res)}")
    print(f"[§5.2] claim section heading ('встановив') found: {res.claim_heading.mean():.1%}")
    by_year = res.groupby("reg_year").filing_date_found.mean().round(3).to_dict()
    print(f"[§5.2] filing date extractable: {res.filing_date_found.mean():.1%} (by year {by_year})")
    gaps = res.filing_gap_days.dropna().quantile([0.25, 0.5, 0.75, 0.9]).tolist()
    print(f"[§5.2] filing -> first ruling, days p25/p50/p75/p90: {gaps}")
    print(f"[§5.3, §6.9] category code in the text: {res.category_code_in_text.mean():.1%}; "
          f"judge name in the text: {res.judge_in_text.mean():.1%}")
    print(f"[§6.3] ruling length, characters p50/p90: {res.chars.quantile([0.5, 0.9]).tolist()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
