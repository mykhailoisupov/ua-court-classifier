"""Case-level statistics behind docs/design.md (sections 2, 3, 4, 5.1, 5.2, 6.6).

Keeps first-instance administrative courts, groups documents by case number, and takes the
earliest published document of each case registered in the analysed years.

    uv run python analysis/case_stats.py --years 2021 2022 2023

Streams documents.csv out of data/raw/edrsr_<year>.zip (from download_snapshots.py) and
writes data/interim/first_docs.parquet (one row per case).
"""

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

ADMINISTRATIVE = "4"  # justice_kinds.csv
FIRST_INSTANCE = "3"  # instances.csv
RULING = "5"  # judgment_forms.csv: Ухвала
COLUMNS = ["doc_id", "court_code", "judgment_code", "justice_kind", "category_code",
           "cause_num", "adjudication_date", "status", "date_publ", "doc_url"]


def read_member(archive: Path, member: str, **kwargs) -> pd.DataFrame:
    """Read one tab-separated file out of a snapshot archive without extracting it."""
    with zipfile.ZipFile(archive) as zf, zf.open(member) as f:
        return pd.read_csv(f, sep="\t", dtype=str, **kwargs)


def load_documents(raw_dir: Path, years: list[int]) -> pd.DataFrame:
    courts = read_member(raw_dir / f"edrsr_{years[0]}.zip", "courts.csv")
    first_instance = set(courts.loc[courts.instance_code == FIRST_INSTANCE, "court_code"])
    parts = []
    for year in years:
        with zipfile.ZipFile(raw_dir / f"edrsr_{year}.zip") as zf, zf.open("documents.csv") as f:
            for chunk in pd.read_csv(f, sep="\t", dtype=str, usecols=COLUMNS, chunksize=1_000_000):
                admin = chunk.justice_kind == ADMINISTRATIVE
                parts.append(chunk[admin & chunk.court_code.isin(first_instance)])
    docs = pd.concat(parts, ignore_index=True).drop_duplicates("doc_id")
    for col in ["adjudication_date", "date_publ"]:
        docs[col] = pd.to_datetime(docs[col].str[:10], errors="coerce")
    # case number = <court>/<sequence>/<two-digit year of registration>
    parsed = docs.cause_num.str.extract(r"^(\d+)/(\d+)/(\d{2})$")
    docs["cn_court"] = parsed[0]
    docs["cn_seq"] = pd.to_numeric(parsed[1])
    docs["reg_year"] = (2000 + pd.to_numeric(parsed[2])).astype("Int64")
    return docs


def quantiles(series: pd.Series) -> list[float]:
    return series.quantile([0.5, 0.9, 0.99]).round().tolist()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/interim/first_docs.parquet"))
    args = parser.parse_args()
    years = sorted(args.years)

    docs = load_documents(args.raw_dir, years)
    print(f"first-instance administrative documents: {len(docs):,}")
    print(f"case number matches <court>/<seq>/<yy>: {docs.reg_year.notna().mean():.1%}")

    docs = docs[docs.reg_year.isin(years)]
    impossible = docs.adjudication_date.dt.year < docs.reg_year
    print(f"[quality gate] documents dated before their case was registered: {impossible.sum():,}")
    docs = docs[~impossible]

    labelled = docs.dropna(subset=["category_code"])
    n_categories = labelled.groupby("cause_num").category_code.nunique()
    print(f"[§2, §5.2] cases with more than one category over their life: "
          f"{(n_categories > 1).mean():.1%} of {len(n_categories):,}")

    first = (docs.sort_values(["cause_num", "adjudication_date", "doc_id"])
             .groupby("cause_num").head(1).copy())
    print(f"[§5.2] cases: {len(first):,}; earliest published document is a ruling: "
          f"{(first.judgment_code == RULING).mean():.1%}")
    per_year = {int(y): n for y, n in first.reg_year.value_counts().sort_index().items()}
    print("[§2] cases per registration year:", per_year)

    rho = []
    for _, g in first.groupby(["cn_court", "reg_year"]):
        if len(g) > 200:
            # Spearman = Pearson on ranks (avoids a scipy dependency)
            rho.append(g.cn_seq.rank().corr(g.adjudication_date.rank()))
    print(f"[§5.2] Spearman(case number order, first-ruling date), median over "
          f"{len(rho)} court-years: {pd.Series(rho).median():.3f}")

    lag = (first.date_publ - first.adjudication_date).dt.days
    for year, g in lag.groupby(first.reg_year):
        print(f"[§4, §5.2] publication lag of the first ruling, {year}, p50/p90/p99 days: "
              f"{quantiles(g)}")

    first["month"] = first.adjudication_date.dt.to_period("M")
    in_range = first[first.month.dt.year.isin(years)]
    monthly = in_range.month.value_counts().sort_index()
    print("[§2, §6.6] new cases per month (month of first ruling):")
    print("  " + " ".join(f"{p.strftime('%y-%m')}:{n / 1000:.0f}k" for p, n in monthly.items()))

    dist = first.groupby(["reg_year", "category_code"]).size().unstack(0, fill_value=0)
    share = dist / dist.sum()
    base = years[0]
    for k in (10, 20, 30):
        top = share[base].nlargest(k).index
        cover = {y: f"{share.loc[top, y].sum():.1%}" for y in years}
        print(f"[§5.1] share of cases in the top-{k} categories of {base}: {cover}")
    for y in years[1:]:
        new = dist[(dist[base] == 0) & (dist[y] > 0)][y]
        print(f"[§4] categories absent in {base} but present in {y}: "
              f"{len(new)} ({new.sum()} cases)")

    names = (read_member(args.raw_dir / f"edrsr_{years[0]}.zip", "cause_categories.csv")
             .drop_duplicates("category_code").set_index("category_code").name)
    print(f"[§2, §6.9] category shares by year, top 10 of {years[-1]}:")
    for code in dist[years[-1]].nlargest(10).index:
        row = " -> ".join(f"{share.loc[code, y]:.1%}" for y in years)
        print(f"  {code}  {row}  {names.get(code, '?')[:70]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    first.drop(columns="month").to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
