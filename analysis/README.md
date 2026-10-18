# Data analysis behind the design document

Every data figure in [`docs/design.md`](../docs/design.md) comes from these three scripts. They read the
public court register (ЄДРСР) open data, so anyone can rerun them:

```bash
make analysis   # ≈ 1 GB download + ~300 small RTF files, ~10 minutes
```

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `download_snapshots.py` | data.gov.ua API | `data/raw/edrsr_<year>.zip` as published + `edrsr_<year>.json` (URL, timestamp, sha256) |
| 2 | `case_stats.py` | the archives (streamed, never extracted: `documents.csv` is ~1.8 GB per year) | `data/interim/first_docs.parquet`, one row per case |
| 3 | `ruling_sample.py` | `first_docs.parquet` + RTF texts of a random sample of first rulings | `data/raw/rulings/*.rtf` (cache), `data/interim/ruling_sample.parquet` |

`data/` is git-ignored; from Lab 2 it is versioned with DVC.

## Figures and where they are used

Results of the run on 2026-10-18 (snapshots published 2026-09-26 to 2026-10-08). Each printed line is
tagged with the design-document section that uses it.

| Figure | Value | Section |
|---|---|---|
| New first-instance administrative cases per year | 461k (2021), 322k (2022), 483k (2023) | §2 |
| Earliest published document of a case is a ruling (ухвала) | 99.9% | §5.2 |
| Order of case numbers vs order of first-ruling dates (Spearman, median over 126 court-years) | 0.998 | §5.2 |
| Cases with more than one category over their life | 3.6% | §2, §3, §5.2, §6.9 |
| Publication lag of the first ruling, p50 / p90 (days) | 3 / 315 (2021), 3 / 47 (2022), 3 / 5 (2023) | §4, §5.2 |
| Share of cases in the 20 most frequent categories of 2021 | 85.7% / 89.2% / 89.1% (2021 / 2022 / 2023) | §5.1 |
| Categories absent in 2021 but present in 2022 | 5 categories, 13 cases | §4 |
| Chernobyl-related claims (40151), share of new cases | 8.0% → 22.2% → 10.6% | §2, §6.9 |
| Claims of persons discharged from public service (40149), share | 20.4% → 22.6% → 37.1% | §2 |
| Monthly volume, Feb → Mar 2022 | 38k → 10k | §6.6 |
| Documents dated before their case was registered | 159 | quality gate |
| Claim-section heading ("встановив") found, sample of 299 rulings | 100% | §5.2 |
| Filing date extractable from the ruling text | 32% | §5.2, §6.9 |
| Filing → first ruling, median (where extractable) | 5 days | §5.2 |
| Category code present in the ruling text | 13.4% | §5.3, §6.9 |
| Judge's name present in the ruling text | 96.7% | §5.3, §6.9 |
| Ruling length, p50 / p90 (characters) | 5,100 / 9,400 | §6.3 |

The register is refreshed monthly and documents can be removed later (`status=0`), so a rerun on newer
snapshots can shift these figures slightly; the sha256 in `edrsr_<year>.json` identifies the snapshot used.
