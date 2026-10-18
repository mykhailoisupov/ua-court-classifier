# ua-court-classifier

When a new administrative claim reaches a Ukrainian court, a clerk assigns it a case category by hand, and the automated assignment of a judge depends on that category.
This project designs a system that reads the claim text and suggests the three most likely categories with their confidence, which the clerk confirms or corrects.
Claims themselves are not public, so the model learns from the first ruling of each case in the Unified State Register of Court Decisions (ЄДРСР), which restates the claim and carries the official category code.
The system replays the register month by month: it ingests each new monthly batch, retrains, and monitors drift such as the change in caseload after martial law began in 2022.
It is built over five labs of the KSE MLOps course; the design is in [`docs/design.md`](docs/design.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and GNU `make`.

```bash
make setup   # install the pinned environment (uv.lock) and the pre-commit hook
make lint    # run ruff, the same check CI runs on every push
```

## Maturity scorecard

| Property | Evidence a reviewer checks | Added in |
|---|---|---|
| Repository reproducible from a clean clone | CI is green; `make setup` and `make lint` succeed on a fresh clone | Lab 1 |
| Design document exists and is current | [`docs/design.md`](docs/design.md) follows the template, every section filled or marked as an assumption | Lab 1 |
