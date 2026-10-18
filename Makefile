.PHONY: setup lint analysis

# Install the pinned environment from uv.lock and the git pre-commit hook.
setup:
	uv sync --locked
	uv run pre-commit install

# Run the linter; CI and the pre-commit hook run exactly this.
lint:
	uv run ruff check .

# Reproduce the data figures in docs/design.md (~1 GB download, see analysis/README.md).
analysis:
	uv run python analysis/download_snapshots.py --years 2021 2022 2023
	uv run python analysis/case_stats.py --years 2021 2022 2023
	uv run python analysis/ruling_sample.py --per-year 100
