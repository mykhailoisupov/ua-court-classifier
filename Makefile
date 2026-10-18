.PHONY: setup lint

# Install the pinned environment from uv.lock and the git pre-commit hook.
setup:
	uv sync --locked
	uv run pre-commit install

# Run the linter; CI and the pre-commit hook run exactly this.
lint:
	uv run ruff check .
