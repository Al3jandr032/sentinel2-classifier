.PHONY: format lint lint-fix check

format:
	uv run ruff format .

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

check: lint-fix format
	@echo "Code formatting and linting completed!"
