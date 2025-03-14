.PHONY: clean build test publish lint type-check format check-all lock check changelog-check lint-fix

clean:
	rm -rf dist/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +

test:
	uv run pytest -v -k "not test_postgresql_checkpoint and not test_postgresql_tool_checkpoint"

test-all:
	uv run pytest -v

build: clean
	uv build

lint:
	uv run ruff check .

format:
	uv run ruff format .

lint-fix:
	uv run ruff check --fix .

# TODO: Remove this in the future when I have time to fix the errors
type-check:
	uv run mypy . --exclude 'primeGraph/graph/llm_tools\.py'

lock:
	uv lock

check: lock lint type-check test
	

check-all: lock lint type-check test	
	uv run pip check

changelog-check:
	@latest_version=$$(grep -m 1 "^#* \[.*\]" CHANGELOG.md | grep -o "\[.*\]" | tr -d '[]'); \
	project_version=$$(grep '^version = ' pyproject.toml | cut -d'"' -f2); \
	if [ "$$latest_version" != "$$project_version" ]; then \
		echo "Error: Version mismatch! CHANGELOG.md ($$latest_version) != pyproject.toml ($$project_version)"; \
		exit 1; \
	fi

publish-test: changelog-check check-all build
	source .env && uv publish --publish-url https://test.pypi.org/legacy/ --token $$PYPI_TOKEN_TEST

publish: changelog-check check-all build
	source .env && uv publish --token $$PYPI_TOKEN_PROD

install:
	uv venv
	uv sync

update:
	uv update

test-actions:
	act push

test-prod-publishing:
	uv run --with primeGraph --no-project -- python -c "import primeGraph"

test-test-publishing:
	uv run --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --with primeGraph --no-project -- python -c "import primeGraph"