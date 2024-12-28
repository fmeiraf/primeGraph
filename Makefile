.PHONY: clean build test publish lint type-check format check-all lock check

clean:
	rm -rf dist/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +

test:
	uv run pytest -v

build: clean
	uv build

lint:
	uv run ruff check .

format:
	uv run ruff format .

type-check:
	uv run mypy .

lock:
	uv lock

check: lock lint type-check test
	

check-all: lock lint type-check test	
	uv run pip check

publish-test: check-all build
	uv publish --publish-url https://test.pypi.org/legacy/

publish: check-all build
	uv publish

install:
	uv venv
	uv sync

update:
	uv update
