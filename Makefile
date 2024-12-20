.PHONY: clean build test publish lint type-check format check-all lock

clean:
	rm -rf dist/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +

test:
	poetry run pytest -v

build: clean
	poetry build

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

type-check:
	poetry run mypy .

lock:
	poetry lock --no-update

check-all: lock lint type-check test
	poetry check
	poetry run pip check

publish-test: check-all build
	poetry config repositories.testpypi https://test.pypi.org/legacy/
	poetry publish -r testpypi

publish: check-all build
	poetry publish

install:
	poetry install

update:
	poetry update