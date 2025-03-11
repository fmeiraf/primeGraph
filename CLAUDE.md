# Prime Graph Developer Guide

## Commands
- **Install/Build**: `make install`, `make build`, `uv build`
- **Lint/Format**: `make lint` (ruff check), `make format` (ruff format), `make type-check` (mypy)
- **Testing**: `make test` (skip PostgreSQL tests), `make test-all` (all tests)
- **Run Single Test**: `uv run pytest -v tests/file_path.py::test_name` or `uv run pytest -v -k "test_pattern"`
- **Full Check**: `make check` (lint+type+test), `make check-all` (all checks)

## Code Style
- **Imports**: stdlib → third-party → project, separated by newlines
- **Typing**: Strong type hints, custom types in `types.py`, Pydantic models
- **Formatting**: 4-space indent, ~100 char line length, ruff
- **Naming**: PascalCase (classes), snake_case (functions), UPPERCASE (constants), _prefixed (private)
- **Error Handling**: Validation in constructors, specific error messages, custom exceptions
- **Documentation**: Docstrings for classes/methods, parameter documentation
- **Architecture**: Modular design, base classes, factory pattern, graph-based workflows