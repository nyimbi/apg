# Installation

## Requirements

- Python 3.10 or newer
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- A POSIX shell (Linux, macOS, WSL)

## Install from source

```bash
git clone https://github.com/nyimbi/apg
cd apg
uv venv .venv
uv pip install -e ".[dev]"
```

Verify the installation:

```bash
apg --help
apg doctor --json
```

If `apg` is not on your PATH, use the virtual environment directly:

```bash
./.venv/bin/apg --help
# or
uv run apg --help
```

## Install from PyPI *(coming soon)*

```bash
pip install apg-language
apg --help
```

## Core dependencies

| Package | Purpose |
|---------|---------|
| `antlr4-python3-runtime` | ANTLR grammar runtime |
| `click` | CLI framework |
| `rich` | Terminal output |
| `flask` | Generated app server |
| `sqlalchemy[asyncio]` | ORM for generated apps |
| `alembic` | Database migrations |
| `pydantic` | Data validation |
| `uuid6` | UUID v7 generation |
| `jinja2` | Code generation templates |
| `watchdog` | File-watch mode |
| `httpx` | HTTP client |
| `asyncpg` | Async PostgreSQL driver |

## Optional extras

```bash
# Language Server Protocol support
uv pip install -e ".[lsp]"

# AI provider integrations
uv pip install -e ".[ai]"

# VS Code extension (build from source)
cd vscode-extension && npm install && npm run compile
```

## Verify with doctor

```bash
apg doctor
```

`apg doctor` checks Python version, ANTLR runtime, grammar file, import chain, and generated-app smoke test. All items should show ✓ before you start developing.

## Uninstall

```bash
uv pip uninstall apg-language
# or remove the cloned directory and .venv
```
