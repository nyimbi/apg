# APG Installation And CLI Reference

This guide describes the current APG developer installation and command surface.
It intentionally avoids older platform-stack instructions that assumed a
monolithic server, Redis, PostgreSQL, or Docker Compose as prerequisites for
basic compiler work.

## Requirements

Minimum:

- Python 3.10 or newer, matching `setup.py`.
- `uv` for local environment creation and repeatable command execution.
- A POSIX-like shell for the documented commands.

Core package dependencies from `setup.py`:

- `antlr4-python3-runtime`
- `click`
- `rich`
- `python-dateutil`
- `Jinja2`
- `watchdog`
- `psutil`
- `pydantic`
- `uuid6`
- `flask`
- `httpx`
- `sqlalchemy[asyncio]`
- `asyncpg`
- `alembic`
- `nats-py`
- `temporalio`

Development extras include `pytest`, `pytest-asyncio`, `pytest-cov`, `numpy`,
`opencv-python`, `Pillow`, `black`, `flake8`, `mypy`, and `pre-commit`.

Optional extras exist for docs, language-server work, AI providers, and vision
features. Generated Flask apps are still self-contained at the browser-asset
level; static UI files are emitted into `static/`.

## Install With uv

From the repository root:

```bash
uv venv .venv
uv pip install -e ".[dev]"
```

If `.venv` already exists:

```bash
uv pip install -e ".[dev]"
```

Verify the CLI:

```bash
apg --help
apg doctor --json
```

If your shell does not pick up the virtual environment entry point, use:

```bash
uv run apg --help
python -m cli.main --help
```

## Compile And Run A Generated App

```bash
apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-minimal --verify
python /tmp/apg-minimal/app.py --self-test
python /tmp/apg-minimal/smoke_test.py
python /tmp/apg-minimal/app.py --host 127.0.0.1 --port 8080
```

Generated apps expose a Flask HTTP server. Open `/ui` for the generated shell
and `/openapi.json` for the generated API contract.

## Runtime Environment Variables

Generated apps understand these common variables:

| Variable | Use |
| --- | --- |
| `APG_HOST` or `HOST` | Default host for `app.py`. |
| `APG_PORT` or `PORT` | Default port for `app.py`. |
| `APG_DEBUG=1` | Enable Flask debug mode in generated apps. |
| `APG_SESSION_SECRET` or `APG_JWT_SECRET` | Session secret for generated auth flows. |
| `APG_API_KEY` | Require API-key authorization for mutations. |
| `APG_DATA_FILE` | Persist generated records to a JSON file. |
| `APG_DATABASE_URL`, `APG_PG_URL`, or `DATABASE_URL` | Optional best-effort PostgreSQL persistence path. |
| `APG_LANDING_STYLE` | Override generated landing style. |

PostgreSQL and Redis are not required for basic compiler, docs, or generated
app smoke-test workflows.

## Full CLI Usage Reference

Top-level command list from the current Click CLI:

```text
apg baseline         Run the compiler bed-down gate over numbered examples.
apg capabilities     Inspect executable APG capability contracts.
apg compile          Compile APG source files to Python artifacts.
apg create           Create new APG projects from templates.
apg deployment       Verify generated APG deployment evidence.
apg diagnostics      Inspect or audit APG diagnostic registry coverage.
apg docs             Audit APG documentation coverage and navigation.
apg doctor           Check APG installation and environment.
apg drift            Detect semantic drift between compiler and generated output.
apg evidence         Build package and verifier evidence for an APG profile.
apg explain          Explain symbols, diagnostics, handlers, and model items.
apg format           Format one APG source file deterministically.
apg graph            Emit one APG graph.
apg graph-suite      Emit every supported APG graph kind and rendering.
apg hygiene          Audit APG repository layout and root cleanliness.
apg ide              Inspect APG IDE integration contracts.
apg init             Initialize APG project in the current directory.
apg language-server  Start or check APG language-server behavior.
apg lint             Lint APG source without writing generated code.
apg migrate-plan     Compare APG sources or semantic models for migration changes.
apg model            Emit the normalized semantic model.
apg nl-plan          Plan a bounded APG DSL patch without mutating source.
apg package          Package generated Python artifacts for an APG profile.
apg package-verify   Verify an existing APG package profile directory.
apg parser-golden    Audit parser golden fixtures and grammar coverage.
apg refactor         Refactor APG source files.
apg release          Compile source and emit generated app release evidence.
apg run              Run APG application output.
apg schema           Generate SQL DDL from APG table declarations.
apg studio           Inspect and round-trip APG Studio designer state.
apg tooling          Run aggregate APG tooling contract checks.
apg validate         Validate APG source and generator readiness.
apg version          Show APG version information.
```

### Compiler Commands

```bash
apg compile <source.apg> --output <dir> --target python --verify
apg lint <source-or-directory> --json
apg validate <source.apg> --target python --json
apg model <source.apg> --json
apg format <source.apg> --check
apg graph <source.apg> --kind er --format mermaid
apg graph-suite <source.apg> --json
apg drift <source.apg> --json
apg release <source.apg> --json
apg schema <source.apg> --json
```

`python` is the only compiler target. Package commands cover web, desktop,
mobile, and container profiles after Python generation.

### Baseline And Audit Commands

```bash
apg baseline examples --json
apg baseline examples --refresh
apg baseline examples --refresh-outputs
apg baseline examples --update
apg parser-golden --json
apg diagnostics --audit-fixtures --json
apg tooling audit --json
apg docs audit --json
apg hygiene audit --json
apg hygiene audit --include-untracked --json
apg doctor --json
```

`apg baseline --refresh` is an alias for `--refresh-outputs`.

### Package, Deployment, And Evidence

```bash
apg package <source.apg> --target web --out <dir> --json
apg package <source.apg> --target desktop --out <dir> --json
apg package <source.apg> --target mobile --out <dir> --json
apg package <source.apg> --target container --out <dir> --json
apg package-verify <package-dir> --json
apg deployment verify <generated-or-package-dir> --json
apg evidence <source.apg> --target web --out <dir> --json
```

### Capability Commands

```bash
apg capabilities list
apg capabilities search <query>
apg capabilities manifest --stats
apg capabilities contracts --json
apg capabilities inspect <capability> --json
apg capabilities evaluate-rules <capability> --context-json '{"tenant_id":"demo"}' --json
apg capabilities validate-contracts --json
apg capabilities audit --json
apg capabilities implementation-audit --json
apg capabilities implementation-audit --strict --json
apg capabilities lifecycle-audit --json
apg capabilities materialize-packages --json
apg capabilities scaffold <domain> <code> --name "Display Name" --json
apg capabilities publish-plan <package-dir> --json
apg capabilities publish-apply <package-dir> --catalog <catalog.json> --json
apg capabilities catalog <catalog.json> --json
```

### IDE, Studio, And Natural-Language Planning

```bash
apg language-server <source.apg> --check --json
apg language-server <source.apg> --code-actions --json
apg language-server <source.apg> --rename OldName --to NewName --json
apg ide audit --json
apg studio snapshot <source.apg> --json
apg studio plan-edit <source.apg> --edit-json '<json>' --json
apg nl-plan <source.apg> --prompt "add a Customer table" --json
apg migrate-plan previous.apg current.apg --json
apg explain <source.apg> --symbol Customer --json
```

## Troubleshooting

Run:

```bash
apg doctor --json
apg docs audit --json
apg tooling audit --json
```

Use `python -m cli.main <command> --help` when debugging entry-point or virtual
environment issues. Use `uv run` to force commands through the managed
environment.
