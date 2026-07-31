# CLI Reference

```
apg [OPTIONS] COMMAND [ARGS]...
```

APG is invoked via the `apg` CLI. All subcommands accept `--help` for full flag documentation.

---

## apg init

Scaffold a new APG project directory.

```bash
apg init [APP_NAME] [--template TEMPLATE]
```

| Flag | Description |
|------|-------------|
| `APP_NAME` | Project name (prompted if omitted) |
| `--template` | Scaffold template: `minimal` (default), `crud`, `saas`, `agent` |

Creates:

```
<APP_NAME>/
  app.apg          ← sample schema
  .env.example     ← all APG_ env vars with comments
  Makefile         ← compile, run, test targets
  .gitignore
```

Example:

```bash
apg init my_crm --template crud
cd my_crm
apg compile app.apg -o out/
python out/app.py
```

---

## apg compile

Compile an APG source file to a generated Python app.

```bash
apg compile SOURCE_FILE [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `SOURCE_FILE` | — | Path to `.apg` file (required) |
| `--output`, `-o` | `./out` | Output directory for generated files |
| `--verify` | — | Run smoke test on generated app after compilation |
| `--format` | `python` | Output format: `python`, `docker` |
| `--dialect` | `sqlite` | SQL dialect for schema: `sqlite`, `postgresql`, `mysql` |
| `--watch` | — | Recompile on file change (alias for `apg watch`) |

Output files:

| File | Description |
|------|-------------|
| `app.py` | Self-contained Flask application |
| `smoke_test.py` | Generated integration test |
| `schema.sql` | DDL for the target dialect |
| `openapi.json` | OpenAPI 3.1 schema |

Example:

```bash
apg compile crm.apg -o generated/ --verify --dialect postgresql
```

---

## apg doctor

Validate the local APG environment.

```bash
apg doctor [--json]
```

| Flag | Description |
|------|-------------|
| `--json` | Output results as JSON |

Checks:

- Python version ≥ 3.10
- `antlr4-python3-runtime` installed
- APG grammar file (`spec/apg.g4`) present
- Compiler import chain healthy
- Generated-app smoke test passes

All items should show ✓ before starting development.

---

## apg watch

Watch an APG file for changes and recompile automatically.

```bash
apg watch SOURCE_FILE [--output OUTPUT_DIR]
```

| Flag | Default | Description |
|------|---------|-------------|
| `SOURCE_FILE` | — | File to watch |
| `--output`, `-o` | `./out` | Output directory |

Recompiles within 200 ms of a save. Prints a diff of changed lines.

---

## apg serve

Compile and immediately serve the generated app.

```bash
apg serve SOURCE_FILE [--host HOST] [--port PORT] [--output OUTPUT_DIR]
```

| Flag | Default | Description |
|------|---------|-------------|
| `SOURCE_FILE` | — | APG source file |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8080` | Listen port |
| `--output`, `-o` | `./out` | Compilation output directory |

Equivalent to `apg compile ... && python out/app.py`.

---

## apg export

Export deployment assets for the generated app.

```bash
apg export SOURCE_FILE --format FORMAT [--output OUTPUT_DIR]
```

| Flag | Default | Description |
|------|---------|-------------|
| `SOURCE_FILE` | — | APG source file |
| `--format` | `docker` | Export format: `docker`, `zip` |
| `--output`, `-o` | `./deploy` | Output directory |

Docker export writes:

```
deploy/
  app.py
  Dockerfile
  docker-compose.yml
  .env.example
```

Build and run:

```bash
apg export myapp.apg --format docker -o deploy/
cd deploy && docker compose up
```

---

## apg lint

Run the APG linter on a source file.

```bash
apg lint SOURCE_FILE [--fix]
```

| Flag | Description |
|------|-------------|
| `--fix` | Auto-fix safe issues |

Reports: unused entities, missing `app` declaration, field name collisions, unknown types.

---

## apg format

Format an APG source file in place.

```bash
apg format SOURCE_FILE [--check]
```

| Flag | Description |
|------|-------------|
| `--check` | Exit non-zero if formatting would change the file (CI use) |

---

## apg diagnostics

Run semantic diagnostics on a source file.

```bash
apg diagnostics SOURCE_FILE [--json]
```

Outputs errors and warnings with line/column positions. Suitable for editor integrations.

---

## apg schema

Generate SQL DDL from an APG source file.

```bash
apg schema SOURCE_FILE [--dialect DIALECT] [--output FILE]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dialect` | `postgresql` | `postgresql`, `sqlite`, `mysql` |
| `--output` | stdout | Write DDL to file |

---

## apg language-server

Start the APG Language Server Protocol (LSP) server.

```bash
apg language-server [SOURCE_FILE] [--host HOST] [--port PORT]
```

| Flag | Default | Description |
|------|---------|-------------|
| `SOURCE_FILE` | — | Optional: run a one-shot check on this file |
| `--host` | `127.0.0.1` | LSP server bind address |
| `--port`, `-p` | `2087` | LSP server port |
| `--check` | — | One-shot diagnostics mode (no server) |
| `--json` | — | Emit JSON output with `--check` |

---

## apg version

Print APG version information.

```bash
apg version
```

---

## apg validate

Validate a compiled app against its source.

```bash
apg validate SOURCE_FILE --app APP_PY
```

---

## apg explain

Explain what a compiled section does in plain English.

```bash
apg explain SOURCE_FILE [ENTITY_NAME]
```

---

## apg refactor

Suggest and apply refactorings to an APG source file.

```bash
apg refactor SOURCE_FILE [--apply]
```

---

## Global flags

| Flag | Description |
|------|-------------|
| `--help` | Show help and exit |

All commands also respect `APG_*` environment variables for defaults.
