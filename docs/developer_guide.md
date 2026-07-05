# APG Developer Guide

This guide is for developers changing APG itself: CLI commands, compiler
stages, generated Flask runtime, generated UI, capability packages, tests, and
documentation.

The development rule is simple: change one public contract, prove it with the
smallest command that exercises that contract, then broaden verification only
when the blast radius requires it.

## Current Architecture

APG is an Agentic Platform Generator. It takes `.apg` source and emits
self-contained Python/Flask applications plus reviewable metadata and tests.

```text
.apg source
  -> spec/apg.g4 parser artifacts
  -> compiler/parser.py
  -> compiler/ast_builder.py
  -> compiler/semantic_model.py
  -> compiler/code_generator.py
  -> generated app.py, static/, templates, manifests, smoke tests
  -> CLI audits, package profiles, release evidence, baseline gate
```

Key directories:

| Directory | Purpose |
| --- | --- |
| `cli/` | Current Click-based CLI exposed by `apg=cli.main:cli`. |
| `cli.py` | Legacy argparse compatibility path. Do not document it as the primary CLI. |
| `compiler/` | Parser, AST, semantic model, diagnostics, generator, packaging, audits, and release tooling. |
| `compiler/templates/` | Jinja templates embedded into generated apps. |
| `compiler/assets/` | Vendored generated-app browser assets copied into `static/`. |
| `capabilities/` | Capability packages and executable contracts. |
| `examples/` | Numbered compiler baseline examples and checked generated output. |
| `tests/` | CLI, compiler, generated runtime, generated UI, and audit tests. |
| `docs/` | Developer and platform documentation. |

## Compiler Pipeline

1. Parse `.apg` source into parser output.
2. Build typed AST declarations for modules, tables, workflows, agents,
   databases, capabilities, apps, screens, themes, i18n, and deployment data.
3. Normalize meaning into `apg.semantic-model.v1`.
4. Run diagnostics, lint, validation, graph, drift, and generator-readiness
   checks from the shared semantic model.
5. Generate a Flask app with:
   - `app.py`
   - `__init__.py`
   - `README.md`
   - `requirements.txt`
   - `semantic_model.json`
   - `smoke_test.py`
   - `Dockerfile`, `.dockerignore`, `.env.example`
   - optional `ai_agents.py`, `apg_capabilities.py`, `apg_application.py`
   - `static/` assets copied from `compiler/assets/`
6. Verify generated behavior with `--self-test`, `smoke_test.py`, OpenAPI,
   route dispatch, component manifest, UI routes, and HTTP probes.

Use these commands to inspect each stage:

```bash
apg lint <source.apg> --json
apg validate <source.apg> --target python --json
apg model <source.apg> --json
apg graph-suite <source.apg> --json
apg compile <source.apg> --output /tmp/apg-out --verify
apg release <source.apg> --json
```

The only compiler target is `python`. Do not add documentation that presents
`django`, `flask-appbuilder`, or similar framework names as compiler targets.

## Generated App Anatomy

Generated apps are Flask applications with no frontend build step.

```text
generated/
  app.py
  __init__.py
  README.md
  requirements.txt
  semantic_model.json
  smoke_test.py
  Dockerfile
  .dockerignore
  .env.example
  ai_agents.py           optional
  apg_capabilities.py    optional
  apg_application.py     optional
  static/
    apg.css
    htmx.min.js
    sortable.min.js
    uplot.min.js
    uplot.min.css
    apg-charts.js
    apg-sse.js
    manifest.webmanifest
    sw.js
    icon.svg
```

Core generated routes include:

- `GET /`, `/home`, `/ui`
- `GET /health`, `/self-test`, `/manifest`, `/component.json`
- `GET /openapi.json`, `/semantic-model.json`, `/metrics`
- `GET /entities`, `/records`, `/relationships`
- `GET|POST /entities/{Entity}/records`
- `PUT|DELETE /entities/{Entity}/records/{id}`
- `GET /ui/entities/{Entity}`
- `GET /ui/entities/{Entity}?view=kanban`
- `GET /ui/entities/{Entity}?view=analytics`
- `GET /ui/entities/{Entity}/{id}`
- `GET /ui/workflows` and workflow wizard routes
- `GET /ui/agents/{Agent}` and `/ui/agent-teams/{Team}`
- `GET /ui/capabilities/{Capability}`
- `GET /ui/databases`
- `GET /ui/debug`
- `GET /ui/marketplace`
- optional `/login`, `/logout`, and `/locale`
- `GET /events` as server-sent events when requested with event-stream accept

Persistence is local by default. Generated apps can use `APG_DATA_FILE` for
JSON persistence and best-effort PostgreSQL persistence through
`APG_DATABASE_URL`, `APG_PG_URL`, or `DATABASE_URL` when the optional driver is
available. Mutation auth can be enforced with `APG_API_KEY` or JWT/session
metadata depending on the generated auth configuration.

## Generated UI

UI output is driven by `compiler/templates/*.html.j2` plus local assets in
`compiler/assets/`. Generated apps load `/static/apg.css`, `/static/htmx.min.js`,
`/static/sortable.min.js`, `/static/uplot.min.js`, `/static/apg-charts.js`, and
`/static/apg-sse.js`. Do not introduce CDN dependencies into generated output.

The generated shell includes sidebar navigation, topbar navigation, command
palette, notifications, theme toggle, i18n switcher, install/update controls,
offline banner, manifest, and service worker.

See [Generated UI](generated_ui.md) for the full workspace and component list.

## Theming

Themes come from APG theme metadata and capability theme contracts. The
generator emits `/theme.css` and uses CSS custom properties from the generated
theme token set. Core token families include:

- color tokens such as primary, accent, success, warning, danger, surface, and
  text
- spacing and density tokens
- border radius and shadow tokens
- typography and shell layout tokens

Generated apps also support a shell theme mode stored in browser local storage:
`system`, `light`, or `dark`.

## I18n

APG source can declare supported languages, default language, and fallback
language. The generated app creates a small shell catalog for English plus
available overrides. Current shell overrides include Swahili (`sw`) and Arabic
(`ar`) strings for core navigation labels. Arabic uses RTL document direction.

The generated `/locale` POST route sets the `apg_lang` cookie and redirects
back to the current UI path. Capability packages should keep user-facing
labels, decisions, and route names stable enough to translate.

## Adding New Capabilities

Use the current CLI scaffold when creating a package-backed capability:

```bash
apg capabilities scaffold <domain> <code> --name "Display Name" --json
```

Then complete these surfaces:

1. `capability_contract.py` with stable id, display name, configuration,
   deterministic rules, UI routes, theme tokens, and i18n/streaming metadata
   where relevant.
2. `cap_spec.md` explaining the business boundary and public contract.
3. Domain models, service behavior, API/view adapters, and focused tests.
4. Package metadata and release evidence when the capability is publishable.
5. Docs that state what is implemented and what remains intentionally out of
   scope.

Useful commands:

```bash
apg capabilities contracts --json
apg capabilities inspect <capability> --json
apg capabilities evaluate-rules <capability> --context-json '{"tenant_id":"demo"}' --json
apg capabilities validate-contracts --json
apg capabilities implementation-audit --json
apg capabilities lifecycle-audit --json
apg capabilities publish-plan <package-dir> --json
```

When extending an existing capability, preserve its public contract unless the
change is intentionally breaking and documented.

## Testing Workflow

Start with the narrowest proof:

```bash
apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-dev --verify
python /tmp/apg-dev/smoke_test.py
```

Then select the gate that matches the changed surface:

| Changed surface | Proof |
| --- | --- |
| CLI registration | `apg --help`; command-specific `--help`; focused CLI test |
| Parser or grammar | `apg parser-golden --json` |
| Diagnostics | `apg diagnostics --audit-fixtures --json` |
| Semantic model | `apg model <source.apg> --json`; semantic fixture tests |
| Formatter | `apg format --audit-fixtures --json` |
| Generator or UI | `apg compile <source.apg> --output /tmp/out --verify`; generated UI tests |
| Capability contract | `apg capabilities validate-contracts --json`; focused package tests |
| Capability depth | `apg capabilities implementation-audit --json` |
| Docs | `apg docs audit --json` |
| Repository layout | `apg hygiene audit --json` |

Before claiming repository-wide readiness, run:

```bash
uv run pytest tests/ -q
```

Current documented evidence: 1486 passed, 1 skipped, and 3 warnings.

## Baseline Gate

The numbered-example gate is the compiler bed-down contract:

```bash
apg baseline examples --json
apg baseline examples --refresh
```

`apg baseline --refresh` rewrites numbered example `output/` directories from
the current compiler before auditing them. Use it only when compiler output
changes are intentional. It checks example count, representative domain
coverage, lint/validate/model readiness, graph suite, release evidence,
generated Python source hygiene, checked-output synchronization, self-tests,
smoke tests, HTTP contract probes, domain route probes, and Python-only
targeting.

## Documentation Rules

- Document current executable behavior first.
- Move historical or aspirational claims into reports, roadmaps, or archive
  docs.
- Do not claim external services are required for the compiler unless the
  source actually requires them.
- Keep generated-app documentation Flask/Python-first.
- Keep static asset claims aligned with `compiler/assets/`.
