# Repository Hygiene

APG keeps the repository root intentionally small. The root should show how to
enter the project, not become a parking area for reports, temporary tests,
generated output, copied PDFs, or agent state.

## Root Allowlist

Tracked files in the repository root are limited to:

```text
.gitignore
LICENSE
README.md
cli.py
pytest.ini
setup.py
uuid_extensions.py
```

`README.md` is the only tracked root Markdown file. Durable documentation lives
under `docs/`. Tests live under `tests/` or capability-local `tests/`
directories when a capability has its own package-local test suite.

## Canonical Locations

| Artifact | Location |
| --- | --- |
| User and developer docs | `docs/` |
| Historical reports | `docs/reports/` |
| Roadmaps and planning docs | `docs/roadmaps/` |
| Specifications | `docs/specifications/` |
| Copied reference documents | `docs/reference/` |
| Archived older docs | `docs/archive/` |
| Repository tests | `tests/` |
| Test fixtures | `tests/fixtures/` |
| Capability package docs | `capabilities/<domain>/<code>/docs/` |
| Capability package tests | `capabilities/<domain>/<code>/tests/` |
| APG examples | `examples/<number>_<name>/` |
| Generated example outputs | `examples/<number>_<name>/output/` |
| Temporary generated output | `/tmp`, `/private/tmp`, or ignored local paths |

## Rules

- Do not add root-level `test_*.py`, `*_test.py`, reports, guides, or copied
  reference documents.
- Do not track runtime output directories such as caches, uploads, generated
  demos, build artifacts, or egg metadata.
- Do not track `.DS_Store`, `__pycache__`, `.pyc`, `.pyo`, or local virtual
  environments.
- Keep APG public docs Python-first and aligned with the executable compiler
  target.
- Keep internal streaming language Bytewax-oriented.
- Move historical or aspirational documents into `docs/archive/`,
  `docs/reports/`, or `docs/roadmaps/` instead of leaving them at source root.
- Keep copied reference PDFs, Word files, spreadsheets, and decks under
  `docs/reference/` when they are intentionally part of the repository. Leave
  local-only references untracked.

## Enforcement

Repository layout is enforced by:

```bash
./.venv/bin/apg hygiene audit --json
./.venv/bin/apg hygiene audit --include-untracked --json
./.venv/bin/python -m pytest -q tests/test_repository_hygiene.py
```

The CLI audit emits `apg.repository-hygiene-audit.v1` and is included in
`apg tooling audit --json`. The default hygiene checks inspect tracked files so
CI is stable and independent of a developer's local agent state. Add
`--include-untracked` when preparing a contribution or cleaning a workspace; it
also reports local untracked runtime output directories, root-level agent state,
root-level Markdown/tests, and copied reference documents that should move under
`docs/reference/` before being committed.
