# APG Developer Guide

This guide is for contributors changing APG itself: grammar, parser artifacts,
compiler, semantic model, generator, CLI, language-server surfaces, capability
contracts, examples, tests, and documentation.

The goal is immediate effectiveness. A new developer should be able to clone the
repository, run one reliable baseline, choose the right implementation surface,
make a vertical slice executable, prove it, document it, and commit it without
waiting for tribal knowledge.

## The APG Contract

APG is not just `spec/apg.g4`. The language is useful only when source text
travels through the full toolchain:

```text
.apg source
  -> spec/apg.g4
  -> generated parser in spec/
  -> compiler/ast_builder.py
  -> compiler/semantic_analyzer.py
  -> compiler/semantic_model.py
  -> compiler/code_generator.py
  -> generated Python application
  -> CLI, examples, packaging, release evidence, and docs
```

A language feature is not done when it parses. It is serviceable when the right
downstream surfaces can consume it:

- parser accepts valid syntax and rejects invalid syntax;
- AST builder normalizes it into stable Python data;
- semantic analyzer validates references and emits diagnostics;
- semantic model exposes it as stable JSON;
- generator executes it or documents that it is metadata-only for now;
- CLI/tooling can inspect, lint, validate, graph, package, or explain it;
- tests and examples prove the behavior;
- docs describe current executable reality, not aspiration.

## Repository Map

| Path | Responsibility |
| --- | --- |
| `spec/apg.g4` | ANTLR grammar source. Keep syntax terse but readable. |
| `spec/apgLexer.py`, `spec/apgParser.py`, `spec/apgVisitor.py` | Generated parser artifacts checked in with grammar changes when needed. |
| `compiler/parser.py` | Parser wrapper and source loading boundary. |
| `compiler/ast_builder.py` | Concrete parse tree to APG AST/normalized metadata. |
| `compiler/semantic_analyzer.py` | Validation, symbol resolution, diagnostics. |
| `compiler/semantic_model.py` | Stable `apg.semantic-model.v1` projection used by tools. |
| `compiler/code_generator.py` | Dependency-light generated Python app writer. |
| `compiler/*` | Formatter, graphs, release evidence, packaging, drift, migrations, NL plans, Studio, audits. |
| `cli/` | Click commands for `apg`. Every durable command needs JSON output. |
| `language_server/` | Dependency-light editor-facing semantic service and LSP entry point. |
| `capabilities/` | Package-backed APG capabilities and executable contracts. |
| `examples/` | Numbered parseable examples plus checked-in generated outputs. |
| `tests/` | Focused regression tests and fixture catalogs. |
| `docs/` | User, contributor, developer, grammar, tooling, standards, and progress docs. |
| `vscode-extension/` | APG VS Code integration metadata. |

## Environment Setup

This repository is currently a `setup.py` project. Use `uv` to create and manage
the virtual environment, then install the editable package:

```bash
uv venv .venv
uv pip install -e ".[dev,language-server]"
./.venv/bin/apg --help
./.venv/bin/apg doctor --json
```

When the virtual environment already exists, refresh it with:

```bash
uv pip install -e ".[dev,language-server]"
```

Use the explicit virtual-environment path in verification logs:

```bash
./.venv/bin/apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-example --verify
```

## Baseline Commands

Run one quick command to prove the CLI is installed:

```bash
./.venv/bin/apg --help
```

Run the aggregate tooling gate when touching CLI, compiler, language server, or
tooling contracts:

```bash
./.venv/bin/apg tooling audit --json
./.venv/bin/apg doctor --json
```

Run a representative compile when touching language, semantic model, generator,
capability composition, or examples:

```bash
./.venv/bin/apg compile examples/20_enterprise_erp_platform/main.apg --output /tmp/apg-erp --verify
./.venv/bin/python /tmp/apg-erp/smoke_test.py
```

Run focused tests rather than the full suite by default when compute or battery
is constrained:

```bash
./.venv/bin/python -m pytest -q tests/test_tooling_audit.py
./.venv/bin/python -m pytest -q tests/test_compiler_baseline.py
./.venv/bin/python -m pytest -q tests/test_cli_capability_scaffold.py
./.venv/bin/python -m pytest -q tests/test_capability_contract_registry.py
```

## Daily Development Loop

1. Check the worktree with `git status --short`.
2. Read the source, tests, docs, and fixtures for the area you are changing.
3. Identify the executable contract that proves success.
4. Make the smallest vertical slice that moves APG closer to executable reality.
5. Add or update focused tests and fixtures.
6. Regenerate checked-in example output only when compiler output changes.
7. Update docs and `docs/progress_log.md`.
8. Run focused verification and inspect the output.
9. Stage only the files in your slice.
10. Commit with a Lore-style commit message and push.

Never stage unrelated dirty files. This repository often contains local agent
state, uploads, generated experiments, and nested capability worktrees.

## Where To Change Things

| Desired change | Primary files | Required proof |
| --- | --- | --- |
| New syntax | `spec/apg.g4`, parser artifacts, `compiler/ast_builder.py` | parser golden or language contract tests |
| New semantic meaning | `semantic_analyzer.py`, `semantic_model.py` | semantic-model JSON and diagnostics tests |
| New generated behavior | `code_generator.py` | compile with `--verify`, generated smoke test |
| New CLI command | `cli/<command>.py`, `cli/main.py` | JSON output test and `apg tooling audit --json` |
| New lint rule | `compiler/linting.py`, fixtures | lint fixture/test proving diagnostic code |
| New graph surface | `compiler/graphs.py` | `apg graph-suite ... --json` and graph tests |
| New capability package | `capabilities/<domain>/<code>/` | contract validation and focused capability tests |
| New capacity | APG source, capability packages, examples, docs | compile, smoke, contract validation, progress log |
| New docs | `docs/` | `git diff --check -- docs` and current command examples |

## Grammar Work

Read [APG Grammar Guide](./apg_grammar_guide.md) before editing
`spec/apg.g4`.

Good APG syntax follows the existing shape:

```apg
entity_type EntityName {
    contract_or_config: value;
}
```

Prefer extending existing generic contracts before adding special-purpose
syntax:

- `contract_object`
- `reference_list`
- `rule_list`
- `ui_contract`
- `theme_contract`
- `screen_set`
- `runtime_contract`

Grammar checklist:

1. Update `spec/apg.g4`.
2. Regenerate parser artifacts when the checked-in generated parser must change.
3. Update AST building.
4. Update semantic validation and semantic model JSON.
5. Update code generation if the construct should execute.
6. Add parser, semantic, generator, or fixture coverage.
7. Add or update a numbered example when the syntax is author-facing.
8. Update language docs and cheat sheet.

## Semantic Model Work

The semantic model is the shared truth for lint, validate, graph, language
server, Studio, drift, release evidence, and generated apps. Downstream tools
should not re-parse source text to discover meaning already present in
`apg.semantic-model.v1`.

When adding semantic data:

- use stable JSON field names;
- include source names and file paths when useful;
- include graph nodes or edges for composed relationships;
- include diagnostics for unresolved references;
- preserve existing JSON keys unless a migration is intentional;
- update docs when a public JSON format changes.

## Code Generation Work

Generated apps should remain dependency-light Python artifacts. The baseline
output includes:

- `app.py`
- `__init__.py`
- `semantic_model.json`
- `README.md`
- `requirements.txt`
- `Dockerfile`
- `smoke_test.py`
- optional sidecars such as `ai_agents.py`, `apg_capabilities.py`, and
  `apg_application.py`

Generated behavior should be inspectable through Python helpers, route
dispatch, OpenAPI metadata, component manifests, validation/self-test checks,
and generated smoke tests.

Representative generator verification:

```bash
./.venv/bin/python -m py_compile compiler/code_generator.py
./.venv/bin/apg compile examples/20_enterprise_erp_platform/main.apg --output /tmp/apg-erp --verify
./.venv/bin/python /tmp/apg-erp/smoke_test.py
```

If checked-in generated examples change, run the checked-in output comparison
test:

```bash
./.venv/bin/python -m pytest -q tests/test_compiler_baseline.py::test_checked_in_example_outputs_match_current_compiler
```

## Capability Work

Capability packages must be composable and executable. Start new package-backed
capabilities with the scaffold command:

```bash
./.venv/bin/apg capabilities scaffold <domain> <code> --name "Display Name" --json
```

A scaffolded package includes:

```text
capabilities/<domain>/<code>/
  __init__.py
  cap_spec.md
  capability_contract.py
  models.py
  service.py
  api.py
  views.py
  app.py
  semantic_model.json
  package_manifest.json
  release_report.json
  tests/
```

Validate, inspect, and publish-plan capabilities through the CLI:

```bash
./.venv/bin/apg capabilities validate-contracts --json
./.venv/bin/apg capabilities inspect <capability_id> --json
./.venv/bin/apg capabilities evaluate-rules <capability_id> --context-json '{}' --json
./.venv/bin/apg capabilities publish-plan capabilities/<domain>/<code> --json
./.venv/bin/apg capabilities publish-apply capabilities/<domain>/<code> --catalog /tmp/apg-capability-catalog.json --dry-run --json
./.venv/bin/apg capabilities catalog /tmp/apg-capability-catalog.json --json
./.venv/bin/apg lint path/to/app.apg --catalog /tmp/apg-capability-catalog.json --json
```

Capability changes should keep configuration, deterministic rules, UI routes,
theme tokens, tenant handling, and tests aligned.

## Capacity Work

A capacity is larger than one capability. It is an executable ability APG can
demonstrate: records, rules, screens, workflows, agents, capability contracts,
generated app behavior, tests, and docs.

Use [Capacity Development Guide](./capacity_development_guide.md) when building
new business or platform abilities. Use [Capability Building Standards](./capability_standards.md)
for the package and contract details.

## CLI And Tooling Work

Every durable command should have:

- stable JSON output with a `format` key;
- human-readable text output;
- focused tests;
- documentation in [Tooling](./tooling.md);
- aggregate audit coverage when it becomes part of the baseline.

Run:

```bash
./.venv/bin/apg tooling audit --json
./.venv/bin/python -m pytest -q tests/test_tooling_audit.py
```

## Language Server And Studio Work

The language server and Studio are projections of APG source. APG source remains
the source of truth.

Language-server proof commands:

```bash
./.venv/bin/apg language-server examples/20_enterprise_erp_platform/main.apg --check --json
./.venv/bin/apg language-server examples/20_enterprise_erp_platform/main.apg --code-actions --json
./.venv/bin/apg language-server --audit-fixtures --json
```

Studio proof commands:

```bash
./.venv/bin/apg studio snapshot examples/20_enterprise_erp_platform/main.apg --json
./.venv/bin/apg studio plan-edit examples/20_enterprise_erp_platform/main.apg --edit-json '{"kind":"noop"}' --json
```

Visual edits must produce reviewable APG diffs or be rejected before source is
written.

## Documentation Work

Docs must separate:

- accepted syntax;
- semantic model contract;
- generated executable behavior;
- package/capability behavior;
- metadata-only behavior;
- known gaps.

Do not write future intent as present behavior. If a command, generated app,
test, fixture, or contract does not prove it, label it as planned or omit it.

Update `docs/progress_log.md` for every coherent slice. Include changed
behavior, verification commands, results, and known remaining gaps.

## Verification Lanes

Use the narrowest lane that proves the claim.

| Lane | Commands |
| --- | --- |
| Docs | `./.venv/bin/apg docs audit --json` and `git diff --check -- docs` |
| Parser | `./.venv/bin/apg parser-golden --json` |
| Semantic model | `./.venv/bin/apg model --audit-fixtures --json` |
| Lint | `./.venv/bin/apg lint --audit-fixtures --json` |
| Validate | `./.venv/bin/apg validate path/to/app.apg --catalog /tmp/apg-capability-catalog.json --target python --json` |
| Generator | `./.venv/bin/apg compile examples/20_enterprise_erp_platform/main.apg --catalog /tmp/apg-capability-catalog.json --output /tmp/apg-erp --verify` |
| Compiler baseline | `./.venv/bin/apg baseline examples --json` for numbered examples, generated source hygiene, checked-in output sync, direct checked-output self-test/smoke-test execution, checked-output HTTP contract route probes, graph, model, and release agreement; use `./.venv/bin/apg baseline examples --refresh-outputs --json` only when intentionally regenerating example outputs |
| Release/package | `./.venv/bin/apg package path/to/app.apg --catalog /tmp/apg-capability-catalog.json --target web --out /tmp/apg-package --json` |
| Evidence bundle | `./.venv/bin/apg evidence path/to/app.apg --catalog /tmp/apg-capability-catalog.json --target web --out /tmp/apg-evidence --json` |
| Capabilities | `./.venv/bin/apg capabilities validate-contracts --json` |
| Tooling | `./.venv/bin/apg tooling audit --json` |
| Environment doctor | `./.venv/bin/apg doctor --json` |
| Repository hygiene | `./.venv/bin/apg hygiene audit --json` and `./.venv/bin/python -m pytest -q tests/test_repository_hygiene.py` |

## Definition Of Done

A developer slice is done when:

- it changes the current repository, not only a plan;
- the feature is reachable through APG source, CLI, generated app, or a package
  API;
- focused tests or commands prove it;
- examples or fixtures cover author-facing behavior;
- docs describe current behavior accurately;
- `docs/progress_log.md` records evidence;
- the verified slice is committed and pushed.
