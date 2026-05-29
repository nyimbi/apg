# APG Developer Guide

This guide is for contributors who need to change APG itself: the grammar,
compiler, semantic model, generator, CLI, language server, tooling audits,
examples, tests, and documentation.

The purpose is immediate effectiveness. A new developer should be able to pick
one concrete APG improvement, make it executable, prove it, document it, and
leave the repository better than they found it.

## Working Model

APG has one source-of-truth path:

```text
.apg source
  -> spec/apg.g4 parser
  -> compiler/ast_builder.py
  -> compiler/semantic_analyzer.py
  -> compiler/semantic_model.py
  -> compiler/code_generator.py
  -> generated Python application
  -> release/package/tooling evidence
```

When adding a language feature, do not stop at parser acceptance. The feature
is not serviceable until it reaches the right downstream contract:

- parsed syntax
- AST or normalized metadata
- semantic validation
- generated Python behavior or explicitly documented metadata-only behavior
- CLI/tooling visibility when relevant
- focused tests
- examples or fixtures
- documentation

## Repository Map

| Path | Purpose |
| --- | --- |
| `spec/apg.g4` | ANTLR grammar for the APG language. |
| `spec/apgLexer.py`, `spec/apgParser.py`, `spec/apgVisitor.py` | Generated parser artifacts. |
| `compiler/` | Parser wrappers, AST builder, semantic model, diagnostics, formatter, graphs, generator, packaging, release evidence, audits. |
| `cli/` | Click commands exposed by the installed `apg` CLI. |
| `language_server/` | Dependency-light language service and LSP entry point. |
| `capabilities/` | Platform capability packages and executable capability contracts. |
| `examples/` | Numbered APG examples and checked-in generated outputs. |
| `tests/` | Repository tests and fixture catalogs. |
| `tests/fixtures/` | Parser, lint, semantic-model, formatter, graph, language-server, migration, NL-plan, and verifier fixtures. |
| `docs/` | User, developer, contributor, and architecture documentation. |
| `vscode-extension/` | APG VS Code extension assets and command integration. |

## Setup

From the repository root:

```bash
uv sync
./.venv/bin/apg --help
./.venv/bin/python -m pytest -q tests/test_tooling_audit.py
```

If dependency state is stale:

```bash
uv venv .venv
uv pip install -e ".[dev,language-server]"
```

The primary CLI command is `apg`. In scripts and verification, prefer the
virtual environment path when you need certainty:

```bash
./.venv/bin/apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-example --verify
```

## Core Commands

Use these commands constantly:

```bash
./.venv/bin/apg --help
./.venv/bin/apg compile examples/20_enterprise_erp_platform/main.apg --output /tmp/apg-erp --verify
./.venv/bin/apg tooling audit --json
./.venv/bin/apg baseline examples --json
./.venv/bin/apg lint examples/20_enterprise_erp_platform/main.apg --json
./.venv/bin/apg model examples/20_enterprise_erp_platform/main.apg --json
./.venv/bin/apg graph-suite examples/20_enterprise_erp_platform/main.apg --json
./.venv/bin/apg release examples/20_enterprise_erp_platform/main.apg --json
```

Use focused pytest slices:

```bash
./.venv/bin/python -m pytest -q tests/test_compiler_baseline.py
./.venv/bin/python -m pytest -q tests/test_capability_composition_runtime.py
./.venv/bin/python -m pytest -q tests/test_generated_workflow_runtime.py
./.venv/bin/python -m pytest -q tests/test_tooling_audit.py
```

Do not run the entire suite by default when working under a battery or compute
constraint. Run the smallest tests that prove the change and one representative
generated-app check.

## Development Workflow

1. Read the current source, tests, and docs for the area you are changing.
2. Identify the current executable contract.
3. Make the smallest coherent change that moves APG toward executable reality.
4. Add or update focused tests.
5. Regenerate example outputs only when compiler output changes.
6. Update docs and `docs/progress_log.md`.
7. Run focused verification.
8. Commit the verified slice with a Lore commit message.
9. Push.

Never stage unrelated dirty files. The repository often contains local agent,
upload, or generated state; keep commits scoped to the slice.

## Changing The Grammar

Start with [APG Grammar Guide](./apg_grammar_guide.md). Then use this checklist:

1. Update `spec/apg.g4`.
2. Regenerate parser artifacts if the generated parser files are part of the
   change.
3. Update `compiler/ast_builder.py`.
4. Update `compiler/semantic_analyzer.py`.
5. Update `compiler/semantic_model.py`.
6. Update `compiler/code_generator.py` if the feature should execute.
7. Add parser/semantic/generator tests or fixtures.
8. Add a numbered example or update an existing example if the feature is
   author-facing.
9. Update docs.

Good grammar changes preserve the universal entity pattern:

```apg
entity_type EntityName {
    contract_or_config: value;
}
```

Prefer reusable `contract_object`, `reference_list`, `rule_list`,
`ui_contract`, `theme_contract`, `screen_set`, and `runtime_contract` rules
before inventing new special syntax.

## Changing The AST And Semantic Model

The AST builder must normalize APG source into stable Python data classes or
metadata dictionaries. The semantic model must then expose the same meaning to
CLI, tooling, language server, Studio, graph builders, release evidence, and
generated apps.

When adding semantic data:

- add a model field with a stable JSON shape
- include the source file and names where possible
- add diagnostics for invalid or unresolved references
- add graph nodes/edges if the construct participates in composition
- avoid making downstream tools re-parse source text

If the generator needs a construct, the semantic model should usually know
about it first.

## Changing Code Generation

`compiler/code_generator.py` emits dependency-light Python applications. The
generated app should remain importable and runnable without optional platform
services.

Generated apps should expose:

- `app.py`
- `__init__.py`
- `semantic_model.json`
- `README.md`
- `requirements.txt`
- `Dockerfile`
- `smoke_test.py`
- optional sidecars such as `ai_agents.py`, `apg_capabilities.py`, and
  `apg_application.py`

Generated behavior should be surfaced through:

- Python helpers
- HTTP route dispatch
- OpenAPI schema entries
- component manifest entries
- validation/self-test checks
- generated smoke tests
- package exports where applicable

When changing generated output, run:

```bash
./.venv/bin/python -m py_compile compiler/code_generator.py
./.venv/bin/apg compile examples/20_enterprise_erp_platform/main.apg --output /tmp/apg-erp --verify
```

If checked-in example outputs change, regenerate all numbered examples and run
the checked-in output comparison test:

```bash
./.venv/bin/python -m pytest -q tests/test_compiler_baseline.py::test_checked_in_example_outputs_match_current_compiler
```

## Changing CLI And Tooling

Every CLI command should have:

- a stable JSON format
- a human-readable text mode
- focused tests
- documentation in `docs/tooling.md`
- aggregate coverage in `apg tooling audit --json` when it is part of the
  documented baseline

Current tooling audits cover parser golden fixtures, diagnostics, lint,
formatter, drift, semantic model, graphs, language server, NL plans, migration,
release evidence, CLI command registration, IDE integration, and Studio
designer contracts.

Run:

```bash
./.venv/bin/apg tooling audit --json
./.venv/bin/python -m pytest -q tests/test_tooling_audit.py tests/test_enhanced_cli.py
```

## Changing The Language Server

The language server should consume the shared semantic model instead of
re-implementing APG meaning.

Required surfaces:

- `apg language-server <file> --check --json`
- `apg language-server <file> --rename <symbol> --to <new-name> --json`
- `apg language-server <file> --code-actions --json`
- `apg language-server --audit-fixtures --json`

Use fixture tests under `tests/fixtures/language_server/` when adding editor
behavior.

## Changing Studio

Studio is a visual-designer projection of APG source. APG source remains the
source of truth.

Required surfaces:

- `apg studio snapshot <file> --json`
- `apg studio plan-edit <file> --edit-json ... --json`

Visual edits must produce reviewable APG diffs. Invalid visual edits should be
rejected before writing source.

## Changing Capabilities

Use [Capability Building Standards](./capability_standards.md) and [Capacity
Development Guide](./capacity_development_guide.md).

At minimum, a capability package should include:

```text
capabilities/<domain>/<code>/
  __init__.py
  cap_spec.md
  capability_contract.py
  models.py
  service.py
  api.py
  views.py
  tests/
```

If a capability is first introduced in APG source, make sure it compiles into
`apg_capabilities.py` and is visible in `component_manifest()`.

## Examples

The numbered examples are the practical language ladder. Keep them parseable,
compiler-clean, and generated.

Rules:

- One example per directory.
- Each example has `main.apg`, `README.md`, and `output/`.
- Examples should increase in complexity.
- Generated output must match the current compiler.
- Example docs should explain the specific language concepts exercised.

Representative verification:

```bash
./.venv/bin/apg compile examples/20_enterprise_erp_platform/main.apg --output /tmp/apg-erp --verify
./.venv/bin/python examples/20_enterprise_erp_platform/output/smoke_test.py
```

## Documentation

Docs must distinguish:

- accepted grammar
- semantic model contract
- generated executable behavior
- metadata-only behavior
- future work

Do not document aspiration as current behavior unless a command, generated app,
test, or fixture proves it.

Update `docs/progress_log.md` for every coherent slice. Include:

- what changed
- verification commands and results
- known remaining gaps

## Verification Lanes

Use the narrowest verification lane that proves the claim.

Parser or grammar:

```bash
./.venv/bin/apg parser-golden --json
./.venv/bin/python -m pytest -q tests/test_apg_language_contract.py
```

Semantic model:

```bash
./.venv/bin/apg model --audit-fixtures --json
./.venv/bin/python -m pytest -q tests/test_semantic_analyzer.py
```

Generator:

```bash
./.venv/bin/apg compile examples/20_enterprise_erp_platform/main.apg --output /tmp/apg-erp --verify
```

Capabilities:

```bash
./.venv/bin/apg capabilities validate-contracts --json
./.venv/bin/python -m pytest -q tests/test_capability_contract_registry.py tests/test_capability_composition_runtime.py
```

Tooling:

```bash
./.venv/bin/apg tooling audit --json
./.venv/bin/python -m pytest -q tests/test_tooling_audit.py
```

Docs:

```bash
git diff --check -- docs
```

## Done Means Executable

A change is done when:

- the current worktree proves it, not just the design intent
- the feature can be invoked through APG source, CLI, generated app, or a
  documented package interface
- tests cover the changed behavior
- examples or fixtures cover author-facing syntax
- docs describe the current truth
- `docs/progress_log.md` records the evidence
- the verified slice is committed and pushed

