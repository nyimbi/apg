# APG Developer Guide

This guide is for people changing APG itself: grammar, compiler, semantic
model, generated Python runtime, CLI tooling, language-server surfaces,
capability packages, examples, tests, and documentation.

The expected outcome is immediate effectiveness. A developer should be able to
enter the repository, prove the current baseline, choose the right owning layer,
make one executable slice better, verify it, update the handoff trail, and
commit it without private context.

## Ten-Minute Start

Run these from the repository root before editing:

```bash
git status --short
./.venv/bin/apg --help
./.venv/bin/apg docs audit --json
./.venv/bin/apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-dev-start --verify
./.venv/bin/python /tmp/apg-dev-start/smoke_test.py
```

Read the output. If the worktree has unrelated changes, leave them unstaged. If
the compile or smoke baseline fails, fix the earliest failing layer before
building new features.

Then write a one-slice packet:

```text
Outcome:
Owning layer:
Public contract:
Representative source or package:
Focused proof:
Docs/progress-log update:
Non-goals:
```

Do not start with a broad rewrite. APG advances through small verified packets
that leave the next developer with a clearer command path.

## Development Mental Model

APG is a chain of contracts:

```text
.apg source
  -> spec/apg.g4
  -> parser artifacts
  -> compiler/ast_builder.py
  -> compiler/semantic_analyzer.py
  -> compiler/semantic_model.py
  -> compiler/code_generator.py
  -> generated Python app
  -> capability packages
  -> CLI audits, examples, docs, and release evidence
```

A feature is not done when it parses. It is serviceable when the promised
downstream surface can consume it and a command proves that claim.

| Contract | Owned by | Proof |
| --- | --- | --- |
| Authoring syntax | `spec/apg.g4`, parser fixtures | parser-focused tests, `apg parser-golden --json` |
| AST projection | `compiler/ast_builder.py` | focused compiler tests, `apg model <file> --json` |
| Semantic meaning | `compiler/semantic_analyzer.py`, `compiler/semantic_model.py` | semantic JSON fields, diagnostics, graph output |
| Generated execution | `compiler/code_generator.py` | `apg compile ... --verify`, generated `smoke_test.py` |
| Capability behavior | `capabilities/<domain>/<code>/` | package pytest, implementation audit, publish-plan |
| Tooling evidence | `cli/`, `compiler/*audit*.py`, `language_server/` | focused CLI tests, `apg tooling audit --json` |
| Handoff | `docs/`, example READMEs, `docs/progress_log.md` | `apg docs audit --json`, `git diff --check` |

Fix the first broken boundary. If a screen declaration parses but is absent from
`apg model --json`, fix semantic projection before changing generated UI code.
If the semantic model has the data but generated smoke tests cannot see it,
work in the generator.

## Repository Map

| Path | Responsibility |
| --- | --- |
| `spec/apg.g4` | Canonical APG grammar. Keep it terse, readable, and practical. |
| `spec/apgLexer.py`, `spec/apgParser.py`, `spec/apgVisitor.py` | Checked-in generated parser artifacts when grammar changes require them. |
| `compiler/parser.py` | Source loading and parser wrapper. |
| `compiler/ast_builder.py` | Parse tree to normalized APG AST data. |
| `compiler/semantic_analyzer.py` | Symbol resolution, reference validation, diagnostics. |
| `compiler/semantic_model.py` | Stable `apg.semantic-model.v1` projection for tools and generators. |
| `compiler/code_generator.py` | Python-first generated application writer. |
| `compiler/*` | Formatter, graphs, drift, release evidence, package, migration, Studio, and audit tooling. |
| `cli/` | Installed `apg` command surface. Durable commands need JSON output. |
| `language_server/` | Dependency-light editor-facing semantic service. |
| `capabilities/` | Package-backed capability contracts and executable runtime packages. |
| `examples/` | Numbered parseable APG examples and intentionally refreshed output. |
| `tests/` | Focused regression tests and fixture catalogs. |
| `docs/` | Contributor, language, tooling, standards, and progress documentation. |

Generated output is evidence, not the primary owner. If generated files are
wrong because the generator is wrong, change the generator and refresh output
only when the output belongs to the packet.

## Primary Work Lanes

### Language And Compiler

Use this lane when APG authors need to express a concept that the compiler
cannot yet parse or project.

Required steps:

1. Add or adjust the grammar in `spec/apg.g4`.
2. Update parser artifacts when required by the grammar workflow.
3. Normalize the construct in `compiler/ast_builder.py`.
4. Validate references in `compiler/semantic_analyzer.py` when the construct
   names other symbols.
5. Expose stable data in `compiler/semantic_model.py`.
6. Add a focused fixture or test.
7. Prove with `apg model <file> --json` and the relevant focused test.

Do not add syntax just because it is expressive. Add syntax when it has a
semantic owner and a path to generated Python, capability behavior, or tooling
inspection.

### Generated Python Runtime

Use this lane when the semantic model already contains the needed data but the
generated application does not expose it.

Required steps:

1. Inspect the representative `.apg` source and `apg model ... --json` output.
2. Update `compiler/code_generator.py` or the relevant runtime helper.
3. Add or update smoke assertions when the generated app surface changes.
4. Compile to `/tmp` first:

   ```bash
   ./.venv/bin/apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-runtime-slice --verify
   ./.venv/bin/python /tmp/apg-runtime-slice/smoke_test.py
   ```

5. Refresh checked-in example output only when the packet intentionally changes
   generated artifacts.

Generated apps should remain dependency-light Python artifacts. Do not introduce
framework-specific targets unless a separate accepted plan changes the target
strategy.

### Capability Packages

Use this lane when a capability package has the right contract shape but still
behaves like a generic materialized record store.

Required steps:

1. Run the implementation audit:

   ```bash
   ./.venv/bin/apg capabilities implementation-audit --json
   ```

2. Pick one package and read only its owning tree first.
3. Replace generic records with domain models, deterministic service behavior,
   API helpers, view models, and focused tests.
4. Keep live integrations behind adapters unless the packet verifies them.
5. Prove the package:

   ```bash
   ./.venv/bin/pytest -q capabilities/<domain>/<code>/test_capability_contract.py capabilities/<domain>/<code>/tests
   ./.venv/bin/apg capabilities implementation-audit --root capabilities/<domain>/<code> --json
   ./.venv/bin/apg capabilities publish-plan capabilities/<domain>/<code> --json
   ```

6. Update the package `cap_spec.md` and `docs/progress_log.md` when readiness
   or implementation-depth evidence changes.

### Examples And Capacity Proof

Use this lane when APG needs to demonstrate an executable application or
business capacity.

Required steps:

1. Choose one numbered example directory.
2. Keep `main.apg` parseable and focused on one event or workflow.
3. Run:

   ```bash
   ./.venv/bin/apg model examples/<nn>_<name>/main.apg --json
   ./.venv/bin/apg compile examples/<nn>_<name>/main.apg --output /tmp/apg-example-slice --verify
   ./.venv/bin/python /tmp/apg-example-slice/smoke_test.py
   ```

4. Update the example README with current readiness, proof commands, generated
   output status, package owners, and next gap.

### Tooling, IDE, And Documentation

Use this lane when the contribution changes the way developers inspect,
validate, package, or understand APG.

Common proof commands:

```bash
./.venv/bin/apg tooling audit --json
./.venv/bin/apg docs audit --json
./.venv/bin/apg language-server examples/01_minimal_customer_records/main.apg --check --json
./.venv/bin/apg studio snapshot examples/01_minimal_customer_records/main.apg --json
git diff --check -- docs
```

Documentation must describe current executable behavior. If the behavior is not
implemented, name it as a gap and point to the owner and proof command.

## Battery-Aware Verification

Prefer focused proof while on battery. Choose the command that proves the
changed contract directly.

| Change | Minimum proof |
| --- | --- |
| Docs only | `apg docs audit --json`, `git diff --check -- docs` |
| One capability package | focused package pytest, implementation audit for that root, publish-plan |
| Shared capability contract surface | package proof plus strict capability artifact audit |
| Compiler semantic key | focused compiler test, `apg model <fixture> --json` |
| Generated runtime behavior | `apg compile ... --verify`, generated smoke test |
| CLI command or audit output | focused CLI test, relevant `apg ... --json` command |
| Grammar-wide or generator-wide change | focused tests plus representative baseline or release command |

Run broader checks when the public contract is shared across examples,
capabilities, or CLI tooling. Do not claim full-suite confidence when only a
focused slice ran.

## Public Contract Rules

Treat these as public unless the packet explicitly migrates them:

- APG command names and JSON `format` values.
- Semantic model top-level keys and stable nested keys.
- Capability IDs, rule IDs, route names, service method names, and theme names.
- Generated file names such as `app.py`, `smoke_test.py`,
  `semantic_model.json`, `apg_capabilities.py`, and `apg_application.py`.
- Example directory names and documented proof commands.

If a public name must change, update tests, docs, examples, and progress-log
evidence in the same verified slice.

## Commit And Handoff

Before committing:

```bash
git status --short
git diff --cached --name-only
git diff --cached --check
```

Stage only the packet. Leave unrelated local files and agent state alone.

Every commit should explain why the change was made and include honest proof:

```text
<intent line>

<context and approach>

Constraint: <constraint that shaped the work>
Rejected: <alternative> | <reason>
Confidence: <low|medium|high>
Scope-risk: <narrow|moderate|broad>
Directive: <future warning>
Tested: <commands run>
Not-tested: <known gaps>
```

Update `docs/progress_log.md` when the work changes executable readiness,
capability implementation depth, compiler baseline evidence, or contributor
workflow. The log should record commands and outcomes, not aspirations.

## Developer Definition Of Done

A developer slice is done when:

- the owning layer is clear;
- the public contract is preserved or intentionally migrated;
- focused proof passed and was inspected;
- docs, package specs, example READMEs, or progress log were updated when
  evidence changed;
- the staged diff contains only the packet;
- the commit uses the Lore protocol;
- the commit has been pushed when the slice is complete.

If those conditions are not true, keep working or narrow the packet.
