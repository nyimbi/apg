# APG Contributors Guide

This guide explains how to contribute to APG without slowing the project down
or adding drift between the language vision and executable reality.

The central rule is simple: make the requested final state more true in the
current repository, prove it with evidence, document what changed, and commit a
reviewable slice.

## Contributor Mindset

APG is not only a grammar. It is a language, compiler, generated runtime,
tooling stack, capability platform, documentation set, and example catalog.

Good contributions:

- move from aspiration toward executable behavior
- preserve the Python-first compiler target
- keep syntax terse but readable
- make AI agents and capabilities first-class citizens
- keep Bytewax as the internal streaming direction
- update tests, examples, docs, and progress evidence
- avoid unrelated cleanup in the same commit

Poor contributions:

- add syntax that nothing consumes
- document behavior that no command or test proves
- add generated code drift without updating examples
- introduce framework targets that multiply the generator matrix
- hide external services behind untestable placeholders
- stage unrelated dirty files

## Before Your First Change

Run:

```bash
uv sync
./.venv/bin/apg --help
./.venv/bin/apg tooling audit --json
./.venv/bin/apg compile examples/20_enterprise_erp_platform/main.apg --output /tmp/apg-erp --verify
```

Read:

- [Quick Start](./quickstart.md)
- [APG Language Guide](./apg_language.md)
- [APG Tutorial](./apg_tutorial.md)
- [Developer Guide](./developer_guide.md)
- [Capability Building Standards](./capability_standards.md)
- [Capacity Development Guide](./capacity_development_guide.md)
- [Tooling Specification](./tooling.md)
- [Goal Progress Log](./progress_log.md)

## Picking Work

Pick work that has a clear executable endpoint:

- a grammar construct reaches generated runtime behavior
- a compiler warning becomes a diagnostic
- a generated app route becomes executable
- a capability contract becomes discoverable and valid
- a numbered example compiles and smokes
- a tooling command emits stable JSON
- documentation reflects the current verified contract

If a task is large, split it into vertical slices. A vertical slice includes
source, behavior, tests, docs, and progress evidence for one coherent outcome.

## Branch And Worktree Hygiene

Check status before editing:

```bash
git status --short
```

There may be existing modified or untracked files from another user or agent.
Do not revert or stage them unless they are part of your task.

Stage explicitly:

```bash
git add compiler/tooling_audit.py tests/test_tooling_audit.py docs/tooling.md docs/progress_log.md
```

Review staged files:

```bash
git diff --cached --stat
git diff --cached --check
```

## Code Style

Follow the existing file style. APG currently uses tabs in several compiler
modules and tests may use tabs or spaces depending on the file. Match the file.

General standards:

- Prefer explicit structured data over string scraping.
- Use existing helpers before adding abstractions.
- Keep comments sparse and useful.
- Do not add dependencies unless the task explicitly requires it.
- Preserve dependency-light generated apps.
- Keep generated Python importable without optional external services.
- Keep public JSON formats stable once documented.

## Documentation Style

Documentation must be actionable and current.

Use:

- concrete commands
- concrete file paths
- current JSON format names
- examples that parse today
- explicit verification instructions
- known gaps when behavior is partial

Avoid:

- broad claims without evidence
- future behavior written as present behavior
- marketing language that hides the actual contract
- old target names such as Flask-AppBuilder or Django as compiler targets

## Testing Expectations

Use focused tests for the changed area. Full-suite verification is useful but
not required for every battery-constrained slice.

Minimum expectations by change type:

| Change | Focused verification |
| --- | --- |
| Docs only | `git diff --check -- docs`; link sanity when adding links |
| CLI/tooling | `apg tooling audit --json`; focused CLI tests |
| Grammar | parser golden audit; language contract tests |
| Semantic model | semantic-model fixture audit; focused semantic tests |
| Generator | compile representative example with `--verify`; generated smoke test |
| Capability contract | `apg capabilities validate-contracts --json`; focused registry tests |
| Package/release | release/evidence fixture audit or package verify command |

Always read command output. Do not claim a command passed unless you inspected
the result.

## Commit Protocol

Commit completed, verified work regularly. APG uses Lore-style commit messages:

```text
<intent line: why the change exists>

<body: context and approach>

Constraint: <external constraint>
Rejected: <alternative> | <reason>
Confidence: <low|medium|high>
Scope-risk: <narrow|moderate|broad>
Directive: <future guidance>
Tested: <verification performed>
Not-tested: <known gaps>
```

Example:

```text
Prove APG tooling surfaces through one audit gate

The tooling spec claims one executable baseline, so the aggregate audit now
verifies CLI, IDE, and Studio surfaces in addition to fixture catalogs.

Constraint: Verification must stay battery-conscious
Rejected: Manual checklist only | the baseline needs machine-readable evidence
Confidence: high
Scope-risk: narrow
Directive: Add documented tooling commands to the aggregate audit
Tested: apg tooling audit --json; pytest -q tests/test_tooling_audit.py
Not-tested: Full repository test suite
```

Push after a verified commit:

```bash
git push
```

## Pull Request Or Handoff Checklist

Before handing off:

- `git status --short` shows only unrelated dirty files or a clean tree.
- Staged/committed files are scoped to the task.
- Tests or commands proving the task are recorded.
- `docs/progress_log.md` has a new entry.
- New docs are linked from `docs/README.md` when they are durable guides.
- New CLI/tooling surfaces are covered by `apg tooling audit --json`.
- New examples compile with `--verify`.
- Known gaps are explicit.

## Common Contribution Patterns

### Add A New APG Language Construct

1. Update grammar.
2. Update AST builder.
3. Update semantic model.
4. Add diagnostics.
5. Add generator behavior or document metadata-only status.
6. Add tests and fixtures.
7. Add an example.
8. Update docs.
9. Run focused verification.

### Add A Generated Runtime Helper

1. Add helper generation in `compiler/code_generator.py`.
2. Export it from generated `__init__.py` when public.
3. Add route dispatch if it is HTTP-visible.
4. Add OpenAPI and component manifest entries.
5. Add generated self-test validation.
6. Add focused generated-runtime tests.
7. Regenerate checked-in examples if output changes.

### Add A CLI Command

1. Add or update a module under `cli/`.
2. Register it in `cli/main.py`.
3. Emit stable JSON.
4. Add focused CLI tests.
5. Update `docs/tooling.md`.
6. Add aggregate audit coverage if it is part of the executable baseline.

### Add A Capability

1. Define the capability boundary and provided services.
2. Add APG source or package files.
3. Add `cap_spec.md` and `capability_contract.py`.
4. Add models/service/API/views as needed.
5. Add focused tests.
6. Validate contracts.
7. Document the capability.

## Review Standards

Review for defects first:

- Does the code actually execute?
- Does the semantic model carry the feature?
- Does generated output expose the behavior?
- Are public JSON formats stable?
- Are tests proving the claim?
- Are docs aligned with current behavior?
- Are unrelated files staged?

Style is secondary to correctness, scope control, and evidence.

## Safety Rules

- Do not put secrets in APG source, generated examples, docs, or tests.
- Do not add external side effects to audits that should be side-effect-free.
- Do not require optional services for dependency-light generated apps.
- Do not replace Bytewax with a broker-first internal streaming direction.
- Do not add provider-specific AI grammar for every new agent tool; use
  adapters.
- Do not mark broad objectives complete from narrow evidence.

## Effective First Issues

Good first contribution areas:

- add a missing diagnostic fixture
- improve one generated app self-test check
- add a small APG example fixture
- extend `apg tooling audit` for one documented command surface
- convert one aspirational doc paragraph into current executable guidance
- add a capability contract wrapper for an existing `cap_spec.md`
- improve a generated README section

Avoid starting with broad refactors, global formatting, dependency changes, or
large capability rewrites.
