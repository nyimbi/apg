# APG Contributors Guide

This guide explains how to contribute to APG without slowing the project down or
creating drift between the APG vision and executable reality.

The contribution rule is simple: make one useful final state more true in the
current repository, prove it with evidence, document it, and commit a reviewable
slice.

## Immediate Contributor Path

Use this path when you want to become useful quickly.

1. Check the worktree and leave unrelated files alone.

   ```bash
   git status --short
   ```

2. Prove the local APG CLI can generate an executable app.

   ```bash
   ./.venv/bin/apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-contributor --verify
   ./.venv/bin/python /tmp/apg-contributor/smoke_test.py
   ```

3. Choose one contribution class:

   | Class | Good first target | Evidence |
   | --- | --- | --- |
   | Compiler | one semantic-model or generated-runtime assertion | focused compiler test and `apg compile ... --verify` |
   | Capability | one materialized package converted to domain behavior | package tests, implementation audit, publish-plan |
   | Capacity | one example that compiles and documents its generated output | model, compile, smoke test, README update |
   | Tooling/docs | one exact command, audit, or guide improvement | docs audit, tooling audit, diff check |

4. Write the packet before editing:

   ```text
   Outcome:
   Owning files:
   Public contract:
   Verification:
   Progress-log entry:
   Non-goals:
   ```

5. Make the smallest vertical change, run the proof, update the progress log
   when evidence changed, then commit and push only that slice.

The fastest contributors are not the ones who touch the most files. They are the
ones who leave the next contributor with fewer unknowns and a command that
proves the current state.

## What APG Is

APG is a Python-first application generation platform built around a terse,
readable DSL. A good contribution improves at least one of these surfaces:

- the APG language grammar;
- parser and AST normalization;
- semantic validation and diagnostics;
- generated Python applications;
- capability contracts and package-backed capabilities;
- AI agent composition;
- screen, workflow, rule, theme, i18n, and Bytewax streaming metadata;
- CLI tooling and JSON evidence;
- examples, docs, and tests.

APG is not helped by syntax nobody consumes, documentation no command proves, or
generated code that cannot be imported and smoke-tested.

## First 30 Minutes

From the repository root:

```bash
uv venv .venv
uv pip install -e ".[dev,language-server]"
./.venv/bin/apg --help
./.venv/bin/apg doctor --json
./.venv/bin/apg tooling audit --json
./.venv/bin/apg compile examples/20_enterprise_erp_platform/main.apg --output /tmp/apg-erp --verify
```

Then read these in order:

1. [Quick Start](./quickstart.md)
2. [APG Language Guide](./apg_language.md)
3. [Developer Guide](./developer_guide.md)
4. [Capability Building Standards](./capability_standards.md)
5. [Capacity Development Guide](./capacity_development_guide.md)
6. [Tooling Specification](./tooling.md)
7. [Goal Progress Log](./progress_log.md)

## First Useful Commit

Your first APG commit should be small enough to understand, verify, and explain
the same day. Use this exact shape:

```text
Goal: make one APG behavior, capacity, package, example, or guide easier to execute.
Files: one owning directory, plus focused tests/docs/progress log.
Proof: one command that proves the changed layer and one whitespace/docs check.
Commit: Lore message with Tested and Not-tested trailers.
Push: after the verified commit lands locally.
```

Good first commits:

- improve one guide section with exact current commands and run the docs audit;
- convert one capability package method from placeholder behavior to
  deterministic service behavior with focused tests;
- add one semantic-model assertion for a syntax shape that already parses;
- make one example README explain its generated output and rerun the focused
  compile;
- add one guardrail test for a capability rule.

Bad first commits mix broad cleanup, grammar redesign, package conversion, and
example-output refreshes in one diff. Split those into separate packets so
reviewers can prove each claim independently.

## Zero-To-PR Runbook

Use this path when you need to make a useful contribution without first
understanding every APG subsystem.

1. Confirm the worktree and avoid unrelated files:

   ```bash
   git status --short
   ```

2. Prove one executable baseline:

   ```bash
   ./.venv/bin/apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-zero-pr --verify
   ./.venv/bin/python /tmp/apg-zero-pr/smoke_test.py
   ```

3. Choose exactly one lane: docs, example, semantic model, generator,
   capability package, CLI/tooling, or capacity.
4. Write a one-paragraph work packet before editing:

   ```text
   Outcome:
   Files:
   Public contract:
   Verification:
   Non-goals:
   ```

5. Make the smallest vertical change that improves that outcome.
6. Run focused verification and inspect the output.
7. Update `docs/progress_log.md` when the change affects executable behavior,
   capability readiness, contributor flow, or verification evidence.
8. Stage only your files and check the staged diff:

   ```bash
   git add <only-your-files>
   git diff --cached --stat
   git diff --cached --check
   ```

9. Commit with the Lore protocol and push.

This runbook is intentionally narrow. APG has many aspirational documents and
historical artifacts; contribution speed comes from making one current,
verified state better.

## Fastest Useful Contribution Path

If you are new and need to help today, use this path:

1. Run the first 30-minute commands above.
2. Open `docs/progress_log.md` and find the latest known gap.
3. Pick one gap that touches one owning directory.
4. Prove the current behavior before editing.
5. Make the smallest vertical slice that improves the evidence.
6. Run the focused verification for that slice.
7. Record the verification in `docs/progress_log.md`.
8. Commit and push only that slice.

The most useful current gaps are usually visible through these commands:

```bash
./.venv/bin/apg tooling audit --json
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg hygiene audit --include-untracked --json
```

Use `implementation-audit` to find packages that are complete in file shape but
still shallow in domain behavior. A good contributor can take one such package,
replace generated baseline service/model/API/view placeholders with
domain-specific behavior, add focused tests, run publish-plan, and leave a
progress-log entry for the next package.

## First 90 Minutes

After the first baseline commands pass, do one small useful thing end to end:

1. Run `git status --short` and note unrelated dirty files.
2. Pick one task lane: docs, example, semantic model, generator, capability,
   CLI, or capacity.
3. Read one existing test for that lane.
4. Make the smallest change that improves executable reality.
5. Run the focused verification for that lane.
6. Update docs or `docs/progress_log.md` when behavior or evidence changed.
7. Stage only your slice.
8. Commit with Lore trailers and push.

Good first 90-minute contributions:

- add a missing docs link and run the docs audit;
- tighten one numbered example README and prove the baseline still passes;
- add one lint fixture for an existing diagnostic;
- improve one generated app smoke assertion;
- add one capability contract default and focused package test;
- document one verified CLI JSON field in `docs/tooling.md`.

Do not use the first contribution to redesign the grammar, replace the
generator, or normalize historical archives. Those can be good later tasks, but
they are poor onboarding slices because they have large blast radius.

## New Contributor Work Packets

Use a work packet when assigning or claiming APG work. It should fit in one
commit unless the evidence says it must be split.

```text
Outcome:
Owner:
Lane:
Files expected:
Public names affected:
Example or fixture:
Verification:
Docs to update:
Known non-goals:
```

Example work packets:

| Packet | Files expected | Verification |
| --- | --- | --- |
| Add one generator route to expose a semantic-model section | `compiler/code_generator.py`, focused generator test, one example output if needed | compile representative example, smoke test |
| Make one capability package publish-plan ready | one `capabilities/<domain>/<code>/` tree | materialize-packages, package tests, capability audit, publish-plan |
| Add one author-facing APG construct | grammar, AST, semantic model, docs, fixture | parser/semantic tests, compile example |
| Improve one onboarding path | `docs/*.md`, `docs/progress_log.md` | docs audit and diff check |

Good work packets have a visible endpoint. Poor packets say "clean up
capabilities" or "improve compiler" without a source file, generated artifact,
or command that proves completion.

## Claiming A Work Packet

Before editing, write down the packet in the issue, branch note, example README,
or progress-log draft. Keep it concrete:

```text
Outcome: AGNT execution plans include runtime assignments and approval evidence.
Owner: <your name>
Lane: capability implementation depth
Files expected: capabilities/common/agnt/*
Public names affected: AgntService, execution_plan, runtime assignments
Example or fixture: capabilities/common/agnt/test_capability_contract.py
Verification: pytest focused AGNT tests; implementation-audit; publish-plan
Docs to update: cap_spec.md; docs/progress_log.md
Known non-goals: live external Codex/Claude/OpenCode/Pi invocation
```

If the packet cannot name a file, public contract, and verification command, it
is not ready to execute. Narrow it until the endpoint is obvious.

## Priority Work Lanes

Use this table to choose work without waiting for architectural context.

| Priority | Lane | Good contribution | Proof |
| --- | --- | --- | --- |
| 1 | Compiler baseline | Keep representative APG examples compiling to runnable Python | `apg compile ... --verify`, `smoke_test.py` |
| 2 | Capability depth | Convert one materialized package into domain-specific behavior | package tests, `implementation-audit`, `publish-plan` |
| 3 | Capacity example | Add or improve a numbered example with README and output | `apg baseline examples --json` or focused compile |
| 4 | Semantic/tooling contract | Add one stable JSON field or audit check | focused CLI tests, `apg tooling audit --json` |
| 5 | Docs and onboarding | Replace vague guidance with exact paths, commands, and gaps | `apg docs audit --json`, `git diff --check -- docs` |

Do not start with broad rewrites. APG advances fastest when many contributors
land small, verified slices that line up into an executable platform.

## Immediate Effectiveness Rules

Follow these rules until you know the repository well:

- Start from a numbered example or one capability package, not from a broad
  search across the whole tree.
- Prefer one vertical slice over many partial edits.
- Keep public names stable: capability IDs, routes, JSON keys, rule names,
  workflow names, agent names, and screen names.
- Treat `apg.*.v1` JSON formats as contracts.
- Preserve dependency-light generated Python output.
- Use package contracts as the bridge between APG source and durable runtime
  behavior.
- Update docs only with behavior that exists now, or label the gap explicitly.
- Record actual verification output in `docs/progress_log.md`.

When in doubt, make the next smallest executable state real: parseable source,
semantic JSON, generated app, capability contract, package evidence, or docs
that make one of those easier to extend.

## Definition Of Done

A contribution is done when a reviewer can verify it without asking for hidden
context.

Minimum done:

- the changed files belong to one coherent slice;
- public names and JSON format keys are stable or intentionally documented;
- focused verification passed and was inspected;
- docs or examples were updated when user-facing behavior changed;
- `docs/progress_log.md` records meaningful platform progress or evidence;
- unrelated dirty or untracked files were not staged;
- the commit message explains why the change exists.

Not done:

- syntax parses but semantic model support is missing;
- semantic data exists but generated apps cannot expose it when execution is
  expected;
- capability contracts exist but package behavior is still only generated
  baseline scaffolding;
- docs describe behavior that no command, test, example, or package proves;
- a broad cleanup is mixed with a feature slice.

Use the smallest applicable proof. For docs-only changes, `apg docs audit
--json` and `git diff --check -- docs` may be enough. For capability behavior,
focused package tests plus implementation/publish audits are usually the right
proof. For compiler-facing changes, compile a representative example and run
the generated smoke test.

## Picking Safe Parallel Work

Multiple contributors can move quickly if each owns a different surface. Use
these patterns:

| Safe in parallel | Why it works | Merge coordination |
| --- | --- | --- |
| One contributor edits a capability package while another edits docs | Mostly disjoint files | Agree on capability ID and public wording |
| One contributor adds examples while another improves package tests | Output paths differ | Coordinate if compiler output is regenerated |
| One contributor works on CLI docs while another works on compiler internals | Docs can track current command names | Re-run docs audit after CLI changes |
| One contributor improves a generated route while another validates contracts | Different runtime layers | Re-run representative compile and capability audit |

Avoid parallel edits to `spec/apg.g4`, `compiler/semantic_model.py`, and
`compiler/code_generator.py` without agreeing on the public semantic model key
first. These files define shared contracts and small naming conflicts can
create large downstream drift.

## Contribution Principles

Good contributions:

- move from aspiration toward executable behavior;
- keep `python` as the compiler target;
- keep syntax terse but readable;
- make capabilities and AI agents first-class where relevant;
- use Bytewax for APG internal streaming semantics;
- preserve dependency-light generated apps;
- add focused tests rather than broad, expensive verification by default;
- update docs and `docs/progress_log.md`;
- commit completed, verified slices regularly.

Avoid:

- adding grammar without semantic-model or generator follow-through;
- documenting future work as current behavior;
- adding framework targets such as Flask-AppBuilder or Django as compiler
  targets;
- hiding external services behind untestable placeholders;
- staging unrelated dirty files;
- broad cleanup mixed into feature work.

## Capacity Contributor Mental Model

When building a new APG capacity, think in layers:

```text
business outcome
  -> APG records and relationships
  -> capability contracts
  -> deterministic rules
  -> screens and workflows
  -> AI agents and Bytewax streaming when needed
  -> generated Python application
  -> tests, docs, and evidence
```

A capacity is not accepted because it has many files. It is accepted when a
contributor can compile it, run it, inspect its generated routes/manifests, and
extend it without guessing the missing contract.

Use these questions before coding:

- What concrete workflow or business ability will exist after this slice?
- Which APG source file proves it?
- Which capability package owns the behavior?
- Which route, helper, CLI report, or test shows it executes?
- Which docs tell the next contributor where to extend it?

## Worktree Hygiene

Always start with:

```bash
git status --short
```

There may be modified or untracked files from another contributor or agent.
Do not revert them. Do not stage them. Stage explicitly:

```bash
git add compiler/code_generator.py tests/test_generated_workflow_runtime.py docs/progress_log.md
```

Check what you are about to commit:

```bash
git diff --cached --stat
git diff --cached --check
```

## Choosing Your First Useful Task

Pick work with a clear executable endpoint. Good first tasks include:

- add a lint diagnostic and fixture;
- improve one generated app helper and smoke test;
- add one capability contract test;
- make one numbered example clearer without changing compiler output;
- add a missing JSON field to an existing CLI report and document it;
- improve one documentation page by replacing aspiration with verified commands.

Avoid first tasks that require changing grammar, generator, and multiple
capability packages at once unless you are already familiar with the toolchain.

## Capability Implementation Packet

Many APG capability packages are valid in shape before they are rich in domain
behavior. A strong contribution can take one package from materialized baseline
to domain-specific implementation.

Use this packet:

```text
Capability id:
Domain outcome:
Services to make real:
Rules to enforce:
Configuration used:
UI/view state exposed:
Tests added:
Publish-plan command:
Known gaps:
```

Implementation steps:

1. Read the package contract and spec first.
2. Remove generic baseline markers only after replacing them with real
   behavior.
3. Keep in-memory runtime state deterministic and tenant-aware where the
   contract says tenant context matters.
4. Use existing rule evaluation helpers for configured rules.
5. Make API helpers and view helpers expose the service state rather than fixed
   demo rows.
6. Add focused tests for at least one happy path and one guardrail.
7. Run implementation audit and publish-plan before claiming readiness.

This is one of the fastest ways to increase APG capacity because it turns an
already-declared capability into something applications can compose.

## Vertical Slices

A vertical slice contains every layer needed for one coherent outcome. For
example, adding a new author-facing screen feature usually means:

1. grammar accepts the screen shape;
2. AST builder captures it;
3. semantic model exposes it;
4. generated app includes it in manifests/routes;
5. graph or Studio surfaces can inspect it when relevant;
6. tests prove it;
7. examples or docs show it;
8. progress log records the evidence.

Small vertical slices are preferred over large partial changes.

## Parallel Contribution Protocol

APG work can proceed quickly when contributors avoid shared-file collisions.
Use these ownership boundaries:

| Contributor lane | Owns | Coordinates before touching |
| --- | --- | --- |
| Grammar | `spec/apg.g4`, parser artifacts, parser fixtures | generator and semantic-model files |
| Compiler semantics | AST, analyzer, semantic model, fixture catalogs | grammar and generator behavior |
| Generator | `compiler/code_generator.py`, generated output fixtures | semantic model keys and example outputs |
| Capability package | one `capabilities/<domain>/<code>/` tree | shared capability registry behavior |
| Example capacity | one `examples/<nn>_*/` tree | compiler output refreshes |
| Docs | relevant `docs/*.md` page and progress log | command examples and public terminology |

Before merging parallel work, align public names: capability IDs, provided
services, route paths, workflow names, screen names, agent names, event names,
and JSON field names.

## Coding Standards

Follow the style of the file you are editing. Some APG modules use tabs; some
tests and newer modules use spaces. Match the local file.

General rules:

- use structured data rather than string scraping;
- reuse existing helpers before adding abstractions;
- keep comments sparse and explanatory;
- avoid new dependencies unless explicitly required;
- preserve importable generated Python artifacts;
- keep public JSON formats stable once documented;
- prefer deterministic local behavior over external-service assumptions.

## Documentation Standards

Docs should make a contributor faster and safer. Use:

- exact commands;
- exact paths;
- current JSON format names;
- parseable APG snippets;
- verification instructions;
- explicit known gaps.

Avoid:

- marketing claims;
- stale target names;
- "will support" phrased as current behavior;
- examples that do not parse;
- descriptions that require reading hidden context to act.

## Testing Expectations

Use focused tests for the changed area. Full-suite runs are not required for
every contribution, especially when compute or battery is constrained.

| Change | Minimum useful verification |
| --- | --- |
| Docs only | `./.venv/bin/apg docs audit --json` and `git diff --check -- docs` |
| CLI/tooling | focused CLI tests plus `./.venv/bin/apg tooling audit --json` |
| Environment doctor | `./.venv/bin/apg doctor --json` |
| Grammar | parser golden audit and relevant language contract tests |
| Semantic model | semantic model fixture audit or focused semantic tests |
| Generator | representative `apg compile ... --verify` and generated smoke test |
| Capability contract | `apg capabilities validate-contracts --json`, `apg capabilities audit --json`, and focused contract tests |
| Capability scaffold/package | scaffold or materialize-packages, implementation-audit, publish-plan, publish-apply dry run, focused package tests |
| Repository hygiene | `./.venv/bin/apg hygiene audit --json` and `./.venv/bin/python -m pytest -q tests/test_repository_hygiene.py` |

Always inspect command output before claiming a pass.

## Capability Contributions

For package-backed capabilities, use the scaffold:

```bash
./.venv/bin/apg capabilities scaffold common demo --name "Demo Capability" --json
```

Then iterate:

```bash
./.venv/bin/python -m pytest -q capabilities/common/demo/tests
./.venv/bin/apg capabilities validate-contracts --json
./.venv/bin/apg capabilities audit --json
./.venv/bin/apg capabilities materialize-packages --capability common_demo --json
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities inspect common_demo --json
./.venv/bin/apg capabilities evaluate-rules common_demo --context-json '{}' --json
./.venv/bin/apg capabilities publish-plan capabilities/common/demo --json
./.venv/bin/apg capabilities publish-apply capabilities/common/demo --catalog /tmp/apg-capability-catalog.json --dry-run --json
./.venv/bin/apg lint examples/08_basic_capability_contract/main.apg --catalog /tmp/apg-capability-catalog.json --json
```

Capability PRs should keep these aligned:

- `cap_spec.md`;
- `capability_contract.py`;
- `models.py`;
- `service.py`;
- `api.py`;
- `views.py`;
- `app.py`;
- `semantic_model.json`;
- `package_manifest.json`;
- tests and docs.

## Capacity Contributions

A capacity contribution makes APG able to do something larger than one package:
for example procurement approvals, inventory operations, general ledger,
customer service, or AI-assisted compliance review.

Use [Capacity Development Guide](./capacity_development_guide.md). A capacity
slice should include:

- records or entities;
- capability contracts;
- deterministic rules;
- screens and relationships;
- workflows where process state matters;
- AI agents where model-backed work is useful;
- Bytewax streaming metadata where event flow matters;
- generated application proof;
- tests and documentation.

Minimum capacity evidence:

```bash
./.venv/bin/apg compile path/to/capacity.apg --output /tmp/apg-capacity --verify
./.venv/bin/python /tmp/apg-capacity/smoke_test.py
./.venv/bin/apg capabilities validate-contracts --json
./.venv/bin/apg capabilities audit --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```

For numbered examples or compiler-facing capacity work, also run:

```bash
./.venv/bin/apg baseline examples --json
```

## Commit Protocol

APG uses Lore-style commit messages. The first line explains why the change
exists, not what files changed.

```text
Make capability scaffolds publish-plan ready

Fresh scaffolds should be usable by contributors immediately, so the generated
package now includes runtime evidence files consumed by the publish planner.

Constraint: Verification must stay battery-conscious
Rejected: Manual catalog edits | they bypass the executable publish path
Confidence: high
Scope-risk: narrow
Directive: Keep scaffold output valid against publish-plan tests
Tested: pytest -q tests/test_cli_capability_scaffold.py; apg tooling audit --json
Not-tested: Full repository test suite
```

Push after a verified commit:

```bash
git push
```

## Handoff Checklist

Before asking for review or handing off:

- `git status --short` contains only unrelated dirty files or is clean.
- The commit contains only the intended slice.
- Verification commands and outcomes are recorded.
- `docs/progress_log.md` has a new entry for meaningful work.
- New durable docs are linked from `docs/README.md`.
- New author-facing syntax is covered by examples or fixtures.
- New CLI/tooling behavior emits stable JSON and is documented.
- Known gaps are explicit.

## Review Standard

Review for correctness before style:

- Does it execute?
- Is meaning represented in the semantic model?
- Does generated output expose the behavior?
- Are public JSON contracts stable?
- Are tests proving the claim?
- Are docs current?
- Are unrelated files absent from the commit?

If a contribution makes APG more executable, easier to extend, and easier to
verify, it is moving in the right direction.
