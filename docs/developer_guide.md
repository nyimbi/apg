# APG Developer Guide

This guide is for people changing APG itself: grammar, compiler, semantic
model, generated Python runtime, CLI tooling, language-server surfaces,
capability packages, examples, tests, and documentation.

The expected outcome is immediate effectiveness. A developer should be able to
enter the repository, prove the current baseline, choose the right owning layer,
make one executable slice better, verify it, update the handoff trail, and
commit it without private context.

## Immediate Effectiveness Contract

APG development is effective when every change moves one observable contract
closer to executable reality. Before editing, write down the contract you are
improving and the command that will prove it.

Use this contract for every packet:

| Question | Acceptable answer |
| --- | --- |
| What user or contributor is helped? | an APG author, generated-app user, capability developer, capacity lead, reviewer, or tool consumer |
| What public surface changes? | grammar syntax, semantic JSON, generated Python, package API, route metadata, rule ID, example, CLI command, or guide |
| Where is the earliest owner? | one file or directory from the repository map, not "the platform" |
| How will it be proven? | one focused command with inspectable output |
| What remains outside scope? | live integrations, broad suites, provider adapters, full ERP module, or unrelated cleanup |

If you cannot fill the table in five minutes, the packet is too broad. Split it
until a reviewer can see the owner, proof, and remaining gap without a meeting.

## New Developer Navigation Loop

Use this loop when you are dropped into an unfamiliar APG area:

1. **Find the public promise.** Read the nearest guide, `cap_spec.md`,
   example README, semantic output, or command help.
2. **Run the narrow proof.** Prefer one package test, one model command, one
   compile command, or one audit command over a broad suite.
3. **Inspect the earliest broken layer.** Parser before AST, AST before
   semantic model, semantic model before generator, package service before UI
   docs.
4. **Make the smallest executable correction.** Avoid sweeping rewrites unless
   the narrow proof demonstrates that the boundary itself is wrong.
5. **Leave a rerunnable trail.** Update the local doc, README, package spec, or
   progress log with the command and result.

This loop makes contributors useful before they understand every APG subsystem.
Depth comes from repeated verified packets, not from private context.

## Ten-Minute Start

Run these from the repository root before editing:

```bash
git status --short
uv sync
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

## First Productive Hour

Use this path when you are new to the repository and need to make a useful
change immediately.

| Minute | Action | Output |
| --- | --- | --- |
| 0-10 | Run the ten-minute baseline and inspect `git status --short` | known clean/dirty state and working CLI |
| 10-20 | Pick one lane from the ownership table below | one directory or module owns the packet |
| 20-30 | Run the focused proof for that owner before editing | baseline command output |
| 30-45 | Make one reversible change | source, package, example, test, or doc update |
| 45-55 | Rerun focused proof and inspect failure/output | evidence or next fix |
| 55-60 | Update the local handoff note, progress log, or README if readiness changed | next contributor can continue |

The first hour is successful when you can answer these five questions:

- Which layer owns the change?
- Which public name or JSON key does it affect?
- Which command proves it?
- Which broader check did you intentionally skip?
- Where should the next contributor continue?

If you cannot answer those questions, narrow the slice before editing more
files.

## Day-One Execution Board

Use this board to choose work that can become a reviewed, pushed slice on the
same day. Each row names the concrete artifact to inspect, the smallest useful
change, and the proof that makes the change credible.

| Lane | Inspect first | Useful same-day packet | Proof |
| --- | --- | --- | --- |
| Grammar | `spec/apg.g4`, one parseable example | add or tighten one terse construct and project it into AST/semantic JSON | `apg parser-golden --json`; `apg model <example> --json` |
| Semantic model | `compiler/ast_builder.py`, `compiler/semantic_model.py` | expose one missing screen, workflow, agent, stream, or capability field | focused pytest; `apg model <example> --json` |
| Generator | `compiler/code_generator.py`, generated `/tmp` output | make one existing semantic field executable in generated Python | `apg compile ... --verify`; generated `smoke_test.py` |
| Capability package | one `capabilities/<domain>/<code>/` tree | replace one generic lifecycle with domain models, service rules, API, views, and tests | package pytest; implementation audit root; publish-plan |
| Capacity example | one `examples/<nn>_<name>/` tree | make one business event parse, compile, and smoke-test | `apg model`; `apg compile --verify`; smoke test |
| Tooling/docs | `cli/`, `compiler/*audit*.py`, `docs/` | add one command proof or guide correction that removes ambiguity | relevant `apg ... --json`; docs audit |

Do not pick a packet because it is interesting. Pick it because it advances one
public APG contract and has a proof command a reviewer can rerun.

## Internal Contract Checklist

Before changing implementation code, name the contract you are changing:

| Contract type | Examples | Required handoff |
| --- | --- | --- |
| Syntax | keywords, block shape, field modifiers | grammar guide or cheat sheet when authoring changes |
| Semantic JSON | `screens`, `workflows`, `agents`, `streams`, graph summaries | fixture or example showing the key |
| Generated runtime | route handlers, manifests, smoke tests, sidecar files | compile proof and generated smoke evidence |
| Capability contract | capability ID, rule IDs, permissions, route names, theme | package tests and `cap_spec.md` |
| Capacity proof | example readiness, package owners, first event | example README and progress log when readiness changes |
| CLI/tooling | command name, JSON `format`, diagnostics | docs/tooling entry and focused command proof |

If the contract is not named, the review will drift into style discussion
instead of whether APG became more executable.

## Cross-Lane Handoff Contract

Many APG changes span more than one specialty. The fastest way to work in
parallel is to keep public names stable and hand off explicit evidence between
lanes.

| From | To | Handoff must include |
| --- | --- | --- |
| grammar author | compiler owner | source fixture, intended AST shape, invalid examples |
| compiler owner | generator owner | semantic JSON keys, reference validation, diagnostic behavior |
| generator owner | example owner | generated file names, route/helper names, smoke-test assertions |
| capability owner | capacity lead | service methods, rule IDs, route metadata, package proof |
| capacity lead | docs owner | readiness level, first event, package owners, proof commands |
| tooling owner | every lane | JSON shape, command status, known side effects |

Do not hand off vague notes such as "screen support is done." Hand off a file,
public key, route name, rule ID, and command output that the next contributor can
rerun.

## Environment Contract

APG is developed as a Python-first repository. Use the checked-in project
metadata and the local virtual environment; do not assume a global `apg`
command reflects the repository you are editing.

```bash
uv sync
./.venv/bin/apg doctor --json
./.venv/bin/apg --version
```

If `.venv` is stale or missing, recreate it with `uv sync`. If the CLI cannot
import after sync, fix packaging or import errors before changing APG behavior.

Use these command forms in docs, tests, and handoffs:

| Need | Command shape |
| --- | --- |
| Current CLI | `./.venv/bin/apg <command>` |
| Current Python | `./.venv/bin/python <script>` |
| Focused tests | `./.venv/bin/pytest -q <path>` |
| Temporary generated output | `/tmp/apg-<packet-name>` |
| Example source | `examples/<nn>_<name>/main.apg` |
| Capability root | `capabilities/<domain>/<code>/` |

Do not write generated output into a source directory unless the packet is
explicitly refreshing checked-in examples. Temporary proof belongs in `/tmp`.

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

## Ownership Decision Tree

Start from the symptom, then work at the earliest owning layer.

| If you need to... | Own the change in... | Do not start in... |
| --- | --- | --- |
| make APG source express a new idea | `spec/apg.g4`, parser fixtures, AST builder | generated output |
| make parsed syntax visible to tools | `compiler/ast_builder.py`, `compiler/semantic_model.py` | capability packages |
| enforce invalid references or policy at compile time | `compiler/semantic_analyzer.py` | generated smoke tests |
| generate executable Python for existing semantics | `compiler/code_generator.py` | example `output/` only |
| make a capability execute real domain behavior | `capabilities/<domain>/<code>/` | global docs first |
| prove a business event end to end | `examples/<nn>_<capacity>/` plus owned packages | a broad platform rewrite |
| expose developer evidence | `cli/`, `compiler/*audit*.py`, docs | hidden scripts |
| clarify contribution flow | `docs/developer_guide.md`, `docs/contributors_guide.md`, `docs/capacity_development_guide.md` | chat-only notes |

When two layers appear responsible, run the earliest proof command. For
example, if generated UI is wrong, inspect `apg model ... --json` first. If the
model is missing screen composition data, fix compiler projection before
changing generator code.

## Command Map

These commands are the fastest way to orient yourself:

| Question | Command |
| --- | --- |
| Does the CLI run? | `./.venv/bin/apg doctor --json` |
| What can APG parse and mean? | `./.venv/bin/apg model <source.apg> --json` |
| Can this source produce executable Python? | `./.venv/bin/apg compile <source.apg> --output /tmp/apg-slice --verify` |
| Does generated Python execute? | `./.venv/bin/python /tmp/apg-slice/smoke_test.py` |
| Which capabilities exist? | `./.venv/bin/apg capabilities list --json` |
| What does one capability expose? | `./.venv/bin/apg capabilities inspect capabilities/<domain>/<code> --json` |
| Which packages need implementation depth? | `./.venv/bin/apg capabilities implementation-audit --json` |
| Is one package publishable? | `./.venv/bin/apg capabilities publish-plan capabilities/<domain>/<code> --json` |
| Are docs navigable and commands known? | `./.venv/bin/apg docs audit --json` |
| Are tooling contracts healthy? | `./.venv/bin/apg tooling audit --json` |
| Is repository layout clean? | `./.venv/bin/apg hygiene audit --json` |

Prefer JSON output for evidence because it is easier to compare, cite, and
inspect in follow-up automation.

## Read A Capability Package In Fifteen Minutes

Use this routine before deepening a package:

1. Open `capability_contract.py` and write down the capability ID, provided
   services, required services, rules, routes, and theme name.
2. Open `cap_spec.md` and compare its promised behavior with the code.
3. Open `models.py`, `service.py`, `api.py`, and `views.py`; look for generic
   record names, placeholder dashboard summaries, and missing guardrails.
4. Open `test_capability_contract.py` and `tests/`; identify the existing
   contract tests and the missing lifecycle tests.
5. Run:

   ```bash
   ./.venv/bin/apg capabilities implementation-audit --root capabilities/<domain>/<code> --json
   ./.venv/bin/apg capabilities publish-plan capabilities/<domain>/<code> --json
   ```

The output of that read should be a packet like:

```text
Package:
Lifecycle to implement:
Models to replace:
Rules to enforce:
View/API surfaces:
Adapter boundaries:
Positive test:
Guardrail tests:
Proof commands:
```

This keeps capability work concrete and prevents generic scaffolding from
surviving behind a polished contract.

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

#### Grammar Change Recipe

For grammar work, make the smallest syntax change that lets an APG author say
the missing thing tersely.

1. Add a representative source fixture or example first.
2. Update `spec/apg.g4`.
3. Regenerate checked-in parser artifacts only when required for the parser
   wrapper to consume the grammar change.
4. Update AST projection with stable field names.
5. Update semantic-model projection with stable JSON keys.
6. Add validation errors in the semantic analyzer when references can be wrong.
7. Update the grammar guide and cheat sheet if the authoring surface changed.

Proof usually looks like:

```bash
./.venv/bin/apg parser-golden --json
./.venv/bin/apg model examples/<nn>_<name>/main.apg --json
./.venv/bin/pytest -q tests/test_apg_language_contract.py tests/test_semantic_analyzer.py
```

Use focused tests that cover the changed construct. Full grammar-wide sweeps
can wait until power and runtime budget allow.

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

#### Generator Change Recipe

Generated output is a contract with APG users. When changing generator behavior:

1. Inspect the semantic JSON key that will drive generation.
2. Update generator code, not checked-in output, unless the output refresh is
   part of the packet.
3. Compile one minimal example and one representative composed example.
4. Run the generated smoke tests.
5. Inspect generated `semantic_model.json` and route/helper files if the change
   affects screens, workflows, agents, streams, or capability manifests.

Keep generated applications readable. A contributor should be able to open the
generated `app.py` and understand the runtime surface without reverse
engineering the compiler.

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

#### Capability Deepening Recipe

The fastest valuable package improvement is to replace one generic storage path
with one real lifecycle.

Minimum package slice:

| File | Expected change |
| --- | --- |
| `models.py` | domain dataclasses or Pydantic models with tenant ownership |
| `<domain>_runtime.py` or similar | deterministic pure helpers for IDs, states, scores, or policy decisions |
| `service.py` | lifecycle methods, guardrails, list/query helpers, compatibility shim |
| `api.py` | dependency-light functions over the service |
| `views.py` | route metadata and UI view models |
| `cap_spec.md` | current behavior, adapter boundaries, proof commands |
| `test_capability_contract.py` and `tests/` | lifecycle test plus negative guardrails |

Keep live systems behind named adapters. A package can be domain-specific and
useful before it has a production OpenTelemetry, payment, model-provider,
database, or device integration.

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

#### New Tooling Surface Recipe

When adding or changing a CLI command:

1. Define the command's public JSON shape before wiring display output.
2. Keep side-effect-free audit commands separate from mutating commands.
3. Add a focused command test or fixture-backed audit.
4. Add the command to the relevant guide only after it runs locally.
5. Include a `--json` mode unless there is a strong reason not to.

Useful proof:

```bash
./.venv/bin/apg <command> --json
./.venv/bin/pytest -q tests/test_<command_or_area>.py
./.venv/bin/apg tooling audit --json
./.venv/bin/apg docs audit --json
```

## Debugging By Layer

When a slice fails, locate the earliest broken boundary:

| Symptom | Start here | Typical fix |
| --- | --- | --- |
| Parser rejects source | `spec/apg.g4`, parser fixtures | grammar rule or lexer ambiguity |
| Parse succeeds but field missing | `compiler/ast_builder.py` | AST visitor normalization |
| AST has data but references are unchecked | `compiler/semantic_analyzer.py` | symbol table or diagnostic rule |
| Semantic JSON omits construct | `compiler/semantic_model.py` | stable projection key |
| Semantic JSON is correct but app lacks behavior | `compiler/code_generator.py` | generated runtime helper or manifest |
| Generated app imports fail | generator output and setup imports | dependency-light import path |
| Package contract passes but behavior is generic | `capabilities/<domain>/<code>/` | domain models, service, API, views, tests |
| Docs command fails audit | `docs/` and CLI command names | stale command or missing navigation link |

Do not fix a downstream symptom by hard-coding around missing upstream meaning.

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

Use this handoff note in the progress log for substantial slices:

```text
### <date time timezone>

<Executable slice name>:

- What changed in source/runtime/package/docs.
- Public contracts preserved or intentionally migrated.
- New behavior and guardrails.
- Adapter boundaries or known gaps.

Battery-conscious verification:

- `<command>` passed with `<specific outcome>`.
- `<command>` passed with `<specific outcome>`.
- Not run: `<broader check>` because `<reason>`.
```

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
