# APG Developer Guide

This guide is for contributors changing APG itself: grammar, parser artifacts,
compiler, semantic model, generator, CLI, language-server surfaces, capability
contracts, examples, tests, and documentation.

The goal is immediate effectiveness. A new developer should be able to clone the
repository, run one reliable baseline, choose the right implementation surface,
make a vertical slice executable, prove it, document it, and commit it without
waiting for tribal knowledge.

## Immediate Effectiveness Spine

Use this spine for every new developer, every pairing session, and every
handoff. It keeps APG development focused on executable reality instead of
broad exploration.

```text
baseline evidence
  -> one work packet
  -> one owning layer
  -> one vertical change
  -> focused proof
  -> docs/progress-log handoff
  -> Lore commit and push
```

The first useful result from a new developer should be a **green slice**: one
small APG behavior that can be run, inspected, and extended. A green slice is
not a large architecture document. It is a current repository state that another
developer can prove with commands.

Minimum green-slice evidence:

| Lane | Required proof |
| --- | --- |
| Grammar or parser | parser fixture or representative `.apg` file parses; invalid shape still fails |
| Semantic model | `apg model <file> --json` exposes the intended stable key |
| Generated app | `apg compile <file> --output /tmp/<name> --verify` and generated `smoke_test.py` pass |
| Capability package | focused package tests, implementation audit, and publish-plan pass |
| Example capacity | example README names readiness, output exists when intentionally refreshed, compile proof passes |
| Documentation | `apg docs audit --json` and `git diff --check -- docs` pass |

If a change cannot be placed in one lane, split it. APG moves faster when each
commit makes one contract more trustworthy.

## Start Here

If you have just joined APG, do not begin by reading every directory. Run one
baseline, choose one owning layer, and make one verified packet.

```bash
git status --short
./.venv/bin/apg --help
./.venv/bin/apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-start --verify
./.venv/bin/python /tmp/apg-start/smoke_test.py
```

Then pick exactly one packet:

| Packet | Start in | Prove with |
| --- | --- | --- |
| Make APG syntax executable | `spec/apg.g4`, `compiler/ast_builder.py`, `compiler/semantic_analyzer.py`, `compiler/semantic_model.py`, `compiler/code_generator.py` | focused parser/semantic/generator tests and `apg compile ... --verify` |
| Deepen one capability | `capabilities/<domain>/<code>/` | package tests, `apg capabilities implementation-audit --json`, `apg capabilities publish-plan ... --json` |
| Build one capacity | `examples/<nn>_<capacity>/`, related packages, docs | `apg model`, `apg compile --verify`, smoke test, package audit when package-backed |
| Improve contributor flow | `docs/`, examples, focused fixture docs | `apg docs audit --json` and `git diff --check` |

Every packet must name:

- the outcome that becomes more executable;
- the public contract being changed or preserved;
- the files that own the change;
- the command that proves the change;
- the next gap left for another contributor.

If a packet cannot name those five things, narrow it before editing.

## First Commit Work Packet

Before editing, write this packet into your issue note, handoff note, example
README, or progress-log draft. Keep it short enough that another contributor
can review scope in one minute.

```text
Outcome:
Lane:
Owning path:
Public contract:
Representative source/package:
Focused proof:
Docs/progress-log update:
Not in this slice:
```

Examples:

```text
Outcome: workflow transitions appear in generated route metadata.
Lane: compiler generated runtime
Owning path: compiler/code_generator.py
Public contract: generated component manifest route entries
Representative source/package: examples/08_sales_workflow/main.apg
Focused proof: apg compile ... --verify; generated smoke_test.py
Docs/progress-log update: yes, compiler baseline evidence changed
Not in this slice: workflow persistence adapters
```

```text
Outcome: one materialized capability becomes domain-specific.
Lane: capability depth
Owning path: capabilities/common/grph/
Public contract: grph service methods, rules, view models, publish-plan package
Representative source/package: capabilities/common/grph/capability_contract.py
Focused proof: package pytest; implementation-audit; publish-plan
Docs/progress-log update: yes, implementation burn-down changed
Not in this slice: live graph database integration
```

## Immediate Effectiveness Contract

A new APG developer is effective when they can take one packet from evidence to
commit without asking where the work belongs. Use this contract for every
developer-facing task:

| Question | Required answer before editing |
| --- | --- |
| What becomes executable? | One APG syntax shape, semantic key, generated artifact, package behavior, example, or guide path |
| Who consumes it? | APG authors, generated app users, capability packages, CLI tooling, language-server clients, or contributors |
| Which public names are affected? | command names, JSON `format` values, semantic keys, route names, capability IDs, rule IDs, service methods |
| Where is the owner? | one primary directory or module, plus focused tests/docs |
| How is it proven? | exact command, expected result, and known verification gap |
| What is not included? | explicit non-goals so parallel contributors do not collide |

When the answer spans many owners, split the work into dependency order:
grammar, AST, semantic model, generated Python, capability package, examples,
docs. Commit each verified slice before starting the next one.

## APG Developer Mental Model

Think of APG as four connected contracts, not as one compiler file:

1. **Authoring contract:** `.apg` source should be terse, readable, and stable.
   The grammar and examples define what authors can write.
2. **Meaning contract:** the AST, semantic analyzer, and semantic model define
   what tools can understand.
3. **Execution contract:** generated Python, package-backed capabilities, rules,
   screens, workflows, agents, and Bytewax metadata define what can run or be
   inspected today.
4. **Evidence contract:** CLI JSON output, tests, audits, release evidence,
   docs, and progress-log entries define what contributors can trust.

Most defects are contract breaks between two adjacent layers. Fix the first
broken boundary instead of patching a later layer around missing earlier
meaning. For example, if a screen relationship parses but does not appear in
`apg model --json`, fix semantic projection before changing generated views.

## High-Leverage First Commits

Use one of these first-commit shapes when onboarding a developer:

| Shape | Files | Proof | Why it helps |
| --- | --- | --- | --- |
| Semantic assertion | one compiler test, one fixture or example | focused pytest, `apg model ... --json` | makes accepted syntax visible to tools |
| Generated-app smoke improvement | `compiler/code_generator.py`, focused test, example output if intentional | `apg compile ... --verify`, generated `smoke_test.py` | moves APG toward runnable apps |
| Capability-depth burn-down | one `capabilities/<domain>/<code>/` tree | package tests, `implementation-audit`, `publish-plan` | turns contracts into composable behavior |
| Example handoff | one numbered example README/source/output | focused compile and smoke test | gives capacity builders a working reference |
| Tooling evidence | one CLI/audit module and docs | focused CLI test, `apg tooling audit --json` | makes future contributors faster |
| Contributor docs | one guide and progress log | `apg docs audit --json`, `git diff --check` | removes private knowledge from the project |

Avoid a first commit that touches grammar, generator, five capability packages,
and docs in one diff. APG velocity comes from many verified packets that stack
cleanly.

## Current Development North Star

APG should rapidly produce executable Python applications from terse, readable
APG source, then let contributors harden the generated apps through reusable
capabilities, deterministic rules, screens, workflows, AI agents, Bytewax
streaming metadata, visual theming, tests, and release evidence.

That means the highest-value development work usually fits one of these lanes:

1. **Compiler baseline:** make `.apg -> generated Python app` more reliable.
2. **Semantic coverage:** expose records, relationships, screens, workflows,
   agents, capabilities, rules, themes, and streaming as stable JSON.
3. **Generated runtime behavior:** make generated apps importable,
   self-testable, smoke-testable, and inspectable.
4. **Capability depth:** replace materialized baseline packages with
   domain-specific service/API/view behavior.
5. **Capacity delivery:** build numbered examples and package-backed
   capabilities that prove APG can assemble real business applications.
6. **Contributor speed:** improve docs, examples, audits, and diagnostics that
   shorten the next contributor's path to a verified slice.

When several tasks look useful, choose the one that moves a representative APG
source file closer to a generated, runnable Python application with capability
evidence. Do not spend early effort adding alternative compiler targets or
framework-specific generation paths; APG's practical target is `python`.

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

## Source Reading Order

Read the codebase by the problem you are solving, not alphabetically.

| Problem | Read first | Then read |
| --- | --- | --- |
| Syntax accepted incorrectly or rejected incorrectly | `spec/apg.g4`, parser fixtures, `compiler/parser.py` | `compiler/ast_builder.py`, parser golden tests |
| Parsed syntax missing from tools | `compiler/ast_builder.py`, `compiler/semantic_model.py` | `compiler/semantic_analyzer.py`, graph/model fixtures |
| Generated app missing behavior | `compiler/code_generator.py`, one numbered example output | generated smoke test and release evidence helpers |
| Capability package shallow or generic | one `capabilities/<domain>/<code>/` tree | `docs/capability_standards.md`, implementation audit output |
| Contributor cannot extend a feature | nearest README/guide/progress-log entry | docs audit, tooling specification |

Stop reading when you can name the owning path, public contract, and focused
proof. Extra reading is useful only when it changes the implementation decision.

## Immediate Operating Model

Use this model for the first day in the codebase:

1. **Orient on the executable path.** Run `./.venv/bin/apg --help`, inspect
   `cli/main.py`, then compile one numbered example. Do not start by reading
   every historical file.
2. **Pick one vertical slice.** A useful slice crosses from APG source to a
   generated app, CLI report, capability package, fixture, or documentation
   proof.
3. **Find the owning layer.** Grammar edits start in `spec/apg.g4`; generated
   runtime edits start in `compiler/code_generator.py`; capability package work
   starts under `capabilities/<domain>/<code>/`; CLI work starts in `cli/`.
4. **Preserve the public contract.** Existing JSON `format` values, generated
   files, route names, semantic model keys, and CLI command names are public
   unless a migration is intentionally documented.
5. **Prove the result narrowly.** Use focused tests and CLI audits first. Run
   broader checks only when shared compiler/tooling contracts changed.
6. **Record the evidence.** Update `docs/progress_log.md` with the actual
   commands and outcomes.
7. **Commit the verified slice.** Stage only the files you changed for the
   slice and use the Lore commit protocol.

The fastest path to understanding APG is to trace one example:

```bash
./.venv/bin/apg model examples/20_enterprise_erp_platform/main.apg --json
./.venv/bin/apg graph-suite examples/20_enterprise_erp_platform/main.apg --json
./.venv/bin/apg compile examples/20_enterprise_erp_platform/main.apg --output /tmp/apg-erp --verify
./.venv/bin/python /tmp/apg-erp/smoke_test.py
```

Then compare the source in `examples/20_enterprise_erp_platform/main.apg` with
the generated `semantic_model.json`, `app.py`, `apg_capabilities.py`, and
`apg_application.py`.

## Seven-Minute Effective Start

Use this path when a new developer joins a session and needs to make a useful
choice immediately.

1. Check the shared worktree:

   ```bash
   git status --short
   ```

2. Prove the CLI and one generated app:

   ```bash
   ./.venv/bin/apg --help
   ./.venv/bin/apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-seven --verify
   ./.venv/bin/python /tmp/apg-seven/smoke_test.py
   ```

3. Pick the next work lane from evidence, not from memory:

   ```bash
   ./.venv/bin/apg capabilities implementation-audit --json
   ./.venv/bin/apg docs audit --json
   ```

4. Claim one packet using this shape:

   ```text
   Outcome:
   Owning path:
   Public contract:
   Focused proof:
   Docs/progress-log update:
   Not doing:
   ```

5. Edit only the owning path plus its focused tests/docs. Verify, update
   `docs/progress_log.md`, stage only the packet, commit, and push.

The point is not to learn every subsystem first. The point is to produce one
small verified improvement while learning the exact layer that owns it.

## Four-Hour Onboarding Plan

Use this plan when a contributor needs to become productive immediately.

Hour 1: prove the installed toolchain.

```bash
git status --short
./.venv/bin/apg version
./.venv/bin/apg doctor --json
./.venv/bin/apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-onboard --verify
./.venv/bin/python /tmp/apg-onboard/smoke_test.py
```

Hour 2: trace one rich example from source to generated behavior.

```bash
./.venv/bin/apg model examples/20_enterprise_erp_platform/main.apg --json
./.venv/bin/apg graph-suite examples/20_enterprise_erp_platform/main.apg --json
./.venv/bin/apg release examples/20_enterprise_erp_platform/main.apg --json
```

Hour 3: inspect one capability package and its implementation depth.

```bash
./.venv/bin/apg capabilities inspect agnt --json
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities publish-plan capabilities/common/agnt --json
```

Hour 4: take one small work packet. Prefer a docs, example, fixture, CLI field,
or single-package implementation-depth slice. The result should be small enough
to verify and commit the same day.

```bash
git diff --check -- <changed-files>
./.venv/bin/apg docs audit --json
git add <only-your-files>
git diff --cached --check
git commit
git push
```

## One-Day Developer Packet

By the end of the first day, a new APG developer should have produced one
verified packet. A packet is deliberately small, but it must cross a useful
boundary instead of leaving isolated edits behind.

```text
Packet name:
Outcome made more executable:
Primary owning path:
Public contract affected:
Representative APG source or package:
Focused verification:
Docs/progress-log update:
Commit:
Known remaining gap:
```

Good first-day packets:

| Packet | Why it matters | Typical proof |
| --- | --- | --- |
| Clarify one contributor workflow | Speeds every later contributor | `apg docs audit --json` |
| Convert one materialized package into domain behavior | Turns a contract into reusable runtime capability | package tests, `implementation-audit`, `publish-plan` |
| Add one semantic-model field consumed by existing tooling | Reduces drift between syntax and tools | semantic fixture or `apg model ... --json` |
| Make one generated route/helper more real | Moves `.apg` source toward executable apps | compile `--verify`, smoke test |
| Improve one numbered example and output | Proves a capacity in source and generated artifacts | focused compile or `apg baseline examples --json` |

The packet should be reviewable without private context. If a reviewer cannot
tell what executable state changed, the packet is too vague or too broad.

## Developer Decision Tree

Use this tree when choosing where to edit:

1. **Does APG source fail to parse?** Start with `spec/apg.g4`, parser golden
   fixtures, and AST builder behavior.
2. **Does source parse but tools cannot see the meaning?** Start with
   `compiler/semantic_analyzer.py` and `compiler/semantic_model.py`.
3. **Does the semantic model contain the data but generated apps do not expose
   it?** Start with `compiler/code_generator.py` and a representative example.
4. **Does a capability contract exist but package behavior is shallow?** Start
   in one `capabilities/<domain>/<code>/` tree.
5. **Does behavior exist but contributors cannot find or extend it?** Update
   the closest guide, example README, and `docs/progress_log.md`.
6. **Does the work require several of the above?** Land it in dependency order:
   parse, semantic model, generated behavior, package behavior, docs.

Do not begin with a broad repository rewrite. Begin where the evidence fails.

## Gap Triage For Core Developers

When you are deciding what APG needs next, prefer current evidence over the
oldest or loudest plan. Run the smallest set of audits that describes the
surface you intend to improve, then pick one packet.

```bash
./.venv/bin/apg baseline examples --json
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg tooling audit --json
./.venv/bin/apg docs audit --json
```

Interpret the results this way:

| Finding | Meaning | Best next packet |
| --- | --- | --- |
| numbered example fails baseline | author-facing executable path is broken | fix parser, semantic model, generator, or checked output for that example |
| capability is `contract_only` | APG can name it but cannot package it | add or materialize package artifacts and focused tests |
| capability is `materialized_baseline` | APG can publish shape but not domain behavior | replace generic records/services/API/views with domain-specific runtime state |
| capability is `mixed` | useful code exists but baseline surfaces remain | preserve custom behavior and remove generated placeholders behind tests |
| tooling audit fails | contributor evidence path is unreliable | fix the failing command, fixture, JSON contract, or guide example |
| docs audit fails | contributor navigation or command examples drifted | update links, command names, or required docs |

The best next task is usually the first failing or shallow executable contract
that can be fixed in one owning directory. If a proposed change spans grammar,
semantic model, generator, capability behavior, examples, and docs, split it in
that dependency order and commit each verified slice separately.

## Executable Reality Review

Before changing a feature, classify its current state. This prevents edits that
improve prose but leave APG no closer to generated applications.

| State | Evidence | Developer action |
| --- | --- | --- |
| idea | docs only | create or update an APG example packet |
| parseable | parser accepts source | add semantic-model exposure |
| semantic | `apg model ... --json` contains it | expose generated runtime behavior |
| generated | `apg compile --verify` emits it | add smoke/self-test or route evidence |
| package-backed | capability contract and artifacts exist | deepen service/API/view behavior |
| operable | tests, audits, publish-plan pass | document extension points and next gap |

Do not claim a feature as implemented at a higher state than the evidence
proves. If the feature is metadata-only for now, say so in the docs and leave a
clear next action.

## First-Day Execution Checklist

Use this checklist when onboarding yourself or another contributor. It is
ordered to create useful context before broad reading.

1. Confirm the worktree:

   ```bash
   git status --short
   ```

2. Confirm the installed APG command:

   ```bash
   ./.venv/bin/apg version
   ./.venv/bin/apg --help
   ```

3. Compile one small application:

   ```bash
   ./.venv/bin/apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-first --verify
   ./.venv/bin/python /tmp/apg-first/smoke_test.py
   ```

4. Inspect one rich application:

   ```bash
   ./.venv/bin/apg model examples/20_enterprise_erp_platform/main.apg --json
   ./.venv/bin/apg graph-suite examples/20_enterprise_erp_platform/main.apg --json
   ```

5. Inspect capabilities:

   ```bash
   ./.venv/bin/apg capabilities validate-contracts --json
   ./.venv/bin/apg capabilities audit --json
   ```

6. Pick one narrow lane from the contributor task table and make a vertical
   slice.

Do not spend the first day reading every capability package. APG has enough
historical material that broad reading can become unproductive. Read the slice
you intend to change, prove one behavior, and leave the next contributor with
better evidence than you found.

## How To Move APG Forward

Every useful APG change should increase one of these forms of executable
reality:

| Improvement | What it means | Best proof |
| --- | --- | --- |
| Parseability | Authors can write accepted APG source | parser golden or example baseline |
| Semantics | Tools can understand the source as stable JSON | `apg model ... --json` |
| Executability | The compiler emits a runnable Python app | `apg compile ... --verify` and smoke test |
| Composability | Capabilities, screens, workflows, agents, and apps connect cleanly | graph/model/capability audits |
| Operability | Package-backed behavior can be inspected and self-tested | capability audit and publish-plan |
| Contributor speed | The next person can extend the slice without guessing | docs, README, progress log |

If a change only adds more files without increasing one of those outcomes, it is
probably not the next best change.

## Turning A Requirement Into A Patch

Most APG requests arrive as product language: "add agent composition", "make ERP
screens composable", "support capacity X", or "make this executable". Convert
that into a patch with this sequence:

1. Name the observable outcome in one sentence.
2. Identify the APG source construct that should express it.
3. Identify the semantic-model key that should carry it.
4. Identify the generated app surface that should expose it.
5. Identify the capability package or example that proves it.
6. Choose the narrowest verification command that proves the claim.
7. Write the docs/progress-log entry using actual command output.

Example:

```text
Requirement: AI agents should be first-class.
Source construct: agent and agent team declarations.
Semantic model: agents, teams, handoffs, runtime assignments.
Generated app: ai_agents.py plus component manifest entries.
Capability package: capabilities/common/agnt.
Proof: focused AGNT tests, implementation-audit, publish-plan, compile example.
```

This prevents a common failure mode: adding syntax or docs without any generated
behavior, semantic evidence, or package-backed runtime path.

## Common Implementation Recipes

Use these recipes to avoid losing time on repo navigation.

Add or adjust APG syntax:

1. Edit `spec/apg.g4`.
2. Regenerate parser artifacts only when the generated files need to change.
3. Update `compiler/ast_builder.py`.
4. Update `compiler/semantic_analyzer.py` and `compiler/semantic_model.py`.
5. Update `compiler/code_generator.py` if the syntax should execute.
6. Add fixtures or example source.
7. Run parser, semantic, and compile checks for the touched feature.

Add a generated runtime surface:

1. Find the semantic model key consumed by the generator.
2. Update `compiler/code_generator.py`.
3. Compile a representative numbered example to `/tmp`.
4. Run the generated `smoke_test.py`.
5. Refresh checked-in example outputs only when the compiler output contract
   intentionally changed.

Add a capability package:

1. Scaffold or create `capabilities/<domain>/<code>/`.
2. Keep `capability_contract.py`, service code, views, `app.py`,
   `semantic_model.json`, package manifest, release report, and tests aligned.
3. Use `apg capabilities materialize-packages --capability <id> --json` when a
   checked-in contract is valid but package artifacts are missing.
4. Run `apg capabilities implementation-audit --json` and confirm the package
   is not still a materialized baseline before claiming domain behavior.
5. Validate contracts, run focused package tests, and run publish-plan.
6. Add or update the APG example that composes the capability.

Add a new capacity:

1. Start with a capacity README blueprint.
2. Add a parseable APG example.
3. Add or update package-backed capability contracts.
4. Compile and smoke-test.
5. Document extension points and known gaps.

Burn down one materialized capability package:

1. Inspect `capability_contract.py`, `cap_spec.md`, `models.py`, `service.py`,
   `api.py`, `views.py`, `app.py`, and package-local tests.
2. Identify the business service the package should actually provide.
3. Replace generic materialized record/service/API/view placeholders with
   domain data structures and deterministic service behavior.
4. Keep behavior dependency-light and in-memory unless the capability already
   has a real integration boundary.
5. Add rule-aware methods that call the existing capability rule engine rather
   than duplicating rule evaluation.
6. Update views/API helpers so the generated app can inspect meaningful state.
7. Update `cap_spec.md` to describe current runtime behavior and known gaps.
8. Run `py_compile`, focused package tests, marker search, implementation
   audit, publish-plan, strict package audit, and `git diff --check`.

Capability package evidence usually looks like:

```bash
./.venv/bin/python -m py_compile capabilities/common/<code>/*.py
./.venv/bin/python -m pytest -q capabilities/common/<code>/test_capability_contract.py capabilities/common/<code>/tests
rg -n "This package materializes|Tenant-scoped dependency-light capability record|Dependency-light service backed|materialized APG capability package" capabilities/common/<code>
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities publish-plan capabilities/common/<code> --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```

## How To Read The Codebase

Read APG from contracts outward:

- **Language contract:** `spec/apg.g4`, parser golden fixtures, language docs.
- **Meaning contract:** AST builder, semantic analyzer, semantic model.
- **Execution contract:** code generator, generated app routes/helpers, smoke
  tests.
- **Composition contract:** capability registry, application composition,
  screen composition, AI agent composition, workflow surfaces.
- **Evidence contract:** CLI JSON reports, tooling audit, compiler baseline,
  docs audit, release/evidence commands.

Do not treat old planning documents, archived grammar drafts, or aspirational
reports as current behavior. Current behavior is what the source, generated
artifacts, CLI reports, and focused tests prove.

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

## New Contributor Task Lanes

Choose one lane and stay inside it until the slice is verified:

| Lane | Good first outcome | Main files | Proof |
| --- | --- | --- | --- |
| Docs | Clarify one current workflow and link it from the docs index | `docs/*.md` | `./.venv/bin/apg docs audit --json` |
| Example | Improve or add a parseable numbered example | `examples/<nn>_*/main.apg`, `README.md`, `output/` | `./.venv/bin/apg baseline examples --json` |
| Generator | Make generated apps expose one more real surface | `compiler/code_generator.py` | compile `--verify`, smoke test, focused generator tests |
| Semantic | Add one stable semantic-model projection | `compiler/ast_builder.py`, `semantic_analyzer.py`, `semantic_model.py` | semantic model fixture tests |
| Capability | Build one package-backed capability contract | `capabilities/<domain>/<code>/` | capability validation and package tests |
| CLI | Add one JSON-producing command or report field | `cli/`, `compiler/<surface>.py` | focused CLI test and tooling audit |
| Capacity | Compose records, rules, screens, workflows, agents, and capabilities into an executable ability | example, capability package, docs | compile, smoke, capability validation |

If a task crosses three or more lanes, split it into smaller commits with a
clear dependency order. For example, add semantic model support first, then
generator behavior, then examples/docs.

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
./.venv/bin/apg capabilities audit --json
./.venv/bin/apg capabilities materialize-packages --dry-run --json
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

For a new capacity, start with this minimum file set:

```text
examples/<nn>_<capacity_name>/
  main.apg
  README.md
  output/

capabilities/<domain>/<code>/
  cap_spec.md
  capability_contract.py
  models.py
  service.py
  api.py
  views.py
  app.py
  tests/

docs/<capacity_name>.md
```

Do not build a large ERP capacity as only Python package code. It should be
visible in APG source, the semantic model, generated app output, capability
contracts, and docs.

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
| Compiler baseline | `./.venv/bin/apg baseline examples --json` for numbered examples, generated source hygiene, checked-in output sync, direct checked-output self-test/smoke-test execution, checked-output HTTP contract route probes, domain HTTP probes for records/workflows/capabilities, graph, model, and release agreement; use `./.venv/bin/apg baseline examples --refresh-outputs --json` only when intentionally regenerating example outputs |
| Release/package | `./.venv/bin/apg package path/to/app.apg --catalog /tmp/apg-capability-catalog.json --target web --out /tmp/apg-package --json` |
| Evidence bundle | `./.venv/bin/apg evidence path/to/app.apg --catalog /tmp/apg-capability-catalog.json --target web --out /tmp/apg-evidence --json` |
| Capabilities | `./.venv/bin/apg capabilities validate-contracts --json` and `./.venv/bin/apg capabilities audit --json` |
| Capability package closure | `./.venv/bin/apg capabilities materialize-packages --dry-run --json`, `./.venv/bin/apg capabilities materialize-packages --json`, and `./.venv/bin/apg capabilities audit --strict-package-artifacts --json` |
| Capability implementation depth | `./.venv/bin/apg capabilities implementation-audit --json` to find remaining materialized baselines; use `--strict` when a capacity is supposed to be domain-implemented |
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
