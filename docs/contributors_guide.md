# APG Contributors Guide

This guide is for anyone making a contribution to APG. It explains how to
become useful quickly, choose a safe work packet, verify it, document it, and
leave the repository easier for the next contributor.

The contribution rule is simple: make one current APG state more executable,
prove it with commands, document the evidence, and commit only that slice.

## New Contributor Operating Contract

You can contribute effectively without understanding the entire platform. Your
first job is to make one public APG contract easier to execute, verify, or
extend.

Use this contract for every contribution:

```text
I am improving:
The public contract is:
The owner is:
The proof command is:
The handoff location is:
The next contributor should:
I will not touch:
```

Good public contracts are concrete:

| Area | Concrete contract |
| --- | --- |
| Language | grammar construct, keyword, diagnostic, semantic JSON key |
| Compiler | AST field, semantic-model field, generated file, route helper |
| Capability | capability ID, service method, rule ID, route, theme token |
| Capacity | first event, example path, package owner, readiness level |
| Tooling | command name, JSON format, audit finding, fixture catalog |
| Documentation | guide link, command block, proof expectation, next-step note |

If you cannot name the public contract, make the contribution a clarification:
update the local README, package spec, or guide so the next contributor can
name it.

## The Fastest Useful Contribution

The fastest useful contribution is not the largest feature. It is the smallest
verified improvement that another contributor can build on without asking for
context.

Use this decision table after the baseline commands:

| You see | Do this | Do not do this |
| --- | --- | --- |
| a broken docs command | fix that command and rerun docs audit | rewrite the whole guide |
| a materialized capability package | implement one domain lifecycle and guardrails | add live integrations first |
| a parseable example with weak README evidence | update the README with current model/compile/smoke proof | refresh unrelated outputs |
| semantic JSON has a field but generated app ignores it | update generator and smoke proof | patch generated output only |
| a capacity idea with no first event | write the event blueprint and proof path | create a broad module inventory |

Your first contribution should leave these facts visible in checked-in files:

```text
Changed contract:
Owning file or package:
Command that proves it:
Known gap:
Next contributor action:
```

If those facts only exist in chat, the contribution is not yet ready.

## Immediate Contributor Runbook

Use this runbook for the first contribution, even if you do not yet understand
the whole APG platform.

1. Inspect local state:

   ```bash
   git status --short
   ```

2. Prove the contributor baseline:

   ```bash
   uv sync
   ./.venv/bin/apg docs audit --json
   ./.venv/bin/apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-contributor-baseline --verify
   ./.venv/bin/python /tmp/apg-contributor-baseline/smoke_test.py
   ```

3. Choose one contribution class:

   | Class | Edit scope | Proof |
   | --- | --- | --- |
   | docs correction | one guide or README | `./.venv/bin/apg docs audit --json` |
   | example readiness | one `examples/<nn>_*/` directory | model, compile, smoke test |
   | capability deepening | one `capabilities/<domain>/<code>/` package | package pytest, implementation audit, publish-plan |
   | compiler exposure | one compiler layer plus fixture/example | focused test and `apg model` |
   | tooling reliability | one CLI/audit surface | relevant `apg ... --json` command |

4. Write the work packet before editing:

   ```text
   I will improve:
   I will edit:
   I will prove it with:
   I will update:
   I will not touch:
   ```

5. Make the change, rerun the proof, stage only the packet, and commit with
   Lore trailers.

This runbook deliberately avoids broad repository reading. Your first useful
job is to make one checked-in fact easier to execute, verify, or continue.

## First-Day Mission Board

Pick one mission. Finish and commit it before starting another.

| Mission | Edit | Commands to run | Handoff |
| --- | --- | --- | --- |
| Make onboarding clearer | one guide or docs index entry | `./.venv/bin/apg docs audit --json`; `git diff --check -- docs` | guide paragraph names current command and next doc |
| Prove one example | one `examples/<nn>_*/README.md` and optional output refresh | `apg model`; `apg compile --verify`; generated smoke test | README readiness level and next slice |
| Deepen one capability guardrail | one package service/test/spec | package pytest; implementation audit root; publish-plan | `cap_spec.md` names behavior and non-goals |
| Expose one compiler fact | one parser/AST/semantic/generator owner plus fixture | focused pytest; `apg model ... --json` | guide or fixture names stable semantic key |
| Seed one capacity | one example directory plus README | model, compile, smoke | README names event, package owners, proof, next gap |

The mission is complete when the handoff lets another contributor continue
without asking what you meant.

## Before You Edit Checklist

Write these answers before changing files:

```text
Mission:
Owner:
Public names:
Unrelated dirty files I will leave alone:
Baseline command:
Proof after:
Docs or progress-log update:
Commit boundary:
```

If `Owner` names more than one major layer, split the work. If `Public names`
is empty, you are probably doing hidden cleanup rather than progressing APG.

## Contributor Effectiveness Pact

APG welcomes contributions that are small, executable, and easy to continue.
You do not need to know the whole platform to help. You do need to make one
claim concrete enough that another contributor can rerun it.

Before starting, write this in your working note or commit prep:

```text
I am improving:
The owner is:
The public contract is:
The proof command is:
I will not change:
The next contributor should:
```

Examples of good contracts:

| Work | Public contract |
| --- | --- |
| documentation | guide link, command spelling, onboarding path, proof expectation |
| example | event name, APG source path, readiness level, generated-output status |
| compiler | grammar construct, semantic JSON key, diagnostic name |
| generated runtime | generated file, route/helper name, smoke-test assertion |
| capability | capability ID, service method, rule ID, route metadata, theme token |
| capacity | first event, package owner, screen route, workflow state, agent boundary |

If the public contract is still unclear, contribute by clarifying the local
README, package spec, or guide before changing behavior.

## First 30 Minutes

Run this baseline from the repository root:

```bash
git status --short
uv sync
./.venv/bin/apg compile examples/01_minimal_customer_records/main.apg --output /tmp/apg-contributor-start --verify
./.venv/bin/python /tmp/apg-contributor-start/smoke_test.py
./.venv/bin/apg docs audit --json
```

Then read only what matches your intended lane:

| Lane | Read first | Useful first proof |
| --- | --- | --- |
| Docs | `docs/README.md`, this guide | `apg docs audit --json` |
| Examples | one `examples/<nn>_*/README.md` and `main.apg` | `apg compile ... --verify` |
| Compiler | [Developer Guide](./developer_guide.md), one focused compiler test | `apg model ... --json` |
| Capability | [Capability Standards](./capability_standards.md), one package tree | package pytest, implementation audit, publish-plan |
| Capacity | [Capacity Development Guide](./capacity_development_guide.md), one example | model, compile, smoke test |
| Tooling | [Tooling Specification](./tooling.md), related CLI module | relevant `apg ... --json` command |

Do not try to understand every subsystem before contributing. Understand one
owner, one public contract, and one proof command.

## Zero-To-Commit Path

Use this path for your first APG contribution. It is intentionally narrow so a
new contributor can finish a real packet without needing private project
history.

1. Run `git status --short` and write down unrelated dirty files you will not
   stage.
2. Pick one packet from the lane table above.
3. Run the packet's focused proof before editing so you know the starting
   state.
4. Edit only files owned by that packet.
5. Rerun the focused proof and inspect the output.
6. Update the local handoff surface: guide, example README, package spec, or
   progress log.
7. Stage exact files, run `git diff --cached --check`, commit with the Lore
   protocol, and push.

Your first contribution does not need to be large. It needs to be executable,
reviewable, and honest about what was and was not tested.

## First Packet Menu

Choose one of these if you are unsure where to start.

| Packet | Edit | Success condition |
| --- | --- | --- |
| Make a guide more runnable | one docs file | `apg docs audit --json` passes and the guide has current commands |
| Refresh one example README | one example README | README names event, readiness, proof, and next gap |
| Add one capability guardrail | one package test and service rule | negative test fails before or is clearly covered after implementation |
| Remove one baseline marker | one capability package | implementation audit root reports `domain_specific` |
| Expose one semantic field | compiler layer plus focused test | `apg model ... --json` shows the new stable key |
| Prove one generated behavior | generator plus smoke assertion | generated app imports and smoke-tests |

When a packet touches more than one row, split it unless the second row is
directly required to prove the first.

## Twenty-Minute Triage

Use this when you have just arrived and want a useful task without asking for
private context.

1. Run:

   ```bash
   git status --short
   ./.venv/bin/apg docs audit --json
   ./.venv/bin/apg capabilities implementation-audit --json
   ```

2. Ignore unrelated dirty files. They may belong to another contributor or a
   local tool.
3. Choose one finding that has a local owner, such as one docs file, one
   example directory, one capability package, or one compiler module.
4. Write the packet in this form:

   ```text
   I will make <public contract> better by editing <owner>.
   I will prove it with <command>.
   I will not touch <unrelated dirty files or broader systems>.
   ```

5. Run the proof once before editing. If it already passes, either improve the
   next concrete gap or choose a different packet.

This triage keeps first-time work useful even when the full APG roadmap is
large. Your job is not to absorb the whole roadmap. Your job is to turn one
visible finding into one verified improvement.

## If You Only Have One Hour

Pick one of these paths and finish it completely.

### Documentation Path

1. Find one stale command, missing link, or unclear guide paragraph.
2. Edit only the relevant docs file.
3. Run:

   ```bash
   ./.venv/bin/apg docs audit --json
   git diff --check -- docs
   ```

4. Commit the docs-only packet.

### Capability Path

1. Run `./.venv/bin/apg capabilities implementation-audit --json`.
2. Pick one package and inspect its contract, service, spec, and tests.
3. Add one guardrail test or one small domain helper.
4. Run the focused package pytest and root implementation audit.
5. Update the package `cap_spec.md` if behavior changed.

### Example Path

1. Choose one numbered example.
2. Compile it to `/tmp`, not into the source tree.
3. Run the generated smoke test.
4. Update the README with current readiness and the next smallest gap.

### Compiler Path

1. Choose one fixture or example with one missing semantic field.
2. Run `./.venv/bin/apg model <source.apg> --json`.
3. Change the earliest compiler layer that owns the missing field.
4. Rerun the focused compiler proof.

Do not open multiple paths unless the first one is proven and committed.

## Contributor Operating Loop

Use the same loop for every contribution, whether it is grammar, compiler,
capability, example, tooling, or documentation work.

1. Pull or inspect the current worktree.
2. Identify unrelated local changes and leave them alone.
3. Choose one packet with one owner.
4. Run the smallest baseline that proves the owner currently works or fails.
5. Make the change.
6. Run focused proof.
7. Update docs, README, cap spec, or progress log if readiness changed.
8. Stage exact files.
9. Commit with the Lore protocol and push the verified slice.

If a packet grows into multiple owners, split it. APG moves faster when each
commit has a clear owner and a clear proof command.

## Pick A Work Packet

A work packet is the smallest coherent change that leaves APG better than it
was before.

Use this template before editing:

```text
Outcome:
Owner:
Lane:
Files expected:
Public contract:
Focused proof:
Docs/progress-log update:
Known non-goals:
```

Good packet:

```text
Outcome: the IOTD package enforces telemetry and command guardrails.
Owner: capabilities/common/iotd/
Lane: capability implementation depth
Files expected: models.py, service.py, api.py, views.py, cap_spec.md, tests
Public contract: iotd capability ID, rule IDs, route metadata, publish-plan shape
Focused proof: package pytest; implementation-audit --root; publish-plan
Docs/progress-log update: yes, implementation-depth burn-down changes
Known non-goals: live MQTT gateway or firmware delivery integration
```

Poor packet:

```text
Improve APG capabilities, rewrite the compiler, refresh examples, and fix docs.
```

Split broad work into dependency order: grammar, AST, semantic model, generated
runtime, capability package, example, docs.

## Contribution Decision Tree

Use this before choosing files:

| Question | Yes | No |
| --- | --- | --- |
| Is a command failing for everyone? | fix that command or its earliest owner first | continue |
| Is a capability still generic? | deepen one package lifecycle | continue |
| Does source parse but generated Python lacks behavior? | inspect semantic JSON, then generator | continue |
| Does the capacity lack a first event? | write or narrow the capacity blueprint | continue |
| Is the problem only stale documentation? | docs-only packet | choose a code owner |
| Does the change need live credentials? | model an adapter boundary and add local deterministic proof first | implement directly |

The highest-value packet is usually the one that turns an existing promise into
an executable local proof without introducing live-provider dependency.

## Same-Day Contribution Choices

If you are new, pick one of these:

| Timebox | Contribution | Edit surface | Proof |
| --- | --- | --- | --- |
| 30 minutes | Fix one stale docs command or link | one guide | docs audit, diff check |
| 60 minutes | Clarify one example README from current output | one example README | compile and smoke test |
| 90 minutes | Add one guardrail test to a capability | one package test | package pytest |
| Half day | Replace one generic package path with domain behavior | one capability package | pytest, implementation audit, publish-plan |
| One day | Expose one semantic-model field | compiler layer and fixture | focused test, `apg model` |

Prefer work that can be reviewed and verified the same day. A small verified
improvement is more valuable than a large branch without proof.

## How To Find Useful Work

Use the repository's own audits instead of guessing:

```bash
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
./.venv/bin/apg tooling audit --json
./.venv/bin/apg hygiene audit --json
```

Pick work in this order:

1. Broken checks that block other contributors.
2. Capability packages still classified as materialized baseline, mixed, or
   contract-only.
3. Examples whose source, README, output, or proof commands have drifted.
4. Generated runtime behavior already visible in `apg model` but not executable
   in generated Python.
5. New capacity slices with one clear first event and proof path.

Avoid work whose first proof requires live credentials, a new dependency, or a
full platform rewrite.

## How To Burn Down Capability Work

Many APG capabilities already have contracts, routes, rules, theme metadata,
and package shape. The useful contribution is often implementation depth:
turning a generic package into domain behavior.

Capability burn-down flow:

1. Run `./.venv/bin/apg capabilities implementation-audit --json`.
2. Pick the next package classified as materialized baseline, mixed, or
   contract-only.
3. Read its `capability_contract.py`, `cap_spec.md`, `__init__.py`, and tests.
4. Implement one lifecycle that matches the contract.
5. Add one positive lifecycle test and negative guardrail tests.
6. Run package proof:

   ```bash
   ./.venv/bin/pytest -q capabilities/<domain>/<code>/test_capability_contract.py capabilities/<domain>/<code>/tests
   ./.venv/bin/apg capabilities implementation-audit --root capabilities/<domain>/<code> --json
   ./.venv/bin/apg capabilities publish-plan capabilities/<domain>/<code> --json
   ```

7. Update `cap_spec.md` with current runtime behavior and adapter boundaries.
8. Update `docs/progress_log.md` when the global audit counts or next target
   changed.

Do not wire live providers first. Local deterministic package behavior gives
the compiler, examples, UI manifests, and publish tooling something stable to
compose.

## Package Deepening Checklist

A package-deepening contribution is effective when it replaces generic shape
with domain behavior while preserving the public contract.

Before editing:

- name the capability ID and route names;
- list the deterministic rules from `capability_contract.py`;
- identify the first lifecycle event;
- decide which live integrations stay behind adapters;
- choose one positive test and at least two guardrail tests.

During implementation:

- replace generic record models with domain records;
- keep tenant and owner checks in the service layer;
- expose dependency-light API helpers;
- return view models that match route/theme metadata;
- keep compatibility shims only when existing tests or tools rely on them;
- update `cap_spec.md` with current behavior and non-goals.

Before commit:

```bash
./.venv/bin/pytest -q capabilities/<domain>/<code>/test_capability_contract.py capabilities/<domain>/<code>/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/<domain>/<code> --json
./.venv/bin/apg capabilities publish-plan capabilities/<domain>/<code> --json
git diff --check -- capabilities/<domain>/<code>
```

## How To Start A New Capacity

A new capacity is not a folder full of aspirations. It starts as one
executable event.

1. Name the event, for example `purchase request submitted` or
   `device telemetry received`.
2. Create or choose one example directory.
3. Write `main.apg` with the records, rules, screens, workflows, agents,
   streams, and capabilities needed for that event.
4. Prove semantics and generation:

   ```bash
   ./.venv/bin/apg model examples/<nn>_<capacity>/main.apg --json
   ./.venv/bin/apg compile examples/<nn>_<capacity>/main.apg --output /tmp/apg-capacity --verify
   ./.venv/bin/python /tmp/apg-capacity/smoke_test.py
   ```

5. Update the example README with readiness level, proof commands, package
   owners, and the next event.
6. Deepen one package when the capacity needs durable behavior.

Use the [Capacity Development Guide](./capacity_development_guide.md) for the
full blueprint.

## Capacity Contribution Contract

When your contribution adds or extends a capacity, reviewers should be able to
find these facts without reading chat history:

| Fact | Where it should appear |
| --- | --- |
| first event | example README and `main.apg` |
| records and relationships | APG source, semantic JSON, and README |
| rule IDs and decisions | APG source, package contract, tests, or README |
| screen route and composition | APG source, generated manifest, and README |
| workflow state movement | APG source and generated smoke evidence when supported |
| AI agent boundary | APG source or README, with runtime/provider behind configuration |
| Bytewax stream boundary | APG source or README, with adapter boundary named |
| durable package owner | capability package path and package `cap_spec.md` |
| proof commands | README, progress log when readiness changed, and commit message |
| next slice | README or progress log |

If one of these facts is unknown, make it the contribution. Naming the event,
route, owner, or proof command is often the work that unlocks parallel
implementation.

## Contribution Standards

Every contribution should satisfy these standards:

| Standard | What it means |
| --- | --- |
| One owner | one primary directory or module owns the change |
| Executable outcome | source parses, generated app runs, package behavior executes, or docs audit proves navigation |
| Public contract named | commands, JSON keys, routes, capability IDs, rules, services, screens, workflows, or agents are explicit |
| Focused proof | one or two commands prove the changed layer |
| Honest boundary | live integrations and untested paths are named |
| Handoff trail | README, guide, cap spec, or progress log explains evidence and next gap |

If a contribution does not meet these standards, narrow it before editing more
files.

## Working In Parallel

Parallel contribution is safe when each contributor owns a different public
surface and the handoff names are stable.

Safe parallel packets:

- one contributor deepens `capabilities/common/<code>/` while another updates
  the matching capacity README;
- one contributor adds a compiler semantic key while another prepares a
  generated-runtime test that consumes that key;
- one contributor fixes documentation commands while another runs package
  implementation burn-down;
- different contributors deepen different capability packages.

Coordinate before changing:

- `spec/apg.g4`;
- semantic model top-level keys;
- generated file names;
- route names used by examples or package specs;
- capability IDs, service names, rule IDs, or theme names;
- checked-in example `output/` directories.

When in doubt, preserve public names and add compatibility shims until the
owning docs, tests, and examples move in the same verified slice.

## Working With The Shared Worktree

The repository may already contain unrelated local changes. Do not revert or
stage them.

Before staging:

```bash
git status --short
git diff -- <your-files>
```

Stage exact files:

```bash
git add <file-1> <file-2> <file-3>
git diff --cached --name-only
git diff --cached --check
```

Unrelated examples of files you should not stage unless they belong to your
packet:

- local agent state such as `.omx/`, `.claude/`, or `.simple-task-master/`;
- copied reference documents not intentionally added to `docs/`;
- generated outputs from unrelated examples;
- another contributor's dirty package files;
- temporary uploads, caches, or editor artifacts.

## Testing Expectations

Use focused tests first, especially on battery.

| Change | Run |
| --- | --- |
| Documentation | `./.venv/bin/apg docs audit --json`; `git diff --check -- docs` |
| Capability package | focused package pytest; implementation audit for that root; publish-plan |
| Capability contract registry | strict package artifact audit |
| APG source or example | `apg model`; `apg compile --verify`; generated smoke test |
| Compiler or generator | focused compiler tests; representative compile and smoke test |
| CLI/tooling | relevant `apg ... --json` command; focused tests when available |

Report exactly what ran. If you did not run the full suite, say so.

## Documentation Expectations

Update documentation when a change affects how someone builds, verifies, or
extends APG.

| Change | Update |
| --- | --- |
| Capability package behavior | package `cap_spec.md`, focused tests, progress log when burn-down changes |
| Example or capacity readiness | example `README.md`, output when intentionally refreshed, progress log |
| Compiler or semantic behavior | focused tests, related example README or guide if workflow changes |
| CLI or audit output | `docs/tooling.md` and command tests where relevant |
| Contributor workflow | this guide, developer guide, capacity guide, or repository hygiene guide |

Write evidence, not optimism. A useful note names the command, outcome, and
remaining gap.

## Handoff Notes

Every non-trivial contribution should leave a short handoff in the most local
place a future contributor will read.

Use this shape:

```text
Current state:
Proof run:
Public contract:
Known gap:
Next useful packet:
```

Where to put it:

| Work | Handoff location |
| --- | --- |
| Example or capacity | example `README.md` |
| Capability package | package `cap_spec.md` and focused tests |
| Global readiness or audit burn-down | `docs/progress_log.md` |
| CLI/tooling workflow | `docs/tooling.md` or relevant guide |
| Contributor workflow | this guide or the developer guide |

Do not leave critical next-step knowledge only in chat, a branch name, or a
private note.

## Review Checklist

Use this before asking for review or committing:

- The diff has one primary owner.
- Public names are stable or the migration is documented.
- The proof command covers the changed contract.
- Docs or progress log changed when readiness evidence changed.
- The staged diff excludes unrelated worktree files.
- The commit message explains why, constraints, proof, and known gaps.

Reviewers should ask for narrower packets or stronger evidence when those items
are unclear.

## Common Mistakes To Avoid

- Claiming a capacity is implemented because source parses but generated Python
  does not run.
- Updating checked-in generated output without fixing the generator.
- Deepening a capability package while leaving the `cap_spec.md` describing old
  generic behavior.
- Adding AI provider details directly into business source instead of modeling
  provider choice behind agent runtime configuration and approvals.
- Modeling a legacy broker as the default stream architecture instead of
  Bytewax-oriented stream semantics with adapters.
- Running a broad suite, seeing unrelated failures, and reporting nothing about
  the focused contract you changed.
- Staging `.omx/`, `.claude/`, uploads, copied references, or another
  contributor's dirty files.

## Commit Protocol

APG commits use the Lore protocol. The first line explains why the change was
made, not just what files changed.

```text
<why this change exists>

<context, constraints, and approach>

Constraint: <external or project constraint>
Rejected: <alternative> | <reason>
Confidence: <low|medium|high>
Scope-risk: <narrow|moderate|broad>
Directive: <warning for future changes>
Tested: <commands run>
Not-tested: <known gaps>
```

Commit and push completed, verified slices regularly. Do not accumulate large
uncommitted batches when a coherent packet is already proven.

## First Useful Contribution Examples

### Documentation Packet

```text
Outcome: new contributors can find capability proof commands from the docs index.
Owner: docs/README.md and docs/contributors_guide.md
Focused proof: apg docs audit --json; git diff --check -- docs
Non-goals: capability implementation changes
```

### Capability Packet

```text
Outcome: one package no longer stores generic materialized records.
Owner: capabilities/common/<code>/
Focused proof: package pytest; implementation-audit --root; publish-plan
Non-goals: live provider integrations
```

### Compiler Packet

```text
Outcome: screen relationships appear in semantic JSON.
Owner: compiler/ast_builder.py and compiler/semantic_model.py
Focused proof: focused compiler test; apg model examples/<nn>/main.apg --json
Non-goals: visual designer changes
```

### Capacity Packet

```text
Outcome: one procurement approval event compiles and smoke-tests.
Owner: examples/<nn>_procurement_approval/
Focused proof: apg model; apg compile --verify; smoke_test.py
Non-goals: full purchasing suite
```

## Contributor Definition Of Done

A contribution is ready when another contributor can pull it and answer:

- What changed?
- Which public contract does it affect?
- Which command proves it?
- What remains intentionally unimplemented?
- Where should the next contributor continue?

If those answers are not visible in code, docs, tests, or the commit message,
the contribution is not finished.
