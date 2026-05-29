# APG Capacity Development Guide

This guide explains how to build new APG capacities. A capacity is a coherent
business or platform ability that combines APG source, generated Python,
capability packages, rules, screens, workflows, AI agents, Bytewax streaming
metadata, tests, documentation, and release evidence.

Use this guide when the goal is to make APG able to do something new, such as
procurement approval, ledger posting, customer onboarding, device management,
agentic operations, or integration monitoring.

## Start With One Executable Thread

A capacity starts when one real event can move through source, semantic model,
generated Python, package behavior, and proof. Do not begin by designing a
whole ERP module. Begin with one event that a user, system, workflow, agent, or
stream can trigger.

Use this thread:

```text
business event
  -> APG source in examples/<nn>_<capacity>/main.apg
  -> semantic JSON from apg model
  -> generated Python from apg compile --verify
  -> package-owned durable behavior
  -> focused package or generated smoke proof
  -> README, cap_spec, and progress-log handoff
```

The first two hours of capacity work should produce these artifacts:

| Artifact | Minimum result |
| --- | --- |
| event statement | one sentence naming the trigger and observable result |
| APG source | parseable `main.apg` with records, rules, screens, workflows, agents, streams, or capability references as needed |
| semantic proof | `./.venv/bin/apg model examples/<nn>_<capacity>/main.apg --json` exposes the intended public names |
| runtime proof | `./.venv/bin/apg compile ... --output /tmp/apg-<capacity> --verify` and generated `smoke_test.py` pass |
| package map | durable behavior maps to one or more `capabilities/<domain>/<code>/` packages |
| handoff | README names readiness, proof commands, owners, gaps, and next slice |

Write the seed before implementation:

```text
Capacity:
First event:
Observable result:
Primary actor:
Tenant/security boundary:
Records:
Rules:
Screens:
Workflow states:
AI agents:
Bytewax streams:
Capability owners:
Proof commands:
Next slice:
```

This seed is the shared contract for parallel work. Grammar, compiler,
generator, package, UI, agent, stream, and documentation contributors can move
quickly only when these names stay stable.

## Capacity Development Principle

A capacity is not a feature list. It is a rerunnable path from one event to one
observable result.

Every capacity must identify:

| Decision | Why it matters |
| --- | --- |
| first event | gives contributors a single executable starting point |
| tenant and security boundary | prevents later package behavior from being globally scoped by accident |
| records and relationships | gives the grammar, semantic model, and generated runtime stable names |
| rules and approvals | turns business policy into deterministic tests and review gates |
| screens and composition | makes the UI contract inspectable instead of implied |
| workflow states | gives generated runtime and package services a lifecycle |
| AI agent boundary | keeps provider choice configurable and auditable |
| Bytewax stream boundary | keeps event flow semantics platform-native and adapter-friendly |
| package owners | tells contributors where durable behavior belongs |
| proof commands | prevents readiness from being based on prose |

The first event should be small enough that a new contributor can model it,
compile it, smoke-test it, and document the next gap in one working session.

## Capacity Versus Capability

A **capability** is a composable unit with a contract and package-backed
behavior: services provided, services required, configuration, rules, UI
routes, theme tokens, tests, and publish-plan evidence.

A **capacity** is an executable ability assembled from one or more capabilities
plus APG language/runtime surfaces.

Example:

```text
Capacity: Procurement approval automation
First event: purchase request submitted
Capabilities: supplier master, budget control, workflow orchestration, audit logging
Rules: supplier required, amount positive, large request review
Screens: request workbench, approval queue
Workflow: draft -> review -> approved -> ordered
Agent: procurement planner under approval
Streaming: Bytewax procurement_events -> procurement_alerts
Evidence: model, compile, smoke test, package tests, publish-plan
```

Build capacities as vertical slices. Do not create a large module inventory
before one event compiles and runs.

## Minimum Executable Capacity

A capacity starts with one event:

```text
event
  -> APG source
  -> semantic model
  -> generated Python app
  -> package-backed durable behavior
  -> focused proof
  -> README/progress handoff
```

Minimum artifacts:

| Artifact | Required content |
| --- | --- |
| `examples/<nn>_<capacity>/main.apg` | records, capabilities, rules, screens, workflows, agents, streams, and app composition for one event |
| `examples/<nn>_<capacity>/README.md` | readiness level, event path, proof commands, generated-output status, next slice |
| generated output | refreshed only when intentionally compiling examples for review |
| capability packages | domain service/API/view behavior where durable state is needed |
| package `cap_spec.md` | executable behavior, guardrails, adapter boundaries, proof commands |
| tests/audits | focused compiler, example, and package proof |
| `docs/progress_log.md` | updated when readiness or implementation-depth evidence changes |

If a capacity cannot name the first event, APG source path, package owners, and
proof commands, reduce scope.

## Capacity Acceptance Gate

Before a capacity can be treated as buildable, it must pass this gate. The gate
is intentionally small enough for a new contributor to evaluate from checked-in
files.

| Gate | Minimum evidence | Fails when |
| --- | --- | --- |
| event | one named event in the README and source | the capacity is described only as a broad module |
| source | `examples/<nn>_<capacity>/main.apg` exists and is parseable | source is prose, pseudocode, or generated output only |
| semantics | `apg model ... --json` exposes the intended records, rules, screens, workflows, agents, streams, or capabilities | meaning is implied but absent from semantic JSON |
| runtime | `apg compile ... --verify` and generated `smoke_test.py` pass for generated behavior | only parser or docs evidence exists |
| package owner | durable behavior maps to `capabilities/<domain>/<code>/` | generated app owns long-lived business behavior by itself |
| rules | at least one deterministic rule or guardrail has a public name | policy is described as "business rules" without inputs or decisions |
| UI | at least one screen route or composition relation is named when humans interact with the capacity | UI is left to later without a route or owner |
| agents | provider choice, tools, approval, and fallback are configuration or adapter concerns | business source hard-codes one fast-changing AI provider |
| streams | Bytewax flow names, event envelopes, and partition keys are named when event flow matters | broker-specific mechanics are treated as the capacity model |
| handoff | README names readiness, proof, package owner, known gaps, and next slice | future contributors need private context |

A capacity can start at level 0, but implementation should not begin until the
first event, owner, and proof path are explicit.

## Capacity Factory Loop

Use this loop to turn an idea into executable APG without waiting for the whole
platform to be complete.

1. **Name the event.** Use an event a user or system can trigger today, such as
   `invoice submitted`, `customer qualified`, `device telemetry received`, or
   `agent plan approved`.
2. **Write the capacity blueprint.** Name actors, records, rules, screens,
   workflows, agents, streams, packages, and proof commands.
3. **Create or update one example.** Put the smallest parseable APG source in
   `examples/<nn>_<capacity>/main.apg`.
4. **Prove semantics.** Run `apg model ... --json` and inspect the relevant
   records, rules, screens, workflows, agents, streams, and capability
   references.
5. **Generate Python.** Run `apg compile ... --verify` into `/tmp`, then run
   the generated smoke test.
6. **Deepen one package.** If durable behavior is needed, implement one
   lifecycle in the owning capability package with positive and guardrail
   tests.
7. **Record the handoff.** Update the example README, package `cap_spec.md`,
   and progress log when readiness or implementation-depth evidence changes.

Repeat the loop for the next event. Do not create a large capacity inventory
that has no compiled first event.

## Capacity-To-Capability Map

Every capacity needs a small map from business intent to package ownership.
Keep it in the example README or package planning note.

| Capacity element | Map it to | Example |
| --- | --- | --- |
| First event | one command, route action, workflow transition, or stream input | `invoice submitted` |
| Durable record | one package model and tenant boundary | `Invoice` in an AP package |
| Rule | one deterministic rule ID and decision vocabulary | `invoice_total_positive -> allow/deny` |
| Screen | one route and composition relationship | `/ap/invoices` contains `Invoice` and actions |
| Workflow | states, transitions, guard rules, approval points | `draft -> submitted -> approved` |
| Agent | runtime adapter, tools, memory, approval rules | `invoice_triage_agent` can draft exceptions |
| Stream | Bytewax flow, input, output, partition key | `invoice_events -> invoice_alerts by tenant_id` |
| Integration | adapter boundary and local deterministic fallback | supplier API adapter, local validation proof |
| Proof | commands that show current readiness | model, compile, smoke test, package pytest |

This map lets compiler, runtime, package, UI, and docs contributors work in
parallel without changing public names underneath each other.

## Capacity Crew Roles

Use these roles when multiple contributors are building a capacity at the same
time. One person may hold several roles on a small slice, but each role should
still have a named artifact and proof.

| Role | Owns | Proof |
| --- | --- | --- |
| capacity lead | first event, readiness level, public names, README | README names event, owners, proof, and next slice |
| language owner | APG source shape and grammar needs | `apg model ... --json` exposes intended constructs |
| runtime owner | generated Python routes, helpers, manifests, smoke assertions | `apg compile ... --verify` and generated smoke test |
| package owner | durable service lifecycle, API helpers, views, rules, theme | package pytest, implementation audit root, publish-plan |
| agent owner | provider-agnostic agent runtime boundary and approval rules | semantic output and generated manifest expose agent policy |
| stream owner | Bytewax flow metadata, event envelopes, partitioning | semantic output and manifest expose stream policy |
| docs owner | guide, README, package spec, progress evidence | docs audit and local diff check |

Parallel work is effective when these roles share stable names: event name,
record names, route names, workflow states, rule IDs, service names, agent IDs,
stream names, and package roots.

## Capacity Packet Template

Put this blueprint in the example README or working note before implementation:

```text
Name:
Outcome:
Primary users:
First event:
Tenant/security boundary:
Records owned:
Capabilities provided:
Capabilities required:
Rules:
Screens:
Workflows:
Agents:
Streaming:
Generated routes/helpers:
Package owners:
Focused proof:
Known gaps:
Next slice:
```

The blueprint describes current executable intent. Keep it updated as evidence
lands.

## Capacity Slice Contract

Each capacity slice should fit in one commit and one reviewer pass. Use this
contract to keep the slice narrow:

```text
Slice name:
Readiness level before:
Readiness level after:
Event:
Source path:
Generated output path:
Package owner:
Public names preserved:
Proof run:
Not tested:
Next slice:
```

Examples:

| Slice | Good boundary |
| --- | --- |
| parseable event | add or repair `main.apg`, README, and semantic proof |
| generated runtime | generator change plus one compile and smoke proof |
| package-backed lifecycle | one package service lifecycle with guardrail tests and `cap_spec.md` update |
| governed capacity | rule inputs, decisions, approval guardrails, and negative tests |
| composable UI | one screen route, contains/composes/binds relations, and generated manifest proof |
| agentic assist | one agent boundary with provider configuration, allowed tools, approval rule, and fallback |
| stream processing | one Bytewax flow with envelope, partition key, state intent, and local proof |

Avoid a slice that introduces new grammar, rewrites packages, refreshes every
example output, and updates global docs at the same time. Split by public
contract and proof.

## Starter Capacity Checklist

Use this checklist to turn a broad idea into a buildable APG slice.

| Decision | Good answer | Bad answer |
| --- | --- | --- |
| First event | `invoice submitted` | `build finance` |
| Primary actor | `accounts payable clerk` | `users` |
| Tenant boundary | `tenant_id scopes invoices and approvals` | `security later` |
| Durable owner | `capabilities/financials/ap/` | `generated app only` |
| First rule | `invoice_total_positive` | `business rules` |
| First screen | `/ap/invoices` with invoice list and action | `dashboard` |
| First workflow | `draft -> submitted -> approved -> posted` | `ERP workflow` |
| First agent | `invoice_triage_agent drafts exceptions, cannot post` | `AI handles it` |
| First stream | `Bytewax invoice_events -> invoice_alerts` | `legacy broker integration` |
| Proof | model, compile, smoke test, package pytest | manual inspection |

When an answer is vague, reduce the scope until the event can compile and
smoke-test.

## Capacity Source Skeleton

Keep the first source file small enough for a new contributor to understand.
The exact syntax should follow the current grammar, but the source should carry
these concepts:

```text
app <capacity_name> targets python

capability <owned_capability> {
  provides <service_name>
  requires <platform_service>
  configure <key> = <value>
}

record <BusinessRecord> {
  tenant_id: str required
  ...
}

rule <stable_rule_name> {
  when <condition>
  then <decision>
}

screen <workbench_name> {
  route "/<area>/<event>"
  contains <BusinessRecord>
  action <event_action>
}

workflow <event_workflow> {
  state draft
  state submitted
  transition submit from draft to submitted when <rule>
}

agent <capacity_agent> {
  runtime <adapter_name>
  can use <service_name>
  approval required for <risky_action>
}

stream <event_flow> {
  engine bytewax
  input <event_name>
  output <alert_name>
  partition by tenant_id
}
```

Do not force every construct into the first slice. Include a construct only
when it is needed for the first event and has a proof path.

## Capacity Design Sprint

Use this one-hour sprint before writing code for a new capacity:

| Minute | Decision | Output |
| --- | --- | --- |
| 0-10 | Name the user and first event | `Primary users` and `First event` in the blueprint |
| 10-20 | Identify records and ownership | record names, tenant boundary, required relationships |
| 20-30 | Identify rules and approvals | deterministic rule names and review routes |
| 30-40 | Identify screens and workflow | screen routes, workflow states, transitions |
| 40-50 | Identify agents and streams | agent role, provider boundary, Bytewax flow names |
| 50-60 | Pick proof and package owners | model/compile/smoke commands and package roots |

If the sprint cannot produce an event, route, owner, and proof command, the
capacity is not ready for implementation. Narrow it until the first event is
clear.

## Build Runbook

1. **Name one event.** Use a concrete phrase such as `request submitted`,
   `journal posted`, `lead qualified`, `device telemetry received`, or
   `agent plan approved`.

2. **Write APG source.** Keep the source terse and readable. Include only the
   records, relationships, capability uses, rules, screens, workflows, agents,
   and Bytewax streams needed for the event.

3. **Inspect semantics.**

   ```bash
   ./.venv/bin/apg model examples/<nn>_<capacity>/main.apg --json
   ```

4. **Compile and smoke-test.**

   ```bash
   ./.venv/bin/apg compile examples/<nn>_<capacity>/main.apg --output /tmp/apg-capacity --verify
   ./.venv/bin/python /tmp/apg-capacity/smoke_test.py
   ```

5. **Deepen package behavior when durable state is needed.**

   ```bash
   ./.venv/bin/pytest -q capabilities/<domain>/<code>/test_capability_contract.py capabilities/<domain>/<code>/tests
   ./.venv/bin/apg capabilities implementation-audit --root capabilities/<domain>/<code> --json
   ./.venv/bin/apg capabilities publish-plan capabilities/<domain>/<code> --json
   ```

6. **Document evidence.** Update the example README, package spec, and progress
   log when readiness changes.

7. **Commit the slice.** Stage only the capacity files, package files, tests,
   and docs that belong to the packet.

If a step fails, fix the earliest failing layer. Do not patch generated output
around missing APG meaning.

## Worked Slice Pattern

Use this pattern to build an ERP capacity without drifting into a full suite.

### Slice 1: Parseable Event

- write `examples/<nn>_<capacity>/main.apg`;
- model one record, one rule, one screen, one workflow transition;
- run `./.venv/bin/apg model ... --json`;
- update the example README to readiness level 1.

### Slice 2: Generated Runtime

- compile the same source to `/tmp`;
- run the generated smoke test;
- inspect generated routes/helpers if screens or workflows changed;
- update the README to readiness level 2.

### Slice 3: Package-Backed Lifecycle

- choose the capability package that owns durable behavior;
- implement domain models, service methods, API helpers, views, and guardrails;
- run package pytest, implementation audit, and publish-plan;
- update package `cap_spec.md` and capacity README.

### Slice 4: Composition Depth

- add screen relationships, workflow routes, agent boundaries, and Bytewax
  stream metadata only when the earlier slices are proven;
- rerun model, compile, smoke, and package proof;
- update `docs/progress_log.md` when global readiness changes.

Each slice should be independently committable.

## Example Directory Shape

Use one directory per capacity or capacity slice:

```text
examples/<nn>_<capacity_name>/
  main.apg
  README.md
  output/
    app.py
    smoke_test.py
    semantic_model.json
```

The `output/` directory is checked in only when intentionally refreshed as
evidence. Otherwise compile to `/tmp/apg-capacity` and keep the source tree
clean.

The README should answer:

- What business event does this capacity execute?
- Which records, capabilities, rules, screens, workflows, agents, and streams
  are involved?
- What readiness level is proven?
- Which commands were run?
- Which package owns durable behavior?
- What is the next smallest executable slice?

## Authoring Standards For Capacity Source

APG capacity source should read as a compact declaration of the business event,
not as generated implementation code.

A useful source file names:

- records and relationships;
- capability dependencies;
- deterministic rules;
- screens and screen relationships;
- workflows and states;
- AI agents, tools, memory, runtime, and approvals;
- Bytewax-oriented stream names and event envelopes;
- application composition and target `python` runtime.

Keep provider details behind adapters. For AI agents, model providers such as
Codex, Claude Code, OpenCode, Pi, local models, or hosted APIs belong behind
runtime configuration and approval rules, not hard-coded business logic.

## Composition Rules

Capacity composition must be explicit.

| Composition point | What to name |
| --- | --- |
| Records | owner, identifiers, required fields, relationships |
| Rules | stable rule names, deterministic inputs, decision vocabulary |
| Screens | route, title, contains, composes, binds, actions, permissions, theme |
| Workflows | states, transitions, guards, approvals, audit events |
| Agents | runtime, model/provider boundary, tools, memory, approvals, budget/risk controls |
| Streams | Bytewax flow name, input event, output event, partition key, retry/error behavior |
| Capabilities | provided services, required services, package owner, adapter boundary |

Implicit relationships slow contributors down. If one screen depends on a
record, rule, workflow, or capability, name the relationship in source,
semantic output, README, or package spec.

## Rule, UI, And Theme Requirements

Each capacity-level capability must be configurable, rule-aware, visible in UI
composition, and themeable. Treat those as minimum composition metadata, not
late polish.

For every package-backed capability in a capacity, document:

| Concern | Required capacity detail |
| --- | --- |
| configuration | tenant-scoped keys, defaults, limits, adapter choices, disabled modes |
| rules | deterministic rule IDs, inputs, decisions, denial reasons, audit events |
| UI | route names, screen ownership, contained records, actions, permissions |
| theme | theme name, semantic color tokens, density needs, dashboard/view variants |
| tests | positive lifecycle and negative guardrails for important rules |
| adapters | live provider boundaries and local deterministic fallback |

Example:

```text
Capability: invoice_triage
Configuration: high_value_threshold, provider_runtime, max_agent_steps
Rules: invoice_total_positive, supplier_active, human_approval_for_posting
UI: /ap/invoices contains Invoice and exposes submit, review, approve actions
Theme: finance_operations with compact tables and status severity tokens
Adapters: AI provider runtime, document extraction service, supplier API
Proof: package pytest, implementation-audit root, publish-plan, compile smoke
```

This keeps capacity work aligned with APG's composable capability model instead
of producing examples that cannot become larger applications.

## Capacity Blueprint Example

Use this as a concrete model for an ERP-style capacity:

```text
Name: Purchase request approval
Outcome: an employee submits a purchase request and receives an approval decision.
Primary users: requester, budget owner, procurement approver
First event: purchase request submitted
Tenant/security boundary: requests are tenant-scoped; approvers need workflow permission.
Records owned: PurchaseRequest, PurchaseRequestLine, ApprovalDecision
Capabilities provided: procurement request intake
Capabilities required: supplier master, budget control, workflow orchestration, audit log, notification
Rules: request_amount_positive, budget_available, high_value_requires_approval
Screens: request_workbench, approval_queue, request_detail
Workflows: draft -> submitted -> review -> approved -> ordered or rejected
Agents: procurement_triage_agent may summarize risk but cannot approve without human review
Streaming: Bytewax purchase_request_events -> purchase_request_alerts by tenant_id/request_id
Generated routes/helpers: request workbench route, approval action helper, semantic manifest
Package owners: capabilities/scm/req/ or a focused procurement package
Focused proof: model, compile --verify, smoke_test.py, package pytest when durable behavior lands
Known gaps: live ERP integration, supplier API adapter, purchasing document export
Next slice: package-backed request lifecycle with approval guardrails
```

This blueprint is deliberately narrow. It avoids building an entire purchasing
suite before APG can execute one request event.

## Readiness Levels

Use these readiness levels in example READMEs and progress-log entries:

| Level | Meaning | Evidence |
| --- | --- | --- |
| 0 - design | event and owners named, no parseable source yet | blueprint only |
| 1 - parseable | `main.apg` parses and appears in semantic JSON | `apg model ... --json` |
| 2 - generated | generated Python app imports and smoke-tests | compile with `--verify`, smoke test |
| 3 - package-backed | durable behavior executes in package services | package pytest, implementation audit, publish-plan |
| 4 - governed | rules, approvals, tenant boundaries, and negative tests are present | guardrail tests and rule evidence |
| 5 - composable | screens, workflows, agents, streams, and capabilities are inspectable and documented | semantic model, generated manifests, README |
| 6 - release-ready | release/package/deployment evidence exists for intended profile | release, package, package-verify, deployment or evidence command |

Advance readiness only when commands prove the new level.

## Capacity Expansion Order

Expand in this order so every layer has something real to consume:

1. First event and APG source.
2. Semantic model visibility.
3. Generated Python runtime and smoke test.
4. One domain-specific capability package.
5. Rule guardrails and negative tests.
6. Screen composition and theme metadata.
7. Workflow transitions and generated route/manifest exposure.
8. AI agent composition with provider-agnostic adapters.
9. Bytewax streaming metadata and deterministic local envelopes.
10. Release evidence, docs, and deployment profile.

Skip a layer only when the README says it is intentionally out of scope for the
current slice.

## Package-Backed Behavior

Durable behavior belongs in capability packages, not only in example prose.

For each package in a capacity, require:

- domain models instead of generic materialized records;
- service methods that execute one real lifecycle;
- API helpers that expose package behavior without framework dependencies;
- view models and route/theme metadata for UI composition;
- deterministic rules and guardrail tests;
- tenant context and ownership checks where relevant;
- adapter boundaries for live external systems;
- `cap_spec.md` describing current behavior and proof commands;
- publish-plan evidence.

Run:

```bash
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```

Use the audit output as the burn-down board for package depth.

## From Capacity To Package Backlog

After the first generated capacity runs, convert remaining work into package
backlog items. Each backlog item should name one lifecycle and one proof.

```text
Package:
Lifecycle:
Models:
Service methods:
Rules:
Views/API:
Adapter boundary:
Positive test:
Guardrail tests:
Publish proof:
```

Example:

```text
Package: capabilities/common/logt/
Lifecycle: ingest log -> correlate trace -> search -> approved export
Models: pipeline, log event, trace, span, query, export, retention policy
Service methods: create_pipeline, ingest_log, ingest_trace, record_span, search_logs, export_logs
Rules: tenant required, trace context required, sensitive log redaction, export approval
Views/API: dashboard, logs, traces, pipelines, retention, analytics
Adapter boundary: OpenTelemetry collector, search index, object store, audit store
Positive test: lifecycle executes with tenant-scoped events
Guardrail tests: missing tenant, missing trace context, unredacted sensitive log, unapproved export
Publish proof: implementation-audit --root and publish-plan
```

This is how APG closes the gap between language-level capacity and executable
capability depth.

## Capacity Backlog Examples

Use these as starting points for new APG capacity packets.

| Capacity | First event | Likely packages | First proof |
| --- | --- | --- | --- |
| Accounts payable | invoice submitted | AP, supplier master, document management, audit | model, compile, AP package pytest |
| General ledger | journal posted | GL, period control, approval, audit | model, compile, GL package pytest |
| Customer onboarding | customer application submitted | CRM, KYC, document management, notification | model, compile, onboarding package pytest |
| Inventory receiving | goods receipt recorded | inventory, purchasing, warehouse, audit | model, compile, inventory package pytest |
| Device telemetry | telemetry received | IOTD, LOGT, alerting, notification | model, compile, IOTD package pytest |
| Agentic operations | plan proposed | AI agent runtime, workflow, audit, approval | model, compile, agent adapter proof |
| Integration monitoring | event failed | integration, logging, alerting, retry queue | model, compile, integration package pytest |

For each backlog item, write the blueprint first, then build only the first
event. The first successful event creates the vocabulary for the rest of the
capacity.

## AI Agent Capacities

AI agents are first-class capacity components when they participate in the
business event.

An agent capacity must name:

- agent role and allowed objective;
- runtime adapter boundary;
- supported provider options;
- tools and capability services it may call;
- memory scope and retention;
- approval rules for risky actions;
- cost, latency, or execution budget where applicable;
- audit events and human review route;
- fallback when the provider is unavailable.

Keep rapidly changing provider integrations behind adapters. A capacity should
be able to switch from one provider runtime to another without rewriting the
business event or package rules.

Agent review rules:

| Action type | Default policy |
| --- | --- |
| Read-only summarization | allowed with audit event |
| Drafting recommendations | allowed with source evidence |
| Mutating records | human approval unless explicitly safe and tested |
| Calling external providers | adapter boundary plus timeout, budget, and fallback |
| Executing code or shell tools | explicit tool allowlist and audit event |
| Sending messages or payments | human approval and policy guardrail |

An AI agent is first-class only when these boundaries are inspectable in APG
source, semantic output, generated manifests, package rules, or documentation.

## Bytewax Streaming Capacities

APG uses Bytewax terminology for streaming capacities. A stream slice should
name:

- flow name;
- input event envelope;
- output event envelope;
- partition key;
- stateful operator intent;
- retry and dead-letter policy;
- capability package that owns durable state;
- generated metadata or manifest output;
- local deterministic proof.

Do not use a legacy broker as the default architecture. If a future integration
needs a broker-specific bridge, model it as an adapter boundary around
Bytewax-oriented APG stream semantics.

## Capacity Verification Matrix

Use this matrix to decide what to run before committing:

| Capacity change | Proof |
| --- | --- |
| New source shape | `apg model ... --json` |
| Generated runtime changed | `apg compile ... --verify`, generated `smoke_test.py` |
| Screen or workflow manifest changed | inspect semantic JSON and generated manifest/routes |
| AI agent surface changed | model JSON, compile proof, agent adapter command when available |
| Bytewax stream metadata changed | model JSON and generated manifest proof |
| Capability package behavior changed | package pytest, implementation audit root, publish-plan |
| Global capability burn-down changed | global implementation audit and progress-log entry |
| Documentation changed only | docs audit and diff check |

Do not advance readiness levels from prose. Advance them only from command
evidence.

## Parallel Capacity Development

Capacity work can be parallel when each contributor owns a different surface.

| Lane | Owns | Coordinates with |
| --- | --- | --- |
| Capacity lead | packet, public names, readiness level, README, progress log | all lanes before public names change |
| APG source owner | example `main.apg` | compiler owner before relying on new syntax |
| Compiler owner | parser, AST, semantic model, generated surfaces | source and runtime owners |
| Capability owner | one package tree | capacity lead for service/rule names |
| Runtime owner | generated routes, helpers, manifests, smoke tests | compiler owner for semantic keys |
| Docs owner | README, guide links, proof commands | every lane for current evidence |

Safe parallel work:

- one contributor deepens a package while another updates the example README;
- one contributor adds generated runtime exposure while another adds package
  guardrail tests;
- different contributors burn down different capability packages.

Unsafe parallel work without coordination:

- multiple contributors changing `spec/apg.g4` and semantic keys independently;
- refreshing example output while generator contracts are moving;
- two contributors changing the same service public methods;
- docs claiming readiness before proof commands pass.

## Review Gate

Before merging or committing a capacity slice, confirm:

- APG source is parseable and readable.
- Semantic JSON exposes the changed records, rules, screens, workflows, agents,
  streams, or capability references.
- Generated Python imports and smoke-tests when runtime output changed.
- Package-owned behavior has domain services, APIs, views, rules, and tests
  when durable behavior changed.
- Tenant, approval, provider, integration, and adapter boundaries are explicit.
- README, package spec, or progress log records proof commands and the next gap.
- Focused verification passed and full-suite gaps are honest.

## Capacity Definition Of Done

A capacity slice is done when another contributor can:

1. find the APG source;
2. run the semantic and compile proof;
3. inspect generated Python evidence;
4. identify package owners for durable behavior;
5. run package proof when package behavior changed;
6. read the README or progress log to understand the next slice.

If those steps require a meeting or private memory, the capacity is not yet
documented well enough.
