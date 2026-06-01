# Human Intelligence Capability Specification

## Purpose

`intel_humint` makes lawful human-intelligence coordination a first-class APG
capability. It provides the application contract for authorities, human
sources, contact plans, contact reports, debriefings, reliability assessments,
leads, dissemination, reviews, Bytewax lifecycle metadata, UI/view composition,
visual theming, and provider-neutral AI-agent support.

The package is intentionally dependency-light and does not implement source
recruitment tradecraft, coercive operations, surveillance, field logistics,
payment handling, covert communications, or physical security operations. It
models lawful authority, source welfare, safety planning, evidence, review,
classification, release control, and adapter boundaries so generated
applications can compose a governed HUMINT workflow.

## Users

- Authorized intelligence, defense, security, law-enforcement, and public
  safety teams.
- Oversight and compliance teams that need authority, welfare, safety,
  classification, audit, and review evidence.
- Application builders composing source-management workflows with graph, RAG,
  NLP, audit, notifications, and authorization capabilities.
- AI-agent teams using Codex, Claude Code, OpenCode, or Pi for authority
  review, source-management assistance, contact planning, debriefing analysis,
  welfare review, and dissemination review.

## Functional Scope

- Record authorities with scope, classification, approver, expiry, and
  evidence.
- Register human sources with type, handling status, risk level, owner,
  authority, protection reference, and evidence.
- Record contact plans with lawful authority, source match, objective, safety
  plan, approval, and evidence.
- Record contact reports with handler, report reference, source-welfare score,
  and evidence.
- Record debriefings, reliability assessments, leads, dissemination decisions,
  and review outcomes.
- Register provider-neutral AI agents and validate privileged or prohibited
  action scopes.

## Rule Engine

Rules are deterministic and enforced before mutation. They cover tenant context,
write policy attachment, authority type/scope/classification/approver/expiry/
evidence, source type/status/risk/owner/authority/protection/evidence, contact
authority/source/source-authority match/method/objective/safety/approval/
evidence, contact-report plan/reference/handler/welfare/evidence, debriefing
report/topic/classification/credibility/analyst/evidence, reliability source/
grade/confidence/analyst/evidence, lead debriefing/type/priority/analyst/
evidence, dissemination lead/audience/release/approval/evidence, review
status/reviewer/evidence, Bytewax routing, supported AI-agent runtimes/roles,
human approval for privileged agent actions, and denial of coercive HUMINT
automation scope.

## UI And Theming

The capability publishes compact APG Python UI metadata for dashboard,
authorities, sources, contact plans, contact reports, debriefings, reliability,
leads, dissemination, reviews, agents, and settings. Theme metadata uses compact
density, 8px radius, operational status indicators, and component icon hints.

## Adapter Boundaries

Field operations, source recruitment, covert communications, payment handling,
physical security, identity protection infrastructure, partner case systems,
storage backends, GraphRAG projections, dissemination delivery, and durable
Bytewax workers remain behind adapters. This package owns the contract,
guardrails, and local executable lifecycle.

## Acceptance Criteria

- `intel_humint` is inspectable by the APG CLI as a valid capability contract.
- The package includes specification, plan, README, capability spec, contract,
  service, models, API helpers, views, app, semantic model, manifest, release
  report, and focused tests.
- Service methods enforce rules before mutation and keep tenant state isolated.
- UI routes include `/intel-humint/agents`.
- Streaming metadata uses Bytewax and stream `apg.intel.humint.lifecycle`.
