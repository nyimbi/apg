# Financial Intelligence Capability Specification

## Purpose

`intel_finint` makes lawful financial-intelligence coordination a first-class
APG capability. It provides the application contract for authorities, financial
sources, subjects, transactions, patterns, risk assessments, referrals,
dissemination, reviews, Bytewax lifecycle metadata, UI/view composition, visual
theming, and provider-neutral AI-agent support.

The package is intentionally dependency-light and does not move funds, execute
payments, freeze accounts, place trades, submit regulatory reports, or perform
live bank/crypto integrations. It models lawful authority, source and subject
lineage, transaction evidence, risk, referrals, release control, review, and
adapter boundaries so generated applications can compose a governed
financial-intelligence workflow.

## Users

- Financial intelligence units, AML teams, sanctions teams, fraud analysts,
  public-sector investigators, risk teams, and compliance teams.
- Oversight teams that need authority, jurisdiction, privacy, audit,
  classification, and review evidence.
- Application builders composing financial-intelligence workflows with KYC,
  AML, graph, RAG, NLP, audit, notifications, authorization, payment, case, and
  reporting adapters.
- AI-agent teams using Codex, Claude Code, OpenCode, or Pi for authority
  review, source stewardship, transaction analysis, pattern analysis, risk
  analysis, and dissemination review.

## Functional Scope

- Record lawful financial-intelligence authorities with scope, classification,
  approver, expiry, and evidence.
- Register financial sources with type, jurisdiction, owner, authority, and
  evidence.
- Record subjects, transactions, patterns, risk assessments, referrals,
  dissemination decisions, and review outcomes.
- Register provider-neutral AI agents and validate privileged or prohibited
  financial automation scopes.

## Rule Engine

Rules are deterministic and enforced before mutation. They cover tenant context,
write policy attachment, authority type/scope/classification/approver/expiry/
evidence, source type/jurisdiction/owner/authority/evidence, subject type/
reference/risk tier/authority/evidence, transaction source/subject authority
alignment/reference/amount/currency/type/timestamp/evidence, pattern
transaction/type/confidence/analyst/evidence, risk pattern/type/level/
confidence/analyst/evidence, referral assessment/type/recipient/approval/
evidence, dissemination assessment/audience/release/approval/evidence, review
status/reviewer/evidence, Bytewax routing, supported AI-agent runtimes/roles,
human approval for privileged actions, and denial of funds-movement automation
scope.

## UI And Theming

The capability publishes compact APG Python UI metadata for dashboard,
authorities, sources, subjects, transactions, patterns, risk, referrals,
dissemination, reviews, agents, and settings. Theme metadata uses compact
density, 8px radius, operational status indicators, and component icon hints.

## Adapter Boundaries

Live bank feeds, payment execution, account freezing, crypto exchange APIs,
sanctions-screening engines, regulatory report submission, case management,
data warehouses, GraphRAG projection, dissemination delivery, and durable
Bytewax workers remain behind adapters. This package owns the contract,
guardrails, and local executable lifecycle.

## Acceptance Criteria

- `intel_finint` is inspectable by the APG CLI as a valid capability contract.
- The package includes specification, plan, README, capability spec, contract,
  service, models, API helpers, views, app, semantic model, manifest, release
  report, and focused tests.
- Service methods enforce rules before mutation and keep tenant state isolated.
- UI routes include `/intel-finint/agents`.
- Streaming metadata uses Bytewax and stream `apg.intel.finint.lifecycle`.
