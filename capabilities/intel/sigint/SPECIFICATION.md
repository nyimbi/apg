# Signals Intelligence Capability Specification

## Purpose

`intel_sigint` makes lawful signals-intelligence coordination a first-class APG
capability. It provides the application contract for authorities, sources,
collection tasks, observations, processing batches, patterns, assessments,
reviews, Bytewax lifecycle metadata, UI/view composition, visual theming, and
provider-neutral AI-agent support.

The package is intentionally dependency-light and does not implement live
interception, receiver control, decryption, or collection hardware operations.
It models lawful authority, minimization, retention, evidence, classification,
review, and adapter boundaries so generated applications can compose a governed
signals-intelligence workflow.

## Users

- Authorized intelligence, defense, security, and public-safety teams.
- Compliance and oversight teams that need authority, minimization, retention,
  classification, audit, and review evidence.
- Application builders composing radio listeners, crawler metadata, graph/RAG,
  NLP, audit, notifications, and authorization into operational tools.
- AI-agent teams using Codex, Claude Code, OpenCode, or Pi for authority review,
  collection planning, processing assistance, pattern analysis, minimization
  review, and dissemination review.

## Functional Scope

- Record lawful authorities with scope, classification, approver, expiry, and
  evidence.
- Register signal sources linked to authority and owner.
- Record collection tasks with collection mode, retention, minimization,
  approval, and evidence.
- Record observations with references, fingerprints, and confidence scores.
- Record processing batches and signal patterns with quality/confidence checks.
- Record classified signal assessments and review decisions.
- Register provider-neutral AI agents and validate privileged agent actions.

## Rule Engine

Rules are deterministic and enforced before mutation. They cover tenant context,
write policy attachment, authority type/scope/classification/approver/expiry/
evidence, source type/band/reference/owner/authority/evidence, task authority/
source/source-authority match/mode/retention/minimization/approval/evidence,
observation task/reference/fingerprint/confidence/evidence, processing type/
quality/analyst/evidence, pattern type/confidence/analyst/evidence, assessment
type/classification/analyst/evidence, review status/reviewer/evidence, Bytewax
routing, supported AI-agent runtimes/roles, and human approval for privileged
agent actions.

## UI And Theming

The capability publishes compact APG Python UI metadata for dashboard,
authorities, sources, collection tasks, observations, processing, patterns,
assessments, reviews, agents, and settings. Theme metadata uses compact density,
8px radius, operational status indicators, and component icon hints.

## Adapter Boundaries

Live RF receivers, lawful-intercept gateways, telecom systems, satellite feeds,
decryptors, speech processing, direction finding, storage backends, search
indexes, GraphRAG projections, dissemination delivery, and durable Bytewax
workers remain behind adapters. This package owns the contract, guardrails, and
local executable lifecycle.

## Acceptance Criteria

- `intel_sigint` is inspectable by the APG CLI as a valid capability contract.
- The package includes specification, plan, README, capability spec, contract,
  service, models, API helpers, views, app, semantic model, manifest, release
  report, and focused tests.
- Service methods enforce rules before mutation.
- UI routes include `/intel-sigint/agents`.
- Streaming metadata uses Bytewax and stream `apg.intel.sigint.lifecycle`.
