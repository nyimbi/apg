# Open Source Intelligence Capability Specification

## Purpose

`intel_osint` makes open-source intelligence a first-class APG capability. It
coordinates collection requirements, source registry records, collection plans,
evidence capture, analyst triage, intelligence assessments, dissemination,
reviews, Bytewax lifecycle metadata, UI/view composition, visual theming, and
provider-neutral AI-agent support.

The package is dependency-light and executable without live external feeds. It
defines the operational contract that generated APG applications can compose
with Intelligence Crawler, Search, Graph Data Management, Retrieval-Augmented
Generation, NLP, audit, notifications, and authorization capabilities.

## Users

- Intelligence analysts and watch teams that need governed OSINT collection.
- Risk, security, and compliance teams that need source-term, evidence, and
  release controls.
- Application builders composing intelligence planning, processing, review, and
  dissemination screens.
- AI-agent operators using Codex, Claude Code, OpenCode, or Pi for source
  scouting, collection planning, evidence triage, assessment drafting, watchlist
  monitoring, and dissemination review.

## Functional Scope

- Register collection requirements with priority, requester, classification,
  and evidence.
- Register OSINT sources with type, owner, source reference, source-terms
  review, risk tier, and evidence.
- Record collection plans linked to requirements and sources, with approval for
  high-risk sources.
- Record evidence with content references, fingerprints, confidence scores, and
  provenance.
- Record analyst triage decisions.
- Record intelligence assessments with type and confidence.
- Record dissemination packages with audience, release marking, approval, and
  evidence.
- Record review decisions and provider-neutral AI-agent registrations.

## Rule Engine

Rules are deterministic and evaluated before state mutation. They enforce
tenant context, write policy attachment, supported priorities, classifications,
source types, risk tiers, methods, decisions, assessment types, review statuses,
Bytewax lifecycle routing, supported AI-agent runtimes/roles, and human approval
for privileged agent actions.

## UI And Theming

The capability publishes APG Python UI metadata for dashboard, requirements,
sources, collection plans, evidence, triage, assessments, dissemination,
reviews, agents, and settings. Theme metadata uses compact operational density,
8px radius, intelligence-focused status indicators, and component icon hints.

## Adapter Boundaries

Live crawler execution, paid source APIs, social-platform access, search-index
queries, GraphRAG projections, storage, source-term verification, and durable
Bytewax workers remain behind adapters. This package owns the application
contract, deterministic guardrails, and local executable lifecycle.

## Acceptance Criteria

- `intel_osint` is inspectable by the APG CLI as a valid capability contract.
- The package includes specification, plan, README, capability spec, contract,
  service, models, API helpers, views, app, semantic model, manifest, release
  report, and focused tests.
- The service enforces rules before mutation.
- UI routes include `/intel-osint/agents`.
- Streaming metadata uses Bytewax and stream `apg.intel.osint.lifecycle`.
