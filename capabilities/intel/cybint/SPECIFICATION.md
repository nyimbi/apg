# Cyber Intelligence Capability Specification

## Purpose

`intel_cybint` makes defensive cyber-intelligence coordination a first-class
APG capability. It provides the application contract for lawful authorities,
indicators, sightings, enrichment, threat profiles, risk assessments, incident
links, dissemination, reviews, Bytewax lifecycle metadata, UI/view composition,
visual theming, and provider-neutral AI-agent support.

The package is intentionally dependency-light and does not implement exploit
development, payload generation, intrusion tooling, vulnerability exploitation,
credential collection, or command-and-control. It models defensive authority,
indicator lineage, TLP, confidence, evidence, risk, incident linkage, release
control, review, and adapter boundaries so generated applications can compose a
governed cyber-intelligence workflow.

## Users

- Security operations, threat-intelligence, incident-response, risk, and
  compliance teams.
- Oversight teams that need authority, TLP, data lineage, classification,
  audit, and review evidence.
- Application builders composing cyber-intelligence workflows with graph, RAG,
  NLP, audit, notifications, authorization, SIEM, SOAR, EDR, and ticketing
  adapters.
- AI-agent teams using Codex, Claude Code, OpenCode, or Pi for authority
  review, indicator triage, enrichment analysis, threat profiling, risk
  analysis, and dissemination review.

## Functional Scope

- Record lawful cyber-intelligence authorities with scope, classification,
  approver, expiry, and evidence.
- Record indicators with type, value, TLP, confidence, authority, and evidence.
- Record sightings, enrichments, threat profiles, risk assessments, incident
  links, dissemination decisions, and review outcomes.
- Register provider-neutral AI agents and validate privileged or prohibited
  cyber automation scopes.

## Rule Engine

Rules are deterministic and enforced before mutation. They cover tenant context,
write policy attachment, authority type/scope/classification/approver/expiry/
evidence, indicator type/value/TLP/confidence/authority/evidence, sighting
indicator/source/observation/severity/evidence, enrichment indicator/type/
provider/confidence/analyst/evidence, profile type/name/classification/
confidence/analyst/evidence, risk indicator/profile/level/confidence/analyst/
evidence, incident assessment/reference/priority/owner/evidence, dissemination
assessment/audience/release/approval/evidence, review status/reviewer/evidence,
Bytewax routing, supported AI-agent runtimes/roles, human approval for
privileged actions, and denial of offensive or exploit automation scope.

## UI And Theming

The capability publishes compact APG Python UI metadata for dashboard,
authorities, indicators, sightings, enrichment, profiles, risk, incidents,
dissemination, reviews, agents, and settings. Theme metadata uses compact
density, 8px radius, operational status indicators, and component icon hints.

## Adapter Boundaries

Live SIEM/EDR/SOAR integrations, malware sandboxes, vulnerability scanners,
ticketing systems, asset inventories, blocklist deployment, containment
execution, storage backends, GraphRAG projections, dissemination delivery, and
durable Bytewax workers remain behind adapters. This package owns the contract,
guardrails, and local executable lifecycle.

## Acceptance Criteria

- `intel_cybint` is inspectable by the APG CLI as a valid capability contract.
- The package includes specification, plan, README, capability spec, contract,
  service, models, API helpers, views, app, semantic model, manifest, release
  report, and focused tests.
- Service methods enforce rules before mutation and keep tenant state isolated.
- UI routes include `/intel-cybint/agents`.
- Streaming metadata uses Bytewax and stream `apg.intel.cybint.lifecycle`.
