# Geospatial Intelligence Capability Specification

## Purpose

`intel_geoint` makes lawful geospatial-intelligence coordination a first-class
APG capability. It provides the application contract for authorities, areas of
interest, imagery/geospatial sources, collection plans, observations, features,
change detections, assessments, dissemination, reviews, Bytewax lifecycle
metadata, UI/view composition, visual theming, and provider-neutral AI-agent
support.

The package is intentionally dependency-light and does not implement live
tasking, weapon targeting, harmful operational planning, surveillance delivery,
sensor control, or raw imagery exploitation engines. It models lawful authority,
area/source governance, retention, evidence, confidence, release control,
review, and adapter boundaries so generated applications can compose a governed
GEOINT workflow.

## Users

- Authorized intelligence, defense, security, public-safety, disaster response,
  and infrastructure teams.
- Oversight teams that need authority, data lineage, retention,
  classification, audit, and release evidence.
- Application builders composing geospatial, graph, RAG, NLP, audit,
  notifications, and authorization capabilities.
- AI-agent teams using Codex, Claude Code, OpenCode, or Pi for authority
  review, area planning, imagery triage, feature analysis, change analysis, and
  dissemination review.

## Functional Scope

- Record lawful geospatial authorities with scope, classification, approver,
  expiry, and evidence.
- Record areas of interest with geometry reference, owner, classification,
  authority, and evidence.
- Register imagery/geospatial sources with source type, sensor type, resolution
  class, owner, authority, and evidence.
- Record collection plans with authority, area, source, mode, retention,
  approval, and evidence.
- Record observations, features, changes, assessments, dissemination decisions,
  and review outcomes.
- Register provider-neutral AI agents and validate privileged or prohibited
  geospatial automation scopes.

## Rule Engine

Rules are deterministic and enforced before mutation. They cover tenant context,
write policy attachment, authority type/scope/classification/approver/expiry/
evidence, area name/geometry/classification/owner/authority/evidence, source
type/sensor/resolution/owner/authority/evidence, collection authority/area/
source/area-authority/source-authority/mode/retention/approval/evidence,
observation plan/reference/capture-time/accuracy/evidence, feature observation/
type/geometry/confidence/analyst/evidence, change feature/type/severity/
confidence/analyst/evidence, assessment change/type/classification/analyst/
evidence, dissemination assessment/audience/release/approval/evidence, review
status/reviewer/evidence, Bytewax routing, supported AI-agent runtimes/roles,
human approval for privileged actions, and denial of targeting or harmful
automation scope.

## UI And Theming

The capability publishes compact APG Python UI metadata for dashboard,
authorities, areas, sources, collection plans, observations, features, changes,
assessments, dissemination, reviews, agents, and settings. Theme metadata uses
compact density, 8px radius, operational status indicators, and component icon
hints.

## Adapter Boundaries

Live satellite/aerial tasking, sensor control, map tile rendering, GIS engines,
large imagery storage, computer vision extraction, geocoding, routing,
dissemination delivery, GraphRAG projection, and durable Bytewax workers remain
behind adapters. This package owns the contract, guardrails, and local
executable lifecycle.

## Acceptance Criteria

- `intel_geoint` is inspectable by the APG CLI as a valid capability contract.
- The package includes specification, plan, README, capability spec, contract,
  service, models, API helpers, views, app, semantic model, manifest, release
  report, and focused tests.
- Service methods enforce rules before mutation and keep tenant state isolated.
- UI routes include `/intel-geoint/agents`.
- Streaming metadata uses Bytewax and stream `apg.intel.geoint.lifecycle`.
