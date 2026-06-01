# Intelligence Analytics Capability Specification

## Purpose

Intelligence Analytics (`intel_analytics`) enables APG applications to compose
governed analytic workbenches across authorized fusion extracts, event streams,
entity tables, geospatial layers, graph projections, document corpora,
transaction sets, partner datasets, and metric series. It records lawful
authority, analytic workspaces, datasets, feature sets, analytic models, runs,
insights, dashboards, narratives, recommendations, reviews, lifecycle metadata,
UI metadata, theming, and provider-neutral AI-agent participation.

The capability is executable without live analytics infrastructure. Generated
applications can use the local runtime for tests and workflows, then provide
adapters for approved data warehouses, feature stores, model registries,
notebook runtimes, Bytewax topologies, evidence storage, graph projection, RAG
indexing, geospatial enrichment, notifications, and dashboard publication.

## Users

- Intelligence analysts building trend, anomaly, risk, forecast, cluster, and
  relationship insights.
- Fraud, threat, public-safety, incident, and operations teams running
  governed analytics.
- Compliance reviewers validating authority, lineage, privacy, evidence,
  model validation, approvals, and release markings.
- Application builders composing APG intelligence, safety, fraud, security, and
  operational analytics products.
- AI-agent operators who need provider-neutral analytic assistance with
  deterministic guardrails.

## Functional Scope

`intel_analytics` provides:

- Authority records with classification, approver, expiry, and evidence.
- Analytic workspaces with type, name, classification, authority, and evidence.
- Dataset registration with workspace, type, owner, lineage, retention, and
  evidence.
- Feature-set records with dataset linkage, references, confidence, analyst,
  and evidence.
- Analytic model records with feature-set linkage, objective, validation,
  risk level, and evidence.
- Run records with model linkage, run type, result reference, confidence,
  analyst, and evidence.
- Insight records with run linkage, claim reference, confidence, analyst, and
  evidence.
- Dashboard, narrative, recommendation, and review workflows with approvals and
  release evidence.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Deterministic rule evaluation for all write-path guardrails.
- Bytewax lifecycle metadata for composable event processing.
- UI route metadata, compact view models, and theme tokens.

## Out Of Scope

The capability does not implement live warehouse connectors, feature stores,
model training engines, notebook runtimes, automated decisioning, model
deployment, training-data disclosure, graph persistence, RAG indexing,
autonomous dissemination, or external dashboard delivery. These behaviors
remain behind explicit adapters or are denied by rule where they would violate
the capability guardrails.

## Lifecycle

1. Record lawful authority.
2. Open an analytic workspace under that authority.
3. Register authorized datasets with owner, lineage, and retention.
4. Record feature sets derived from datasets.
5. Record analytic models with validation evidence.
6. Record analytic runs and result references.
7. Record evidence-backed insights.
8. Record dashboards, narratives, or recommendations with approval.
9. Record review outcomes.
10. Emit Bytewax lifecycle metadata for every accepted mutation.
11. Allow AI agents only inside configured roles and approved scopes.

## Rules

All service methods evaluate rules before mutating state. Guardrails require
tenant context, write policy, lawful authority, supported types, dataset
lineage, retention class, confidence scores between 0 and 1, analyst
ownership, model validation evidence, publication approvals, review evidence,
Bytewax batch routing, supported AI-agent runtimes and roles, human approval
for privileged agent actions, and denial of hallucinated insights,
training-data leakage, privacy bypass, unsupported automated decisions,
unapproved model deployment, and autonomous dissemination.

## UI And Theme

The capability exposes generated-screen metadata for dashboards, authorities,
workspaces, datasets, feature sets, models, runs, insights, analytic
dashboards, narratives, recommendations, reviews, agents, and settings. Theme
tokens are compact, operational, and suitable for dense analytic review and
model-governance workflows.

## Adapter Boundaries

Adapters own live data ingestion, warehouses, feature stores, notebook runtimes,
model registries, model execution, graph writes, RAG indexing, geospatial
enrichment, evidence storage, durable Bytewax worker topology, notification
delivery, dashboard rendering, search, and dissemination delivery.
