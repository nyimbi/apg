# Data Correlation Capability Specification

## Purpose

Data Correlation (`intel_correlation`) enables APG applications to compose
governed cross-source correlation workflows across authorized fusion extracts,
graph projections, entity tables, event streams, geospatial layers,
transactions, document corpora, and partner datasets. It records lawful
authority, correlation workspaces, source lineage, entities, observations,
correlation rules, runs, match clusters, resolution decisions, referrals,
reviews, lifecycle metadata, UI metadata, theming, and provider-neutral
AI-agent participation.

The capability is executable without live matching infrastructure. Generated
applications can use the local runtime for tests and workflows, then provide
adapters for approved entity-resolution engines, fuzzy matching, graph
projection, geospatial joins, Bytewax topologies, evidence storage,
notifications, case writes, and search.

## Users

- Intelligence analysts correlating entities, events, locations, transactions,
  and documents across sources.
- Fraud, threat, public-safety, incident, and operations teams reviewing
  potential matches.
- Compliance reviewers validating authority, source lineage, evidence,
  identity-resolution approvals, and referral decisions.
- Application builders composing APG intelligence, safety, fraud, security, and
  operational correlation products.
- AI-agent operators who need provider-neutral matching assistance with
  deterministic guardrails.

## Functional Scope

`intel_correlation` provides:

- Authority records with classification, approver, expiry, and evidence.
- Correlation workspaces with type, name, classification, authority, and
  evidence.
- Source registration with workspace, type, custodian, lineage, and evidence.
- Entity and observation records with source/entity linkage, confidence, and
  evidence.
- Correlation rule records with thresholds, analyst, and evidence.
- Correlation runs with result references, confidence, analyst, and evidence.
- Match-cluster workflows with cluster type, reference, confidence, analyst,
  and evidence.
- Resolution decisions and referrals with rationale, approvals, and evidence.
- Review workflows with supported statuses and evidence.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Deterministic rule evaluation for all write-path guardrails.
- Bytewax lifecycle metadata for composable event processing.
- UI route metadata, compact view models, and theme tokens.

## Out Of Scope

The capability does not implement live matching engines, automatic identity
merge, fuzzy matching providers, graph persistence, geospatial join execution,
RAG indexing, autonomous referral, source alteration, evidence creation,
privacy bypass, or case-system delivery. These behaviors remain behind
explicit adapters or are denied by rule where they would violate the
capability guardrails.

## Lifecycle

1. Record lawful authority.
2. Open a correlation workspace under that authority.
3. Register authorized sources with custodian and lineage.
4. Record source-backed entities and observations.
5. Record a governed correlation rule.
6. Record a run against the rule.
7. Record a match cluster from the run.
8. Record a resolution decision with approval.
9. Record referral or review outcomes.
10. Emit Bytewax lifecycle metadata for every accepted mutation.
11. Allow AI agents only inside configured roles and approved scopes.

## Rules

All service methods evaluate rules before mutating state. Guardrails require
tenant context, write policy, lawful authority, supported types, source
lineage, entity/observation evidence, threshold and confidence scores between
0 and 1, analyst ownership, decision/referral approvals, review evidence,
Bytewax batch routing, supported AI-agent runtimes and roles, human approval
for privileged agent actions, and denial of unapproved identity merge, source
tampering, privacy bypass, evidence fabrication, autonomous referral, and
unreviewed high-impact match scopes.

## UI And Theme

The capability exposes generated-screen metadata for dashboards, authorities,
workspaces, sources, entities, observations, rules, runs, clusters, decisions,
referrals, reviews, agents, and settings. Theme tokens are compact,
operational, and suitable for dense match-review and identity-resolution
workflows.

## Adapter Boundaries

Adapters own live source ingestion, entity-resolution engines, fuzzy matching,
graph writes, geospatial joins, RAG indexing, evidence storage, durable Bytewax
worker topology, notification delivery, case-management writes, search, and
referral delivery.
