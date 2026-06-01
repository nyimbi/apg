# Intelligence Fusion Capability Specification

## Purpose

Intelligence Fusion (`intel_fusion`) enables APG applications to compose
evidence-led fusion workspaces across authorized OSINT, SIGINT, HUMINT, GEOINT,
CYBINT, FININT, SOCINT, dark-web, radio, monitoring, and partner-report
sources. It records lawful authority, fusion workspaces, source lineage,
evidence artifacts, correlations, hypotheses, assessments, referrals,
dissemination, reviews, lifecycle metadata, UI metadata, theming, and
provider-neutral AI-agent participation.

The capability is executable without live source connectors. Generated
applications can use the local runtime for tests and workflows, then provide
adapters for approved collection feeds, Bytewax topologies, evidence storage,
graph projection, RAG indexing, geospatial enrichment, notifications, and
dissemination delivery.

## Users

- Intelligence analysts building cross-source operational pictures.
- Fraud, threat, public-safety, and incident teams correlating source evidence.
- Compliance reviewers validating authority, source lineage, privacy, evidence,
  approvals, and release markings.
- Application builders composing APG intelligence, safety, fraud, security, and
  operational command products.
- AI-agent operators who need provider-neutral analytic assistance with
  deterministic guardrails.

## Functional Scope

`intel_fusion` provides:

- Authority records with classification, approver, expiry, and evidence.
- Fusion workspaces with type, name, classification, authority, and evidence.
- Source registration with source type, custodian, authority, lineage, and
  evidence.
- Artifact records with workspace/source alignment, references, fingerprints,
  confidence, and evidence.
- Correlation workflows with artifact linkage, correlation type, analyst,
  confidence, and evidence.
- Hypothesis workflows with correlation linkage, claim reference, analyst,
  confidence, and evidence.
- Assessment workflows with hypothesis linkage, assessment type, risk level,
  analyst, confidence, and evidence.
- Referral, dissemination, and review workflows with approval and release
  evidence.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Deterministic rule evaluation for all write-path guardrails.
- Bytewax lifecycle metadata for composable event processing.
- UI route metadata, compact view models, and theme tokens.

## Out Of Scope

The capability does not implement live collection feeds, cross-tenant data
movement, automated identity resolution, graph persistence, RAG indexing,
autonomous dissemination, source alteration, evidence creation, privacy bypass,
or external release delivery. These behaviors remain behind explicit adapters
or are denied by rule where they would violate the capability guardrails.

## Lifecycle

1. Record lawful authority.
2. Open a fusion workspace under that authority.
3. Register authorized sources with custodians and lineage.
4. Record evidence artifacts aligned to a workspace and source authority.
5. Record correlations from artifacts.
6. Record hypotheses from correlations.
7. Record assessments from hypotheses.
8. Record referrals or dissemination with approval.
9. Record review outcomes.
10. Emit Bytewax lifecycle metadata for every accepted mutation.
11. Allow AI agents only inside configured roles and approved scopes.

## Rules

All service methods evaluate rules before mutating state. Guardrails require
tenant context, write policy, lawful authority, supported types, source lineage,
workspace/source authority alignment, artifact fingerprints, confidence scores
between 0 and 1, analyst ownership, referral/dissemination approvals, review
evidence, Bytewax batch routing, supported AI-agent runtimes and roles, human
approval for privileged agent actions, and denial of evidence fabrication,
source tampering, privacy bypass, unsupported identity resolution, autonomous
dissemination, and unapproved attribution.

## UI And Theme

The capability exposes generated-screen metadata for dashboards, authorities,
workspaces, sources, artifacts, correlations, hypotheses, assessments,
referrals, dissemination, reviews, agents, and settings. Theme tokens are
compact, operational, and suitable for dense evidence review and analyst
workbench workflows.

## Adapter Boundaries

Adapters own live source ingestion, partner feeds, graph writes, RAG indexing,
geospatial enrichment, entity-resolution engines, evidence storage, durable
Bytewax worker topology, notification delivery, case-management writes, search,
and dissemination delivery.
