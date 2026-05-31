# MDM Capability Specification

## Purpose

`common/mdm` is the APG capability for governed master-data composition. It
provides a first-class, tenant-scoped control plane for the entity lifecycle,
quality evidence, duplicate review, golden-record composition, source-system
cross references, publish readiness, generated UI state, and audit evidence.

The specification is intentionally executable: every requirement below maps to
contract data, service behavior, view-model output, or focused tests in this
capability directory.

## Users

- Application builders who compose APG capabilities into executable ERP,
  CRM, commerce, HR, supply-chain, asset, and finance applications.
- Data stewards who review quality, duplicates, merges, and publish gates.
- Integration engineers who map source-system identifiers and publish mastered
  changes.
- Platform operators who connect persistence, matching, quality, lineage, audit,
  cache, and Bytewax stream adapters.
- AI/data-agent operators who register Codex, Claude Code, opencode, Pi, or
  other APG-compatible agents as governed contributors to stewardship and
  publishing workflows.

## Runtime Surfaces

### Generated-App Control Plane

`MdmService` is dependency-light and safe for generated applications. It stores
in-memory lifecycle records, evaluates deterministic guardrails, and emits audit
events. It is the composition baseline for APG package publication.

### Production Runtime

`MDMService` remains the async database-backed runtime. It owns durable
persistence, database sessions, AI matching engines, quality services, audit
logging, and integration adapters. It must follow decisions made by the
capability contract.

### First-Class Data Agents

AI/data agents are first-class MDM citizens. They can participate in review,
quality, duplicate matching, golden-record, survivorship, lineage, and publish
gate workflows when they are explicitly registered with runtime, role, scope,
owner, purpose, and contribution disclosure metadata. Privileged agent roles
must be approved for human-supervised operation before they can affect
publication-sensitive lifecycle steps.

The contract starts with supported runtimes `codex`, `claude_code`,
`opencode`, and `pi`. New runtimes must be added through the contract rather
than hidden inside service code.

## Functional Requirements

### Entity Registration

- Register entities per tenant.
- Require entity ID, entity type, name, business key, source system, and tenant.
- Support customer, product, supplier, employee, location, asset, account,
  contract, organization, and custom entity types.
- Require business keys for all entities.
- Reject unsupported entity types.
- Require owner, audit evidence, and classification evidence for restricted or
  sensitive data.

### Quality Management

- Store quality assessments for each entity.
- Support completeness, accuracy, consistency, validity, uniqueness, and
  timeliness scores.
- Reject scores outside `0..100`.
- Update the entity's latest quality evidence after accepted assessments.
- Preserve issues and recommendations as structured evidence.

### Duplicate Review

- Create duplicate candidates between tenant-local entities.
- Store confidence, reason, status, steward, notes, and review decision.
- Route likely duplicates to steward review.
- Require review notes for review decisions.

### Golden Records

- Create golden records from source entity IDs.
- Support survivorship policies:
  `most_recent`, `most_complete`, `most_trusted_source`, `highest_quality`,
  `custom_rules`, and `ai_determined`.
- Require survivorship policy for merge requests.
- Require independent steward review when conflicts are present.

### Cross References

- Map mastered entity IDs to source-system identifiers.
- Require source-system evidence before accepting cross-reference updates.

### Publish Readiness

- Publish decisions must require tenant context, data owner, current quality
  assessment, and quality score above the blocking threshold.
- Accepted publish decisions update entity status to `published`.
- Denied publish decisions must preserve matched rules for UI explanation.

### Retirement

- Entity retirement must require lineage evidence.
- Retirement decisions must be audited.

### Data-Agent Composition

- Register agent contributors per tenant.
- Require supported runtime, supported role, declared scope, owner, purpose,
  and machine-contribution disclosure.
- Require human approval for privileged roles that can influence duplicate
  matching, golden-record, survivorship, and publish-gate outcomes.
- Persist data-agent records in the generated-app control plane for UI display
  and audit evidence.
- Surface agent registration failures as matched guardrails, not generic
  runtime errors.

### Bytewax Lifecycle Batches

- Validate MDM lifecycle batch processing through Bytewax.
- Require lifecycle processors to declare the `bytewax` event stream.
- Track batch status, mutation count, and matched guardrails for UI/runtime
  evidence.
- Keep broker transports out of the core lifecycle contract; event brokers may
  exist behind adapters, but Bytewax is the required lifecycle processing
  engine for this packet.

### UI and Theming

- Expose generated-application routes for dashboard, entities, golden records,
  quality, duplicates, stewardship, lineage, cross references, publish,
  analytics, audit, adapters, data agents, lifecycle batches, and settings.
- Expose theme tokens and component metadata through the capability contract.
- Provide view models that turn service state into composable UI state without
  requiring a web framework.

### Audit

- Every lifecycle operation must append a tenant-scoped audit event.
- Audit events must record subject, actor, decision, matched rules, details, and
  timestamp.

## Guardrails

The deterministic rule engine must include at least these decisions:

- `tenant_context_required`
- `entity_type_must_be_supported`
- `business_key_required_for_entity`
- `restricted_entity_requires_data_owner`
- `entity_publish_requires_data_owner`
- `publish_requires_latest_quality_assessment`
- `low_quality_blocks_publish`
- `invalid_quality_score_blocks_assessment`
- `duplicate_candidates_require_review`
- `auto_merge_requires_high_confidence`
- `golden_record_merge_requires_survivorship`
- `conflicted_merge_requires_independent_steward`
- `restricted_entity_requires_audit_trail`
- `restricted_entity_requires_classification_evidence`
- `cross_reference_requires_source_evidence`
- `retire_requires_lineage_evidence`
- `review_decisions_require_notes`
- `data_agent_runtime_supported`
- `data_agent_role_supported`
- `data_agent_requires_scope`
- `data_agent_requires_owner`
- `data_agent_requires_purpose`
- `data_agent_requires_contribution_disclosure`
- `data_agent_privileged_role_requires_human_approval`
- `bytewax_mdm_stream_required`

## Adapter Requirements

Adapters can provide richer implementation behind the control plane:

- PostgreSQL persistence and transaction boundaries.
- AI or rules-based quality scoring.
- AI or deterministic duplicate matching.
- Metadata catalog synchronization.
- Bytewax lifecycle streams for mastered entity changes.
- Cache invalidation and lookup acceleration.
- Graph lineage persistence.
- External audit and security integration.

Adapters must receive and preserve guardrail decisions. They must not publish,
merge, retire, or mutate restricted data by bypassing `capability_contract.py`.

## Non-Goals For This Packet

- Rendering a production browser UI.
- Embedding Codex, Claude Code, opencode, Pi, or any other agent runtime
  client directly in MDM.
- Training matching or quality ML models.
- Running external databases, event brokers, caches, or metadata catalogs.
- Treating any broker as the core lifecycle processor.
- Replacing the existing database-backed `MDMService`.
- Running the full repository test suite during battery-constrained capability
  delivery.

## Acceptance Criteria

- Root `README.md`, `SPECIFICATION.md`, and `PLAN.md` exist.
- `capability_contract.py` exposes configuration, schema, rules, UI routes, and
  theme data.
- `service.py` exposes `MdmService` lifecycle behavior and audit records.
- `service.py` registers data agents and validates Bytewax lifecycle batches
  using deterministic guardrails.
- `view_models.py` exposes generated-application view models.
- `app.py` builds semantic model data from the live contract.
- Focused package tests prove guardrails, lifecycle, view models, registration,
  and publishable app evidence.
- Package manifest and release evidence reference current packet artifacts.
