# META Capability Specification

## Purpose

`common/meta` is the APG capability for metadata cataloging and governance. It
provides an executable, tenant-scoped control plane for asset registration,
approved discovery, classification, lineage, quality, certification, glossary
management, publication, retirement, generated UI state, and audit evidence.

## Users

- Application builders composing APG capabilities into executable business
  systems.
- Data stewards reviewing sensitive classifications and certification evidence.
- Data engineers registering source systems, discovery jobs, and lineage.
- Governance teams managing glossary terms, ownership, and audit evidence.
- Platform operators connecting discovery, classification, lineage, search,
  metadata-store, and Bytewax stream adapters.
- AI/catalog-agent operators who register Codex, Claude Code, opencode, Pi, or
  other APG-compatible agents as governed metadata contributors.

## Runtime Surfaces

### Generated-App Control Plane

`MetaService` is dependency-light and safe for generated applications. It stores
in-memory lifecycle records, evaluates guardrails, exposes dashboard summaries,
and records audit events without requiring a database, search index, or AI
runtime.

### Production Runtime

`APGMetadataService` remains the production orchestration surface for durable
persistence, discovery connectors, AI classification, lineage engines, search
indexes, and APG integration adapters.

### First-Class Catalog Agents

AI/catalog agents are first-class metadata participants. They can assist with
discovery review, classification review, lineage review, glossary ownership,
certification, stewardship, and publish gates when they are registered with
runtime, role, scope, owner, purpose, and contribution disclosure metadata.
Privileged catalog-agent roles must be human approved before they can influence
classification, lineage, certification, or publication-sensitive decisions.

The contract starts with supported runtimes `codex`, `claude_code`,
`opencode`, and `pi`. New runtimes must be added through the contract instead
of hidden inside service code.

## Functional Requirements

### Asset Catalog

- Register tenant-scoped assets.
- Require asset ID, supported asset type, name, business key, source system,
  and tenant.
- Support database, schema, table, column, file, API, stream, report,
  dashboard, model, pipeline, and business-term assets.
- Track owner, steward, sensitivity, tags, metadata, quality, classification,
  lineage, status, and audit evidence.

### Discovery

- Schedule discovery jobs with connector type, source system, schedule, and
  approval evidence.
- Deny unapproved connectors.
- Route stale or unreviewed schedules to review.
- Record discovery results as asset IDs.

### Classification

- Store classification label, confidence, completion state, and review state.
- Route low-confidence classifications to steward review.
- Require review notes for classification review decisions.
- Require complete classification for restricted assets before publication.

### Lineage

- Capture lineage edges between registered assets.
- Deny lineage when source or target asset is missing.
- Route excessive lineage depth to review.
- Mark participating assets as lineage-covered after accepted lineage capture.

### Quality

- Store metadata quality scores and dimensions.
- Reject scores outside `0..100`.
- Update asset quality readiness after accepted assessment.

### Certification

- Require lineage evidence.
- Require quality above the configured certification threshold.
- Require freshness review for stale assets.
- Mark accepted assets as certified.

### Glossary

- Register terms with definitions, owners, and linked asset IDs.
- Deny terms without accountable owners.

### Publication And Retirement

- Publish only when owner, quality, classification, and steward checks pass.
- Retire only when impact-analysis evidence is present.
- Audit every publication and retirement decision.

### Catalog-Agent Composition

- Register agent contributors per tenant.
- Require supported runtime, supported role, declared scope, owner, purpose,
  and machine-contribution disclosure.
- Require human approval for privileged roles that can influence
  classification, lineage, certification, and publish-gate outcomes.
- Persist catalog-agent records in the generated-app control plane for UI
  display and audit evidence.
- Surface catalog-agent registration failures as matched guardrails.

### Bytewax Lifecycle Batches

- Validate metadata lifecycle batch processing through Bytewax.
- Require lifecycle processors to declare the `bytewax` event stream.
- Track batch status, mutation count, and matched guardrails for UI/runtime
  evidence.
- Keep broker transports out of the core lifecycle contract; event brokers may
  exist behind adapters, but Bytewax is the required lifecycle processing
  engine for this packet.

### UI And Theme

- Expose routes and view models for dashboard, catalog, discovery, lineage,
  classification, quality, certification, glossary, impact analysis, search,
  audit, adapters, catalog agents, lifecycle batches, and settings.
- Expose compact catalog-console theme tokens and component metadata.

## Guardrails

The deterministic rule engine must include at least:

- `tenant_context_required`
- `asset_type_must_be_supported`
- `asset_registration_requires_business_key`
- `asset_registration_requires_source_system`
- `published_asset_requires_owner`
- `publish_requires_quality_assessment`
- `restricted_asset_requires_classification`
- `sensitive_asset_requires_steward`
- `certified_asset_requires_lineage`
- `certification_requires_quality_threshold`
- `classification_review_requires_notes`
- `low_classification_confidence_requires_review`
- `discovery_requires_approved_connector`
- `discovery_schedule_requires_review`
- `lineage_requires_registered_assets`
- `lineage_depth_requires_review`
- `glossary_term_requires_owner`
- `retire_asset_requires_impact_analysis`
- `stale_asset_requires_review`
- `catalog_agent_runtime_supported`
- `catalog_agent_role_supported`
- `catalog_agent_requires_scope`
- `catalog_agent_requires_owner`
- `catalog_agent_requires_purpose`
- `catalog_agent_requires_contribution_disclosure`
- `catalog_agent_privileged_role_requires_human_approval`
- `bytewax_meta_stream_required`

## Adapter Requirements

Production adapters may provide:

- Durable metadata store persistence.
- Connector execution and schema extraction.
- AI or deterministic classification.
- Lineage graph persistence and traversal.
- Search index updates and query execution.
- Bytewax lifecycle streams.
- APG audit, auth, MDM, ETL, connector, monitoring, and notification
  integrations.

Adapters must preserve and expose guardrail decisions rather than bypassing the
contract.

## Non-Goals For This Packet

- Running live discovery connectors.
- Embedding Codex, Claude Code, opencode, Pi, or any other agent runtime
  client directly in META.
- Training or serving AI classification models.
- Rendering production browser UI.
- Running external databases, graph stores, search indexes, caches, event
  brokers, or Bytewax flows.
- Treating any broker as the core lifecycle processor.
- Replacing `APGMetadataService`.
- Running the full repository test suite during battery-constrained capability
  delivery.

## Acceptance Criteria

- Root `README.md`, `SPECIFICATION.md`, and `PLAN.md` exist.
- `capability_contract.py` exposes configuration, schema, rules, routes, and
  theme data.
- `service.py` exposes `MetaService` lifecycle behavior and audit records.
- `service.py` registers catalog agents and validates Bytewax lifecycle batches
  using deterministic guardrails.
- `view_models.py` exposes generated-application view models.
- `app.py` builds semantic model data from the live contract.
- Focused tests prove guardrails, lifecycle behavior, view models,
  registration metadata, and publishable app evidence.
- Package manifest and release evidence reference current packet artifacts.
