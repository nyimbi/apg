# APG Master Data Management Capability

`common/mdm` provides the master-data governance layer for APG applications.
It lets generated applications register tenant-scoped entities, score data
quality, review duplicate candidates, compose golden records, manage
cross-system identifiers, evaluate publish readiness, and retain audit evidence.
It preserves durable policy and review evidence for generated stewardship
queues and audit timelines.

The capability has two runtime surfaces:

- `MdmService`: a dependency-light control plane for generated applications,
  tests, local composition, UI models, and guardrail decisions.
- `MDMService`: the database-backed async runtime for production persistence,
  AI matching adapters, quality engines, metadata sync, cache, and event
  delivery.

The dependency-light control plane is the default surface for APG composition.
Adapters can provide richer scoring, matching, lineage, and persistence, but
they must honor the same capability contract and guardrails.

## What It Provides

- Tenant-scoped entity registration for customers, products, suppliers,
  employees, locations, assets, accounts, contracts, organizations, and custom
  entity types.
- Business-key and source-system tracking for every mastered entity.
- Six-dimensional quality assessment: completeness, accuracy, consistency,
  validity, uniqueness, and timeliness.
- Duplicate candidate creation with confidence scores and steward review.
- Golden-record creation and merge requests with survivorship policies.
- Per-field survivorship strategies: `most_recent`, `most_trusted`,
  `most_complete`, `majority_vote`, `concatenate`.
- Cross-reference mapping for external source-system identifiers.
- Publish-readiness gates that require owner and current quality evidence.
- Configurable per-(entity_type, channel) quality threshold enforcement.
- Restricted-data checks for data owner, audit evidence, and classification
  evidence.
- First-class data-agent registration for Codex, Claude Code, opencode, Pi,
  and future APG-compatible runtimes.
- Data-agent guardrails for supported roles, declared scope, owner, purpose,
  machine-contribution disclosure, and human approval for privileged roles,
  including durable `pending_review` records for otherwise valid privileged
  agents awaiting approval.
- Bytewax lifecycle batch validation for mastered entity, quality, duplicate,
  golden-record, publish, and data-agent streams, including persisted denial
  evidence before `PermissionError` on non-Bytewax batches.
- Hierarchical entity relationships: `parent_of`, `part_of`, `affiliated_with`,
  `supersedes`.
- In-memory entity resolution graph with edge types, confidence, and transitive
  cluster analysis.
- Point-in-time entity state reconstruction via audit event replay.
- Streaming quality degradation alerts and trend analysis.
- Async bulk quality assessment pipeline with semaphore-bounded concurrency.
- Cross-tenant federated entity search with attribute masking and consent
  auditing.
- Consent and purpose-limitation enforcement (GDPR Article 5(1)(b) alignment).
- Attribute change propagation from source entities to parent golden records.
- Probabilistic completeness profiling per attribute across entity-type cohorts.
- Business-key normalization and collision detection per entity type.
- Stewardship SLA tracker classifying pending reviews by urgency.
- Pending-review queues and policy evidence fields for entities, quality
  assessments, duplicate candidates, merge requests, cross references, publish
  records, data agents, and lifecycle batches.
- Generated-application route, theme, adapter, and view-model contracts.
- Bytewax lifecycle adapter boundary for publishing mastered changes.

## Core Lifecycle

1. Register a tenant-scoped entity with an entity type, business key, source
   system, owner, classification, and attributes.
2. Assess quality and store the latest quality evidence.
3. Create duplicate candidates from matching evidence.
4. Record steward review decisions for likely duplicates.
5. Create a golden record with a survivorship policy.
6. Evaluate merge requests, requiring independent stewardship when conflicts
   exist.
7. Attach source-system cross references with evidence.
8. Register data agents that can contribute to stewardship and publish-gate
   workflows.
9. Validate lifecycle batches through Bytewax before publishing operational
   evidence.
10. Publish the entity only when ownership and quality gates pass.
11. Preserve policy decisions, review reasons, review evidence, and audit
    events for every lifecycle decision.

## Quick Use

```python
from capabilities.common.mdm.service import MdmService

service = MdmService()

entity = service.register_entity(
    tenant_id="tenant-a",
    entity_id="cust-1",
    entity_type="customer",
    name="Acme Limited",
    business_key="ACME-001",
    source_system="crm",
    data_owner="steward-a",
)

service.assess_quality(
    tenant_id="tenant-a",
    entity_id=entity.entity_id,
    overall_score=92.0,
    dimensions={
        "completeness": 96.0,
        "accuracy": 91.0,
        "consistency": 90.0,
        "validity": 94.0,
        "uniqueness": 88.0,
        "timeliness": 93.0,
    },
    assessor="quality-engine",
)

publish = service.publish_entity(
    tenant_id="tenant-a",
    entity_id=entity.entity_id,
    channel="bytewax.entity_stream",
)

assert publish.status == "published"
```

Register a governed agent contributor:

```python
agent = service.register_data_agent(
    tenant_id="tenant-a",
    agent_id="publish-reviewer",
    name="Publish Reviewer",
    runtime="codex",
    role="publish_gate_reviewer",
    scope="customer publish readiness",
    owner="data-office",
    purpose="review mastered records before publication",
    human_approval_required=True,
)

batch = service.validate_mdm_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=12,
)

assert agent.runtime == "codex"
assert batch.status == "accepted"
```

## API Reference

| Method | Surface | Description |
|--------|---------|-------------|
| `register_entity` | `MdmService` | Register a tenant-scoped entity with guardrail evaluation |
| `assess_quality` | `MdmService` | Record six-dimensional quality evidence for an entity |
| `create_duplicate_candidate` | `MdmService` | Create a scored duplicate candidate and route for review |
| `review_duplicate_candidate` | `MdmService` | Record steward decision on a duplicate candidate |
| `create_golden_record` | `MdmService` | Compose a golden record from governed source entities |
| `merge_golden_record` | `MdmService` | Merge source entities into an existing golden record |
| `split_record` | `MdmService` | Remove source entities from a golden record composition |
| `update_cross_reference` | `MdmService` | Attach a source-system identifier mapping with evidence |
| `retire_entity` | `MdmService` | Retire an entity with lineage evidence requirement |
| `publish_entity` | `MdmService` | Evaluate and record publish readiness for a channel |
| `register_data_agent` | `MdmService` | Register a governed AI/ML data agent with role guardrails |
| `validate_mdm_lifecycle_batch` | `MdmService` | Validate that lifecycle mutations flow through Bytewax |
| `survivorship_rule` | `MdmService` | Define a per-field survivorship strategy for merges |
| `match_score` | `MdmService` | Compute a lightweight attribute match score between two entities |
| `entity_search` | `MdmService` | Filter entities by type, status, or minimum quality score |
| `entity_bulk_register` | `MdmService` | Register multiple entities with per-entity outcome reporting |
| `data_lineage` | `MdmService` | Return upstream/downstream lineage graph for an entity |
| `steward_assign` | `MdmService` | Assign a data steward to an entity |
| `dashboard_summary` | `MdmService` | Return summary metrics for generated MDM dashboards |
| `list_pending_reviews` | `MdmService` | Return all MDM records awaiting steward or human review |
| `subscription_notify` | `MdmService` | Notify downstream subscribers of an entity change event |
| `workflow_approve` | `MdmService` | Approve or reject a duplicate candidate (alias) |
| `domain_publish` | `MdmService` | Publish to a domain channel (alias for `publish_entity`) |
| `EntityService.create_entity` | `MDMService` | Create entity with version tracking and quality bootstrap |
| `EntityService.update_entity` | `MDMService` | Update entity with version tracking and quality re-assessment |
| `EntityService.search_entities` | `MDMService` | Advanced entity search with JSONB attribute filtering |
| `QualityService.assess_quality` | `MDMService` | AI-enhanced or rule-based quality assessment |
| `MatchingService.detect_duplicates` | `MDMService` | AI-powered duplicate detection with confidence tiers |

## World-Class Enhancements (v2.0)

The following 15 improvements define the v2.0 roadmap. They are specified in
`WORLD_CLASS_IMPROVEMENTS.md` and drive the next implementation cycle.

1. **Probabilistic deduplication via LSH blocking + Fellegi-Sunter scoring.**
   MinHash/SimHash blocking reduces comparison space 10-100x; Jaro-Winkler,
   Soundex, and token-set comparators produce calibrated m/u probability
   weights instead of hand-tuned thresholds.

2. **Async bulk quality assessment pipeline.**
   `bulk_assess_quality_async` fans out quality coroutines across a
   semaphore-bounded pool (default 32), accumulates results, and streams
   progress events. Enables nightly quality sweeps without blocking the event
   loop.

3. **Golden record attribute survivorship engine.**
   `resolve_golden_attributes` applies per-field strategies: `most_recent`,
   `most_trusted`, `most_complete`, `majority_vote`, `concatenate`.
   Field-level `survivorship_rule()` overrides the record-level policy.

4. **Deterministic entity resolution graph.**
   In-memory adjacency list keyed by `golden_record_id` tracks edges with
   type (`member`, `suspect_duplicate`, `split_from`), confidence, and
   timestamp. `entity_graph()` returns a networkx-compatible adjacency dict
   for transitive cluster analysis without a graph database.

5. **Data stewardship SLA tracker.**
   `stewardship_sla_report` computes age for every pending-review record,
   classifies items as within-SLA / warning / breached, and returns a ranked
   urgency list for dashboard and alert integrations.

6. **Incremental lineage graph with impact analysis.**
   `lineage_impact_analysis` performs BFS across golden-record and
   cross-reference chains to a configurable depth, annotating nodes with
   entity type and status. Critical for change-impact assessment before
   entity retirement or attribute rename.

7. **Composite business-key normalization and collision detection.**
   `normalize_business_key` applies entity-type-specific normalization (E.164,
   email domain stripping, tax-ID country prefixing). `detect_key_collision`
   checks the normalized form before registration, preventing silent duplicates
   from case/format variants.

8. **Configurable quality threshold enforcement.**
   `configure_quality_gate` stores per-(entity_type, channel) minimum scores
   and per-dimension floors. `publish_entity` returns a structured
   `gate_failures` list rather than a boolean pass/fail, enabling different
   quality bars for internal vs. external-facing channels.

9. **Audit event replay and point-in-time state projection.**
   `project_entity_state_at` replays audit events in chronological order up to
   a given timestamp, reconstructing the entity attribute snapshot. Enables
   point-in-time recovery and regulatory lookback without separate version
   tables.

10. **Cross-tenant entity federation.**
    `federated_entity_search` executes authorized cross-tenant queries returning
    attribute-masked results (non-restricted fields only) with per-tenant
    consent records audited. Enables hub-and-spoke MDM for shared reference
    data.

11. **Attribute change propagation to downstream golden records.**
    `propagate_attribute_changes` re-runs survivorship resolution for affected
    golden record fields on source entity update and emits a
    `golden_record.attribute_updated` audit event, keeping golden records
    continuously fresh.

12. **Consent and purpose-limitation enforcement.**
    `check_access_purpose` evaluates whether an accessor's declared purpose
    (e.g., `marketing`, `fraud_detection`, `regulatory_audit`) is permitted for
    the entity's classification and owner-defined purpose list. Returns
    allow/deny/require_review, aligned to GDPR Article 5(1)(b).

13. **Streaming quality degradation alerts.**
    `quality_trend_analysis` computes per-dimension score deltas across the last
    N assessments, flags entities where any dimension has degraded beyond a
    configurable threshold in the rolling window, and returns a prioritized
    alert list.

14. **Hierarchical entity relationships.**
    `register_entity_relationship` records typed relationships (`parent_of`,
    `part_of`, `affiliated_with`, `supersedes`) with evidence and actor.
    `get_entity_hierarchy` returns the relationship tree rooted at a given
    entity, enabling product hierarchies, org structures, and account-contact
    relationships.

15. **Probabilistic completeness profiling.**
    `profile_entity_completeness` scans all entities of a given type, returns
    per-attribute population rates (% non-null, % non-empty, % type-conforming),
    a recommended `required_fields` list for high-population attributes, and a
    sparse-attribute list indicating data model drift.

## New Methods

### `entity_bulk_register` — register a fleet of entities in one call

```python
outcomes = service.entity_bulk_register(
    tenant_id="tenant-a",
    data_owner="steward-a",
    entities=[
        {"entity_id": "prod-1", "entity_type": "product", "name": "Widget A",
         "business_key": "SKU-001", "source_system": "erp"},
        {"entity_id": "prod-2", "entity_type": "product", "name": "Widget B",
         "business_key": "SKU-002", "source_system": "erp"},
    ],
)
# [{"status": "registered", "entity_id": "prod-1", ...}, ...]
```

### `match_score` — lightweight attribute similarity between two entities

```python
result = service.match_score(
    tenant_id="tenant-a",
    entity_id_a="cust-1",
    entity_id_b="cust-2",
)
# {"match_score": 83.33, "confidence": "high", "matching_attribute_count": 5, ...}
```

### `survivorship_rule` — field-level override for golden record merges

```python
service.survivorship_rule(
    tenant_id="tenant-a",
    rule_id="rule-email-trusted",
    entity_type="customer",
    field="email",
    strategy="most_trusted",
    priority=10,
    owner="data-office",
)
```

### `data_lineage` — upstream / downstream lineage graph

```python
lineage = service.data_lineage(
    tenant_id="tenant-a",
    entity_id="cust-1",
    lineage_direction="both",
)
# {
#   "cross_references": [...],
#   "golden_record_id": "gr-...",
#   "golden_record_sources": ["cust-1", "cust-3"],
#   ...
# }
```

### `steward_assign` + `subscription_notify` — stewardship and event fan-out

```python
service.steward_assign(
    tenant_id="tenant-a",
    entity_id="cust-1",
    steward_id="steward-jane",
    role="data_steward",
    actor="data-office",
)

service.subscription_notify(
    tenant_id="tenant-a",
    entity_id="cust-1",
    event_type="entity.updated",
    subscriber_ids=["downstream-crm", "analytics-pipeline"],
    payload={"changed_field": "email"},
)
```

## Generated UI Surfaces

The capability contract exposes routes and view models for:

- Dashboard
- Entity workbench
- Golden records
- Quality console
- Duplicate review
- Stewardship queue
- Lineage trace
- Cross-reference console
- Publish readiness
- Analytics
- Audit timeline
- Adapter health
- Data-agent roster
- Lifecycle batch monitor
- Settings

`view_models.py` turns service state into generated-application models for these
surfaces. Rendering technology is intentionally outside this packet so APG can
target different UI shells.

## Guardrail Summary

MDM evaluates deterministic rules before lifecycle decisions. Key guardrails:

- Tenant context is required.
- Entity type must be supported.
- Business key is required.
- Restricted entities require a data owner, audit evidence, and classification
  evidence.
- Quality scores must be within range.
- Publish requires an owner and current quality assessment.
- Low-quality entities cannot be published.
- Likely duplicates require steward review.
- Golden-record merges require a survivorship policy.
- Conflicted merges require an independent steward.
- Cross-reference updates require source-system evidence.
- Entity retirement requires lineage evidence.
- Review decisions require notes.
- Data-agent runtime and role must be supported.
- Data-agent scope, owner, purpose, and machine-contribution disclosure are
  required.
- Privileged data-agent roles require human approval.
- Lifecycle batch processing must use Bytewax.
- Review-required and denied records must preserve policy and review evidence.

## Adapter Boundaries

This packet defines the executable control plane. Production adapters may supply:

- Database persistence through the existing async `MDMService`.
- AI-assisted entity matching and quality scoring via locally hosted Ollama
  models.
- Metadata catalog synchronization.
- Lineage graph persistence.
- Bytewax stream processing for mastered entity and data-agent lifecycle events.
- Cache, audit, search, and security integrations.

Adapters must not bypass the contract in `capability_contract.py`.
They must preserve `policy_decision`, `matched_rules`, `review_reasons`, and
`review_evidence` when syncing MDM records into external systems.

The MDM packet intentionally does not embed SDK clients for Codex, Claude Code,
opencode, Pi, or future agent providers. Those runtimes connect through
adapters that preserve the APG contract, guardrail decisions, audit events, and
human-approval requirements.

## Local Proof

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/mdm/capability_contract.py \
  capabilities/common/mdm/service.py \
  capabilities/common/mdm/api.py \
  capabilities/common/mdm/view_models.py \
  capabilities/common/mdm/app.py \
  capabilities/common/mdm/test_capability_contract.py \
  capabilities/common/mdm/tests/test_package_contract.py

./.venv/bin/pytest -q \
  capabilities/common/mdm/test_capability_contract.py \
  capabilities/common/mdm/tests/test_package_contract.py
```
