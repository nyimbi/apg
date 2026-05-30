# APG Master Data Management Capability

`common/mdm` provides the master-data governance layer for APG applications.
It lets generated applications register tenant-scoped entities, score data
quality, review duplicate candidates, compose golden records, manage
cross-system identifiers, evaluate publish readiness, and retain audit evidence.

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
- Cross-reference mapping for external source-system identifiers.
- Publish-readiness gates that require owner and current quality evidence.
- Restricted-data checks for data owner, audit evidence, and classification
  evidence.
- Generated-application route, theme, adapter, and view-model contracts.
- Bytewax-ready event-stream adapter boundary for publishing mastered changes.

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
8. Publish the entity only when ownership and quality gates pass.
9. Preserve audit events for every lifecycle decision.

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

## Adapter Boundaries

This packet defines the executable control plane. Production adapters may supply:

- Database persistence through the existing async `MDMService`.
- AI-assisted entity matching and quality scoring.
- Metadata catalog synchronization.
- Lineage graph persistence.
- Bytewax stream processing for mastered entity events.
- Cache, audit, search, and security integrations.

Adapters must not bypass the contract in `capability_contract.py`.

## Local Proof

Focused proof for this package:

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
