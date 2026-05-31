# APG Metadata Management Capability

`common/meta` provides the metadata catalog and governance layer for APG
applications. It lets generated applications register metadata assets, schedule
approved discovery, classify sensitive assets, capture lineage, assess metadata
quality, certify governed assets, manage glossary terms, evaluate publication
and retirement decisions, and preserve audit evidence.

The capability has two runtime surfaces:

- `MetaService`: a dependency-light control plane for generated applications,
  tests, local composition, UI models, and guardrail decisions.
- `APGMetadataService`: the production runtime that orchestrates database,
  discovery, AI classification, lineage, search, and APG integration adapters.

Generated APG applications should use `MetaService` first. Production adapters
can attach richer discovery, classification, lineage, search, persistence, and
Bytewax event streams behind the same contract.

## What It Provides

- Tenant-scoped asset catalog for databases, schemas, tables, columns, files,
  APIs, streams, reports, dashboards, models, pipelines, and glossary terms.
- Approved discovery scheduling for databases, files, APIs, streams, ML
  systems, and external catalogs.
- Classification evidence with confidence, sensitivity labels, and steward
  review.
- Lineage capture between registered source and target assets.
- Metadata quality assessment across completeness, freshness, accuracy,
  lineage, classification, and usage.
- Certification gates for governed assets.
- Business glossary ownership and asset links.
- Publication and retirement guardrails.
- First-class catalog-agent registration for Codex, Claude Code, opencode, Pi,
  and future APG-compatible runtimes.
- Catalog-agent guardrails for supported roles, declared scope, owner, purpose,
  machine-contribution disclosure, and human approval for privileged roles.
- Bytewax lifecycle batch validation for asset, discovery, classification,
  lineage, quality, certification, glossary, and catalog-agent streams.
- Generated-application UI routes, view models, theme tokens, and adapter
  metadata.

## Core Lifecycle

1. Register metadata assets with tenant, type, business key, source system,
   owner, steward, sensitivity, tags, and metadata.
2. Schedule discovery only with approved connectors and reviewed schedules.
3. Classify sensitive assets and route low-confidence results to stewardship.
4. Capture lineage between registered assets.
5. Assess metadata quality.
6. Request certification after quality and lineage evidence are present.
7. Publish assets only when owner, quality, classification, and steward gates
   pass.
8. Register catalog agents that can contribute to metadata governance and
   publish-gate workflows.
9. Validate lifecycle batches through Bytewax before publishing operational
   evidence.
10. Manage glossary terms with accountable owners.
11. Retire assets only after impact-analysis evidence exists.
12. Preserve audit events for every lifecycle decision.

## Quick Use

```python
from capabilities.common.meta.service import MetaService

service = MetaService()

asset = service.register_asset(
    tenant_id="tenant-a",
    asset_id="warehouse.customers",
    asset_type="table",
    name="customers",
    business_key="warehouse.public.customers",
    source_system="warehouse",
    owner="data-owner",
    steward="data-steward",
    sensitivity="restricted",
)

service.classify_asset(
    tenant_id="tenant-a",
    asset_id=asset.asset_id,
    label="pii",
    confidence=0.96,
    classification_complete=True,
    steward_review_recorded=True,
)

service.assess_quality(
    tenant_id="tenant-a",
    asset_id=asset.asset_id,
    score=91.0,
    dimensions={"completeness": 95.0, "freshness": 90.0},
    assessor="quality-engine",
)

published = service.publish_asset(
    tenant_id="tenant-a",
    asset_id=asset.asset_id,
)

assert published.status == "published"
```

Register a governed catalog-agent contributor:

```python
agent = service.register_catalog_agent(
    tenant_id="tenant-a",
    agent_id="classification-reviewer",
    name="Classification Reviewer",
    runtime="codex",
    role="classification_reviewer",
    scope="restricted metadata classification",
    owner="metadata-office",
    purpose="review sensitive classification evidence",
    human_approval_required=True,
)

batch = service.validate_meta_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=8,
)

assert agent.runtime == "codex"
assert batch.status == "accepted"
```

## Generated UI Surfaces

`capability_contract.py` and `view_models.py` expose:

- Dashboard
- Asset catalog
- Discovery console
- Lineage viewer
- Classification review
- Quality console
- Certification queue
- Business glossary
- Impact analysis
- Search
- Audit timeline
- Adapter health
- Catalog-agent roster
- Lifecycle batch monitor
- Settings

The packet does not require a particular web framework. Generated APG targets
can render these models in their own UI shells.

## Guardrail Summary

META evaluates deterministic rules before lifecycle decisions. Key guardrails:

- Tenant context is required.
- Asset type must be supported.
- Business key and source system are required for registration.
- Published assets require owners and quality evidence.
- Restricted assets require classification and stewards.
- Certification requires lineage and quality above threshold.
- Low-confidence classifications require steward review.
- Classification review decisions require notes.
- Discovery requires approved connectors and current schedule review.
- Lineage requires registered source and target assets.
- Excessive lineage depth requires review.
- Glossary terms require owners.
- Asset retirement requires impact analysis.
- Stale assets require freshness review before certification.
- Catalog-agent runtime and role must be supported.
- Catalog-agent scope, owner, purpose, and machine-contribution disclosure are
  required.
- Privileged catalog-agent roles require human approval.
- Lifecycle batch processing must use Bytewax.

## Adapter Boundaries

This packet defines the executable control plane. Production adapters may supply:

- Durable metadata store persistence.
- Discovery connector execution.
- AI or rules-based classification.
- Lineage graph persistence and traversal.
- Search index maintenance.
- Bytewax lifecycle streams for metadata and catalog-agent events.
- APG audit, auth, MDM, ETL, connector, monitoring, and notification
  integration.

Adapters must not bypass `capability_contract.py` decisions.

The META packet intentionally does not embed SDK clients for Codex, Claude
Code, opencode, Pi, or future agent providers. Those runtimes connect through
adapters that preserve the APG contract, guardrail decisions, audit events, and
human-approval requirements.

## Local Proof

Focused proof for this package:

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/meta/__init__.py \
  capabilities/common/meta/capability_contract.py \
  capabilities/common/meta/service.py \
  capabilities/common/meta/api.py \
  capabilities/common/meta/view_models.py \
  capabilities/common/meta/app.py \
  capabilities/common/meta/test_capability_contract.py \
  capabilities/common/meta/tests/test_package_contract.py

./.venv/bin/pytest -q \
  capabilities/common/meta/test_capability_contract.py \
  capabilities/common/meta/tests/test_package_contract.py
```
