# MDM Capability — User Guide

`capabilities/common/mdm` — Master Data Management control plane for APG.

Author: Nyimbi Odero | © 2025 Datacraft

---

## Overview

The MDM capability provides a dependency-light `MdmService` control plane plus an optional database-backed `MDMService` runtime. Generated applications use `MdmService` for all lifecycle operations; production deployments may add adapters for persistence, AI matching, and stream processing.

---

## Installation

The capability ships as part of the APG workspace. No additional dependencies are required for the `MdmService` control plane. The database-backed `MDMService` requires `asyncpg`.

---

## Core Concepts

| Concept | Description |
|---|---|
| Entity | A tenant-scoped master data record (customer, product, supplier, etc.) |
| Business Key | Unique identifier for an entity within its source system |
| Golden Record | The canonical, survivorship-resolved view of merged source entities |
| Quality Assessment | Six-dimensional (completeness, accuracy, consistency, validity, uniqueness, timeliness) score 0–100 |
| Duplicate Candidate | A suspected match between two entities requiring steward review |
| Cross Reference | A source-system identifier mapped to a master entity |
| Survivorship Policy | Strategy for resolving conflicting attribute values during merges |
| Data Agent | A registered AI/automation contributor to MDM stewardship workflows |
| Lifecycle Batch | A validated Bytewax event-stream batch for mastered entity mutations |

---

## Lifecycle

```
register_entity
    → assess_quality
    → create_duplicate_candidate  (if match found)
        → review_duplicate_candidate  (steward decision)
    → create_golden_record
        → merge_golden_record  (add sources)
        → resolve_golden_attributes  (apply survivorship)
    → update_cross_reference
    → publish_entity
```

---

## Quickstart

```python
from capabilities.common.mdm.service import MdmService

svc = MdmService(tenant_id="acme")

# 1. Register an entity
entity = svc.register_entity(
    tenant_id="acme",
    entity_id="cust-001",
    entity_type="customer",
    name="Acme Corp",
    business_key="ACME-001",
    source_system="crm",
    data_owner="steward-jane",
)

# 2. Score data quality
svc.assess_quality(
    tenant_id="acme",
    entity_id="cust-001",
    overall_score=91.0,
    dimensions={
        "completeness": 95.0, "accuracy": 90.0, "consistency": 89.0,
        "validity": 93.0, "uniqueness": 88.0, "timeliness": 91.0,
    },
    assessor="quality-engine-v2",
)

# 3. Publish
pub = svc.publish_entity(tenant_id="acme", entity_id="cust-001", channel="erp.master")
assert pub.status == "published"
```

---

## Entity Registration

```python
entity = svc.register_entity(
    tenant_id="acme",
    entity_id="prod-42",
    entity_type="product",          # customer, product, supplier, employee, location, asset, account, contract, organization
    name="Widget Pro",
    business_key="WGT-PRO-001",
    source_system="pim",
    data_owner="product-team",
    classification="internal",      # internal | restricted | confidential | sensitive
    attributes={"sku": "WGT-001", "category": "hardware", "price": 49.99},
    audit_evidence="audit-ref-2025-06",          # required for restricted/confidential
    classification_evidence="class-ref-2025-06", # required for restricted/confidential
)
```

**Guardrails enforced at registration:**
- Tenant context required.
- Entity type must be in the supported list.
- Business key required (non-empty).
- Restricted/confidential entities require `data_owner`, `audit_evidence`, and `classification_evidence`.

---

## Quality Assessment

```python
qa = svc.assess_quality(
    tenant_id="acme",
    entity_id="cust-001",
    overall_score=88.5,
    dimensions={
        "completeness": 92.0,
        "accuracy": 87.0,
        "consistency": 86.0,
        "validity": 90.0,
        "uniqueness": 85.0,
        "timeliness": 90.0,
    },
    assessor="auto-scorer",
    issues=[{"field": "phone", "type": "accuracy", "severity": "low", "message": "Non-standard format"}],
    recommendations=["Standardise phone to E.164"],
)
print(qa.status)         # "accepted"
print(qa.overall_score)  # 88.5
```

---

## Bulk Quality Assessment (async)

Fan-out quality scoring across many entities with bounded concurrency:

```python
import asyncio

result = asyncio.run(svc.bulk_assess_quality_async(
    tenant_id="acme",
    entity_ids=["cust-001", "cust-002", "cust-003"],
    assessor="nightly-sweep",
    concurrency=32,
))
print(result["assessed"], result["avg_score"])
```

Provide an optional `dimensions_fn` async callable `(entity_id) -> (float, dict[str, float])` to plug in an external scorer.

---

## Duplicate Detection and Review

```python
# Create a candidate
candidate = svc.create_duplicate_candidate(
    tenant_id="acme",
    entity_id="cust-001",
    candidate_entity_id="cust-002",
    confidence=87.5,           # 0–100
    reason="Matching name and postcode",
    steward_review_recorded=False,
)

# Steward decision
reviewed = svc.review_duplicate_candidate(
    candidate_id=candidate.candidate_id,
    steward="steward-jane",
    review_decision="merge",   # merge | keep_separate | defer
    review_notes="Confirmed same legal entity via Companies House",
)
```

Candidates with `confidence >= 85` automatically enter `review_required` status. Steward notes are mandatory.

---

## Golden Records

```python
# Create
gr = svc.create_golden_record(
    tenant_id="acme",
    entity_type="customer",
    source_entity_ids=["cust-001", "cust-002"],
    survivorship_policy="most_recent",   # most_recent | most_trusted | most_complete | majority_vote
    attributes={"segment": "enterprise"},
)

# Add more sources
svc.merge_golden_record(
    tenant_id="acme",
    golden_record_id=gr.golden_record_id,
    source_entity_ids=["cust-003"],
    survivorship_policy="most_recent",
    conflict_present=False,
)

# Resolve canonical attributes
import asyncio
resolved = asyncio.run(svc.resolve_golden_attributes(
    tenant_id="acme",
    golden_record_id=gr.golden_record_id,
))
print(resolved["resolved_attributes"])
print(resolved["field_provenance"])   # which source entity won each field
```

### Survivorship Strategies

| Strategy | Behaviour |
|---|---|
| `most_recent` | Picks value from the source entity with the latest `updated_at` |
| `most_trusted` | Uses source system trust ranking: crm > erp > hr > legacy > bulk |
| `most_complete` | Picks from the source entity with the most populated attributes |
| `majority_vote` | Uses the modal value across all sources |

Field-level overrides can be defined with `survivorship_rule()`.

---

## Cross References

```python
xref = svc.update_cross_reference(
    tenant_id="acme",
    entity_id="cust-001",
    source_system="sap",
    source_identifier="SAP-CUST-99201",
    evidence_reference="mapping-doc-2025-06",
)
```

Evidence reference is required; omitting it triggers `require_review`.

---

## Entity Relationships and Hierarchy

```python
import asyncio

# Register a parent–child relationship
asyncio.run(svc.register_entity_relationship(
    tenant_id="acme",
    parent_entity_id="org-001",
    child_entity_id="dept-finance",
    relationship_type="parent_of",   # parent_of | part_of | affiliated_with | supersedes
    evidence="org-structure-2025",
    actor="hr-system",
))

# Retrieve the full hierarchy tree
tree = asyncio.run(svc.get_entity_hierarchy(
    tenant_id="acme",
    root_entity_id="org-001",
    relationship_type="parent_of",
    max_depth=5,
))
for node in tree["tree"]:
    print(node["entity_id"], "->", [c["entity_id"] for c in node["children"]])
```

---

## Lineage and Impact Analysis

```python
impact = asyncio.run(svc.lineage_impact_analysis(
    tenant_id="acme",
    entity_id="cust-001",
    direction="both",    # upstream | downstream | both
    max_depth=5,
))
print(impact["node_count"], impact["edge_count"])
for node in impact["nodes"]:
    print(node["entity_id"], node["entity_type"], "depth:", node["depth"])
```

---

## Stewardship SLA Monitoring

```python
report = asyncio.run(svc.stewardship_sla_report(
    tenant_id="acme",
    warning_hours=24.0,
    breach_hours=72.0,
))
print(f"Pending: {report['total_pending']}, Breached: {report['breached']}, Warning: {report['warning']}")
for item in report["items"]:
    if item["sla_status"] == "breached":
        print("BREACHED:", item.get("entity_id") or item.get("candidate_id"), f"{item['age_hours']:.1f}h")
```

---

## Quality Trend Analysis

```python
# Single entity trend
trend = asyncio.run(svc.quality_trend_analysis(
    tenant_id="acme",
    entity_id="cust-001",
    window=5,
    degradation_threshold=5.0,
))

# Cohort-level trend
cohort_trend = asyncio.run(svc.quality_trend_analysis(
    tenant_id="acme",
    entity_type="customer",
    window=10,
    degradation_threshold=3.0,
))
for flagged in cohort_trend["flagged_entities"]:
    print(flagged["entity_id"], "degraded dims:", flagged["degraded_dimensions"])
```

---

## Attribute Completeness Profiling

```python
profile = asyncio.run(svc.profile_entity_completeness(
    tenant_id="acme",
    entity_type="customer",
    population_threshold=70.0,
))
print("Recommended required fields:", profile["recommended_required_fields"])
print("Sparse attributes:", profile["sparse_attributes"])
for attr, stats in profile["attribute_profiles"].items():
    print(f"  {attr}: {stats['population_rate']}% populated")
```

---

## Access Purpose Enforcement

```python
decision = asyncio.run(svc.check_access_purpose(
    tenant_id="acme",
    entity_id="cust-001",
    accessor_id="analyst-bob",
    purpose="marketing",
    permitted_purposes=["regulatory_audit", "fraud_detection", "operations"],
))
print(decision["decision"])  # "deny" — marketing not in permitted list
```

Default permitted purposes: `regulatory_audit`, `fraud_detection`, `operations`, `data_quality`, `stewardship`.

---

## Business Key Normalisation and Collision Detection

```python
# Normalise before registering
norm = asyncio.run(svc.normalize_business_key(
    entity_type="customer",
    raw_key="Acme Ltd",
    source_system="crm",
))
# norm["normalized_key"] == "acme"

# Check for collision before insert
collision = asyncio.run(svc.detect_key_collision(
    tenant_id="acme",
    entity_type="customer",
    normalized_key=norm["normalized_key"],
))
if collision["collision"]:
    print("Duplicate key — conflicting entities:", collision["conflicting_entity_ids"])
```

Normalisation strategies by entity type:

| Entity type | Strategy |
|---|---|
| `customer` | Lowercase, strip legal suffixes (ltd, inc, llc, plc, corp) |
| `product` | Uppercase, spaces → hyphens |
| `supplier` / `vendor` | Uppercase trim |
| default | Uppercase trim |

---

## Data Agent Registration

```python
agent = svc.register_data_agent(
    tenant_id="acme",
    agent_id="quality-scorer-v2",
    name="Quality Scorer V2",
    runtime="codex",           # codex | claude_code | opencode | pi
    role="quality_assessor",   # non-privileged
    scope="customer quality scoring",
    owner="data-office",
    purpose="automated quality scoring pipeline",
    contribution_disclosed=True,
    human_approval_required=False,
)

# Privileged roles (merge_decision_maker, golden_record_curator, etc.)
# require human_approval_required=True — record lands in pending_review
priv_agent = svc.register_data_agent(
    tenant_id="acme",
    agent_id="merge-agent-v1",
    name="Merge Decision Agent",
    runtime="claude_code",
    role="merge_decision_maker",
    scope="customer golden record merges",
    owner="data-office",
    purpose="AI-assisted merge decisions with human sign-off",
    contribution_disclosed=True,
    human_approval_required=True,
)
assert priv_agent.status == "pending_review"
```

---

## Lifecycle Batch Validation (Bytewax)

```python
batch = svc.validate_mdm_lifecycle_batch(
    tenant_id="acme",
    event_stream="bytewax",
    mutation_count=150,
)
assert batch.status == "accepted"

# Non-Bytewax streams raise PermissionError with audit evidence preserved
try:
    svc.validate_mdm_lifecycle_batch(tenant_id="acme", event_stream="kafka", mutation_count=10)
except PermissionError as e:
    print("Blocked:", e)
```

---

## Publishing an Entity

```python
pub = svc.publish_entity(
    tenant_id="acme",
    entity_id="cust-001",
    channel="data-platform.master-customer",
)
# Requires: data_owner assigned AND latest quality assessment present AND quality_score > 0
assert pub.status == "published"
```

---

## Dashboard Summary

```python
summary = svc.dashboard_summary(tenant_id="acme")
# Keys: entity_count, quality_assessment_count, duplicate_review_count,
#       golden_record_count, pending_merge_count, published_entity_count,
#       data_agent_count, pending_data_agent_review_count,
#       lifecycle_batch_count, denied_lifecycle_batch_count,
#       pending_review_count, audit_event_count
```

---

## Pending Review Queue

```python
items = svc.list_pending_reviews(tenant_id="acme")
for item in items:
    print(item["status"], item.get("entity_id") or item.get("candidate_id"))
```

Records enter the pending queue when guardrails return `require_review`. All records carry `policy_decision`, `matched_rules`, `review_reasons`, and `review_evidence` for audit and UI rendering.

---

## Audit Trail

All operations append to `svc.audit_events`. Each event carries:

- `event_id`, `tenant_id`, `event_type`, `subject`, `actor`
- `decision`, `matched_rules`, `policy_decision`
- `review_reasons`, `review_evidence`
- `details`, `created_at`

```python
for evt in svc.audit_events[-5:]:
    print(evt.event_type, evt.subject, evt.decision)
```

---

## Production Adapters

`MDMService` (database-backed) adds:

- PostgreSQL persistence via `asyncpg`
- Ollama-powered AI entity matching and quality scoring
- Bytewax stream publishing
- Metadata catalog synchronisation
- Redis caching

```python
from capabilities.common.mdm.service import MDMService

svc = MDMService(
    database_url="postgresql+asyncpg://user:pass@localhost/mdm",
    config={"enable_ai": True, "ollama_url": "http://localhost:11434"},
)
await svc.initialize()
```

Adapters must preserve `policy_decision`, `matched_rules`, `review_reasons`, and `review_evidence` when syncing records into external systems. They must not bypass `capability_contract.py` guardrails.

---

## Running Tests

```bash
# Syntax check
python -m py_compile capabilities/common/mdm/service.py

# Contract tests
./.venv/bin/pytest -q capabilities/common/mdm/tests/
```
