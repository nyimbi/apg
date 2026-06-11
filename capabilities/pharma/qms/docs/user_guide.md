# Quality Management System — User Guide

**Capability ID**: `pharma_qms` | **Domain**: `pharma` | **Version**: `1.1.0`

## Description

End-to-end pharmaceutical QMS covering GxP batch records, CAPA management, deviation handling, OOS investigations, controlled document management, audit management, validation lifecycle, quality risk assessment, supplier qualification, batch release risk scoring, regulatory impact classification, and inspection readiness monitoring. All workflows enforce cGMP/GDP compliance and 21 CFR Part 11 electronic signature requirements. Integrates with the APG NATS/Bytewax event mesh for real-time quality intelligence.

---

## Installation

```bash
pip install apg-pharma-qms
```

---

## Quick Start

```python
import asyncio
from apg_pharma_qms.service import QualityManagementService

svc = QualityManagementService(tenant_id="acme_pharma", actor_id="qa_user_01")

# Initiate a change control
change = svc.change_control(
    change_type="major",
    description="Update granulation step parameters for Product X",
    impact_assessment="Risk assessment REF-2025-001 completed. GMP impact: high.",
    originator_id="eng_01",
    tenant_id="acme_pharma",
    regulatory_impact=True,
)

# Classify regulatory submission obligation
notification = asyncio.run(svc.classify_regulatory_impact(
    tenant_id="acme_pharma",
    change_id=change.id,
    jurisdictions=["FDA", "EMA"],
    change_category="major",
    description=change.description,
))
print(notification["jurisdiction_notifications"])
# [{'jurisdiction': 'FDA', 'notification_type': 'cbee', 'submission_deadline_iso': ..., ...}, ...]
```

---

## Core Workflows

### 1. Change Control

Change control enforces a mandatory phase sequence: **Draft → Impact Assessment → Approval → Implementation → Effectiveness Check → Closed**.

```python
# Initiate
change = svc.change_control(
    change_type="minor",
    description="Update label artwork for EU market",
    impact_assessment="No GMP impact. Cosmetic label update only.",
    originator_id="reg_01",
    tenant_id="acme",
)

# Approve (requires impact + risk assessment)
approved = svc.approve_change(
    change_id=change.id,
    tenant_id="acme",
    approval_reference="ECO-2025-042",
    impact_assessed=True,
    risk_assessed=True,
)

# Implement
from datetime import datetime
implemented = svc.implement_change(change.id, "acme", datetime.utcnow())

# Close with effectiveness check
closed = svc.close_change(
    change_id=change.id,
    tenant_id="acme",
    effectiveness_checked=True,
    effectiveness_reference="EFFCK-2025-042",
)
```

#### Regulatory Impact Classification
For changes with regulatory impact, classify the submission obligation before implementation:

```python
notification = await svc.classify_regulatory_impact(
    tenant_id="acme",
    change_id=change.id,
    jurisdictions=["FDA", "EMA", "MHRA"],
    change_category="major",
    description=change.description,
)
# notification["jurisdiction_notifications"] contains per-jurisdiction deadlines
```

---

### 2. CAPA Management

CAPAs must have a documented root cause and completed effectiveness check before closure.

```python
# Create a corrective action
capa = svc.capa_creation(
    source="audit_finding",
    root_cause="Inadequate operator training on gowning procedure",
    action_plan="Update training SOP, retrain all personnel, add competency assessment",
    responsible_person="qa_mgr_01",
    deadline=datetime(2025, 9, 1),
    tenant_id="acme",
    capa_type="corrective_action",
    severity="major",
)

# Check overdue CAPAs
overdue = svc.check_overdue_capas("acme")

# Predict effectiveness before closing (requires Ollama)
prediction = await svc.predict_capa_effectiveness(
    tenant_id="acme",
    capa_id=capa.id,
)
# {"predicted_effectiveness": "highly_effective", "recurrence_risk_score": 0.1, ...}

# Close with root cause and effectiveness result
closed_capa = svc.close_capa(
    capa_id=capa.id,
    tenant_id="acme",
    root_cause="Inadequate operator training on gowning procedure",
    root_cause_method="fishbone_5why",
    effectiveness_checked=True,
    effectiveness_result="effective",
)
```

---

### 3. Deviation Management

```python
# Raise a process deviation
deviation = svc.deviation_management(
    deviation_type="process_deviation",
    description="Granulation end-point not reached within specified time",
    batch_id="BATCH-2025-0412",
    impact="Batch hold pending OOS investigation",
    tenant_id="acme",
    severity="major",
    raised_by="prod_op_01",
    affected_products=["PROD-X-500MG"],
)

# Detect recurring patterns across all deviations
clusters = await svc.cluster_similar_deviations(
    tenant_id="acme",
    similarity_threshold=0.75,
    min_cluster_size=3,
)
# clusters["draft_capa_recommendations"] lists systemic CAPA suggestions

# Close the deviation
closed_dev = svc.close_deviation(
    deviation_id=deviation.id,
    tenant_id="acme",
    root_cause="Binder solution concentration out of specification",
    capa_reference=capa.id,
)
```

---

### 4. OOS / OOT Investigation

Follows the FDA OOS Guidance (2006) two-phase investigation structure.

```python
oos = await svc.initiate_oos_investigation(
    tenant_id="acme",
    sample_id="SAMPLE-2025-0887",
    test_name="Assay by HPLC",
    specification="95.0–105.0%",
    result_obtained="88.3%",
    analyst_id="lab_analyst_02",
    batch_id="BATCH-2025-0412",
    product_id="PROD-X-500MG",
    phase="phase1_lab",
    assignee_id="qa_scientist_01",
)
# oos["sla_deadline_iso"] — Phase 1 must close within 120 hours (configurable)
# Breach triggers oos_sla_breach event on NATS apg.pharma.qms.lifecycle
```

---

### 5. Document Control

```python
# Create and approve a new SOP
doc = svc.create_document(
    tenant_id="acme",
    document_number="SOP-QA-001",
    title="Gowning and De-gowning Procedure",
    document_type="sop",
    version="1.0",
    department="Quality Assurance",
    owner_id="qa_mgr_01",
    created_by="qa_author_01",
)

# Electronic signature approval (21 CFR Part 11)
result = await svc.sign_and_approve_document(
    doc_id=doc.id,
    tenant_id="acme",
    approver_id="qp_director_01",
    meaning="I approve this SOP for GMP use effective immediately",
)

# Identify documents due for periodic review
schedule = await svc.schedule_periodic_reviews(
    tenant_id="acme",
    lead_time_days=60,
    publish_to_nats=True,  # publishes to apg.pharma.qms.scheduling
)
print(f"{schedule['total_items']} documents/validations due within 60 days")
```

---

### 6. SPC Trend Analysis

```python
data_points = [
    {"value": 9.8, "timestamp": "2025-06-01T08:00Z", "batch_id": "B001"},
    {"value": 10.1, "timestamp": "2025-06-02T08:00Z", "batch_id": "B002"},
    # ... more data points ...
    {"value": 13.5, "timestamp": "2025-06-15T08:00Z", "batch_id": "B015"},  # OOC candidate
]

analysis = await svc.run_spc_trend_analysis(
    tenant_id="acme",
    process_parameter="tablet_hardness_N",
    data_points=data_points,
    control_chart_type="individuals",
)
# {"out_of_control": True, "signals": [...], "recommended_action": "raise_preventive_capa"}
```

---

### 7. Batch Release Risk Scoring

```python
risk = await svc.compute_batch_risk_score(
    tenant_id="acme",
    batch_id="BATCH-2025-0412",
    product_id="PROD-X-500MG",
    cpp_excursions=1,
    equipment_oq_status="qualified",
    material_coa_compliant=True,
    environmental_excursions=0,
    process_cpk=1.41,
)
# {"batch_risk_score": 8.75, "risk_level": "low", "release_recommendation": "release_standard"}
```

---

### 8. Inspection Readiness Scoring

```python
readiness = await svc.generate_inspection_readiness_score(
    tenant_id="acme",
    inspection_type="fda_gmp",
)
print(readiness["score"])   # e.g. 87
print(readiness["grade"])   # "B"
for gap in readiness["gap_items"]:
    print(f"  [{gap['priority']}] {gap['issue']} (−{gap['deduction']} pts)")
```

---

### 9. Supplier Qualification

```python
qual = svc.supplier_qualification(
    supplier_id="SUP-RAWMAT-042",
    qualification_type="initial_audit",
    result="qualified",
    tenant_id="acme",
    quality_agreement_ref="QAG-2025-SUP042",
    approved_materials=["LACTOSE-MONOHYDRATE", "MCC-PH102"],
    next_audit_days=730,
)
```

---

### 10. Quality Metrics Dashboard

```python
metrics = svc.quality_metrics(period="2025-Q2", tenant_id="acme")
# Keys: capa_closure_rate_pct, capa_effectiveness_rate_pct, overdue_capas,
#       open_deviations, critical_deviations, total_audit_findings, ...
```

---

## Audit Trail

Every state mutation emits an immutable audit event:

```python
svc._audit_events  # list of {"tenant_id", "event_type", "reference_id", "processor", "stream"}
```

Events are processed by the Bytewax dataflow on NATS subject `apg.pharma.qms.lifecycle`. Downstream subscribers include: overdue CAPA escalation, deviation clustering, KPI aggregation, and notification delivery.

---

## NATS Integration

When NATS is configured, scheduling and alert events are published to:

| NATS Subject | Events |
|---|---|
| `apg.pharma.qms.lifecycle` | All core QMS lifecycle events |
| `apg.pharma.qms.scheduling` | `review_due`, `revalidation_due` |

Configure NATS by setting `NATS_URL` in the environment. Bytewax dataflows subscribe and process events for escalation, aggregation, and cross-capability fan-out.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `NATS_URL` | NATS server URL | `nats://localhost:4222` |
| `OLLAMA_BASE_URL` | Ollama inference server for ML features | (unset — ML disabled) |
| `PHARMA_QMS_TENANT` | Default tenant ID for CLI usage | `default` |
| `PHARMA_QMS_DB_URL` | PostgreSQL connection string | (in-memory if unset) |

---

## Permissions Reference

| Permission | Grants Access To |
|---|---|
| `pharma_qms:view` | Dashboard read-only |
| `pharma_qms:change_control` | Change control CRUD |
| `pharma_qms:capa` | CAPA CRUD and effectiveness prediction |
| `pharma_qms:deviations` | Deviations and OOS investigations |
| `pharma_qms:documents` | Document control and e-signature |
| `pharma_qms:audits` | Audit planning and closure |
| `pharma_qms:validation` | Validation lifecycle |
| `pharma_qms:batch_release` | Batch risk scoring |
| `pharma_qms:metrics` | Quality KPIs and SPC analysis |
| `pharma_qms:admin` | Settings and periodic review scheduling |

---

## Further Reading

- `/capabilities/pharma/qms/service.py` — Business logic and all async methods
- `/capabilities/pharma/qms/models.py` — Pydantic v2 data models
- `/capabilities/pharma/qms/api.py` — REST API endpoints
- `/capabilities/pharma/qms/capability_contract.py` — Rules engine and configuration
- `/capabilities/pharma/qms/WORLD_CLASS_IMPROVEMENTS.md` — 15 strategic enhancements
- `/capabilities/pharma/qms/SPECIFICATION.md` — Full capability specification
