# Customer Management — User Guide

**Capability ID**: `telecom_cus` | **Domain**: `telecom` | **Version**: `1.1.0`
**Company**: Datacraft | **Copyright**: © 2025

---

## Overview

`telecom_cus` manages the full subscriber lifecycle from onboarding through churn.
It covers KYC verification, plan activation, SIM and device management, complaint
handling with SLA tracking, churn risk scoring, dunning workflows, GDPR/POPIA
data erasure, and customer segmentation.

All state is tenant-scoped.  Every write operation emits an audit event to the
`apg.telecom.cus.lifecycle` stream via Bytewax.

---

## Installation

```bash
pip install apg-telecom-cus
```

---

## Quick Start

```python
import asyncio
from capabilities.telecom.cus.service import TelecomCustomerService

svc = TelecomCustomerService()

async def demo():
    # 1. Onboard a customer
    customer = await svc.create_account(
        customer_type="individual",
        legal_name="Alice Wanjiku",
        id_number="12345678",
        contact={"phone": "+254712345678", "email": "alice@example.com"},
        address={"street": "Kimathi St", "city": "Nairobi", "country": "KE"},
        tenant_id="acme",
        created_by="agent-001",
    )

    # 2. Run KYC
    kyc = await svc.kyc_check(
        customer_id=customer["id"],
        documents=[{"doc_type": "national_id", "reference": "NID-9876", "expiry_date": "2030-01-01"}],
        tenant_id="acme",
    )

    # 3. Activate a service plan
    activation = await svc.activate_service(
        customer_id=customer["id"],
        service_type="prepaid_voice",
        tenant_id="acme",
    )

    # 4. Get Customer 360 profile
    profile = await svc.get_customer_360(customer_id=customer["id"], tenant_id="acme")
    print(profile)

asyncio.run(demo())
```

---

## Core Workflows

### Customer Onboarding

```python
# Low-level: create_customer with explicit IDs
customer = svc.create_customer(
    customer_id="cust-001",
    tenant_id="acme",
    customer_type="individual",   # individual | corporate | government | mvno | roaming
    msisdn="+254712345678",
    name="Alice Wanjiku",
    created_by="agent-001",
)

# High-level: create_account generates the ID automatically
account = await svc.create_account(
    customer_type="corporate",
    legal_name="Acme Corp Ltd",
    id_number="KRA-PIN-P00123",
    contact={"phone": "+254700000001"},
    address={"city": "Mombasa", "country": "KE"},
    tenant_id="acme",
)
```

### KYC Verification

```python
# Submit a single document
doc = svc.submit_kyc_document(
    doc_id="doc-001", tenant_id="acme", customer_id="cust-001",
    document_type="national_id", document_reference="NID-9876",
    expires_at="2030-01-01",
)

# Verify or reject
svc.verify_kyc(doc_id="doc-001", tenant_id="acme", verified_by="agent-002")
svc.reject_kyc(doc_id="doc-001", tenant_id="acme")

# Batch KYC check (submits and auto-verifies each document)
result = await svc.kyc_check(
    customer_id="cust-001",
    documents=[
        {"doc_type": "national_id", "reference": "NID-9876"},
        {"doc_type": "passport", "reference": "A1234567", "expiry_date": "2028-06-01"},
    ],
    tenant_id="acme",
)
# result["overall_status"] in ("verified", "pending")

# KYC compliance report (regulatory)
report = await svc.kyc_compliance_report(tenant_id="acme")
```

### Plan Management

```python
# Activate a plan
plan = svc.activate_plan(
    plan_id="plan-001", tenant_id="acme", customer_id="cust-001",
    plan_type="postpaid",   # prepaid | postpaid | hybrid | data_bundle | ...
    plan_name="Postpaid 2000",
    plan_reference="PP-2000",
    activated_at="2025-01-15T08:00:00Z",
    credit_check_completed=True,
)

# Activate a value-added service
vas = await svc.activate_service(
    customer_id="cust-001", service_type="roaming_data", tenant_id="acme"
)

# Suspend / restore
suspension = await svc.suspend_service(
    customer_id="cust-001", reason="non_payment", tenant_id="acme"
)
restored = await svc.restore_service(
    customer_id="cust-001", service_id="plan-001", tenant_id="acme"
)
```

### SIM Management

```python
# Provision a SIM
sim = svc.provision_sim(
    sim_id="sim-001", tenant_id="acme", customer_id="cust-001",
    iccid="8954307700012345678", imsi="639070000012345",
    msisdn="+254712345678", provisioned_at="2025-01-15T08:00:00Z",
)

# Block a stolen SIM
svc.update_sim_status(sim_id="sim-001", tenant_id="acme", new_status="stolen_blocked")

# SIM swap with 30-day cooling-off fraud detection
swap = await svc.sim_swap(
    old_sim_id="sim-001", new_sim_id="sim-002",
    new_iccid="8954307700099999999", new_imsi="639070000099999",
    msisdn="+254712345678", reason="physical_damage",
    tenant_id="acme", swapped_by="agent-003",
)
# swap["cooling_off_violation"] is True → fraud_report case created automatically
```

### Case Management & Complaints

```python
# Log a complaint
complaint = await svc.complaint_log(
    customer_id="cust-001",
    complaint_type="billing",
    description="Overcharged on last invoice",
    tenant_id="acme",
    channel="phone",
)
# complaint["sla_due_at"] is set based on complaint type (billing=48h, network=4h)

# Resolve a complaint
resolution = await svc.complaint_resolution(
    complaint_id=complaint["id"],
    resolution="Credit note issued for 500 KES",
    resolved_by="agent-004",
    tenant_id="acme",
)

# Escalate a case (tier_1 → tier_2 → specialist → management)
escalated = await svc.escalate_case(
    case_id=complaint["id"],
    escalation_reason="Customer threatening legal action",
    escalated_to_tier="specialist",
    tenant_id="acme",
)

# Check SLA breaches across all open cases
breaches = await svc.get_sla_breaches(tenant_id="acme")
# breaches["breached_cases"]  — already past SLA due date
# breaches["at_risk_cases"]   — within 2 hours of breach
```

### Churn Management

```python
# Score deterministic churn risk
risk = await svc.score_churn_risk(customer_id="cust-001", tenant_id="acme")
# risk["churn_risk_score"]  0.0–1.0
# risk["risk_level"]        "low" | "medium" | "high"
# risk["threshold_breached"] True if score >= 0.65  →  lifecycle event emitted

# Execute a retention intervention
intervention = await svc.churn_risk_intervention(
    customer_id="cust-001",
    intervention_type="discount_offer",   # retention_call | discount_offer | loyalty_reward | ...
    tenant_id="acme",
)

# Record an NPS survey response
nps = await svc.record_nps(
    customer_id="cust-001", score=3, channel="sms", tenant_id="acme",
    comment="Service was very slow last week",
)

# NPS analytics (% promoters − % detractors)
analytics = await svc.nps_analytics(tenant_id="acme", period="last_90_days")
```

### Dunning Workflow

```python
# Trigger dunning based on days overdue
# 1–7 days   → reminder (notification only)
# 8–14 days  → warning
# 15–30 days → soft suspension (data only)
# 31+ days   → full deactivation
dunning = await svc.trigger_dunning(
    customer_id="cust-001",
    invoice_id="inv-2025-001",
    amount_due=2500.00,
    days_overdue=20,
    tenant_id="acme",
)
# dunning["dunning_step"]  "soft_suspension"
# dunning["case"]          auto-created billing_query case
# dunning["suspension"]    suspension record if step >= soft_suspension
```

### Customer Segmentation

```python
# Filter customers by multiple criteria (paginated)
segment = await svc.segment_customers(
    criteria={
        "status": "active",
        "kyc_status": "verified",
        "plan_type": "postpaid",
    },
    tenant_id="acme",
    page=1,
    page_size=100,
)
# segment["total_matches"]  total count matching criteria
# segment["records"]        page of customer dicts
```

### GDPR / POPIA Data Erasure

```python
# Initiate right-to-erasure
erasure = await svc.request_data_erasure(
    customer_id="cust-001",
    reason="Customer withdrawal of consent",
    requested_by="dpo@acme.com",
    tenant_id="acme",
)
# PII fields pseudonymised, kyc_status → "erased"
# 30-day compliance case opened automatically
# Audit trail preserved with anonymised references
```

### Bulk Import

```python
# Import a list of customers with per-record results
result = await svc.bulk_import_customers(
    records=[
        {"customer_type": "individual", "msisdn": "+254700000010", "name": "Bob Kimani", "created_by": "migration"},
        {"customer_type": "individual", "msisdn": "+254700000011", "name": "Carol Auma", "created_by": "migration"},
    ],
    tenant_id="acme",
    dry_run=False,   # set True for pre-flight validation
)
# result["created"]  count of new records
# result["skipped"]  count of duplicate MSISDNs
# result["failed"]   count of validation failures
# result["results"]  per-record status list
```

### Analytics and Reporting

```python
# Customer lifecycle period report
report = await svc.customer_lifecycle_report(period="2025-Q1", tenant_id="acme")

# KPI dashboard data
kpis = await svc.customer_analytics(tenant_id="acme", period="monthly")

# Export customers to JSON or CSV
export = await svc.export_customers(tenant_id="acme", format="csv")

# Dashboard summary (sync)
summary = svc.dashboard_summary(tenant_id="acme")

# Service health
health = await svc.health_check(tenant_id="acme")
```

---

## Device Management

```python
svc.register_device(
    device_id="dev-001", tenant_id="acme", customer_id="cust-001",
    device_type="smartphone", imei="359301234567890",
    model="Samsung Galaxy A35", registered_at="2025-01-15T08:00:00Z",
)
```

---

## Lifecycle Events

Record arbitrary lifecycle milestones:

```python
svc.record_lifecycle_event(
    event_id="evt-001", tenant_id="acme", customer_id="cust-001",
    event_type="plan_upgrade", event_reference="plan-001->plan-002",
    occurred_at="2025-03-01T12:00:00Z", recorded_by="agent-005",
)
```

---

## Automation Agents

Register rule-based or ML-backed agents:

```python
svc.register_agent(
    agent_id="agent-churn-001", tenant_id="acme",
    name="ChurnPatrol", runtime="langgraph",
    role="churn_manager", scope="read:customers,write:interventions",
)
```

---

## Capability Contract

```python
# Describe the contract (supported types, rules, streaming config)
contract = svc.describe(tenant_id="acme")

# Evaluate a policy context explicitly
decision = svc.evaluate({
    "tenant_id": "acme",
    "operation": "create_customer",
    "kyc_initiated": True,
    "policy_attached": True,
    "operation_type": "write",
})
# decision["decision"] in ("allow", "deny")
```

---

## Interoperability

```apg
use telecom_cus;
```

`telecom_cus` composes with:
| Downstream | Data Provided |
|-----------|---------------|
| `telecom_bil` | Customer identity, dunning events |
| `telecom_ord` | Order validation, subscriber status |
| `telecom_ana` | Churn risk scores, segmentation output |
| `telecom_pro` | SIM and device provisioning data |
| `comp` | KYC compliance reports |

---

## Configuration Reference

All keys are tenant-scoped and managed via the `conf` capability or environment
variables prefixed with `TELECOM_CUS_`.

| Key | Default | Description |
|-----|---------|-------------|
| `customers.kyc_required` | `true` | Enforce KYC at creation |
| `plans.credit_check_for_postpaid` | `true` | Credit check gate for postpaid |
| `sims.max_sims_per_customer` | `10` | Hard SIM limit per customer |
| `devices.imei_check` | `true` | IMEI validation mandatory |
| `devices.blacklist_check` | `true` | Blacklist check mandatory |
| `cases.sla_hours.billing` | `48` | Billing case SLA in hours |
| `cases.sla_hours.network` | `4` | Network case SLA in hours |
| `cases.sla_hours.complaint` | `24` | General complaint SLA in hours |
| `churn.risk_threshold` | `0.65` | Score above which churn_risk_flagged fires |

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Dataclass models
- `api.py` — REST API endpoints (Flask-AppBuilder blueprints)
- `views.py` — Pydantic request/response schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Improvement roadmap (15 items, 9 implemented)
- `SPECIFICATION.md` — Full capability specification
- `tests/` — Unit and integration tests
