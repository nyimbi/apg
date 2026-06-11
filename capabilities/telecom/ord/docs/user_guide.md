# Order Management — User Guide

**Capability ID**: `telecom_ord` | **Domain**: `telecom` | **Version**: `1.1.0`

---

## Description

`telecom_ord` manages the complete lifecycle of telecom service orders: capture, validation, decomposition into provisioning tasks, fallout handling, number portability, bulk processing, contract management, cost estimation, jeopardy prediction, and operational metrics export. All operations are tenant-scoped and audit-trailed.

---

## Installation

```bash
pip install apg-telecom-ord
```

---

## Quick Start

```python
from capabilities.telecom.ord.service import TelecomOrderManagementService

svc = TelecomOrderManagementService()

# Submit and validate a single order
order = svc.submit_order(
    order_id="ord-001",
    tenant_id="acme",
    order_type="new_service",
    customer_id="cust-42",
    channel="api",
    priority="normal",
    submitted_at="2026-06-11T08:00:00Z",
)
svc.validate_order("ord-001", "acme")
svc.decompose_order("ord-001", "acme")

# Create and complete a task
svc.create_task("task-001", "acme", "ord-001", "sim_provisioning")
svc.complete_task("task-001", "acme", "2026-06-11T09:00:00Z")
svc.complete_order("ord-001", "acme", "2026-06-11T09:05:00Z")
```

---

## Core Workflow

### 1. Order Capture

**Single order**

```python
order = svc.submit_order(
    order_id="ord-002",
    tenant_id="acme",
    order_type="upgrade",
    customer_id="cust-99",
    channel="retail",
    priority="high",
    submitted_at="2026-06-11T10:00:00Z",
)
```

**Multi-product bundle** (async)

```python
import asyncio

bundle = asyncio.run(svc.capture_order(
    customer_id="cust-99",
    product_ids=["voice-plan-xl", "data-10gb", "sms-500"],
    channel="api",
    sales_agent_id="agent-007",
    tenant_id="acme",
    priority="normal",
))
print(bundle["bundle_id"], bundle["order_count"])
```

---

### 2. Validation

```python
# Structured validation with check-by-check results
result = asyncio.run(svc.order_validation("ord-002", "acme"))
# result["valid"] → True/False
# result["checks"] → {"order_exists": True, "not_duplicate": True, ...}
```

---

### 3. Credit Check

```python
credit = asyncio.run(svc.credit_check_order(
    customer_id="cust-99",
    monthly_value=12500.0,
    tenant_id="acme",
))
# credit["decision"] → "approved" | "conditional" | "declined"
# credit["approved_limit"] → 50000.0
```

---

### 4. Cost Estimation

Obtain an itemised cost breakdown before committing the order:

```python
estimate = asyncio.run(svc.estimate_order_cost(
    customer_id="cust-99",
    products=[
        {"product_id": "voice-plan-xl", "unit_price": 1200.0, "quantity": 1},
        {"product_id": "data-10gb",     "unit_price": 800.0,  "quantity": 2},
    ],
    duration_months=12,
    tenant_id="acme",
    currency="KES",
))
print(f"Total: {estimate['currency']} {estimate['grand_total']}")
```

---

### 5. Order Decomposition and Task Management

```python
svc.decompose_order("ord-002", "acme")

# Create tasks with dependency chain
svc.create_task("task-A", "acme", "ord-002", "sim_provisioning")
svc.create_task("task-B", "acme", "ord-002", "number_activation", depends_on="task-A")
svc.create_task("task-C", "acme", "ord-002", "billing_setup",     depends_on="task-A")

# Inspect execution plan (DAG + parallel groups)
plan = asyncio.run(svc.get_task_execution_plan("ord-002", "acme"))
# plan["execution_groups"] → [["task-A"], ["task-B", "task-C"]]
# task-B and task-C can run concurrently after task-A completes
```

---

### 6. Fallout Handling

```python
# Record a fallout via error code
fallout = asyncio.run(svc.order_fallout(
    order_id="ord-002",
    error_code="NET_001",
    description="BSS provisioning API timeout",
    tenant_id="acme",
))

# Retry
svc.retry_fallout(fallout["id"], "acme")

# Resolve
svc.resolve_fallout(fallout["id"], "acme", "re-routed to backup NE", "2026-06-11T11:00:00Z")
```

---

### 7. Order Amendment and Cancellation

```python
# Amend an in-flight order (status must be submitted/validated/decomposed)
amended = asyncio.run(svc.order_amendment(
    order_id="ord-002",
    change_type="product_swap",
    new_parameters={"product_id": "voice-plan-xxl"},
    tenant_id="acme",
    requested_by="customer",
))

# Cancel
cancelled = asyncio.run(svc.order_cancellation(
    order_id="ord-003",
    reason="customer_requested",
    cancelled_by="agent-007",
    tenant_id="acme",
))
```

---

### 8. Number Portability

```python
# Pre-validate eligibility
eligibility = asyncio.run(svc.validate_portability_eligibility(
    msisdn="+254712345678",
    donor_operator="SAFARICOM",
    recipient_operator="AIRTEL",
    tenant_id="acme",
))
# eligibility["eligible"] → True | False
# eligibility["checks"]["msisdn_e164_format"] → True
# eligibility["checks"]["no_concurrent_port"] → True

# Submit the request only if eligible
if eligibility["eligible"]:
    pr = svc.submit_portability_request(
        request_id="port-001",
        tenant_id="acme",
        order_id="ord-002",
        msisdn="+254712345678",
        donor_operator="SAFARICOM",
        recipient_operator="AIRTEL",
        submitted_at="2026-06-11T10:00:00Z",
    )
```

---

### 9. Bulk Orders

**Pre-approved bulk submission**

```python
bulk = svc.submit_bulk_order(
    bulk_id="bulk-2026-06-11",
    tenant_id="acme",
    order_type="new_service",
    item_count=500,
    approval_reference="APPROVAL-REF-99",
    submitted_by="ops-team",
    submitted_at="2026-06-11T08:00:00Z",
)
```

**CSV bulk import**

```python
csv_payload = (
    "order_id,customer_id,order_type,channel,priority\n"
    "ord-101,cust-A,new_service,api,normal\n"
    "ord-102,cust-B,upgrade,retail,high\n"
)
result = asyncio.run(svc.bulk_order_import(csv_payload, tenant_id="acme"))
print(result["success_count"], result["error_count"])
```

---

### 10. Contract Management

```python
# Create contract after order validation
contract = asyncio.run(svc.contract_creation(
    order_id="ord-002",
    contract_terms={"monthly_fee": 1200, "data_cap_gb": 10},
    duration_months=24,
    tenant_id="acme",
    template_id="standard_postpaid",
))

# Confirm customer signature
active_contract = asyncio.run(svc.confirm_contract_signature(
    contract_id=contract["contract_id"],
    signed_by="cust-99",
    signature_hash="sha256-abc123...",
    tenant_id="acme",
))

# Renew 12 months later
renewed = asyncio.run(svc.renew_contract(
    contract_id=contract["contract_id"],
    extension_months=12,
    tenant_id="acme",
    renewed_by="cust-99",
))
```

---

### 11. Webhook Registration

Receive HMAC-signed order lifecycle events in real time:

```python
webhook = asyncio.run(svc.register_webhook(
    order_id="ord-002",
    callback_url="https://my-crm.example.com/webhooks/orders",
    events=["order_completed", "order_fallout", "order_cancelled"],
    secret="super-secret-key",
    tenant_id="acme",
))
# webhook["webhook_id"] → "wh-ord-002-2026-06-11"
```

The `X-APG-Signature` header on each delivery is HMAC-SHA256 of the payload body with `secret`.

---

### 12. Jeopardy Prediction

Score an in-flight order for completion risk:

```python
jeopardy = asyncio.run(svc.predict_order_jeopardy(
    order_id="ord-002",
    tenant_id="acme",
    sla_hours=4.0,  # high-priority SLA
))
# jeopardy["risk_score"]  → 0.61
# jeopardy["risk_band"]   → "high"
# jeopardy["recommended_action"] → "expedite_processing"
# jeopardy["features"]["age_ratio"] → 0.73
```

Risk bands:
| Score | Band | Action |
|-------|------|--------|
| 0.75 – 1.0 | critical | escalate_to_supervisor |
| 0.50 – 0.75 | high | expedite_processing |
| 0.25 – 0.50 | medium | monitor_closely |
| 0.00 – 0.25 | low | no_action_required |

---

### 13. SLA Monitoring and Analytics

```python
# SLA compliance snapshot
sla = asyncio.run(svc.order_sla_monitoring(
    period="2026-06",
    tenant_id="acme",
    sla_hours=24.0,
))
print(sla["sla_compliance_rate"], sla["breaching_count"])

# Channel / period analytics
analytics = asyncio.run(svc.order_analytics(
    period="2026-06",
    channel="all",
    tenant_id="acme",
))
print(analytics["completion_rate"], analytics["fallout_rate"])
```

---

### 14. Metrics Export

```python
# Prometheus text format (scrape endpoint)
prometheus_text = asyncio.run(svc.export_metrics(tenant_id="acme", fmt="prometheus"))
print(prometheus_text)

# JSON format for dashboards
import json
metrics_json = asyncio.run(svc.export_metrics(tenant_id="acme", fmt="json"))
data = json.loads(metrics_json)
```

---

### 15. Audit Trail and Replay

```python
# Full audit trail
trail = asyncio.run(svc.get_audit_trail(tenant_id="acme"))

# Replay order state as of a specific timestamp
snapshot = asyncio.run(svc.replay_order(
    order_id="ord-002",
    as_of="2026-06-11T10:30:00Z",
    tenant_id="acme",
))
print(snapshot["event_count"], snapshot["events"])
```

---

## Dashboard Summary

```python
summary = svc.dashboard_summary("acme")
# Keys: order_count, task_count, open_fallout_count, portability_count,
#       bulk_order_count, agent_count, contract_count, amendment_count,
#       cancellation_count, audit_event_count, streaming
```

---

## Agent Integration

```python
svc.register_agent(
    agent_id="agt-001",
    tenant_id="acme",
    name="FalloutRemediation",
    runtime="langchain",
    role="remediation",
    scope="fallout",
)

# Validate agent action before execution
svc.validate_agent_action(
    tenant_id="acme",
    privileged_scope=False,
    human_approval_recorded=True,
)
```

---

## Interoperability

`telecom_ord` integrates with APG capabilities via the composition engine:

```apg
use telecom_ord;
```

Integration points:
- **telecom_pro**: Provisioning workflows triggered on decomposition
- **telecom_cus**: Customer validation and lifecycle events
- **telecom_inv**: Network resource availability checks
- **telecom_bil**: Charge setup on order completion; tariff lookup for cost estimation
- **comp**: Portability regulatory compliance registry

---

## Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `orders.sla_hours.emergency` | `1` | SLA hours for emergency priority |
| `orders.sla_hours.urgent` | `2` | SLA hours for urgent priority |
| `orders.sla_hours.high` | `4` | SLA hours for high priority |
| `orders.sla_hours.normal` | `24` | SLA hours for normal priority |
| `orders.sla_hours.low` | `72` | SLA hours for low priority |
| `fallout.max_retries` | `3` | Max auto-retries before escalation |
| `fallout.escalation_threshold_minutes` | `30` | Minutes in fallout before escalation |
| `decomposition.parallel_execution` | `true` | Run independent tasks concurrently |
| `idempotency.ttl_hours` | `24` | Idempotency key TTL for safe replay |
| `credit.standard_limit_kes` | `50000` | Default credit approval threshold |
| `metrics.latency_buffer_size` | `1000` | Max latency samples retained in memory |

---

## Further Reading

- `service.py` — Business logic implementation (all service methods)
- `models.py` — OrdOrder, OrdTask, OrdFallout, OrdPortabilityRequest, OrdBulkOrder, OrdAgent
- `api.py` — REST API endpoint definitions
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — Detailed improvement catalogue with rationale
- `README.md` — Quick reference
