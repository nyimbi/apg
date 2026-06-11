# telecom_pro — User Guide

**Capability ID**: `telecom_pro` | **Domain**: `telecom` | **Version**: `2.0.0`
**Copyright**: © 2025 Datacraft | www.datacraft.co.ke

---

## Overview

`telecom_pro` is the APG service provisioning and product catalogue capability for telecom operators. It covers:

- **Product Catalogue** — TMF620-aligned product definitions with lifecycle state machine and versioning
- **Bundle Management** — Compose products into bundles, evaluate eligibility, and decompose to provisioning orders
- **Price List** — Effective-dated prices with tiered rate tables and bulk update support
- **Promotional Campaigns** — Time-limited discount campaigns with usage gating
- **Provisioning Workflows** — End-to-end service activation, resource reservation, config push, rollback

---

## Installation

```bash
pip install apg-telecom-pro
```

---

## Quick Start

```python
import asyncio
from capabilities.telecom.pro.service import ServiceProvisioningService

svc = ServiceProvisioningService()

async def main():
    # 1. Create a product in the catalogue
    product = await svc.create_product(
        product_id="FIBRE-100",
        name="Fibre 100 Mbps",
        category="broadband",
        characteristics={"speed_mbps": 100, "technology": "FTTH"},
        tenant_id="acme",
    )
    print(product["status"])  # draft

    # 2. Advance lifecycle: draft -> review -> approved -> active
    await svc.update_product_status("FIBRE-100", "review", "Ready for review", "acme")
    await svc.update_product_status("FIBRE-100", "approved", "Pricing approved", "acme")
    await svc.update_product_status("FIBRE-100", "active", "Launched Q1 2025", "acme")

    # 3. Create a price
    price = await svc.create_price(
        price_id="PR-FIBRE-100-KES",
        product_id="FIBRE-100",
        amount=2999.00,
        currency="KES",
        charge_type="recurring",
        effective_from="2025-01-01",
        tenant_id="acme",
    )

    # 4. Receive and provision a service order
    order = await svc.service_order_receive(
        order_id="ORD-001",
        customer_id="CUST-42",
        product_code="FIBRE-100",
        parameters={"address": "Westlands, Nairobi"},
        tenant_id="acme",
    )
    print(order["workflow_id"])  # wf-ORD-001

asyncio.run(main())
```

---

## Product Catalogue

### Creating Products

Products follow the TMF620 schema. Initial status is always `draft`.

```python
product = await svc.create_product(
    product_id="VOIP-PBX-10",
    name="VoIP PBX 10-seat",
    category="voice",
    characteristics={"seats": 10, "codec": "G.711"},
    tenant_id="acme",
    status="draft",  # default
)
```

**Characteristics** is a free-form dict. Use it to carry technical attributes that drive provisioning templates.

### Product Lifecycle

Products must traverse the state machine in order:

```
draft -> review -> approved -> active -> deprecated -> retired
```

Backward transitions are allowed for draft/review/approved (to support corrections). Once `active`, a product can only move forward. `retired` is a terminal state.

```python
# Advance to review
await svc.update_product_status("VOIP-PBX-10", "review", "Product spec finalised", "acme")

# Approve
await svc.update_product_status("VOIP-PBX-10", "approved", "Pricing confirmed by finance", "acme")

# Launch
await svc.update_product_status("VOIP-PBX-10", "active", "Q2 2025 product launch", "acme")

# Deprecate (allow order wind-down)
await svc.update_product_status("VOIP-PBX-10", "deprecated", "Replaced by VOIP-PBX-FLEX", "acme")

# Retire (no more orders)
await svc.update_product_status("VOIP-PBX-10", "retired", "EOL confirmed by product board", "acme")
```

Attempting an invalid transition raises `ValueError` with the full allowed-transitions map:
```
ValueError: Cannot transition product VOIP-PBX-10 from 'active' to 'draft'.
Allowed transitions: ['deprecated']
```

### Searching the Catalogue

```python
results = await svc.search_catalogue(
    tenant_id="acme",
    category="broadband",
    status="active",
    keyword="fibre",
    min_price=1000,
    max_price=5000,
    offset=0,
    limit=20,
)
# results["facets"]["category"] — counts per category
# results["facets"]["status"] — counts per status
# results["total_matches"] — total before pagination
```

---

## Bundle Management

### Creating a Bundle

Bundles reference existing active/approved products as components.

```python
bundle = await svc.create_bundle(
    bundle_id="TRIPLE-PLAY-BASIC",
    name="Triple Play Basic",
    components=[
        {"product_id": "FIBRE-100", "quantity": 1, "mandatory": True},
        {"product_id": "VOIP-PBX-10", "quantity": 1, "mandatory": True},
        {"product_id": "IPTV-SD",    "quantity": 1, "mandatory": False},  # optional
    ],
    pricing_tier="standard",
    tenant_id="acme",
    eligibility_rules=["segment:residential", "geography:nairobi"],
    incompatible_with=["ENTERPRISE-WAN"],
)
```

Non-existent or non-active component products raise `ValueError` immediately.

### Decomposing a Bundle to Orders

```python
decomp = await svc.decompose_bundle_to_orders(
    bundle_id="TRIPLE-PLAY-BASIC",
    customer_id="CUST-42",
    tenant_id="acme",
    parameters={"address": "Westlands, Nairobi"},
)
# decomp["created_orders"] — order_ids for mandatory components (auto-provisioned)
# decomp["pending_optional"] — optional components awaiting customer confirmation
```

Each mandatory component generates a `service_order_receive` call with the bundle_id stamped in parameters for traceability.

---

## Price List Management

### Creating a Price

```python
price = await svc.create_price(
    price_id="PR-FIBRE-100-KES-PROMO",
    product_id="FIBRE-100",
    amount=2499.00,
    currency="KES",
    charge_type="recurring",           # one-time | recurring | usage
    effective_from="2025-03-01",
    effective_to="2025-06-30",         # None = open-ended
    tenant_id="acme",
    rate_table=[                       # optional tiered pricing
        {"threshold": 0,     "unit_price": 2499.00},
        {"threshold": 5,     "unit_price": 2299.00},   # >5 units
        {"threshold": 10,    "unit_price": 1999.00},   # >10 units
    ],
)
```

### Querying Effective Price

```python
ep = await svc.get_effective_price(
    product_id="FIBRE-100",
    as_of_date="2025-04-15",
    tenant_id="acme",
    charge_type="recurring",
)
# Returns the price whose effective_from <= 2025-04-15 and effective_to >= 2025-04-15
# Returns None if no price covers the date
```

### Bulk Price Update

Bulk updates are atomic — validation runs for all records before any change is applied.

```python
result = await svc.bulk_update_prices(
    updates=[
        {"price_id": "PR-FIBRE-100-KES", "new_amount": 2799.00, "effective_from": "2025-07-01", "reason": "Annual review"},
        {"price_id": "PR-VOIP-KES",      "new_amount": 999.00,  "effective_from": "2025-07-01", "reason": "Annual review"},
    ],
    approval_reference="FINANCE-APPROVAL-2025-Q3",
    tenant_id="acme",
)
# result["updated_count"], result["results"] per-item status
```

---

## Promotional Campaigns

### Creating a Promotion

```python
promo = await svc.create_promotion(
    campaign_id="FIBRE-LAUNCH-2025",
    discount_type="percentage",        # percentage | fixed | free-month
    discount_value=15.0,               # 15% off
    applies_to=["FIBRE-100", "FIBRE-200"],
    valid_from="2025-03-01",
    valid_to="2025-03-31",
    tenant_id="acme",
    usage_limit=500,                   # 0 = unlimited
)
```

### Applying to an Order

```python
discount = await svc.apply_promotion_to_order(
    order_id="ORD-001",
    campaign_id="FIBRE-LAUNCH-2025",
    tenant_id="acme",
)
# Idempotent — calling twice returns the same discount record
```

Checks enforced:
- Campaign must be `active`
- Order date must fall within `valid_from`/`valid_to`
- `usage_count < usage_limit` (if limit set)
- Order `product_code` must be in `applies_to`

---

## Provisioning Workflows

### Service Order Lifecycle

```python
# 1. Receive order
order = await svc.service_order_receive("ORD-001", "CUST-42", "FIBRE-100", {}, "acme")

# 2. Decompose to tasks
tasks = await svc.order_decomposition("ORD-001", "acme")

# 3. Allocate resources
alloc = await svc.resource_allocation("ORD-001", "ip_address", "10.0.0.42", "acme")

# 4. Push network config
cfg = await svc.network_configuration(
    "ORD-001",
    ["interface GigE0/0/1", "ip address 10.0.0.42 255.255.255.0"],
    tenant_id="acme",
    ne_id="PE-NBI-01",
)

# 5. Activation check
check = await svc.activation_check("ORD-001", "acme")
assert check["activation_ready"]

# 6. Complete
result = await svc.order_completion("ORD-001", "2025-04-01", "acme")
```

### Rollback

```python
svc.trigger_rollback("RB-001", "acme", "wf-ORD-001", "activation_failure",
                      "E2E test failed", _utcnow())
svc.complete_rollback("RB-001", "acme", _utcnow())
```

### Fallout Management

```python
fallout = await svc.fallout_management(
    order_id="ORD-001",
    error_type="ne_unreachable",
    retry_action="reschedule_config_push",
    tenant_id="acme",
    max_retries=3,
)
# fallout["escalated"] = True after 3rd retry -> NOC attention required
```

### Jeopardy Handling

```python
jeopardy = await svc.order_jeopardy(
    order_id="ORD-001",
    reason="SLA breach imminent — 48h without activation",
    tenant_id="acme",
    assigned_to="noc-lead@datacraft.co.ke",
)
```

---

## Catalogue Health Dashboard

```python
health = await svc.catalogue_health_dashboard(tenant_id="acme")
# {
#   "total_products": 42,
#   "status_breakdown": {"active": 30, "draft": 8, "deprecated": 4},
#   "price_coverage_pct": 85.71,
#   "bundle_count": 6,
#   "promotions_expiring_in_7d": ["FIBRE-LAUNCH-2025"],
#   "completeness_score": 78.57,
#   "computed_at": "2025-04-01T..."
# }
```

`completeness_score` is computed as `(active_products + priced_products) / (2 * total) * 100`, clamped to 100. A score above 80 indicates a healthy catalogue.

---

## Provisioning Analytics

```python
kpis = await svc.provisioning_analytics("2025-Q1", tenant_id="acme")
# total_orders, completion_rate, fallout_count, jeopardy_count, bulk_job_count
```

---

## Bulk Operations

### Bulk Provisioning

```python
result = await svc.bulk_provisioning(
    order_ids=["ORD-100", "ORD-101", "ORD-102"],
    tenant_id="acme",
    workflow_type="new_service",
    approval_reference="OPS-APPROVAL-2025-042",
    submitted_by="ops.team@datacraft.co.ke",
)
# result["success_count"], result["error_count"], result["results"] per-order
```

### Bulk Cancel

```python
result = await svc.bulk_cancel_orders(
    order_ids=["ORD-100", "ORD-101"],
    reason="Customer withdrawal",
    tenant_id="acme",
)
```

### Bulk Price Update

See [Price List Management](#price-list-management) section above.

---

## Analytics Methods

| Method | Description |
|--------|-------------|
| `provisioning_analytics(period)` | Provisioning KPIs — completion rate, MTTP, fallout |
| `workflow_analytics()` | Workflow completion and failure breakdown |
| `resource_reservation_summary()` | Reservations by resource type |
| `rollback_analytics()` | Rollback frequency and trigger distribution |
| `config_push_analytics()` | Config push success rate |
| `activation_analytics()` | Activation rate |
| `catalogue_health_dashboard()` | Catalogue completeness score and health KPIs |

---

## Export and Compliance

```python
# Export service orders as CSV
export = await svc.export_orders(tenant_id="acme", format="csv")
# export["content"] — CSV string

# Provisioning compliance report
compliance = await svc.provisioning_compliance(tenant_id="acme")
# completion_rate_pct, jeopardy_orders count

# Audit trail
trail = await svc.get_audit_trail(tenant_id="acme")
```

---

## Health Check

```python
status = await svc.health_check(tenant_id="acme")
# {"service": "ServiceProvisioningService", "status": "healthy", ...}
```

---

## Permissions Reference

| Permission | Grants Access To |
|-----------|-----------------|
| `telecom_pro:catalogue` | Product catalogue CRUD, search, health dashboard |
| `telecom_pro:bundles` | Bundle management and decomposition |
| `telecom_pro:prices` | Price list management and bulk update |
| `telecom_pro:promotions` | Promotion campaign management |
| `telecom_pro:workflows` | Provisioning workflow console |
| `telecom_pro:resources` | Resource reservation management |
| `telecom_pro:config_push` | Network config push console |
| `telecom_pro:activation` | Service activation management |
| `telecom_pro:rollback` | Rollback console |
| `telecom_pro:bulk` | Bulk provisioning operations |
| `telecom_pro:network_elements` | NE health console |

---

## Composability

```apg
use telecom_pro;
use telecom_ord;   -- sends decomposed orders to telecom_pro
use telecom_inv;   -- resource reservation lookups
use telecom_bil;   -- charge activation on service completion
use telecom_cus;   -- customer lifecycle event on activation
```

---

## Further Reading

- `service.py` — Full business logic implementation
- `models.py` — Dataclass models for provisioning entities
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder view models
- `capability_contract.py` — Policy rules and contract definition
- `WORLD_CLASS_IMPROVEMENTS.md` — Detailed improvement roadmap
- `SPECIFICATION.md` — Full capability specification
