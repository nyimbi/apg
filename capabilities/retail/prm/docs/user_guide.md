# Promotions Management — User Guide

**Capability ID**: `retail_prm` | **Domain**: `retail` | **Version**: `1.1.0` | **© 2025 Datacraft**

---

## Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Promotion Lifecycle](#promotion-lifecycle)
5. [Coupon Management](#coupon-management)
6. [Dynamic Pricing](#dynamic-pricing)
7. [Markdowns](#markdowns)
8. [Analytics and Simulation](#analytics-and-simulation)
9. [Competitor Price Intelligence](#competitor-price-intelligence)
10. [Approval Governance](#approval-governance)
11. [Customer Promotion Fatigue](#customer-promotion-fatigue)
12. [Audit Trail](#audit-trail)
13. [Configuration Reference](#configuration-reference)
14. [Composability](#composability)

---

## Overview

`retail_prm` is the promotions engine for the APG retail domain. It governs the full lifecycle of a promotion from authoring through approval, activation, basket application, and post-campaign analytics. Key capabilities introduced in v1.1.0:

- **Dynamic pricing** — demand-sensing price adjustments within configured corridors
- **Budget burn-rate alerting** — proactive notification before budget exhaustion
- **Competitor price intelligence** — ingest external prices and compute gap analysis
- **Promotion simulation** — model impact before activation
- **Bulk coupon issuance** — mass personalised code generation
- **Approval SLA tracking** — escalation for overdue reviews
- **Immutable audit trail** — append-only field-level change history
- **Customer fatigue scoring** — prevent over-promotion of individual customers
- **Preflight validator** — conflict detection before plan activation

All operations are tenant-isolated. No cross-tenant data leaks are possible by design.

---

## Installation

```bash
pip install apg-retail-prm
```

Or add to your `pyproject.toml`:

```toml
[project.dependencies]
apg-retail-prm = ">=1.1.0"
```

---

## Quick Start

```python
import asyncio
from apg_retail_prm.service import PrmService

svc = PrmService(tenant_id="acme", actor_id="admin@acme.com")

async def main():
    # Create and activate a promotion
    promo = await svc.create_promotion(
        name="Summer Flash Sale",
        promo_type="percentage",
        discount_value=15.0,
        start_date="2026-07-01",
        end_date="2026-07-31",
        conditions={"budget_cap": 50000, "min_spend": 500, "margin_floor_pct": 10},
    )
    await svc.activate_promotion(promo["id"])

    # Check eligibility for a cart
    eligibility = await svc.check_promotion_eligibility(
        cart_id="cart-001", customer_id="cust-abc"
    )
    print(eligibility["eligible_promotions"])

asyncio.run(main())
```

---

## Promotion Lifecycle

### 1. Author

```python
promo = await svc.create_promotion(
    name="Q3 Clearance",
    promo_type="percentage",
    discount_value=20.0,
    start_date="2026-09-01",
    end_date="2026-09-30",
    conditions={
        "budget_cap": 100_000,
        "margin_floor_pct": 8,
        "eligible_skus": ["SKU-A", "SKU-B"],
        "eligible_segments": ["loyalty_gold", "loyalty_platinum"],
    },
)
promotion_id = promo["id"]
```

### 2. Submit for Approval

```python
await svc.submit_for_approval(tenant_id="acme", promotion_id=promotion_id, by="alice")
```

### 3. Approve

```python
await svc.approve_promotion(tenant_id="acme", promotion_id=promotion_id, by="manager")
```

### 4. Activate

```python
await svc.activate_promotion(promotion_id)
```

### 5. Apply to Basket

```python
result = await svc.apply_promotion(
    tenant_id="acme",
    promotion_id=promotion_id,
    basket_value=1200.0,
    item_count=3,
)
# {"applied": True, "discount_amount": 240.0, ...}
```

### 6. Pause / Reject

```python
await svc.pause_promotion(tenant_id="acme", promotion_id=promotion_id)
await svc.reject_promotion(tenant_id="acme", promotion_id=promotion_id, reason="budget_cut", by="cfo")
```

### Stacking

Multiple promotions on a single cart are subject to the stacking policy. Incompatible pairs (e.g. two percentage discounts) are blocked:

```python
compat = await svc.promotion_stacking_rules(["pid-1", "pid-2", "pid-3"])
# {"stackable": True/False, "conflicts": [...], "allowed_combinations": [...]}
```

Pre-screen a plan before activation to avoid runtime conflicts:

```python
plan = await svc.preflight_promotion_plan(["pid-1", "pid-2"])
# {"plan_viable": True, "conflicts": [], "suggestions": []}
```

---

## Coupon Management

### Issue a Single Coupon

```python
coupon = await svc.coupon_issue(
    customer_id="cust-001",
    discount_pct=10.0,
    expiry="2026-12-31",
)
```

### Bulk Issue

Generate up to 50,000 personalised codes in one call:

```python
result = await svc.bulk_issue_coupons(
    customer_ids=["c1", "c2", "c3", ...],
    promotion_id=promotion_id,
    expiry="2026-12-31",
    code_prefix="VIP",
)
# {"issued_count": 3, "duplicate_skipped": 0, "coupon_ids": [...]}
```

### Redeem

```python
redemption = await svc.coupon_redemption(
    coupon_code="VIP-ABCDE1234",
    transaction_id="TXN-20260611-001",
)
```

Redemption validates:
- Coupon exists and belongs to tenant
- Status is `active`
- Not past `valid_to`
- `times_used < max_uses`

---

## Dynamic Pricing

`compute_dynamic_price` adjusts a base price using active pricing rules, sell-through velocity, days-to-expiry pressure, and competitor price gap observations. The adjusted price is never allowed to drop below 60% of the base price (configurable floor).

```python
result = await svc.compute_dynamic_price(
    sku="SKU-WINTER-JACKET",
    base_price=4500.0,
    channel="online",
    sell_through_rate=0.18,   # 18% sold — below 30% threshold
    days_to_expiry=5,         # seasonal end approaching
)
# {
#   "sku": "SKU-WINTER-JACKET",
#   "base_price": 4500.0,
#   "adjusted_price": 3690.0,
#   "adjustment_pct": -18.0,
#   "reasoning": ["sell_through_pressure:-6.0%", "expiry_pressure:-6.0%", ...]
# }
```

### Pricing Rules

Static rules are evaluated before dynamic signals and take highest priority:

```python
rule = await svc.create_pricing_rule(PrmPricingRuleCreate(
    tenant_id="acme",
    name="Electronics -5%",
    rule_type="category",
    category_path=["electronics"],
    adjustment_type="percentage",
    adjustment_value=-5.0,
    priority=10,
    valid_from=datetime.utcnow(),
    created_by="admin",
))
```

---

## Markdowns

Markdowns reduce the price of slow-moving or clearance SKUs. They flow through the same draft → approved workflow as promotions.

```python
md = await svc.markdown_schedule(
    sku="SKU-OLDSTOCK-007",
    markdown_pct=30.0,
    effective_date="2026-07-15",
    reason="clearance",
)

await svc.approve_markdown(tenant_id="acme", markdown_id=md["id"], by="manager")
```

Use `create_markdown` for multi-SKU or category-level markdowns with cascade:

```python
from apg_retail_prm.models import PrmMarkdownCreate

md = await svc.create_markdown(PrmMarkdownCreate(
    tenant_id="acme",
    name="Winter Clearance Wave 2",
    markdown_type="clearance",
    sku_list=["SKU-W001", "SKU-W002", "SKU-W003"],
    markdown_pct=40.0,
    floor_margin_pct=5.0,
    cascade_enabled=True,
    cascade_interval_days=7,
    cascade_increment_pct=10.0,
    effective_from=datetime(2026, 8, 1),
    effective_to=datetime(2026, 8, 31),
    created_by="admin",
))
```

---

## Analytics and Simulation

### Single Promotion Performance

```python
perf = await svc.promotion_performance(promotion_id)
# {
#   "redemption_count": 412,
#   "total_discount_issued": 74160.0,
#   "budget_utilisation_pct": 74.16,
#   "coupons_issued": 500,
#   "coupons_redeemed": 412,
#   ...
# }
```

### Tenant Analytics

```python
analytics = await svc.promotion_analytics(period="2026-Q3")
# Includes: active count, total discount, redemption rate, top 5 promotions, markdown count
```

### Simulate Before Activating

Model the ROI and margin impact of a promotion before going live:

```python
sim = await svc.simulate_promotion_impact(
    promotion_id=promotion_id,
    expected_redemptions=300,
    avg_basket_value=2500.0,
)
# {
#   "projected_total_discount": 112500.0,
#   "projected_budget_utilisation_pct": 112.5,  # over budget — reconsider!
#   "effective_margin_pct": 94.0,
#   "margin_floor_headroom_pct": 86.0,
#   "break_even_redemptions": 30,
#   "projected_roi_pct": 11.1,
#   "confidence": "low"
# }
```

### Budget Burn Rate

```python
burn = await svc.check_budget_burn_rate(promotion_id)
# {
#   "burn_rate_health": "accelerating",
#   "actual_daily_burn": 3200.0,
#   "expected_daily_burn": 1612.9,
#   "days_remaining_at_current_rate": 8.1
# }
```

When `burn_rate_health == "exhaustion_imminent"` (utilisation >= 95%), a notification event is automatically published to the `ntfy` capability.

---

## Competitor Price Intelligence

Ingest external price observations (from web scrapers, data feeds, or manual entry):

```python
await svc.ingest_competitor_price(
    sku="SKU-PHONE-128GB",
    competitor="TechMart",
    price=42999.0,
    source_url="https://techmart.example/phones/128gb",
    captured_at="2026-06-10T14:00:00",
)
```

Analyse the gap:

```python
gap = await svc.compute_price_gap_analysis("SKU-PHONE-128GB")
# {
#   "tenant_price": 45999.0,
#   "comp_median": 42999.0,
#   "gap_vs_median_pct": 6.98,  # tenant is 7% above median
#   "competitor_count": 3
# }
```

The `compute_dynamic_price` method automatically incorporates competitor observations when computing adjusted prices, applying up to a 10% markdown to close a price gap.

---

## Approval Governance

### SLA Tracking

Promotions in `pending_review` for longer than the configured SLA are surfaced for escalation:

```python
overdue = await svc.list_overdue_approvals(sla_hours=24.0)
# [
#   {
#     "promotion_id": "...",
#     "name": "Flash Weekend",
#     "age_hours": 31.4,
#     "overdue_by_hours": 7.4
#   }
# ]
```

Integrate with `wflo` to trigger an escalation chain when `overdue_by_hours > 0`.

### Triggers

Attach trigger conditions to promotions to define threshold-based activation:

```python
trigger = await svc.add_trigger(PrmTriggerCreate(
    tenant_id="acme",
    promotion_id=promotion_id,
    trigger_type="basket_value",
    trigger_value=1000.0,
    trigger_operator="gte",
    created_by="admin",
))
```

---

## Customer Promotion Fatigue

Prevent over-promotion by scoring individual customers:

```python
fatigue = await svc.get_customer_promotion_fatigue(
    customer_id="cust-vip-001",
    window_days=30,
)
# {
#   "fatigue_score": 40,
#   "fatigue_level": "medium",
#   "exposure_count": 4
# }
```

Use the fatigue score to gate campaign delivery:

```python
if fatigue["fatigue_level"] != "high":
    await svc.coupon_issue(customer_id="cust-vip-001", ...)
```

---

## Audit Trail

Every promotion state change is recorded in an append-only audit ledger:

```python
trail = await svc.get_promotion_audit_trail(promotion_id)
# [
#   {"field": "approval_status", "old_value": "draft", "new_value": "pending_review",
#    "changed_by": "alice", "changed_at": "2026-06-01T09:00:00"},
#   {"field": "approval_status", "old_value": "pending_review", "new_value": "approved",
#    "changed_by": "manager", "changed_at": "2026-06-01T14:30:00"},
#   ...
# ]
```

In production deployments, the ledger is persisted to the `prm_audit_log` PostgreSQL table via the `audl` capability for compliance and dispute resolution.

---

## Configuration Reference

All keys are tenant-scoped and settable via the `conf` capability or environment variables prefixed `RETAIL_PRM_`.

| Key | Default | Description |
|---|---|---|
| `promotions.max_active_promotions` | 500 | Concurrent active promotion limit |
| `promotions.approval_required` | true | Require approval before activation |
| `promotions.end_date_required` | true | Block open-ended promotions |
| `stacking.default_policy` | best_of | Default stacking resolution |
| `stacking.max_concurrent_promotions` | 3 | Max promotions per basket |
| `markdown.floor_margin_pct` | 5 | Minimum margin post-markdown |
| `dynamic_pricing.max_markdown_pct` | 40 | Maximum automated price reduction |
| `dynamic_pricing.sell_through_threshold` | 0.30 | Velocity below which markdown is triggered |
| `fatigue.max_exposures_per_month` | 10 | Exposures above which level is "high" |
| `approval.sla_hours` | 24 | Pending review SLA in hours |
| `competitor.max_gap_markdown_pct` | 10 | Cap on competitor-driven markdown |

---

## Composability

```apg
use retail_prm;
```

| Capability | Integration |
|---|---|
| `retail_pos` | Calls `apply_promotion` at checkout for line-item discounts |
| `retail_omc` | Uses `compute_dynamic_price` and `apply_promotion_to_cart` for online basket |
| `retail_loy` | `loyalty_multiplier` promotion type modifies earn rate; fatigue score gates delivery |
| `retail_sin` | Shopper journey stages trigger `activate_promotion` |
| `intel_moni` | Subscribes to `prm.budget.exhaustion_imminent` for ops dashboards |
| `audl` | Receives promotion audit ledger entries for compliance persistence |
| `ntfy` | Receives budget burn alerts and SLA escalations |
| `wflo` | Orchestrates multi-step approval chains |
| `schd` | Schedules `advance_markdown_cascade` and promotion expiry checks |

---

*Further reading: `service.py` (business logic), `models.py` (Pydantic models), `api.py` (REST endpoints), `WORLD_CLASS_IMPROVEMENTS.md` (roadmap).*
