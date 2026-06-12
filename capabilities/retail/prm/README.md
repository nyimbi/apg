# Promotions Management

## Overview
Provides complete promotion lifecycle management: authoring 12 promotion types with multi-trigger conditions, an approval workflow, stack-policy enforcement, budget and margin-floor governance, coupon issuance and redemption with expiry validation, channel and audience targeting, clearance/markdown optimisation with cascade support, real-time dynamic pricing, budget burn rate alerting, competitor price intelligence, promotion simulation, bulk coupon issuance, approval SLA tracking, immutable audit trail, and promotion effectiveness analytics. All operations are tenant-isolated and governed by 24 deterministic rules.

## Capability ID
`retail_prm`

## Provides
| Service | Description |
|---|---|
| promotion_authoring | Draft, author, and manage promotion definitions |
| promotion_activation | Approval-gated activation with budget enforcement |
| pricing_rules_engine | Priority-ordered pricing rule evaluation |
| dynamic_pricing | Demand-sensing price adjustment within configured corridors |
| coupon_management | Issue, track, and expire coupon codes (single + bulk) |
| coupon_redemption | Validated redemption with use-count tracking |
| markdown_optimisation | Clearance and slow-mover markdown with cascade |
| promotion_effectiveness_analytics | ROI, redemption rate, and basket uplift |
| audience_targeting | Segment-based and personalised targeting |
| promotion_budget_management | Real-time spend tracking with burn-rate alerting |
| promotion_stacking_engine | Exclusive, best-of, and additive stacking policies |
| competitor_price_intelligence | Ingest and analyse external price observations |
| promotion_simulation | Pre-activation margin and ROI impact modelling |
| approval_sla_tracking | Overdue approval detection and escalation |
| audit_trail | Append-only field-level change history |
| promotion_fatigue | Customer over-exposure detection and scoring |
| preflight_validator | Conflict pre-screening before plan activation |

## Requires
| Capability | Reason |
|---|---|
| auth | Promotion author and approver authentication |
| audl | Immutable promotion and redemption audit trail |
| mten | Tenant isolation |
| conf | Programme and stacking configuration |
| ntfy | Budget cap and activation notifications |
| wflo | Approval workflow for promotions and markdowns |
| mqeb | Bytewax batch apply stream |
| moni | Budget burn rate and effectiveness monitoring |
| nlpc | Audience description NLP |
| schd | Promotion start/end scheduling |

## Configuration
| Key | Default | Description |
|---|---|---|
| promotions.max_active_promotions | 500 | Concurrent active limit |
| promotions.approval_required | true | All promotions require approval |
| promotions.end_date_required | true | Open-ended promotions blocked |
| stacking.default_policy | best_of | Default stacking resolution |
| stacking.max_concurrent_promotions | 3 | Max promotions per basket |
| markdown.floor_margin_pct | 5 | Minimum margin after markdown |
| dynamic_pricing.max_markdown_pct | 40 | Maximum automated price reduction |
| fatigue.max_exposures_per_month | 10 | Customer promotion fatigue threshold |
| approval.sla_hours | 24 | Hours before approval is overdue |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /retail-prm/api/v1/promotions | GET/POST | List/create promotions | retail_prm:view/write |
| /retail-prm/api/v1/promotions/\<id\> | GET/PUT/DELETE | Detail/update/reject | retail_prm:view/write |
| /retail-prm/api/v1/promotions/\<id\>/submit | POST | Submit for approval | retail_prm:write |
| /retail-prm/api/v1/promotions/\<id\>/approve | POST | Approve promotion | retail_prm:approve |
| /retail-prm/api/v1/promotions/\<id\>/activate | POST | Activate promotion | retail_prm:approve |
| /retail-prm/api/v1/promotions/\<id\>/apply | POST | Apply to basket | retail_prm:write |
| /retail-prm/api/v1/promotions/\<id\>/simulate | POST | Simulate impact | retail_prm:view |
| /retail-prm/api/v1/promotions/\<id\>/burn-rate | GET | Budget burn rate | retail_prm:view |
| /retail-prm/api/v1/promotions/\<id\>/audit | GET | Audit trail | retail_prm:admin |
| /retail-prm/api/v1/promotions/preflight | POST | Pre-activation conflict check | retail_prm:write |
| /retail-prm/api/v1/promotions/overdue-approvals | GET | SLA-overdue approvals | retail_prm:approve |
| /retail-prm/api/v1/coupons | GET/POST | List/create coupons | retail_prm:view/write |
| /retail-prm/api/v1/coupons/bulk | POST | Bulk coupon issuance | retail_prm:write |
| /retail-prm/api/v1/coupons/redeem | POST | Redeem coupon | retail_prm:write |
| /retail-prm/api/v1/pricing | GET/POST | Pricing rules | retail_prm:admin |
| /retail-prm/api/v1/pricing/dynamic | POST | Compute dynamic price | retail_prm:view |
| /retail-prm/api/v1/markdown | GET/POST | Markdown plans | retail_prm:view/write |
| /retail-prm/api/v1/markdown/\<id\>/approve | PUT | Approve markdown | retail_prm:approve |
| /retail-prm/api/v1/effectiveness/\<id\> | GET | Effectiveness history | retail_prm:view |
| /retail-prm/api/v1/competitor-prices | POST | Ingest competitor price | retail_prm:admin |
| /retail-prm/api/v1/competitor-prices/\<sku\>/gap | GET | Price gap analysis | retail_prm:view |
| /retail-prm/api/v1/customers/\<id\>/fatigue | GET | Customer fatigue score | retail_prm:view |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| unapproved_activation_denied | activate without approval | deny |
| budget_exceeded_denied | budget_consumed >= budget_cap | deny |
| margin_floor_breach_denied | effective margin < floor | deny |
| expired_coupon_denied | coupon past valid_to | deny |
| coupon_already_redeemed_denied | times_used >= max_uses | deny |
| promotion_requires_end_date | no end_date set | deny |
| promotion_requires_budget_cap | budget_cap=0 | deny |
| markdown_exceeds_floor_margin | markdown breaches floor | deny |
| max_concurrent_promotions_exceeded | exclusive + another active | deny |
| excluded_item_protection | item in exclusion list | deny |
| dynamic_price_floor | adjusted price < 60% base | clamp to floor |
| bulk_coupon_limit | request > 50,000 codes | deny |
| competitor_gap_markdown_cap | competitor-driven markdown > 10% | cap at 10% |
| exhaustion_imminent_notification | utilisation >= 95% | notify via ntfy |

## Data Models
| Model | Key Fields |
|---|---|
| PrmPromotionResponse | id, promotion_code, approval_status, budget_consumed, redemption_count |
| PrmTriggerResponse | id, trigger_type, trigger_value, trigger_operator |
| PrmCouponResponse | id, coupon_code, coupon_type, times_used, status |
| PrmCouponRedemptionResponse | id, coupon_id, discount_applied, redeemed_at |
| PrmPricingRuleResponse | id, rule_type, adjustment_type, priority |
| PrmMarkdownResponse | id, markdown_type, markdown_pct, items_affected |
| PrmEffectivenessResponse | id, redemption_rate, roi, basket_uplift_pct |

## Key Service Methods

### Promotion Lifecycle
- `create_promotion()` — Draft a new promotion
- `get_promotion()` / `update_promotion()` — Read/modify draft
- `submit_for_approval()` / `approve_promotion()` / `reject_promotion()` — Workflow
- `activate_promotion()` / `pause_promotion()` — State control
- `list_promotions()` — Filtered list

### Basket Application
- `check_promotion_eligibility(cart_id, customer_id)` — Which promos apply
- `apply_promotion(tenant_id, promotion_id, basket_value, item_count)` — Apply + margin check
- `apply_promotion_to_cart(cart_id, promotion_id)` — Cart-level application with stacking
- `promotion_stacking_rules(promotion_ids)` — Compatibility matrix
- `preflight_promotion_plan(promotion_ids)` — Conflict check before activation

### Coupons
- `coupon_issue(customer_id, discount_pct, expiry)` — Single personalised coupon
- `bulk_issue_coupons(customer_ids, promotion_id, expiry, code_prefix)` — Mass issuance
- `coupon_redemption(coupon_code, transaction_id)` — Validate and redeem
- `create_coupon()` / `redeem_coupon()` — Lower-level model-driven methods
- `list_coupons()` — Filtered coupon list

### Dynamic Pricing
- `compute_dynamic_price(sku, base_price, channel, sell_through_rate, days_to_expiry)` — Demand-adjusted price
- `create_pricing_rule()` / `list_pricing_rules()` — Rule management

### Markdowns
- `markdown_schedule(sku, markdown_pct, effective_date, reason)` — Schedule a markdown
- `create_markdown()` / `approve_markdown()` / `list_markdowns()` — Lifecycle

### Analytics & Intelligence
- `promotion_performance(promotion_id)` — Single promotion metrics
- `promotion_analytics(period)` — Tenant-level summary
- `simulate_promotion_impact(promotion_id, expected_redemptions, avg_basket_value)` — Pre-activation projection
- `check_budget_burn_rate(promotion_id)` — Burn rate health + alerting
- `get_customer_promotion_fatigue(customer_id, window_days)` — Fatigue score
- `record_effectiveness()` / `get_effectiveness()` — Effectiveness records
- `promotion_summary()` — Aggregated summary

### Competitor Intelligence
- `ingest_competitor_price(sku, competitor, price, source_url, captured_at)` — Record external price
- `compute_price_gap_analysis(sku)` — Gap vs. competitor stats

### Approval Governance
- `list_overdue_approvals(sla_hours)` — SLA breach detection
- `get_promotion_audit_trail(promotion_id)` — Full change history

## Streaming Events
- `promotion_created`, `promotion_approved`, `promotion_activated`, `promotion_paused`, `promotion_expired`
- `coupon_issued`, `coupon_redeemed`, `coupon_voided`
- `markdown_applied`
- `budget_cap_reached`, `margin_floor_breached`, `prm.budget.exhaustion_imminent`
- `effectiveness_calculated`

## Edge Cases Handled
- Activation without prior approval: assertion failure
- Budget already consumed at activation: blocked
- Margin floor breach: apply returns applied=False gracefully
- Expired coupon redemption: assertion with expiry check
- Duplicate coupon code per tenant: ValueError raised
- Exclusive stacking with concurrent active: denied
- Markdown deeper than floor margin: assertion at create time
- Open-ended promotions (no end_date): rule engine denies
- Dynamic price below 60% base: clamped to floor
- Bulk coupon code collision: up to 5 retries, then skip with counter
- Budget burn rate exhaustion_imminent: notification event fired

## Composability Notes
- **retail_pos** calls `apply_promotion` at checkout to compute line discounts
- **retail_omc** applies pricing rules and `compute_dynamic_price` to online cart
- **retail_loy** loyalty_multiplier campaign type modifies earn rate; fatigue score gates campaign delivery
- **retail_sin** shopper journey stages can trigger campaign activation
- **intel_moni** subscribes to `prm.budget.exhaustion_imminent` for ops alerting

---

## World-Class Enhancements (v2.0)

1. **Real-Time Dynamic Pricing Engine** — demand-sensing price adjustment within pre-approved corridors using sell-through velocity and time-to-expiry signals
2. **Multi-Buy / Bundle Offer Types** — `BundleOffer` and `MultiBuyOffer` models with per-line discount breakdowns at basket apply time
3. **Promotion Canary / A-B Testing Framework** — deterministic bucket assignment (hash mod 100) with `control`/`treatment` variant tracking on effectiveness records
4. **Promotion Fatigue Detection** — rolling 30-day exposure counter per customer with configurable `max_customer_exposures_per_month` gate
5. **Cascade Markdown Automation** — `advance_markdown_cascade` auto-progresses markdown tiers on schedule via the `schd` capability
6. **Promotion ROI Attribution Pipeline** — `compute_promotion_roi` aggregates incremental revenue with counterfactual baseline and writes `PrmEffectivenessResponse` automatically
7. **Budget Burn Rate Alerting** — `check_budget_burn_rate` returns `on_track | accelerating | exhaustion_imminent` and fires `ntfy` events at 95% utilisation
8. **Competitor Price Parity Monitoring** — ingest external prices, compute gap analysis, and get margin-safe competitive markdown suggestions per SKU
9. **Approval Workflow SLA Tracking** — `list_overdue_approvals` detects promotions in `pending_review` beyond `sla_hours` and escalates via `wflo`/`ntfy`
10. **Personalised Coupon Generation at Scale** — `bulk_issue_coupons` generates up to 50,000 unique codes in a single async call with CSV export URL
11. **Promotion Conflict Pre-flight Validator** — `preflight_promotion_plan` returns a conflict matrix before any promotion is activated, preventing impossible stacking combinations
12. **Tenant Analytics Benchmarking** — percentile rank for redemption rate, discount depth, and ROI against anonymised cross-tenant peer stats
13. **Promotion Lifecycle Event Bus** — structured CloudEvents published via `mqeb` on `retail_prm.promotion.{activated|expired|budget_exhausted|margin_breach}` topics
14. **Margin-Aware Promotion Simulation** — `simulate_promotion_impact` projects ROI, margin headroom, and break-even redemption count before activation
15. **Immutable Audit Trail via Append-Only Ledger** — field-level `_audit_ledger` with `get_promotion_audit_trail` persisted to PostgreSQL `prm_audit_log` via the `audl` capability

---

## New Methods

### `compute_dynamic_price` — demand-adjusted pricing

```python
result = await svc.compute_dynamic_price(
    sku="SKU-001",
    base_price=100.0,
    channel="online",
    sell_through_rate=0.85,   # 85% of stock sold
    days_to_expiry=5,         # perishable urgency
)
# result["adjusted_price"]  — clamped to 60% floor
# result["markdown_pct"]    — e.g. 12.5
# result["signal"]          — "high_sell_through" | "expiry_urgency" | etc.
```

### `simulate_promotion_impact` — pre-activation margin modelling

```python
simulation = await svc.simulate_promotion_impact(
    promotion_id="prm-uuid",
    expected_redemptions=2500,
    avg_basket_value=45.00,
)
# simulation["projected_roi"]          — e.g. 2.3 (230%)
# simulation["margin_floor_headroom"]  — pct points above floor
# simulation["break_even_redemptions"] — minimum redemptions to be ROI-positive
# simulation["confidence_interval"]    — [low, high] based on historical data
```

### `bulk_issue_coupons` — mass personalised coupon issuance

```python
summary = await svc.bulk_issue_coupons(
    customer_ids=["c-001", "c-002", ...],  # up to 50,000
    promotion_id="prm-uuid",
    expiry=datetime(2026, 12, 31),
    code_prefix="VIP-",
)
# summary["issued_count"]       — codes successfully created
# summary["duplicate_skipped"]  — collision retries exhausted
# summary["export_url"]         — signed URL to CSV of all codes
```
