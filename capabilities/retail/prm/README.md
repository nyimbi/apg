# Promotions Management

## Overview
Provides complete promotion lifecycle management: authoring 12 promotion types with multi-trigger conditions, an approval workflow, stack-policy enforcement, budget and margin-floor governance, coupon issuance and redemption with expiry validation, channel and audience targeting, clearance/markdown optimisation with cascade support, real-time budget tracking, and promotion effectiveness analytics. All operations are tenant-isolated and governed by 24 deterministic rules.

## Capability ID
`retail_prm`

## Provides
| Service | Description |
|---|---|
| promotion_authoring | Draft, author, and manage promotion definitions |
| promotion_activation | Approval-gated activation with budget enforcement |
| pricing_rules_engine | Priority-ordered pricing rule evaluation |
| coupon_management | Issue, track, and expire coupon codes |
| coupon_redemption | Validated redemption with use-count tracking |
| markdown_optimisation | Clearance and slow-mover markdown with cascade |
| promotion_effectiveness_analytics | ROI, redemption rate, and basket uplift |
| audience_targeting | Segment-based and personalised targeting |
| promotion_budget_management | Real-time spend tracking against caps |
| promotion_stacking_engine | Exclusive, best-of, and additive stacking policies |

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

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /retail-prm/api/v1/promotions | GET/POST | List/create promotions | retail_prm:view/write |
| /retail-prm/api/v1/promotions/<id> | GET/PUT/DELETE | Detail/update/reject | retail_prm:view/write |
| /retail-prm/api/v1/promotions/<id>/submit | POST | Submit for approval | retail_prm:write |
| /retail-prm/api/v1/promotions/<id>/approve | POST | Approve promotion | retail_prm:approve |
| /retail-prm/api/v1/promotions/<id>/activate | POST | Activate promotion | retail_prm:approve |
| /retail-prm/api/v1/promotions/<id>/apply | POST | Apply to basket | retail_prm:write |
| /retail-prm/api/v1/coupons | GET/POST | List/create coupons | retail_prm:view/write |
| /retail-prm/api/v1/coupons/redeem | POST | Redeem coupon | retail_prm:write |
| /retail-prm/api/v1/pricing | GET/POST | Pricing rules | retail_prm:admin |
| /retail-prm/api/v1/markdown | GET/POST | Markdown plans | retail_prm:view/write |
| /retail-prm/api/v1/markdown/<id>/approve | PUT | Approve markdown | retail_prm:approve |
| /retail-prm/api/v1/effectiveness/<id> | GET | Effectiveness history | retail_prm:view |

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

## Streaming Events
- `promotion_created`, `promotion_approved`, `promotion_activated`, `promotion_paused`, `promotion_expired`
- `coupon_issued`, `coupon_redeemed`, `coupon_voided`
- `markdown_applied`
- `budget_cap_reached`, `margin_floor_breached`
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

## Composability Notes
- **retail_pos** calls apply_promotion at checkout to compute line discounts
- **retail_omc** applies pricing rules and promotions to online cart
- **retail_loy** loyalty_multiplier campaign type modifies earn rate
- **retail_sin** shopper journey stages can trigger campaign activation
