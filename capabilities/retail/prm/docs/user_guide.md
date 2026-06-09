# Promotions Management

**Capability ID**: `retail_prm` | **Domain**: `retail` | **Version**: `1.0.0`

## Description

Provides complete promotion lifecycle management: authoring 12 promotion types with multi-trigger conditions, an approval workflow, stack-policy enforcement, budget and margin-floor governance, coupon issuance and redemption with expiry validation, channel and audience targeting, clearance/markdown optimisation with cascade support, real-time budget tracking, and promotion effectiveness analytics. All operations are tenant-isolated and governed by 24 deterministic rules.

## Installation

```bash
pip install apg-retail-prm
```

## Provides

- `promotion_authoring`
- `promotion_activation`
- `pricing_rules_engine`
- `coupon_management`
- `coupon_redemption`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/retail-prm/dashboard` | `retail_prm:view` | Overview |
| `/retail-prm/promotions` | `retail_prm:view` | Promotions |
| `/retail-prm/promotions/<id>` | `retail_prm:view` | Promotions |
| `/retail-prm/promotions/create` | `retail_prm:write` | Promotions |
| `/retail-prm/coupons` | `retail_prm:view` | Coupons |
| `/retail-prm/coupons/create` | `retail_prm:write` | Coupons |
| `/retail-prm/coupons/redeem` | `retail_prm:write` | Coupons |
| `/retail-prm/pricing` | `retail_prm:admin` | Pricing |

## Key Service Methods

- `create_promotion()`
- `get_promotion()`
- `update_promotion()`
- `activate_promotion()`
- `check_promotion_eligibility()`
- `apply_promotion_to_cart()`
- `promotion_stacking_rules()`
- `submit_for_approval()`
- `approve_promotion()`
- `reject_promotion()`

_(See `service.py` for complete API.)_

## Interoperability

`retail_prm` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use retail_prm;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `RETAIL_PRM_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
