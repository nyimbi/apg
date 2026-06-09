# Loyalty & Rewards

**Capability ID**: `retail_loy` | **Domain**: `retail` | **Version**: `1.0.0`

## Description

Provides end-to-end loyalty programme management for retail tenants: member enrolment with consent and identity verification, points earn/redeem/adjust transactions, tier qualification and downgrade management, coalition partner integration, targeted campaign authoring with approval workflows, a reward catalogue, customer lifetime value (CLV) segmentation, and configurable points-expiry policies. All operations are tenant-isolated, streamed to Bytewax, and governed by 28 deterministic rules.

## Installation

```bash
pip install apg-retail-loy
```

## Provides

- `loyalty_member_enrolment`
- `loyalty_points_earn`
- `loyalty_points_redeem`
- `loyalty_tier_management`
- `loyalty_campaign_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/retail-loy/dashboard` | `retail_loy:view` | Overview |
| `/retail-loy/members` | `retail_loy:view` | Members |
| `/retail-loy/members/<id>` | `retail_loy:view` | Members |
| `/retail-loy/members/enrol` | `retail_loy:write` | Members |
| `/retail-loy/transactions` | `retail_loy:view` | Transactions |
| `/retail-loy/earn` | `retail_loy:write` | Transactions |
| `/retail-loy/redeem` | `retail_loy:write` | Transactions |
| `/retail-loy/tiers` | `retail_loy:admin` | Programme |

## Key Service Methods

- `create_programme()`
- `get_programme()`
- `list_programmes()`
- `enrol_member()`
- `get_member()`
- `get_member_by_number()`
- `update_member()`
- `list_members()`
- `freeze_member()`
- `reactivate_member()`

_(See `service.py` for complete API.)_

## Interoperability

`retail_loy` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use retail_loy;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `RETAIL_LOY_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
