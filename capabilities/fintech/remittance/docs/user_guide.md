# Cross-Border Remittance

**Capability ID**: `fintech_remittance` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Cross-Border Remittance manages the lifecycle of international money transfers: corridor and currency eligibility checks, FX quote creation with rate and fee locking, transfer creation with dual-side KYC and source-of-funds evidence, AML screening with sanctions blocking, fraud decisioning, payout release with provider receipt, and refund handling. Same-country transfers are architecturally blocked — the capability is strictly cross-border.

## Installation

```bash
pip install apg-fintech-remittance
```

## Provides

- `remittance_corridor_governance`
- `remittance_quote_lifecycle`
- `cross_border_transfer_workflow`
- `remittance_payout_workflow`
- `remittance_refund_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-remittance/dashboard` | `fintech_remittance:view` | Overview |
| `/fintech-remittance/corridors` | `fintech_remittance:govern_corridors` | Corridors |
| `/fintech-remittance/quotes` | `fintech_remittance:quote` | Quotes |
| `/fintech-remittance/transfers` | `fintech_remittance:transfer` | Transfers |
| `/fintech-remittance/payouts` | `fintech_remittance:payout` | Payouts |
| `/fintech-remittance/refunds` | `fintech_remittance:refund` | Exceptions |
| `/fintech-remittance/agents` | `fintech_remittance:admin` | Automation |
| `/fintech-remittance/settings` | `fintech_remittance:admin` | Administration |

## Key Service Methods

- `describe()`
- `evaluate()`
- `get_fx_quote()`
- `initiate_remittance()`
- `compliance_check()`
- `partner_routing()`
- `track_remittance()`
- `recipient_notification()`
- `payout_methods()`
- `deliver_to_mobile_money()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_remittance` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_remittance;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_REMITTANCE_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
