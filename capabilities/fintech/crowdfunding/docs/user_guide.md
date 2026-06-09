# Crowdfunding Platform

**Capability ID**: `fintech_crowdfunding` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Crowdfunding Platform manages the lifecycle of alternative finance campaigns: issuer due diligence, campaign publishing across equity, debt, reward, donation, and revenue-share structures, investor disclosure management, commitment recording, escrow funding, milestone tracking, payout authorization, investor updates, compliance alerts, and review workflows. It is designed for regulated crowdfunding operations where every campaign requires disclosure review before investors can commit.

## Installation

```bash
pip install apg-fintech-crowdfunding
```

## Provides

- `crowdfunding_issuer_workflow`
- `crowdfunding_campaign_workflow`
- `crowdfunding_disclosure_workflow`
- `crowdfunding_commitment_workflow`
- `crowdfunding_escrow_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-crowdfunding/dashboard` | `fintech_crowdfunding:view` | Overview |
| `/fintech-crowdfunding/issuers` | `fintech_crowdfunding:issuers` | Issuers |
| `/fintech-crowdfunding/campaigns` | `fintech_crowdfunding:campaigns` | Campaigns |
| `/fintech-crowdfunding/disclosures` | `fintech_crowdfunding:disclosures` | Campaigns |
| `/fintech-crowdfunding/commitments` | `fintech_crowdfunding:commitments` | Investors |
| `/fintech-crowdfunding/escrow` | `fintech_crowdfunding:escrow` | Funds |
| `/fintech-crowdfunding/milestones` | `fintech_crowdfunding:milestones` | Funds |
| `/fintech-crowdfunding/payouts` | `fintech_crowdfunding:payouts` | Funds |

## Key Service Methods

- `describe()`
- `evaluate()`
- `onboard_issuer()`
- `get_issuer()`
- `launch_campaign()`
- `campaign_status()`
- `campaign_analytics()`
- `contribute()`
- `refund_failed_campaign()`
- `disburse_funds()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_crowdfunding` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_crowdfunding;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_CROWDFUNDING_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
