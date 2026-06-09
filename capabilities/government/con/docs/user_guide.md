# Government Contracts and Procurement

**Capability ID**: `government_con` | **Domain**: `government` | **Version**: `1.0.0`

## Description

End-to-end public procurement process covering tender management, bid evaluation, contract award, contract lifecycle management, variation control, performance monitoring, and PPDA compliance. Enforces the Public Procurement and Disposal Act requirements including debarment register and mandatory notifications.

## Installation

```bash
pip install apg-government-con
```

## Provides

- `tender_management_workflow`
- `evaluation_workflow`
- `contract_award_workflow`
- `contract_lifecycle_workflow`
- `contract_variation_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/government-con/dashboard` | `government_con:view` | Overview |
| `/government-con/tenders` | `government_con:tenders` | Procurement |
| `/government-con/evaluations` | `government_con:evaluate` | Procurement |
| `/government-con/awards` | `government_con:award` | Procurement |
| `/government-con/contracts` | `government_con:contracts` | Contracts |
| `/government-con/variations` | `government_con:vary` | Contracts |
| `/government-con/performance` | `government_con:performance` | Monitoring |
| `/government-con/ppda` | `government_con:ppda` | Compliance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `publish_tender()`
- `tender_publish()`
- `bid_submission()`
- `evaluation_committee()`
- `evaluate_bid()`
- `award_contract()`
- `contract_performance()`
- `variation_order()`

_(See `service.py` for complete API.)_

## Interoperability

`government_con` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use government_con;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GOVERNMENT_CON_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
