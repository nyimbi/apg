# Regulatory Technology

**Capability ID**: `fintech_regtech` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Regulatory Technology provides automated tracking and management of regulatory obligations: regulatory source registration, change intake (new rules, updates, guidance, enforcement actions, consultations), obligation mapping with policy references, impact assessment across APG capabilities, regulatory filing preparation and submission, regulatory inquiry management, and approved response recording. It is the regulatory horizon scanning and filing layer that feeds obligation evidence into `fintech_compliance`.

## Installation

```bash
pip install apg-fintech-regtech
```

## Provides

- `regulatory_source_workflow`
- `regulatory_change_workflow`
- `regulatory_obligation_mapping_workflow`
- `regulatory_policy_mapping_workflow`
- `regulatory_impact_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-regtech/dashboard` | `fintech_regtech:view` | Overview |
| `/fintech-regtech/sources` | `fintech_regtech:sources` | Sources |
| `/fintech-regtech/changes` | `fintech_regtech:changes` | Horizon |
| `/fintech-regtech/obligations` | `fintech_regtech:obligations` | Obligations |
| `/fintech-regtech/impact` | `fintech_regtech:impact` | Impact |
| `/fintech-regtech/filings` | `fintech_regtech:filings` | Filings |
| `/fintech-regtech/submissions` | `fintech_regtech:submissions` | Filings |
| `/fintech-regtech/inquiries` | `fintech_regtech:inquiries` | Inquiries |

## Key Service Methods

- `describe()`
- `evaluate()`
- `regulatory_calendar()`
- `compliance_obligation_check()`
- `auto_report_generation()`
- `regulatory_change_monitoring()`
- `compliance_gap_analysis()`
- `prepare_filing()`
- `regulatory_filing()`
- `record_submission()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_regtech` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_regtech;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_REGTECH_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
