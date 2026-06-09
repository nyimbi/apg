# Multi-Country Operations

**Capability ID**: `loc_mco` | **Domain**: `loc` | **Version**: `1.0.0`

## Description

Multi-Country Operations (MCO) provides country entity management, local regulatory compliance mapping, cross-border intercompany transaction governance, and statutory reporting for organisations operating across multiple jurisdictions. It enforces arms-length transfer pricing, tenant-scoped entity isolation, and audit-trailed compliance workflows across any combination of supported jurisdictions.

## Installation

```bash
pip install apg-loc-mco
```

## Provides

- `country_entity_management`
- `regulatory_compliance_mapping`
- `intercompany_transaction_workflow`
- `statutory_reporting_workflow`
- `transfer_pricing_validation`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/loc-mco/dashboard` | `loc_mco:view` | Overview |
| `/loc-mco/countries` | `loc_mco:countries` | Setup |
| `/loc-mco/countries/create` | `loc_mco:countries_write` | Setup |
| `/loc-mco/entities` | `loc_mco:entities` | Setup |
| `/loc-mco/entities/create` | `loc_mco:entities_write` | Setup |
| `/loc-mco/compliance` | `loc_mco:compliance` | Compliance |
| `/loc-mco/compliance/create` | `loc_mco:compliance_write` | Compliance |
| `/loc-mco/intercompany` | `loc_mco:intercompany` | Transactions |

## Key Service Methods

- `uuid7str()`
- `uuid7str()`
- `describe()`
- `evaluate()`
- `register_country()`
- `get_country()`
- `list_countries()`
- `update_country()`
- `register_entity()`
- `get_entity()`

_(See `service.py` for complete API.)_

## Interoperability

`loc_mco` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use loc_mco;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `LOC_MCO_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
