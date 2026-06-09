# Pharmaceutical Distribution

**Capability ID**: `pharma_dis` | **Domain**: `pharma` | **Version**: `1.0.0`

## Description

Manages pharmaceutical distribution operations including cold chain monitoring, product serialisation and verification, wholesale distribution authorisations, product recalls, GDP compliance, and import/export shipment tracking. Enforces WDA requirements, temperature monitoring, serialisation verification, and recall timeline obligations at every distribution boundary.

## Installation

```bash
pip install apg-pharma-dis
```

## Provides

- `wholesale_distribution_workflow`
- `cold_chain_management_workflow`
- `serialisation_verification_workflow`
- `recall_management_workflow`
- `gdp_compliance_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-dis/dashboard` | `pharma_dis:view` | Overview |
| `/pharma-dis/shipments` | `pharma_dis:shipments` | Operations |
| `/pharma-dis/shipments/<id>` | `pharma_dis:shipments` | Operations |
| `/pharma-dis/cold-chain` | `pharma_dis:cold_chain` | Cold Chain |
| `/pharma-dis/cold-chain/excursions` | `pharma_dis:cold_chain` | Cold Chain |
| `/pharma-dis/serialisation` | `pharma_dis:serialisation` | Traceability |
| `/pharma-dis/recalls` | `pharma_dis:recalls` | Recalls |
| `/pharma-dis/recalls/<id>` | `pharma_dis:recalls` | Recalls |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_shipment()`
- `dispatch_shipment()`
- `deliver_shipment()`
- `get_shipment()`
- `list_shipments()`
- `create_cold_chain_record()`
- `report_excursion()`
- `list_excursions()`

_(See `service.py` for complete API.)_

## Interoperability

`pharma_dis` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pharma_dis;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PHARMA_DIS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
