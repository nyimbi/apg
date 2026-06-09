# Laboratory Information System

**Capability ID**: `healthcare_lab` | **Domain**: `healthcare` | **Version**: `1.0.0`

## Description

Full-featured LIS capability providing lab order management, specimen tracking with chain of custody, result entry and verification, critical value alerting with mandatory acknowledgement, QC management with Westgard rule evaluation, and instrument status tracking. Critical value workflow blocks result release until notification is confirmed.

## Installation

```bash
pip install apg-healthcare-lab
```

## Provides

- `lab_order_management`
- `specimen_tracking`
- `result_entry_verification`
- `critical_value_alerting`
- `qc_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/healthcare-lab/dashboard` | `healthcare_lab:view` | Overview |
| `/healthcare-lab/orders` | `healthcare_lab:orders` | Orders |
| `/healthcare-lab/orders/new` | `healthcare_lab:orders_write` | Orders |
| `/healthcare-lab/orders/<id>` | `healthcare_lab:orders` | Orders |
| `/healthcare-lab/specimens` | `healthcare_lab:specimens` | Specimens |
| `/healthcare-lab/specimens/<id>` | `healthcare_lab:specimens` | Specimens |
| `/healthcare-lab/results` | `healthcare_lab:results` | Results |
| `/healthcare-lab/results/entry` | `healthcare_lab:results_write` | Results |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_order()`
- `receive_lab_order()`
- `cancel_order()`
- `get_order()`
- `list_orders()`
- `collect_specimen()`
- `label_specimen()`
- `track_specimen_chain_of_custody()`

_(See `service.py` for complete API.)_

## Interoperability

`healthcare_lab` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use healthcare_lab;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `HEALTHCARE_LAB_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
