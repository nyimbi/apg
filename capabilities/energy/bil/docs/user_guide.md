# Energy Billing & Tariffs

**Capability ID**: `energy_bil` | **Domain**: `energy` | **Version**: `1.0.0`

## Description

Energy Billing & Tariffs manages the complete revenue cycle from tariff configuration through bill generation, payment processing, credit issuance, dispute resolution, and revenue assurance. It supports 13 tariff structures including time-of-use, demand charges, and net metering. Collection rates, write-off approvals, and revenue assurance flagging ensure financial governance across all customer classes.

## Installation

```bash
pip install apg-energy-bil
```

## Provides

- `tariff_management`
- `consumption_billing`
- `demand_charge_calculation`
- `renewable_credits_management`
- `revenue_assurance`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/energy-bil/dashboard` | `energy_bil:view` | Overview |
| `/energy-bil/tariffs` | `energy_bil:tariffs` | Configuration |
| `/energy-bil/tariffs/<id>` | `energy_bil:tariffs` | Configuration |
| `/energy-bil/bills` | `energy_bil:billing` | Billing |
| `/energy-bil/bills/<id>` | `energy_bil:billing` | Billing |
| `/energy-bil/payments` | `energy_bil:payments` | Payments |
| `/energy-bil/credits` | `energy_bil:credits` | Credits |
| `/energy-bil/disputes` | `energy_bil:disputes` | Customer Service |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_tariff()`
- `approve_tariff()`
- `activate_tariff()`
- `list_tariffs()`
- `get_active_tariff()`
- `generate_bill()`
- `issue_bill()`
- `write_off_bill()`

_(See `service.py` for complete API.)_

## Interoperability

`energy_bil` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use energy_bil;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ENERGY_BIL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
