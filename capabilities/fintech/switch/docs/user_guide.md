# Payment Switch

**Capability ID**: `fintech_switch` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Payment Switch provides a world-class, standalone-deployable implementation of payment switch capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Installation

```bash
pip install apg-fintech-switch
```

## Provides

- `iso8583_message_switching`
- `payment_routing_engine`
- `channel_key_management`
- `pin_block_translation`
- `mac_generation_verification`

## Requires

- `auth`
- `audl`
- `ntfy`
- `keym`
- `encr`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-switch/dashboard` | `fintech_switch:view` | Overview |
| `/fintech-switch/routing` | `fintech_switch:manage_routing` | Routing |
| `/fintech-switch/transactions` | `fintech_switch:monitor` | Transactions |
| `/fintech-switch/channels` | `fintech_switch:manage_channels` | Channels |
| `/fintech-switch/security` | `fintech_switch:manage_keys` | Security |
| `/fintech-switch/mobile-money` | `fintech_switch:mobile_money` | Mobile Money |
| `/fintech-switch/settlement` | `fintech_switch:settle` | Settlement |
| `/fintech-switch/networks` | `fintech_switch:manage_networks` | Networks |

## Key Service Methods

- `_audit_event()`
- `route_transaction()`
- `switch_authorisation()`
- `_velocity_check_internal()`
- `settlement_routing()`
- `interchange_fee_calculation()`
- `scheme_compliance_check()`
- `switch_analytics()`
- `downtime_failover()`
- `transaction_replay()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_switch` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_switch;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_SWITCH_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
