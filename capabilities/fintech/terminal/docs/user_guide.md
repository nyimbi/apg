# Terminal Management System

**Capability ID**: `fintech_terminal` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Terminal Management System provides a world-class, standalone-deployable implementation of terminal management system capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Installation

```bash
pip install apg-fintech-terminal
```

## Provides

- `terminal_lifecycle_management`
- `terminal_key_injection_workflow`
- `terminal_parameter_deployment`
- `terminal_certificate_management`
- `terminal_health_monitoring`

## Requires

- `auth`
- `audl`
- `ntfy`
- `keym`
- `encr`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-terminal/dashboard` | `fintech_terminal:view` | Overview |
| `/fintech-terminal/terminals` | `fintech_terminal:manage` | Terminals |
| `/fintech-terminal/keys` | `fintech_terminal:manage_keys` | Security |
| `/fintech-terminal/parameters` | `fintech_terminal:deploy_parameters` | Configuration |
| `/fintech-terminal/certificates` | `fintech_terminal:manage_certificates` | Security |
| `/fintech-terminal/compliance` | `fintech_terminal:compliance` | Compliance |
| `/fintech-terminal/mobile-money` | `fintech_terminal:mobile_money` | Mobile Money |
| `/fintech-terminal/health` | `fintech_terminal:monitor` | Operations |

## Key Service Methods

- `_audit_event()`
- `_get_terminal()`
- `_assert_active()`
- `register_terminal()`
- `activate_terminal()`
- `terminal_transaction()`
- `cash_deposit()`
- `cash_withdrawal()`
- `fund_transfer_terminal()`
- `bill_payment_terminal()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_terminal` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_terminal;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_TERMINAL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
