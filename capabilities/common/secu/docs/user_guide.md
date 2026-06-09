# Security Framework

**Capability ID**: `secu` | **Domain**: `common` | **Version**: `1.0.0`

## Description

The Security Framework capability (`secu`) is the executable security control plane for generated APG applications. It provides tenant-scoped security policy, device posture, threat indicator, access assessment, compliance

## Installation

```bash
pip install apg-common-secu
```

## Provides

- `risk_assessment`
- `threat_detection`
- `security_policies`
- `compliance_automation`
- `incident_response_governance`

## Requires

- `auth`
- `conf`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/secu/dashboard` | `secu:view` | Operations |
| `/secu/risk` | `secu:view_risk` | Operations |
| `/secu/threats` | `secu:view_threats` | Operations |
| `/secu/policies` | `secu:manage_policies` | Governance |
| `/secu/exceptions` | `secu:approve_exception` | Governance |
| `/secu/incidents` | `secu:respond` | Operations |
| `/secu/quarantine` | `secu:respond` | Operations |
| `/secu/compliance` | `secu:view_compliance` | Governance |

## Key Service Methods

- `initialize()`
- `_load_default_configurations()`
- `_load_security_policies()`
- `_set_config()`
- `get_config()`
- `update_policy()`
- `get_policies_for_context()`
- `_evaluate_policy_conditions()`
- `_evaluate_condition()`
- `_get_context_value()`

_(See `service.py` for complete API.)_

## Interoperability

`secu` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use secu;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `SECU_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
