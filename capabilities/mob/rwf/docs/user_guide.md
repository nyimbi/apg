# Remote Workforce

**Capability ID**: `mob_rwf` | **Domain**: `mob` | **Version**: `1.0.0`

## Description

The Remote Workforce (RWF) capability provides a complete remote and hybrid work governance runtime. It manages remote work policy authoring, activation, and employee acknowledgment; VPN access provisioning with MFA enforcement and split-tunneling prevention; consent-based productivity tracking; equipment requisition with per-employee limits; digital onboarding orchestration with step tracking; remote compliance checks; and remote incident management — all governed by tenant-scoped deterministic rules with full audit trails.

## Installation

```bash
pip install apg-mob-rwf
```

## Provides

- `remote_work_policy_management`
- `vpn_access_governance`
- `productivity_tracking_workflow`
- `equipment_requisition_workflow`
- `digital_onboarding_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mob-rwf/dashboard` | `mob_rwf:view` | Overview |
| `/mob-rwf/policies` | `mob_rwf:policies:list` | Policies |
| `/mob-rwf/policies/<policy_id>` | `mob_rwf:policies:view` | Policies |
| `/mob-rwf/policies/<policy_id>/acknowledge` | `mob_rwf:policies:acknowledge` | Policies |
| `/mob-rwf/vpn` | `mob_rwf:vpn:list` | VPN |
| `/mob-rwf/vpn/provision` | `mob_rwf:vpn:provision` | VPN |
| `/mob-rwf/productivity` | `mob_rwf:productivity:view` | Productivity |
| `/mob-rwf/productivity/<employee_id>` | `mob_rwf:productivity:view` | Productivity |

## Key Service Methods

- `uuid7str()`
- `describe()`
- `evaluate()`
- `create_work_policy()`
- `activate_work_policy()`
- `update_work_policy()`
- `list_work_policies()`
- `get_work_policy()`
- `acknowledge_policy()`
- `list_acknowledgments()`

_(See `service.py` for complete API.)_

## Interoperability

`mob_rwf` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mob_rwf;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MOB_RWF_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
