# Mobile Device Management

**Capability ID**: `mob_mdm` | **Domain**: `mob` | **Version**: `1.0.0`

## Description

The Mobile Device Management (MDM) capability provides an enterprise-grade device lifecycle management runtime. It covers device enrolment across multiple platforms and methods; deterministic policy creation, activation, and assignment; continuous compliance evaluation with automatic alert generation; silent app distribution; remote wipe with mandatory dual approval; MDM configuration profile deployment; and a device inventory registry — all tenant-scoped with full audit trails.

## Installation

```bash
pip install apg-mob-mdm
```

## Provides

- `device_enrolment_workflow`
- `mdm_policy_enforcement`
- `compliance_monitoring`
- `remote_wipe_workflow`
- `app_distribution_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mob-mdm/dashboard` | `mob_mdm:view` | Overview |
| `/mob-mdm/devices` | `mob_mdm:devices:list` | Devices |
| `/mob-mdm/devices/<device_id>` | `mob_mdm:devices:view` | Devices |
| `/mob-mdm/enrolment` | `mob_mdm:enrolment:manage` | Devices |
| `/mob-mdm/policies` | `mob_mdm:policies:list` | Policies |
| `/mob-mdm/policies/<policy_id>` | `mob_mdm:policies:view` | Policies |
| `/mob-mdm/compliance` | `mob_mdm:compliance:view` | Compliance |
| `/mob-mdm/compliance/<device_id>` | `mob_mdm:compliance:view` | Compliance |

## Key Service Methods

- `uuid7str()`
- `describe()`
- `evaluate()`
- `enrol_device()`
- `get_device()`
- `list_devices()`
- `update_device()`
- `unenrol_device()`
- `suspend_device()`
- `create_policy()`

_(See `service.py` for complete API.)_

## Interoperability

`mob_mdm` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mob_mdm;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MOB_MDM_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
