# Zero Trust Network Access

**Capability ID**: `ztna` | **Domain**: `common` | **Version**: `1.0.0`

## Description

`ztna` is APG's package-backed Zero Trust Network Access capability. It gives generated applications a tenant-scoped access broker for identity, device posture, protected resources, access requests, access reviews, governed

## Installation

```bash
pip install apg-common-ztna
```

## Provides

_(see capability contract)_

## Requires

_(none)_

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/ztna/dashboard` | `ztna:view` | Overview |
| `/ztna/policies` | `ztna:manage_policies` | Policies |
| `/ztna/identities` | `ztna:manage_policies` | Identity |
| `/ztna/devices` | `ztna:manage_devices` | Devices |
| `/ztna/resources` | `ztna:manage_policies` | Resources |
| `/ztna/access` | `ztna:approve_access` | Access |
| `/ztna/sessions` | `ztna:view` | Operations |
| `/ztna/risk` | `ztna:view` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_identity()`
- `verify_identity()`
- `register_device()`
- `update_device_posture()`
- `register_resource()`
- `attach_resource_policy()`
- `request_access()`
- `approve_access_request()`

_(See `service.py` for complete API.)_

## Interoperability

`ztna` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use ztna;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ZTNA_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
