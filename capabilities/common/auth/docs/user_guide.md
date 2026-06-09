# Authentication & RBAC

**Capability ID**: `auth` | **Domain**: `common` | **Version**: `1.0.0`

## Description

AUTH is the APG identity, session, role, access-decision, privacy-budget, and security-agent governance capability. It gives generated applications a dependency-light control plane for registering tenant identities, defining

## Installation

```bash
pip install apg-common-auth
```

## Provides

- `identity_registry`
- `role_governance`
- `session_control`
- `access_decisions`
- `privacy_budget_governance`

## Requires

- `audl`
- `mten`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/auth/access/login` | `public` | Access |
| `/auth/access/dashboard` | `auth:view` | Overview |
| `/auth/roles/workbench` | `auth:manage_roles` | Authorization |
| `/auth/roles/approvals` | `auth:approve_roles` | Authorization |
| `/auth/sessions` | `auth:manage_sessions` | Access |
| `/auth/access/decisions` | `auth:view` | Authorization |
| `/auth/biometric/enroll` | `auth:manage_biometrics` | Assurance |
| `/auth/biometric/manage` | `auth:manage_biometrics` | Assurance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_identity()`
- `list_identities()`
- `define_role()`
- `list_roles()`
- `request_role_assignment_approval()`
- `decide_role_assignment_approval()`
- `list_role_assignment_approvals()`
- `assign_role()`

_(See `service.py` for complete API.)_

## Interoperability

`auth` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use auth;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `AUTH_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
