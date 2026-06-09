# User Management

**Capability ID**: `usrm` | **Domain**: `common` | **Version**: `1.0.0`

## Description

USRM is the APG capability for governed user lifecycle management. It gives generated applications a composable runtime for user identity, profiles, consented invitations, role assignment, privileged MFA, access reviews, privacy

## Installation

```bash
pip install apg-common-usrm
```

## Provides

- `user_directory`
- `profile_management`
- `consented_invitations`
- `role_assignment_governance`
- `access_review_workflows`

## Requires

- `auth`
- `mfau`
- `cons`
- `audl`
- `idfd`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/usrm/dashboard` | `usrm:view` | Overview |
| `/usrm/users` | `usrm:manage_users` | Users |
| `/usrm/profiles` | `usrm:manage_users` | Users |
| `/usrm/lifecycle` | `usrm:manage_users` | Lifecycle |
| `/usrm/access` | `usrm:review_access` | Access |
| `/usrm/privacy` | `usrm:view` | Privacy |
| `/usrm/deprovisioning` | `usrm:deprovision` | Lifecycle |
| `/usrm/agents` | `usrm:admin` | Automation |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_user()`
- `update_profile()`
- `invite_user()`
- `assign_role()`
- `record_access_review()`
- `deprovision_user()`
- `bulk_suspend_users()`
- `register_usrm_agent()`

_(See `service.py` for complete API.)_

## Interoperability

`usrm` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use usrm;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `USRM_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
