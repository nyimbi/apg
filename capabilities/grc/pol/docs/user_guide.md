# Policy Management

**Capability ID**: `grc_pol` | **Domain**: `grc` | **Version**: `1.0.0`

## Description

Policy Management provides a world-class, standalone-deployable implementation of policy management capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Installation

```bash
pip install apg-grc-pol
```

## Provides

- `policy_lifecycle_management`
- `policy_acknowledgement_workflow`
- `policy_exception_workflow`
- `policy_review_workflow`
- `policy_publication_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/grc-pol/dashboard` | `grc_pol:view` | Overview |
| `/grc-pol/policies` | `grc_pol:manage_policies` | Policies |
| `/grc-pol/policies/:id` | `grc_pol:view` | Policies |
| `/grc-pol/acknowledgements` | `grc_pol:manage_acknowledgements` | Compliance |
| `/grc-pol/exceptions` | `grc_pol:manage_exceptions` | Governance |
| `/grc-pol/reviews` | `grc_pol:review` | Governance |
| `/grc-pol/review-calendar` | `grc_pol:view` | Planning |
| `/grc-pol/gap-analysis` | `grc_pol:view` | Analysis |

## Key Service Methods

- `_audit_event()`
- `_get_policy()`
- `create_policy()`
- `draft_policy_content()`
- `policy_review()`
- `approve_policy()`
- `publish_policy()`
- `acknowledge_policy()`
- `policy_exception_request()`
- `approve_exception()`

_(See `service.py` for complete API.)_

## Interoperability

`grc_pol` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use grc_pol;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GRC_POL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
