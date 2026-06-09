# Message Queue Event Bus

**Capability ID**: `mqeb` | **Domain**: `common` | **Version**: `1.0.0`

## Description

MQEB is APG's package-backed event fabric. It provides tenant-scoped topic management, governed message publishing, subscription lifecycle state, delivery/dead-letter evidence, replay review, priority quota review, rule

## Installation

```bash
pip install apg-common-mqeb
```

## Provides

- `mqeb_event_fabric`
- `message_governance`
- `event_agent_composition`
- `review_evidence`

## Requires

- `conf`
- `auth`
- `audl`
- `secu`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mqeb/dashboard` | `mqeb:view` | Overview |
| `/mqeb/topics` | `mqeb:manage_topics` | Operations |
| `/mqeb/publish` | `mqeb:publish` | Operations |
| `/mqeb/subscriptions` | `mqeb:subscribe` | Operations |
| `/mqeb/delivery` | `mqeb:view_metrics` | Reliability |
| `/mqeb/dead-letters` | `mqeb:manage_routing` | Reliability |
| `/mqeb/quota-exceptions` | `mqeb:admin` | Governance |
| `/mqeb/replays` | `mqeb:admin` | Governance |

## Key Service Methods

- `to_dict()`
- `to_dict()`
- `to_dict()`
- `to_dict()`
- `to_dict()`
- `to_dict()`
- `to_dict()`
- `to_dict()`
- `to_dict()`
- `describe()`

_(See `service.py` for complete API.)_

## Interoperability

`mqeb` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mqeb;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MQEB_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
