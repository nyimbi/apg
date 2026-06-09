# Chat and Messaging

**Capability ID**: `chat` | **Domain**: `common` | **Version**: `1.0.0`

## Description

`chat` provides the APG common capability for tenant-scoped team messaging. It is a dependency-light generated-application packet that can be composed into larger APG applications while keeping live WebSocket servers, durable brokers, identity providers, and notification providers behind adapter boundaries.

## Installation

```bash
pip install apg-common-chat
```

## Provides

_(see capability contract)_

## Requires

_(none)_

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/chat/dashboard` | `chat:view` | Overview |
| `/chat/rooms` | `chat:manage_rooms` | Rooms |
| `/chat/direct` | `chat:send` | Messaging |
| `/chat/messages` | `chat:send` | Messaging |
| `/chat/presence` | `chat:view` | Messaging |
| `/chat/agents` | `chat:manage_rooms` | Messaging |
| `/chat/lifecycle` | `chat:admin` | Operations |
| `/chat/moderation` | `chat:moderate` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_room()`
- `approve_room()`
- `join_room()`
- `leave_room()`
- `list_rooms()`
- `room_members()`
- `room_permissions()`
- `send_message()`

_(See `service.py` for complete API.)_

## Interoperability

`chat` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use chat;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `CHAT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
