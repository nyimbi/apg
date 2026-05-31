# Chat and Messaging Capability

`chat` provides the APG common capability for tenant-scoped team messaging. It is a dependency-light generated-application packet that can be composed into larger APG applications while keeping live WebSocket servers, durable brokers, identity providers, and notification providers behind adapter boundaries.

## What It Provides

- Tenant-local rooms, direct-message surfaces, room membership, owners, external guests, and large-room review.
- Message delivery with message fingerprints, thread keys, attachments, delivery receipts, moderation status, and audit records.
- Presence state for online status and typing indicators.
- Moderation queues for restricted content and room-access review.
- First-class AI-agent composition for registered, scoped, owned, purpose-bound, disclosed chat agents across runtimes such as Codex, Claude Code, OpenCode, and Pi.
- Bytewax lifecycle batch validation for room, message, thread, reaction, presence, moderation, retention, guest-access, and chat-agent mutations.
- Deterministic rules for room governance, sender identity, membership, moderation, attachments, DLP, audit, retention exports, and Bytewax batch mutation.
- APG UI route metadata, view models, theme tokens, package manifest, semantic model, and release evidence.

## Runtime Shape

The package runtime is `service.ChatService`. It is intentionally in-memory and deterministic so generated applications can execute a complete chat lifecycle without external infrastructure.

Primary methods:

- `create_room(...)`
- `approve_room(...)`
- `send_message(...)`
- `update_presence(...)`
- `review_moderation(...)`
- `register_chat_agent(...)`
- `validate_chat_lifecycle_batch(...)`
- `list_rooms(...)`
- `list_messages(...)`
- `list_presence(...)`
- `list_moderation_items(...)`
- `list_chat_agents(...)`
- `list_lifecycle_batches(...)`
- `list_audit_events(...)`
- `conversation_summary(...)`

API helpers in `api.py` wrap the same runtime for generated applications.

## Configuration And Rules

`capability_contract.py` is the source of truth for:

- configuration defaults
- configuration schema
- deterministic rule engine
- UI routes
- theme tokens
- APG adapter map

The contract declares Bytewax as the event-stream adapter. Batch chat mutations must use Bytewax; Kafka is intentionally not part of this packet.

## Agent And Lifecycle Composition

CHAT treats AI agents as first-class application citizens. The contract exposes a top-level `agents` manifest with:

- supported runtimes: `codex`, `claude_code`, `opencode`, and `pi`
- supported roles for room, message, moderation, retention, presence, guest access, attachment, thread, lifecycle, and chat-steward review
- privileged roles that require human approval evidence before they become active
- a provider-neutral adapter contract: `aicr_provider_neutral_chat_agent_adapter`

The contract also exposes a top-level `streaming` manifest. `validate_chat_lifecycle_batch(...)` accepts only Bytewax-backed lifecycle batches with declared operations and non-empty mutation counts. This gives generated applications a clear lifecycle guardrail before durable workers are attached.

## UI Surfaces

The generated application exposes these route contracts:

- dashboard
- rooms
- direct
- messages
- presence
- agents
- lifecycle
- moderation
- retention
- audit
- analytics
- settings

`views.py` provides dependency-light view models for these surfaces.

## How To Use

```python
from capabilities.common.chat.service import ChatService

service = ChatService()
room = service.create_room(
    "ops-room",
    "tenant-1",
    "Operations",
    "owner",
    ["owner", "operator"],
    "retain-90-days",
)
message = service.send_message(
    "msg-1",
    "tenant-1",
    room["id"],
    "operator",
    "handover complete",
)
agent = service.register_chat_agent(
    "agent-1",
    "tenant-1",
    "Ops Chat Steward",
    "codex",
    "chat_steward",
    "room:ops-room",
    "owner",
    "review operational chat lifecycle",
    human_approval_required=True,
)
batch = service.validate_chat_lifecycle_batch(
    "tenant-1",
    "bytewax",
    2,
    "chat_agent_batch",
)
```

Use `register_capability()` to expose the full APG registration payload to the composition engine.

## Integration Boundaries

This packet does not open sockets, call brokers, scan files, invoke identity providers, or call AI-agent CLIs directly. Those integrations belong behind APG adapters:

- `mqeb` for message/event bus integration
- `ntfy` for notifications
- `auth` for identity enforcement
- `mten` for multi-tenant context
- `audl` for audit sinks
- `nlpc` for language/content classification
- `colb` for collaboration workflows
- `secu` for security inspection
- `cach` for cache and presence acceleration

## Verification

Focused verification for this packet should use:

```bash
./.venv/bin/python -m py_compile capabilities/common/chat/__init__.py capabilities/common/chat/capability_contract.py capabilities/common/chat/chat_engine.py capabilities/common/chat/models.py capabilities/common/chat/service.py capabilities/common/chat/api.py capabilities/common/chat/views.py capabilities/common/chat/app.py capabilities/common/chat/test_capability_contract.py capabilities/common/chat/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/chat/test_capability_contract.py capabilities/common/chat/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/chat --json
./.venv/bin/apg capabilities publish-plan capabilities/common/chat --json
```
