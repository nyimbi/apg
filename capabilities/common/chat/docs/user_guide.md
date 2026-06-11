# Chat and Messaging — User Guide

**Capability ID**: `chat` | **Domain**: `common` | **Version**: `1.1.0`
**© 2025 Datacraft** | www.datacraft.co.ke | nyimbi@gmail.com

---

## Overview

The `chat` capability provides tenant-scoped team messaging for APG applications. It is an in-process, dependency-light service that you import directly — no broker, socket server, or cloud API required for basic use. Advanced features (semantic search, summarisation, intent classification) activate automatically when a locally-hosted Ollama instance is reachable.

---

## Quick Start

```python
import asyncio
from capabilities.common.chat.service import ChatService

svc = ChatService()

# 1. Create a room
room = svc.create_room(
    room_id="ops-1",
    tenant_id="acme",
    name="Operations",
    owner="alice",
    members=["alice", "bob"],
    retention_policy="retain-90-days",
)

# 2. Send a message
msg = svc.send_message(
    message_id="m-001",
    tenant_id="acme",
    room_id="ops-1",
    sender="alice",
    body="System check passed.",
)

# 3. Inspect summary
print(svc.conversation_summary("acme"))
```

---

## Core Concepts

### Tenant Isolation

Every call requires a `tenant_id`. The service enforces tenant boundaries at the storage layer — no cross-tenant data leakage is possible in the in-memory implementation.

### Rooms

A room is the unit of collaboration. Rooms have:

- an **owner** (required)
- a **member list** (deduplicated)
- an optional **external_guests** list (triggers review workflow if large)
- a **retention_policy** string (e.g. `retain-90-days`, `retain-1-year`)
- a **visibility** of `private` (default) or `public`
- a **status** of `active` or `pending_review`

Large rooms (member count > contract maximum) enter `pending_review` and require an explicit `approve_room(...)` call.

### Messages

Messages carry:

- `body` (text payload)
- `attachments` (list of storage refs)
- `fingerprint` (deterministic hash of content)
- `thread_key` (for grouping related messages)
- `delivery_receipts`
- `moderation_status`: `clear`, `approved`, `pending`, `flagged`, or `deleted`

Restricted-term detection runs synchronously on `send_message`. Triggered messages enter the moderation queue and are delivered after moderation review.

### Presence

`update_presence(tenant_id, user_id, status, room_id, typing)` records user availability. `typing_indicator(...)` is a convenience wrapper.

### Moderation

`message_moderation(tenant_id, message_id, moderator, action, reason)` accepts `approve`, `flag`, or `remove`. `review_moderation(...)` closes the loop for queued items.

---

## Room Operations

```python
# Join / leave
svc.join_room("ops-1", "charlie", "acme", invited_by="alice")
svc.leave_room("ops-1", "charlie", "acme")

# Membership list
members = svc.room_members("ops-1", "acme")

# Per-room permission configuration
svc.room_permissions("ops-1", "acme", actor="alice", permissions={
    "can_send": ["alice", "bob"],
    "can_moderate": ["alice"],
})

# Analytics
stats = svc.room_analytics("acme", "ops-1")
print(stats["message_count"], stats["top_senders"])
```

---

## Messaging Operations

```python
# Edit (sender only)
svc.edit_message("m-001", "acme", editor="alice", new_body="System check passed (updated).")

# Soft-delete
svc.delete_message("m-001", "acme", actor="alice")

# Emoji reaction (toggle)
svc.react_to_message("m-001", "acme", user_id="bob", emoji="thumbsup")

# Thread reply
svc.thread_reply("m-002", "acme", parent_message_id="m-001", room_id="ops-1", sender="bob", body="Confirmed.")

# Pin
svc.pin_message("ops-1", "acme", message_id="m-001", actor="alice")

# Direct message (auto-creates synthetic DM room)
svc.direct_message("dm-001", "acme", from_user="alice", to_user="bob", body="Can we talk?")

# Broadcast to multiple rooms
svc.broadcast_message("bc-001", "acme", sender="alice", body="Maintenance at 22:00", room_ids=["ops-1", "dev-1"])

# File share
svc.file_share("fs-001", "acme", "ops-1", "alice", "report.pdf", 204800, "application/pdf", "s3://bucket/report.pdf")

# Read receipt
svc.read_receipts("acme", "m-001", user_id="bob")

# Lexical search
results = svc.search_messages("acme", "system check", room_id="ops-1")
```

---

## AI-Powered Features

All AI features require `OLLAMA_BASE_URL` to be set. They degrade gracefully when the environment variable is absent or the service is unreachable.

### Semantic Search

```python
import asyncio, os
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

result = asyncio.run(svc.semantic_search_messages(
    "acme",
    "what was decided about the database migration?",
    room_id="ops-1",
    limit=10,
))
# result["semantic"] == True means vector search was used
# result["scores"] gives cosine similarity per result
```

Model used: `nomic-embed-text` (configurable). Falls back to lexical search on any error.

### Conversation Summarisation

```python
summary = asyncio.run(svc.summarise_conversation("acme", "ops-1", last_n=50))
# {
#   "summary": "The team discussed deployment strategy and agreed on a Friday window.",
#   "decisions": ["Ship on Friday 22:00 UTC"],
#   "action_items": ["Alice to notify stakeholders", "Bob to prepare rollback plan"],
#   "message_count": 50,
#   "ml_enhanced": True
# }
```

Model: `CHAT_SUMMARY_MODEL` env var (default `mistral`). Results are cached by `(tenant_id, room_id, last_message_id)`.

### Intent Classification and Agent Dispatch

```python
# Send a message that might match a registered bot or agent
svc.send_message("m-003", "acme", "ops-1", "alice", "/status db-primary")

intent = asyncio.run(svc.classify_message_intent("acme", "m-003"))
# {
#   "intent": "database_status_check",
#   "confidence": 0.91,
#   "handler_id": "bot-db",
#   "rationale": "Message matches database status command pattern.",
#   "ml_enhanced": True
# }
```

Model: `CHAT_INTENT_MODEL` env var (default `phi3`). High-confidence results (>= 0.85) can be auto-dispatched when `ai_agent_participant=True` on `send_message`.

---

## Retention Policy Enforcement

```python
# Dry run: see what would be purged
report = asyncio.run(svc.enforce_retention_policy("acme", "ops-1", dry_run=True))
print(f"Would purge {report['expired_message_count']} messages beyond {report['retention_days']}-day window")

# Apply the policy
asyncio.run(svc.enforce_retention_policy("acme", "ops-1"))

# Tenant-wide compliance report
compliance = asyncio.run(svc.retention_compliance_report("acme"))
for room in compliance["non_compliant"]:
    print(f"Room {room['room_id']}: {room['overdue_messages']} overdue messages")
```

Supported policy string formats: `retain-N-days`, `retain-N-year`, `retain-N-years`.

---

## Token Cost Accounting

Uses `Decimal` throughout — never `float` for monetary values.

```python
from decimal import Decimal

# Configure per-agent rate (USD per 1K tokens)
asyncio.run(svc.set_token_rate("acme", "agent-1", "0.002", actor="admin"))

# Usage report
report = asyncio.run(svc.token_usage_report("acme"))
print(f"Total: {report['total_tokens']} tokens, ${report['total_cost_usd']} USD")
for row in report["rows"]:
    print(row["agent_id"], row["total_tokens"], row["cost_usd"])

# Filter by date prefix
report_june = asyncio.run(svc.token_usage_report("acme", date_prefix="2026-06"))
```

---

## Rate Limiting

Token-bucket algorithm. Refills at `messages_per_minute` (from capability contract, default 60) tokens per minute per `(tenant_id, user_id)`.

```python
# Check and deduct (returns False when exhausted)
allowed = asyncio.run(svc.check_rate_limit("acme", "alice", cost=1))
if not allowed:
    raise PermissionError("You are sending too quickly. Please wait.")

# Dashboard view
status = asyncio.run(svc.rate_limit_status("acme", "alice"))
# {"tokens_remaining": 58, "capacity": 60, "refill_rate_per_minute": 60}
```

---

## Federated Guest Access

```python
# Grant time-boxed access
grant = asyncio.run(svc.grant_guest_access(
    "acme", "ops-1",
    guest_email="partner@vendor.com",
    granted_by="alice",
    expiry_hours=24,
    permissions=["read"],
))
token = grant["token"]
print(f"Share this token: {token}")

# Validate (call on each guest request)
try:
    verified = asyncio.run(svc.verify_guest_token(token))
    print(f"Grant valid until {verified['expires_at']}")
except PermissionError as e:
    print(f"Access denied: {e}")

# Revoke early
asyncio.run(svc.revoke_guest_access("acme", token, revoked_by="alice"))
```

Tokens are SHA-256 hashes. Expiry and revocation are checked on every `verify_guest_token` call.

---

## Workspace Search with Facets

```python
results = asyncio.run(svc.workspace_search(
    "acme",
    "deployment window",
    filters={
        "sender": "alice",
        "has_attachment": False,
        "after_date": "2026-06-01",
    },
    limit=25,
    page=0,
))
print(results["total"], "matching messages")
print(results["facets"]["room_id"])   # {"ops-1": 12, "dev-1": 3}
print(results["facets"]["sender"])    # {"alice": 8, "bob": 7}
```

Available filter keys: `room_id`, `sender`, `after_date`, `before_date`, `has_attachment`, `moderation_status`, `thread_only`, `semantic`.

---

## AI Agents

```python
# Register a governed agent
agent = svc.register_chat_agent(
    agent_id="agent-ops",
    tenant_id="acme",
    name="Ops Steward",
    runtime="claude_code",
    role="chat_steward",
    scope="room:ops-1",
    owner="alice",
    purpose="Monitor operations chat for compliance gaps",
    human_approval_required=True,
)

# List agents
for a in svc.list_chat_agents("acme"):
    print(a["id"], a["status"])
```

Privileged roles (e.g. `chat_steward`) enter `pending_review` until a human approver calls `approve_room(...)` for the associated room.

---

## Webhooks and Bots

```python
# Register outgoing webhook
webhook = svc.webhook_integration(
    tenant_id="acme",
    room_id="ops-1",
    webhook_url="https://hooks.example.com/ops",
    events=["message_sent", "member_joined"],
    owner="alice",
)

# Register a bot
bot = svc.bot_registration(
    tenant_id="acme",
    bot_id="bot-status",
    name="StatusBot",
    owner="alice",
    allowed_rooms=["ops-1"],
    commands=["/status", "/ping"],
)
```

---

## Analytics and Exports

```python
# Room-level analytics
room_stats = svc.room_analytics("acme", "ops-1")
# top_senders, message_count, attachment_count, moderated_message_count

# Tenant-wide analytics
tenant_stats = svc.chat_analytics("acme")
# total_rooms, active_rooms, total_messages, busiest_room, webhook_count, bot_count

# Export history
export = svc.export_chat_history("acme", "ops-1", format="json")
export_csv = svc.export_chat_history("acme", "ops-1", format="csv")
```

---

## Lifecycle Batch Validation (Bytewax)

```python
batch = svc.validate_chat_lifecycle_batch(
    tenant_id="acme",
    event_stream="bytewax",
    mutation_count=15,
    operation="chat_agent_batch",
)
print(batch["status"])  # "accepted" or "denied"
```

Only Bytewax-backed streams with recognised operations and non-zero mutation counts are accepted.

---

## Audit Trail

Every state-changing operation emits a `ChatAuditEvent`. Retrieve them via:

```python
events = svc.list_audit_events("acme")
for e in events:
    print(e["event_type"], e["actor"], e["decision"])
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | (unset) | Local Ollama API, e.g. `http://localhost:11434` |
| `CHAT_SUMMARY_MODEL` | `mistral` | LLM for conversation summarisation |
| `CHAT_INTENT_MODEL` | `phi3` | LLM for intent classification |

---

## Integration Boundaries

| Adapter | Purpose |
|---------|---------|
| `mqeb` | Message/event bus |
| `ntfy` | Notification delivery |
| `auth` | Identity enforcement |
| `mten` | Multi-tenant context |
| `audl` | External audit sinks |
| `nlpc` | Language/content classification |
| `colb` | Collaboration workflows |
| `secu` | Security inspection |
| `cach` | Cache and presence acceleration |

---

## Further Reading

- `service.py` — Full business logic implementation
- `models.py` — Frozen dataclass domain models
- `api.py` — REST API endpoint wrappers
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Rules engine, configuration, and adapter map
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 improvement opportunities
- `SPECIFICATION.md` — Formal capability specification
- `README.md` — Quick reference
