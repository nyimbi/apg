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
- **Semantic RAG search** via locally-hosted Ollama embedding models (`nomic-embed-text`), falling back to lexical.
- **Conversation summarisation** via Ollama LLM with structured decisions and action-item extraction.
- **LLM intent classification** routing messages to registered bots and agents automatically.
- **Retention policy enforcement** with compliance reporting and dry-run support.
- **Token cost accounting** with `Decimal`-safe per-tenant, per-agent billing rates.
- **Adaptive per-user rate limiting** using a token-bucket algorithm with dashboard-visible state.
- **Federated guest access grants** with cryptographic tokens, time-boxed expiry, and revocation.
- **Workspace-wide cross-room search** with faceting over sender, date range, room, attachment presence, and moderation status.

## Runtime Shape

The package runtime is `service.ChatService`. It is intentionally in-memory and deterministic so generated applications can execute a complete chat lifecycle without external infrastructure.

### Synchronous Methods

- `create_room(...)` / `approve_room(...)` / `join_room(...)` / `leave_room(...)`
- `room_members(...)` / `room_permissions(...)` / `list_rooms(...)`
- `send_message(...)` / `edit_message(...)` / `delete_message(...)`
- `react_to_message(...)` / `thread_reply(...)` / `pin_message(...)`
- `direct_message(...)` / `broadcast_message(...)` / `file_share(...)`
- `search_messages(...)` / `message_search(...)` / `typing_indicator(...)`
- `read_receipts(...)` / `list_messages(...)`
- `update_presence(...)` / `list_presence(...)`
- `message_moderation(...)` / `review_moderation(...)` / `list_moderation_items(...)`
- `webhook_integration(...)` / `bot_registration(...)` / `mention_notification(...)`
- `room_analytics(...)` / `chat_analytics(...)` / `export_chat_history(...)`
- `register_chat_agent(...)` / `validate_chat_lifecycle_batch(...)`
- `list_chat_agents(...)` / `list_lifecycle_batches(...)` / `list_audit_events(...)`
- `conversation_summary(...)` / `health_check(...)`

### Async Methods (AI, Compliance, Billing)

| Method | Purpose |
|--------|---------|
| `semantic_search_messages(tenant_id, query, room_id, limit)` | RAG vector search via Ollama embeddings, lexical fallback |
| `summarise_conversation(tenant_id, room_id, last_n)` | LLM-generated summary with decisions and action items |
| `classify_message_intent(tenant_id, message_id)` | Intent classification for agent/bot auto-dispatch |
| `enforce_retention_policy(tenant_id, room_id, dry_run)` | Soft-delete messages exceeding retention window |
| `retention_compliance_report(tenant_id)` | Per-room compliance status across all retention policies |
| `token_usage_report(tenant_id, date_prefix)` | Decimal-accurate LLM token usage and cost per agent |
| `set_token_rate(tenant_id, agent_id, rate_per_1k_tokens, actor)` | Configure billing rate for an agent |
| `rate_limit_status(tenant_id, user_id)` | Token-bucket state for a user (capacity, remaining, refill rate) |
| `check_rate_limit(tenant_id, user_id, cost)` | Deduct from bucket; returns False when exhausted |
| `grant_guest_access(tenant_id, room_id, guest_email, granted_by, expiry_hours, permissions)` | Issue a time-boxed cryptographic guest token |
| `verify_guest_token(token)` | Validate a guest access token; raises on expiry or revocation |
| `revoke_guest_access(tenant_id, token, revoked_by)` | Invalidate a guest access token immediately |
| `workspace_search(tenant_id, query, filters, limit, page)` | Cross-room paginated search with facets |

API helpers in `api.py` wrap the same runtime for generated applications.

## Configuration And Rules

`capability_contract.py` is the source of truth for:

- configuration defaults
- configuration schema
- deterministic rule engine
- UI routes
- theme tokens
- APG adapter map

The contract declares Bytewax as the event-stream adapter. Batch chat mutations must use Bytewax; broker-specific queue is intentionally not part of this packet.

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

### Basic room and message lifecycle

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

### Semantic search (requires OLLAMA_BASE_URL)

```python
import asyncio, os
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

result = asyncio.run(service.semantic_search_messages(
    "tenant-1",
    "what did we decide about the deployment?",
    room_id="ops-room",
    limit=10,
))
# result["semantic"] == True when Ollama is reachable, False on lexical fallback
for msg in result["results"]:
    print(msg["sender"], msg["body"])
```

### Conversation summarisation

```python
summary = asyncio.run(service.summarise_conversation("tenant-1", "ops-room", last_n=100))
print(summary["summary"])
print("Decisions:", summary["decisions"])
print("Action items:", summary["action_items"])
```

### Retention policy enforcement

```python
# Check what would be purged (dry run)
report = asyncio.run(service.enforce_retention_policy("tenant-1", "ops-room", dry_run=True))
print(f"Would purge {report['expired_message_count']} messages")

# Apply the policy
asyncio.run(service.enforce_retention_policy("tenant-1", "ops-room"))

# Compliance dashboard
compliance = asyncio.run(service.retention_compliance_report("tenant-1"))
print(f"Non-compliant rooms: {compliance['non_compliant_rooms']}")
```

### Token cost accounting

```python
from decimal import Decimal

# Set rate: $0.002 per 1K tokens for the ops agent
asyncio.run(service.set_token_rate("tenant-1", "agent-1", "0.002", actor="admin"))

# Get usage report
report = asyncio.run(service.token_usage_report("tenant-1"))
print(f"Total cost: ${report['total_cost_usd']}")
```

### Federated guest access

```python
grant = asyncio.run(service.grant_guest_access(
    "tenant-1", "ops-room",
    guest_email="partner@example.com",
    granted_by="owner",
    expiry_hours=48,
    permissions=["read", "send"],
))
token = grant["token"]

# Validate later
verified = asyncio.run(service.verify_guest_token(token))

# Revoke early
asyncio.run(service.revoke_guest_access("tenant-1", token, revoked_by="owner"))
```

### Workspace cross-room search with facets

```python
results = asyncio.run(service.workspace_search(
    "tenant-1",
    "deployment",
    filters={"has_attachment": False, "sender": "operator"},
    limit=20,
    page=0,
))
print(results["total"], "hits")
print(results["facets"])  # {"room_id": {...}, "sender": {...}, ...}
```

### Rate limiting

```python
allowed = asyncio.run(service.check_rate_limit("tenant-1", "operator", cost=1))
if not allowed:
    print("Rate limited")

status = asyncio.run(service.rate_limit_status("tenant-1", "operator"))
print(f"{status['tokens_remaining']} / {status['capacity']} tokens remaining")
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

Ollama integration (semantic search, summarisation, intent classification) is optional and degrades gracefully when `OLLAMA_BASE_URL` is unset.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | (unset) | Base URL for local Ollama API, e.g. `http://localhost:11434` |
| `CHAT_SUMMARY_MODEL` | `mistral` | Ollama model used for conversation summarisation |
| `CHAT_INTENT_MODEL` | `phi3` | Ollama model used for intent classification |

## Verification

Focused verification for this packet should use:

```bash
./.venv/bin/python -m py_compile capabilities/common/chat/__init__.py capabilities/common/chat/capability_contract.py capabilities/common/chat/chat_engine.py capabilities/common/chat/models.py capabilities/common/chat/service.py capabilities/common/chat/api.py capabilities/common/chat/views.py capabilities/common/chat/app.py capabilities/common/chat/test_capability_contract.py capabilities/common/chat/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/chat/test_capability_contract.py capabilities/common/chat/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/chat --json
./.venv/bin/apg capabilities publish-plan capabilities/common/chat --json
```
