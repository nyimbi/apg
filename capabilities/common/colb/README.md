# Collaboration Capability (colb)

`colb` provides APG's common capability for tenant-scoped collaborative workspaces. It composes workspaces, documents, co-editing sessions, comments, tasks, notifications, presence, AI collaborators, and audit trails into a generated-application packet that runs without live collaboration infrastructure.

**Author:** Nyimbi Odero | **Copyright:** © 2025 Datacraft

---

## What It Provides

- Collaborative workspaces with owners, members, external participants, retention policies, and visibility controls (`private` / `internal` / `public`).
- Document lifecycle: create, share, update, version history, restore, soft-delete, and export (JSON / Markdown / plain text / CSV).
- Real-time co-editing sessions with operational-transform op tracking, session replay, and conflict resolution.
- Threaded comments anchored to text ranges, replies, resolution, and reactions.
- Task assignments linked to documents with priority, due dates, bulk creation, and deadline reminders.
- @mention tracking with per-user notification queues and read-state management.
- Full-text document and workspace search.
- Activity feed, tenant KPI analytics, and compliance reports (ISO 27001 / GDPR).
- Complete audit trail on every state mutation.
- First-class AI-agent composition for registered, scoped, purpose-bound collaboration agents across Codex, Claude Code, OpenCode, and Pi runtimes.
- Bytewax lifecycle batch validation for workspace, session, document, annotation, decision, presence, protocol, guest-access, and agent mutations.
- Protocol adapter metadata for WebSocket, WebRTC, MQTT, gRPC, and Bytewax event streams.

---

## Runtime Shape

**In-memory service** (`CollaborationService`) — 44+ async methods, dependency-light, suitable for generated applications, tests, and local development without external databases or WebSocket servers.

**DB-backed mode** — pass a SQLAlchemy async session as the first constructor argument; the service routes `get_chat_messages` and related queries through it automatically.

**Generated runtime** (`collaboration_runtime.CollaborationRuntime`) — thin deterministic wrapper for exercising the full collaboration lifecycle in a single object.

---

## Quick Start

```python
import asyncio
from capabilities.common.colb.service import CollaborationService

async def main():
    svc = CollaborationService(actor_id_or_db="alice", tenant_id="acme")

    # workspace
    ws = await svc.workspace_create("Product Docs", owner_id="alice", visibility="internal")
    await svc.workspace_invite(ws["workspace_id"], user_id="bob", role="editor")

    # document
    doc = await svc.document_create(ws["workspace_id"], "Roadmap Q3", content="## Goals\n...", created_by="alice")

    # co-edit
    session = await svc.co_edit_session(doc["doc_id"], initiator_id="alice", participants=["bob"])
    await svc.co_edit_apply_op(session["session_id"], user_id="bob", op_type="insert", payload={"pos": 10, "text": "updated "})
    await svc.co_edit_close_session(session["session_id"], closed_by="alice")

    # comment and task
    comment = await svc.comment_add(doc["doc_id"], author_id="bob", body="Should we add OKRs?")
    await svc.task_assign(doc["doc_id"], title="Add OKRs section", assigned_to="alice", created_by="bob", priority="high")

    # analytics
    report = await svc.collaboration_analytics()
    print(report)

asyncio.run(main())
```

Using the generated runtime:

```python
from capabilities.common.colb.collaboration_runtime import CollaborationRuntime

runtime = CollaborationRuntime()
workspace = runtime.create_workspace("tenant-1", "workspace-1", "Finance Close", "owner", ["owner", "analyst"], "retain-180-days")
session   = runtime.start_session("tenant-1", "session-1", workspace["id"], "owner")
agent     = runtime.register_collaboration_agent(
    "tenant-1", "agent-1", "Workspace Steward", "codex",
    "collaboration_steward", "workspace:workspace-1", "owner",
    "review collaboration lifecycle", human_approval_required=True,
)
batch = runtime.validate_colb_lifecycle_batch("tenant-1", "bytewax", 2, "collaboration_agent_batch")
```

---

## API Reference

| # | Method | Description |
|---|--------|-------------|
| 1 | `workspace_create(name, owner_id, description, visibility)` | Create a tenant-scoped workspace |
| 2 | `workspace_invite(workspace_id, user_id, role, invited_by)` | Invite a user; roles: `viewer / commenter / editor / admin` |
| 3 | `workspace_remove_member(workspace_id, user_id)` | Remove a member |
| 4 | `list_workspace_members(workspace_id)` | List all workspace members |
| 5 | `document_create(workspace_id, title, content, doc_type, created_by)` | Create a versioned document |
| 6 | `document_share(doc_id, user_ids, permission)` | Share with specific users; `view / comment / edit` |
| 7 | `document_update(doc_id, content, updated_by, title)` | Update content and auto-increment version |
| 8 | `co_edit_session(doc_id, initiator_id, participants)` | Open a co-editing session |
| 9 | `co_edit_apply_op(session_id, user_id, op_type, payload)` | Apply an OT operation |
| 10 | `co_edit_close_session(session_id, closed_by)` | Close session and persist final state |
| 11 | `comment_add(doc_id, author_id, body, anchor)` | Add a comment, optionally anchored to a text range |
| 12 | `comment_reply(doc_id, comment_id, author_id, body)` | Reply to a comment thread |
| 13 | `comment_resolve(doc_id, comment_id, resolved_by)` | Resolve a comment |
| 14 | `mention_notify(mentioned_user_id, source_doc_id, mentioned_by, context)` | Record @mention and notify |
| 15 | `mention_resolve(user_id, mention_id)` | Mark a mention as read |
| 16 | `task_assign(doc_id, title, assigned_to, created_by, due_date, priority)` | Assign a task to a document |
| 17 | `task_update(task_id, **kwargs)` | Update `status / priority / due_date / title` |
| 18 | `deadline_reminder(lookahead_hours)` | Scan open tasks and send upcoming-deadline notifications |
| 19 | `version_history(doc_id)` | Return full version list for a document |
| 20 | `version_restore(doc_id, version, restored_by)` | Restore document to a prior version |
| 21 | `conflict_resolve(doc_id, winning_content, resolved_by, strategy)` | Resolve co-edit conflict; `manual / last_write_wins / merge` |
| 22 | `export_document(doc_id, fmt)` | Export as `json`, `markdown`, or `txt` |
| 23 | `activity_feed(workspace_id, limit)` | Tenant activity feed, newest-first |
| 24 | `collaboration_analytics()` | Aggregate KPIs: workspaces, documents, sessions, tasks, comments |
| 25 | `list_workspaces(status)` | List workspaces, optionally by status |
| 26 | `list_documents(workspace_id)` | List active documents |
| 27 | `list_tasks(assigned_to, status)` | List tasks with optional filters |
| 28 | `list_comments(doc_id, resolved)` | List document comments |
| 29 | `delete_document(doc_id, deleted_by)` | Soft-delete a document |
| 30 | `delete_workspace(workspace_id, deleted_by)` | Soft-delete workspace and all documents |
| 31 | `bulk_create_documents(workspace_id, docs, created_by)` | Batch document creation |
| 32 | `bulk_assign_tasks(doc_id, tasks, created_by)` | Batch task assignment |
| 33 | `export_workspace_csv(workspace_id)` | Export document metadata as CSV |
| 34 | `export_tasks_json(assigned_to)` | Export tasks as JSON |
| 35 | `health_check()` | Service health and storage summary |
| 36 | `dashboard()` | Alias for `collaboration_analytics()` |
| 37 | `compliance_report(framework)` | Data governance report (default: `ISO_27001`) |
| 38 | `audit_trail(event_type)` | Full audit log, filterable by event type |
| 39 | `get_notifications(user_id, unread_only)` | Retrieve user notifications |
| 40 | `mark_notifications_read(user_id, notification_ids)` | Mark notifications read |
| 41 | `workspace_search(query)` | Search workspaces by name/description |
| 42 | `document_search(query, workspace_id)` | Full-text search across title and content |
| 43 | `co_edit_session_ops(session_id)` | Return all ops applied in a session |
| 44 | `user_activity_summary(user_id)` | Per-user activity roll-up |
| — | `get_chat_messages(session_id, tenant_id, page, limit)` | DB-backed: merged chat messages from `RTCPageCollaboration` and `RTCMessage` |

---

## World-Class Enhancements (v2.0)

Fifteen improvements elevate COLB from in-memory prototype to production-grade collaboration infrastructure. Ordered by implementation effort (ascending); items 1–5 address correctness, 6–10 usability, 11–15 production readiness.

| # | Enhancement | What It Adds |
|---|-------------|-------------|
| 1 | **OT Conflict-Free Merge** | Lightweight `insert / delete / retain` engine with position rebasing makes concurrent ops produce a single canonical document state; `conflict_resolve` becomes the fallback, not the default |
| 2 | **CRDT Shared Data Structures** | `SharedStructure` abstraction (grow-only counter + LWW-map CRDT) for shared lists, kanbans, and whiteboards — commutative, idempotent, no external dependency |
| 3 | **Pluggable Persistence Backend** | `StorageBackend` protocol (`get / set / delete / scan`) with `InMemoryBackend` and `PostgresBackend` (SQLAlchemy async); service layer unchanged between environments |
| 4 | **Workspace Roles & Permission Matrix** | `PermissionMatrix` maps `(role, action)` to allow/deny; `_assert_permission` helper enforces checks declaratively and auditably |
| 5 | **Document Templates & Schemas** | `template_create` / `document_from_template`; templates carry a JSON Schema, documents validate on every update — enables typed downstream integrations |
| 6 | **Presence & Cursor Broadcasting** | `PresenceBus` stores `{session_id → {user_id → {cursor, selection, last_seen}}}`; `presence_heartbeat`, `presence_cursor_update`, `presence_snapshot` connect to `websocket_manager.py` |
| 7 | **Reactions & Emoji Annotations** | `comment_react` / `comment_unreact` add `reactions: dict[str, list[str]]` to comment threads — high-signal engagement metric at minimal cost |
| 8 | **Smart Activity Digest** | `DigestQueue` accumulates events per user over a configurable window (default 5 min); `flush_digest` collapses N notifications into one summary, with real-time escalation for `urgent` items |
| 9 | **Document Access-Link Sharing** | `document_share_link(doc_id, permission, expires_in_hours)` mints UUID7 tokens; `document_resolve_token(token)` validates expiry — covers external collaborators without workspace membership |
| 10 | **Full-Text Search with Ranking** | TF-IDF scorer over title + content + comment bodies; results carry `relevance_score` sorted descending; same API contract over PostgreSQL `ts_vector` in production |
| 11 | **Co-Edit Session Replay** | `co_edit_replay(doc_id, from_version, to_version)` applies ops sequentially and returns intermediate document states — enables audits and time-travel debugging |
| 12 | **Async Webhooks** | `register_webhook(event_pattern, url, secret)` + `_dispatch_webhooks(event_type, payload)` deliver signed HTTP POSTs on audit events — Slack, Jira, custom dashboards, no polling |
| 13 | **Workspace Analytics Time-Series** | `collaboration_analytics(from_date, to_date)` returns a `time_series` array bucketed by day: `documents_created`, `comments_added`, `tasks_completed`, `co_edit_ops` |
| 14 | **AI Summarisation Adapter** | `document_summarise(doc_id, model)` and `thread_summarise(doc_id, comment_id)` via pluggable `SummarisationAdapter`; default stub, production via Ollama — consistent with APG open-weight strategy |
| 15 | **Compliance Data-Retention Enforcement** | `retention_policy_create(workspace_id, retain_days)` + `retention_policy_enforce()` soft-deletes expired documents with `retention_expired` audit events; `compliance_report` gains `retention_violations`; satisfies ISO 27001 A.8.3 and GDPR Article 5(1)(e) |

---

## New Methods — Usage Examples

### 1. Document Share Link (time-limited external access)

```python
# Mint a 48-hour read-only share token
token_record = await svc.document_share_link(doc["doc_id"], permission="view", expires_in_hours=48)
token = token_record["token"]

# External collaborator resolves the token
resolved = await svc.document_resolve_token(token)
# resolved = {"doc": {...}, "permission": "view", "expires_at": "..."}
```

### 2. Presence & Cursor Broadcasting

```python
# User joins a co-edit session; broadcast cursor position
await svc.presence_heartbeat(session_id=session["session_id"], user_id="bob")
await svc.presence_cursor_update(
    session_id=session["session_id"],
    user_id="bob",
    cursor={"line": 12, "col": 4},
    selection={"start": 200, "end": 215},
)

# Snapshot current presence for UI rendering
state = await svc.presence_snapshot(session_id=session["session_id"])
# state = {"bob": {"cursor": {...}, "selection": {...}, "last_seen": "..."}, ...}
```

### 3. Comment Reactions

```python
comment = await svc.comment_add(doc["doc_id"], author_id="alice", body="Ship this by Friday?")

# Colleagues react
await svc.comment_react(doc["doc_id"], comment["comment_id"], user_id="bob",   emoji="👍")
await svc.comment_react(doc["doc_id"], comment["comment_id"], user_id="carol", emoji="👍")
await svc.comment_react(doc["doc_id"], comment["comment_id"], user_id="dave",  emoji="🚀")

# comment["reactions"] == {"👍": ["bob", "carol"], "🚀": ["dave"]}
await svc.comment_unreact(doc["doc_id"], comment["comment_id"], user_id="bob", emoji="👍")
```

### 4. Co-Edit Session Replay (time-travel audit)

```python
# After a session with many ops, replay between version checkpoints
states = await svc.co_edit_replay(doc["doc_id"], from_version=3, to_version=7)
for intermediate in states:
    print(intermediate["version"], intermediate["content"][:80])
```

### 5. AI Summarisation

```python
# Summarise a long document using the local Ollama adapter
summary = await svc.document_summarise(doc["doc_id"], model="local/mistral")
print(summary["summary"])

# Summarise a comment thread to surface the decision
thread_summary = await svc.thread_summarise(doc["doc_id"], comment["comment_id"])
print(thread_summary["summary"])
```

---

## Configuration and Rules

`capability_contract.py` is the source of truth for:

- configuration defaults and schema
- deterministic capability rules
- UI route contracts and theme tokens
- APG adapter map

The event stream adapter is Bytewax. Batch collaboration mutations must use Bytewax; broker-specific queuing is intentionally outside this packet.

---

## Agent and Lifecycle Composition

COLB treats AI collaborators as first-class application citizens. The `agents` manifest defines:

- supported runtimes: `codex`, `claude_code`, `opencode`, `pi`
- supported roles: workspace, session, artifact, annotation, decision, presence, protocol, guest-access, lifecycle, and collaboration-steward review
- privileged roles requiring human approval evidence before activation
- provider-neutral adapter contract: `aicr_provider_neutral_collaboration_agent_adapter`

The `streaming` manifest defines the `colb.lifecycle` stream, `event_time` watermark, required `bytewax` processor, supported lifecycle operations, and topics.

---

## UI Surfaces

Route contracts exposed by `views.py` and `view_models.py`:

`dashboard` | `workspaces` | `sessions` | `presence` | `artifacts` | `annotations` | `decisions` | `agents` | `lifecycle` | `protocols` | `analytics` | `audit` | `settings`

---

## Verification

```bash
./.venv/bin/python -m py_compile \
    capabilities/common/colb/__init__.py \
    capabilities/common/colb/capability_contract.py \
    capabilities/common/colb/collaboration_runtime.py \
    capabilities/common/colb/service.py \
    capabilities/common/colb/views.py \
    capabilities/common/colb/app.py

./.venv/bin/pytest -q \
    capabilities/common/colb/test_capability_contract.py \
    capabilities/common/colb/tests/test_package_contract.py

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/colb --json
./.venv/bin/apg capabilities publish-plan capabilities/common/colb --json
```
