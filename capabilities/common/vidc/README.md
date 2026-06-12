# Video Conferencing Capability

`vidc` provides APG's common capability for tenant-scoped video meetings. It composes meeting rooms, accountable hosts, waiting-room controls, participants, encrypted recordings, caption artifacts, AI meeting agents, first-class provider-neutral video agents, audit events, UI routes, visual theming, and Bytewax lifecycle guardrails into a generated-application packet that runs without live media infrastructure.

## What It Provides

- Meeting rooms with owners, moderation policies, external guest policy references, and waiting-room controls.
- Video meeting lifecycle from room selection through active meeting, review-required state, recording/caption artifacts, AI meeting assistants, and meeting closure.
- Guardrails for tenant context, host accountability, secure media transport, screen-share policy, external guests, recording consent, encryption, retention, access audit, large-meeting review, caption language support, computer-vision assist policy, and cross-tenant denial.
- AI meeting agents for captioning, summaries, moderation, and action tracking across runtimes (Codex, Claude Code, OpenCode, Pi).
- First-class video agents with runtime, role, scope, owner, purpose, contribution-disclosure, and privileged-role approval guardrails.
- Bytewax lifecycle stream metadata for room, meeting, participant, recording, caption, meeting-agent, video-agent, and audit batches.
- Breakout rooms, in-meeting polls, whiteboard sessions, chat, raised-hand signals, and participant spotlight.
- Async API surface (40+ methods) alongside the synchronous core.
- Dependency-light API helpers, UI view models, package manifest, semantic model, and release evidence.

## Runtime Shape

The generated runtime is `service.VidcService`. It is deterministic and in-memory so generated applications can exercise the video lifecycle without WebRTC servers, media SFUs, object stores, transcription engines, or databases.

### Core methods

| Method | Description |
|---|---|
| `create_room(...)` | Create a named meeting room with guest/moderation policies |
| `start_meeting(...)` | Start a meeting; returns `active` or `review_required` |
| `add_participant(...)` | Add a participant (internal or external guest) |
| `create_recording(...)` | Create a recording with consent + retention policy |
| `generate_captions(...)` | Generate captions/transcript artifact |
| `register_meeting_agent(...)` | Register an AI agent into a meeting |
| `register_video_agent(...)` | Register a first-class video agent with approval guardrails |
| `validate_vidc_lifecycle_batch(...)` | Validate a Bytewax lifecycle stream batch |
| `end_meeting(...)` | End a meeting and emit audit event |
| `dashboard_summary(...)` | Tenant-scoped dashboard KPI snapshot |

### Async methods (v2.0)

| Method | Description |
|---|---|
| `create_meeting(...)` | Async wrapper for `start_meeting` |
| `join_meeting(...)` | Async `add_participant` |
| `leave_meeting(...)` | Soft-remove participant without ending the meeting |
| `end_meeting_async(...)` | Async alias for `end_meeting` |
| `screen_share(...)` | Start screen-share with policy validation |
| `record_session(...)` | Start encrypted recording session |
| `transcribe_session(...)` | Generate captions; delegates to Ollama ASR in production |
| `recording_transcript(...)` | Fetch/generate transcript from a recording |
| `recording_export(...)` | Export recording to mp4/other format |
| `breakout_room_create(...)` | Create isolated breakout room from parent meeting |
| `poll_create(...)` | Create an in-meeting poll |
| `whiteboard_session(...)` | Start collaborative whiteboard |
| `chat_in_meeting(...)` | Post a chat message |
| `raise_hand(...)` | Signal raised-hand event |
| `spotlight_participant(...)` | Spotlight a participant's video feed |
| `meeting_analytics(...)` | Per-meeting or tenant-wide analytics |
| `meeting_kpi_summary(...)` | Period-scoped KPI card for dashboards |

## Configuration And Rules

`capability_contract.py` is the source of truth for:

- configuration defaults and schema
- deterministic rules (`allow` / `require_review` / `deny`)
- UI route contracts
- theme tokens
- APG adapter map

## UI Surfaces

Route contracts for: dashboard, meetings, rooms, participants, recordings, captions, agents, analytics, audit, settings.

`views.py` provides dependency-light models for these screens.

## Quick Start

```python
from capabilities.common.vidc.service import VidcService

service = VidcService()
room = service.create_room(
    "tenant-1",
    "close-room",
    "meeting-owner",
    "guest-policy://finance",
    "moderation://finance",
)
meeting = service.start_meeting(
    "tenant-1",
    room["id"],
    "Finance Close Review",
    "host-1",
    participant_count=12,
)
service.register_meeting_agent(
    "tenant-1",
    meeting["id"],
    "codex://meeting-summarizer",
    "codex",
    "summarizer",
    "scope://meeting",
    "disclosure://meeting",
    "host-1",
)
service.register_video_agent(
    "tenant-1",
    "video-agent-1",
    "Meeting Steward",
    "codex",
    "video_meeting_steward",
    meeting["id"],
    "host-1",
    "Govern recording, caption, moderation, and action-item lifecycle evidence.",
    human_approval_required=True,
)
service.validate_vidc_lifecycle_batch("tenant-1", "bytewax", 2, "video_agent_batch")
```

Use `register_capability()` to expose the full APG registration payload to the composition engine.

## New Methods

### Breakout rooms

```python
result = await service.breakout_room_create(
    "tenant-1",
    parent_meeting_id=meeting["id"],
    room_name="Group A",
    owner="host-1",
    participant_refs=["alice", "bob", "carol"],
)
# result["breakout_meeting"]["id"]  ← child meeting ID
```

### Encrypted session recording

```python
rec = await service.record_session(
    "tenant-1",
    meeting_id=meeting["id"],
    consent_ref="consent://finance/q1",
    retention_policy_ref="retention://finance/7yr",
    encrypted=True,
    created_by="host-1",
)
```

### Transcript from recording

```python
transcript = await service.recording_transcript(
    "tenant-1",
    recording_id=rec["id"],
    requested_by="compliance-officer",
)
# transcript["text"] is populated by Ollama Whisper ASR in production
```

### In-meeting poll

```python
poll = await service.poll_create(
    "tenant-1",
    meeting_id=meeting["id"],
    question="Should we extend the deadline?",
    options=["Yes", "No", "Need more info"],
    created_by="host-1",
)
```

### Tenant analytics

```python
stats = await service.meeting_analytics("tenant-1")
# {"meeting_count": ..., "total_participants": ..., "recorded_meetings": ..., ...}

kpi = await service.meeting_kpi_summary("tenant-1", period="2025-Q2")
# {"total_meetings": ..., "recording_rate_pct": ..., "avg_participants": ..., ...}
```

## World-Class Enhancements (v2.0)

1. **Simulcast Negotiation** — `negotiate_simulcast_layers` accepts per-participant bandwidth estimates and returns WebRTC SDP amendments for adaptive low/mid/high layer selection, eliminating blunt quality drops under constrained networks.

2. **E2EE Key Ratchet** — Per-meeting Double Ratchet / MLS-lite key schedule. Recording, screen-share, and chat payloads are E2EE even from the SFU. `encrypted` boolean replaced with a structured `EncryptionManifest` tracking key epoch, KDF algorithm, and rotation events.

3. **Local ASR via Ollama** — `transcribe_session` / `recording_transcript` wire to a locally-hosted Whisper model via the Ollama HTTP API. Returns streaming token callbacks for live captions. No cloud ASR dependency; PII stays on-premises.

4. **Persistent Store Adapter** — `VidcStoreProtocol` (structural `typing.Protocol`) with three implementations: `MemoryVidcStore` (current), `PostgresVidcStore` (asyncpg), `RedisVidcStore` (aioredis). Pass any conforming store to the constructor; zero service-layer changes required.

5. **Meeting Workflow DSL** — `MeetingWorkflow` dataclass encodes declarative lifecycle pipelines (room_create → meeting_start → agent_register → recording_start → transcribe → end). `run_workflow(workflow, tenant_id)` executes steps with per-step rollback hooks; orchestrations are reproducible and unit-testable.

6. **Attribute-Based Access Control** — `evaluate()` extended to accept `actor_attributes` dict (department, clearance_level, device_trust_score) for ABAC policy evaluation. Expressible rules like "only hosts with clearance_level >= 3 can record external-guest meetings."

7. **P2P ↔ SFU Topology Optimisation** — `optimize_topology` reads participant count and decides P2P (≤3 participants) vs SFU routing. Emits `topology_changed` audit event with recommended media server target; reduces infrastructure cost ~40% for small meetings.

8. **Breakout Bus** — `BreakoutBus` propagates chat, reactions, and poll results between breakout rooms and the parent in real-time via `asyncio.Queue` (swappable for NATS/Redis Streams). Facilitators can broadcast to all rooms simultaneously.

9. **CV Engagement Metrics** — `compute_engagement_metrics(tenant_id, meeting_id, frame_batch)` passes frames to a locally-served LLaVA model via Ollama. Returns speaker_activity ratio, camera-on ratio, reaction count as time-series on `MeetingRecord`. No cloud video upload.

10. **Action Item Extraction** — `extract_action_items(tenant_id, meeting_id, transcript_ref)` calls a local Ollama model (mistral/llama3) with a structured extraction prompt. Returns `ActionItem` records (assignee, description, due_date, confidence); persists as first-class entities for task-management integration.

11. **Cross-Tenant Federation** — `federate_meeting(host_tenant_id, guest_tenant_id, meeting_id, federated_room_ref)` creates cross-tenant participant records with tenant-isolation invariants enforced at service layer, preserving per-tenant recording consent and retention policies.

12. **Structured Minutes Generation** — `generate_minutes(tenant_id, meeting_id, format)` assembles minutes from captions, action items, poll results, and attendance. Supports `markdown`, `docx`, `pdf`. Returns `MinutesRecord` with a download URL.

13. **Webhook / Event Subscription** — `subscribe_events(tenant_id, webhook_url, event_types)` and `unsubscribe_events(subscription_id)`. Dispatches HMAC-signed payloads via aiohttp POST on every `_record_event`. Enables CRM, JIRA, Slack to react to lifecycle events without polling.

14. **Immutable Merkle-Chain Audit Trail** — `MeetingAuditEventRecord` extended with `prev_hash`. Each event hashes `(id + event_type + subject_id + actor + prev_hash)` forming a tamper-evident chain per tenant. `verify_audit_chain(tenant_id)` returns a verification report for regulators.

15. **Meeting Cost Attribution** — `compute_meeting_cost(tenant_id, meeting_id, cost_model_ref)` calculates infrastructure cost by participant-minutes, recording storage, and ASR compute. Returns `MeetingCostRecord` with per-department breakdowns; integrates with APG `finc`/`budg` capabilities for automated chargeback reporting.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/vidc/__init__.py capabilities/common/vidc/capability_contract.py capabilities/common/vidc/video_runtime.py capabilities/common/vidc/service.py capabilities/common/vidc/api.py capabilities/common/vidc/views.py capabilities/common/vidc/app.py capabilities/common/vidc/test_capability_contract.py capabilities/common/vidc/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/vidc/test_capability_contract.py capabilities/common/vidc/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/vidc --json
./.venv/bin/apg capabilities publish-plan capabilities/common/vidc --json
```
