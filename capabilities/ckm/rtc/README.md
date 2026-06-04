# Real-Time Collaboration

## Overview

The Real-Time Collaboration capability (`ckm_rtc`) provides synchronous and asynchronous collaboration infrastructure across chat, presence, voice, video, screen sharing, co-editing, and whiteboarding modes. It manages collaboration sessions with participant policies, real-time messaging with audit retention, and structured decision capture with voting and consensus building — all scoped per tenant and wired into the APG audit trail.

Beyond basic conferencing, `ckm_rtc` introduces page-level collaboration on any Flask-AppBuilder view: users share presence on the same page, delegate form fields to colleagues, request in-context assistance, and co-edit with field-level locking. The multi-protocol signaling layer (WebSocket, WebRTC, gRPC, SIP, RTMP, Socket.IO) is surfaced through a unified protocol manager, enabling Teams/Zoom/Meet feature parity within the APG shell without requiring external platform accounts.

## Capability ID

`ckm_rtc`  Version: 1.0.0

## Provides

| Service | Description |
|---------|-------------|
| collaboration_sessions | Tenant-scoped session lifecycle with owner accountability and participant policies |
| presence_awareness | Heartbeat-driven presence with stale detection at 90 seconds |
| real_time_messaging | Threaded chat with reactions, pinning, retention policies, and sensitive-content review |
| media_collaboration | Video calls with recording consent, screen sharing permissions, and breakout rooms |
| decision_capture | Structured decisions with voting, consensus threshold, and implementation tracking |
| page_collaboration | Per-page presence, form field delegation, assistance requests, and field locking |
| rtc_agents | AI agent assist for session facilitation, decision review, transcription, and risk moderation |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Identity context, participant authentication, and RBAC |
| conf | Tenant-scoped configuration for session limits and protocol settings |
| audl | Audit log sink for session events, messages, and decision traces |
| ckm_not | Sends session invitations, join notifications, decision alerts, and recording consent requests |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant scoping for all operations |
| sessions.owner_required | bool | true | Sessions require accountable owner |
| sessions.participant_policy_required | bool | true | Participant policy required at creation |
| sessions.max_participants | int | 250 | Hard cap on session participants |
| presence.stale_after_seconds | int | 90 | Presence entry considered stale without heartbeat |
| presence.context_disclosure_required | bool | true | Presence context must be disclosed |
| messaging.message_audit_required | bool | true | All messages must be audited |
| messaging.retention_policy_required | bool | true | Messages require retention policy |
| media.recording_requires_consent | bool | true | Recording blocked without consent |
| media.screen_share_requires_permission | bool | true | Explicit permission required for screen share |
| collaboration.co_edit_locking_required | bool | true | Field locks required for co-editing |
| collaboration.decision_capture_required | bool | true | Decisions require trace evidence |
| rtc_agents.agent_registration_required | bool | true | Agents must be registered before use |
| governance.batch_event_stream | string | "bytewax" | Batch mutations must route through Bytewax |

Supported protocols: `websocket`, `webrtc`, `grpc`, `sip`, `rtmp`, `socketio`

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /ckm-rtc/dashboard | GET | ckm_rtc:view | Overview |
| rooms | /ckm-rtc/rooms | GET | ckm_rtc:manage_rooms | Collaboration |
| presence | /ckm-rtc/presence | GET | ckm_rtc:view | Collaboration |
| messages | /ckm-rtc/messages | GET | ckm_rtc:participate | Collaboration |
| media | /ckm-rtc/media | GET | ckm_rtc:participate | Media |
| decisions | /ckm-rtc/decisions | GET | ckm_rtc:participate | Governance |
| agents | /ckm-rtc/agents | GET | ckm_rtc:govern | Governance |
| rules | /ckm-rtc/rules | GET | ckm_rtc:govern | Governance |
| analytics | /ckm-rtc/analytics | GET | ckm_rtc:view | Insights |
| audit | /ckm-rtc/audit | GET | ckm_rtc:view | Governance |
| settings | /ckm-rtc/settings | GET | ckm_rtc:admin | Administration |

API prefix: `/ckm-rtc/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny |
| session_requires_owner | create_session without owner | deny |
| session_requires_participant_policy | create_session without participant policy | deny |
| join_requires_allowed_participant | join_session by participant not in policy | deny |
| presence_requires_heartbeat | update_presence without heartbeat evidence | deny |
| message_requires_active_session | post_message to inactive session | deny |
| sensitive_message_requires_review | Sensitive content detected without review | require_review |
| screen_share_requires_permission | start_screen_share without permission | deny |
| recording_requires_consent | start_recording without consent | deny |
| decision_requires_trace | capture_decision without decision trace | deny |
| rtc_agent_requires_registration | Agent present but not registered | deny |
| rtc_agent_runtime_supported | Agent uses unsupported runtime | deny |
| rtc_agent_role_supported | Agent uses unsupported role | deny |
| rtc_agent_requires_scope | Agent without explicit scope | deny |
| rtc_agent_requires_disclosure | Agent contribution not disclosed | deny |
| rtc_state_change_requires_audit | State change without audit event | deny |
| batch_rtc_mutation_requires_bytewax | Batch mutation not using Bytewax | deny |

Supported agent runtimes: `codex`, `claude_code`, `opencode`, `pi`

Supported agent roles: `session_facilitator`, `decision_reviewer`, `transcript_reviewer`, `risk_moderator`, `workflow_assistant`

## Data Models

| Model | Key Fields |
|-------|-----------|
| RTCSession | session_id, tenant_id, session_name, session_type, digital_twin_id, owner_user_id, is_active, max_participants, collaboration_mode, recording_enabled, current_view_state, access_code, require_approval |
| RTCParticipant | participant_id, session_id, user_id, display_name, role, joined_at, is_online, can_edit, can_annotate, can_share_screen, can_run_simulations, activity_count, connection_quality |
| RTCActivity | activity_id, session_id, participant_id, activity_type, action, target_object, old_values, new_values, impact_level, requires_sync, sync_status, conflicts_with |
| RTCMessage | message_id, session_id, participant_id, message_type, content, reply_to_message_id, thread_root_id, is_private, target_participants, reactions, is_pinned, read_by |
| RTCDecision | decision_id, session_id, title, decision_type, proposed_by, decision_method, options, votes, consensus_threshold, status, selected_option, implementation_status |
| RTCWorkspace | workspace_id, tenant_id, digital_twin_id, owner_user_id, collaborators, saved_views, persistent_annotations, field_locks, total_collaboration_hours |
| RTCVideoCall | call_id, session_id, call_type, host_user_id, status, teams_meeting_url, zoom_meeting_id, meet_url, enable_recording, breakout_rooms_enabled, end_to_end_encryption |
| RTCVideoParticipant | video_participant_id, call_id, participant_id, audio_enabled, video_enabled, is_muted, is_screen_sharing, hand_raised, role, in_waiting_room, breakout_room_id |
| RTCScreenShare | share_id, call_id, presenter_id, share_type, status, remote_control_enabled, annotations_enabled, annotations, hide_sensitive_content, bandwidth_usage_kbps |
| RTCRecording | recording_id, call_id, recording_type, status, duration_minutes, auto_transcription_enabled, ai_highlights, ai_summary, ai_action_items, cloud_provider, auto_delete_after_days |
| RTCPageCollaboration | page_collab_id, tenant_id, page_url, blueprint_name, view_name, record_id, current_users, delegated_fields, field_locks, assistance_requests, form_data_state |
| RTCThirdPartyIntegration | integration_id, tenant_id, platform, integration_type, access_token, webhook_url, status, teams_tenant_id, zoom_account_id, google_workspace_domain, monthly_api_limit |

## Streaming Events

Events emitted to the ckm event stream via Bytewax.

Topic: `apg.ckm_rtc.lifecycle`

| Event | Trigger |
|-------|---------|
| rtc_session_created | New collaboration session opened |
| rtc_participant_joined | Participant joined a session |
| rtc_presence_updated | Presence heartbeat or status change |
| rtc_message_posted | Message posted in session |
| rtc_screen_share_started | Screen sharing initiated |
| rtc_recording_started | Session recording begun |
| rtc_decision_captured | Decision recorded in session |
| rtc_agent_registered | AI collaboration agent registered |

Batch mutation guardrail: `batch_rtc_mutation_requires_bytewax`

## Edge Cases Handled

- Blocked users are checked in `RTCSession.can_user_join()` before participant count is evaluated, preventing a race where a blocked user slips in just as capacity opens up.
- Sensitive message review (`sensitive_message_requires_review`) fires at the rule layer before storage, giving the reviewer workflow a chance to quarantine content before other participants see it.
- `RTCDecision.calculate_consensus()` handles the zero-vote case explicitly, returning `consensus_reached: false` rather than dividing by zero.
- Page-level field locking in `RTCPageCollaboration.lock_field()` is idempotent on first-lock and returns false (not an error) when a field is already locked, letting callers distinguish contention from success without raising exceptions.
- Recording consent is enforced at the `start_recording` operation by the `recording_requires_consent` rule; consent state is independent of whether participants later mute or leave, so recordings cannot be started retroactively.
- `RTCThirdPartyIntegration.log_api_call()` enforces monthly quota by returning false when `current_month_usage >= monthly_api_limit`, allowing callers to gracefully degrade rather than silently blow the quota.
- Screen share annotations survive a presenter disconnect because they are persisted on `RTCScreenShare.annotations` (JSON column) rather than kept in memory.

## Composability

- **Upstream**: `auth` provides identity and RBAC; `conf` supplies tenant session limits; `audl` receives all collaboration audit events; `ckm_not` sends notifications triggered by session events.
- **Downstream**: `ckm_wfa` uses `ckm_rtc` as a collaboration surface for approval reviews, exception escalation calls, and process design sessions. Workflow tasks can spawn collaboration sessions and capture decisions back as task completion evidence.
- **Peer**: `ckm_not` is the canonical dependency for all outbound notifications. Third-party integrations (Teams, Zoom, Google Meet) are bridged through `RTCThirdPartyIntegration` rather than being treated as separate capabilities.

## Development Notes

- The unified protocol manager (`unified_protocol_manager.py`) abstracts WebSocket, WebRTC, gRPC, SIP, RTMP, and Socket.IO behind a single interface. Protocol selection is tenant/session-configurable; do not hard-code protocol references in service logic.
- `RTCRecording.ai_highlights`, `ai_summary`, and `ai_action_items` are populated asynchronously post-processing. Recording status progresses `recording → processing → completed`; callers must poll or subscribe to the event stream rather than expecting synchronous AI output.
- Page collaboration (`RTCPageCollaboration`) integrates at the Flask-AppBuilder blueprint level via `blueprint.py`. `form_data_state` and `field_locks` are JSON columns — they are not relational locks. Optimistic concurrency must be implemented at the service layer.
- `RTCVideoCall` models Teams/Zoom/Meet meeting IDs independently (`teams_meeting_id`, `zoom_meeting_id`, `meet_url`). Only one external platform per call is typical, but the schema allows hybrid scenarios for tenants bridging platforms.
- OAuth token refresh for third-party integrations (`RTCThirdPartyIntegration.refresh_access_token()`) is stubbed and must be implemented per platform. Monitor `token_expires_at` proactively — the `is_token_expired()` helper is available.
- Conflict resolution for concurrent activities is tracked via `RTCActivity.conflicts_with` (list of conflicting activity IDs) and `resolution_strategy`. The default strategy is last-writer-wins; implement operational transforms at the service layer for co-edit scenarios requiring stronger guarantees.

## Quick Use

```python
from capabilities.ckm.rtc import RtcLifecycleService

service = RtcLifecycleService("tenant-acme")

service.create_session(
    session_id="close-room-001",
    name="Month-end close review",
    owner_id="user-cfo",
    context_ref="fin.glr/period/2026-05",
    participant_policy=["user-cfo", "user-controller", "user-auditor"],
)

service.join_session("close-room-001", "user-controller", role="editor")
service.update_presence(
    session_id="close-room-001",
    user_id="user-controller",
    status="active",
    heartbeat_id="heartbeat-001",
    context_ref="fin.glr/journal-review",
)

message = service.post_message(
    session_id="close-room-001",
    author_id="user-controller",
    body="Variance review is ready.",
)
assert message["status"] == "posted"

decision = service.capture_decision(
    session_id="close-room-001",
    owner_id="user-cfo",
    decision_text="Approve accrual adjustment batch A.",
    trace_ref="audit/decision/close-room-001/1",
)
assert decision["trace_ref"]
```

## AI Agent Registration

AI agents are first-class contributors only after registration:

```python
agent = service.register_rtc_agent(
    name="Decision reviewer",
    runtime="codex",
    role="decision_reviewer",
    scope="review captured decisions for trace and policy gaps",
    contribution_disclosed=True,
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles are `session_facilitator`, `decision_reviewer`, `transcript_reviewer`,
`risk_moderator`, and `workflow_assistant`.

## Bytewax Batch Mutation

Batch collaboration mutation must use the Bytewax event stream:

```python
allowed = service.validate_batch_rtc_mutation("bytewax")
blocked = service.validate_batch_rtc_mutation("other-stream")

assert allowed["decision"] == "allow"
assert blocked["decision"] == "deny"
```

The contract declares topic `apg.ckm_rtc.lifecycle` and state for sessions,
participants, presence, messages, media events, decisions, RTC agents, and
audit events.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/ckm/rtc/__init__.py capabilities/ckm/rtc/capability_contract.py capabilities/ckm/rtc/lifecycle.py capabilities/ckm/rtc/app.py capabilities/ckm/rtc/test_capability_contract.py
./.venv/bin/pytest -q capabilities/ckm/rtc/test_capability_contract.py
./.venv/bin/python -c "import importlib; pkg = importlib.import_module('capabilities.ckm.rtc'); service = pkg.RtcLifecycleService('tenant-proof'); print(service.dashboard_summary())"
./.venv/bin/apg capabilities implementation-audit --root capabilities/ckm/rtc --json
./.venv/bin/apg capabilities publish-plan capabilities/ckm/rtc --json
```
