# CKM Real-Time Collaboration

`ckm_rtc` is the APG Collaboration and Knowledge Management real-time
collaboration capability. It lets generated applications compose collaboration
sessions, participant policy, presence, messaging, media guardrails, decision
capture, audit evidence, analytics metadata, and AI-agent assistance.

The package is dependency-light. It defines the executable lifecycle, rules, UI
route metadata, theme metadata, Bytewax stream declaration, and semantic
evidence. Live WebSocket, WebRTC, SIP, RTMP, Socket.IO, gRPC, database, media,
notification, scheduler, and stream-worker deployments are adapter
responsibilities. The preserved legacy FastAPI/WebSocket runtime is available
as `runtime_app.py`.

## What It Provides

- Collaboration sessions tied to APG business context.
- Participant policy and controlled join lifecycle.
- Presence heartbeat and context-state tracking.
- Real-time message lifecycle with sensitive-content review.
- Media guardrails for screen sharing and recording consent.
- Decision capture with trace evidence.
- AI RTC-agent registration for Codex, Claude Code, OpenCode, Pi, and future
  runtimes behind the same contract.
- Bytewax stream guardrail for batch RTC mutation.
- UI routes and visual theme tokens for generated APG applications.

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

## Guardrails

The deterministic rules deny or require review when:

- tenant context is missing;
- session owner or participant policy is missing;
- a participant is not allowed by session policy;
- presence lacks heartbeat evidence;
- a message targets an inactive session;
- sensitive message content lacks review;
- screen sharing lacks permission;
- recording lacks consent evidence;
- decision capture lacks trace evidence;
- an AI RTC agent is unregistered, unsupported, unscoped, or undisclosed;
- lifecycle state changes lack audit evidence;
- batch RTC mutation does not use Bytewax.

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

## Composition

Generated APG applications should compose `ckm_rtc` through:

- capability ID: `ckm_rtc`;
- provided services: collaboration sessions, presence awareness, real-time
  messaging, media collaboration, decision capture, page collaboration, and RTC
  agents;
- required services: `auth`, `conf`, `audl`, and `ckm_not`;
- API prefix: `/ckm-rtc/api/v1`;
- UI routes: dashboard, rooms, presence, messages, media, decisions, agents,
  rules, analytics, audit, and settings;
- theme: `ckm_rtc_collaboration_ops`;
- stream processor: `bytewax`.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/ckm/rtc/__init__.py capabilities/ckm/rtc/capability_contract.py capabilities/ckm/rtc/lifecycle.py capabilities/ckm/rtc/app.py capabilities/ckm/rtc/test_capability_contract.py
./.venv/bin/pytest -q capabilities/ckm/rtc/test_capability_contract.py
./.venv/bin/python -c "import importlib; pkg = importlib.import_module('capabilities.ckm.rtc'); service = pkg.RtcLifecycleService('tenant-proof'); print(service.dashboard_summary())"
./.venv/bin/apg capabilities implementation-audit --root capabilities/ckm/rtc --json
./.venv/bin/apg capabilities publish-plan capabilities/ckm/rtc --json
```
