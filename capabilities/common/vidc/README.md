# Video Conferencing Capability

`vidc` provides APG's common capability for tenant-scoped video meetings. It composes meeting rooms, accountable hosts, waiting-room controls, participants, encrypted recordings, caption artifacts, AI meeting agents, audit events, UI routes, visual theming, and Bytewax event-stream guardrails into a generated-application packet that runs without live media infrastructure.

## What It Provides

- Meeting rooms with owners, moderation policies, external guest policy references, and waiting-room controls.
- Video meeting lifecycle from room selection through active meeting, review-required state, recording/caption artifacts, AI meeting assistants, and meeting closure.
- Guardrails for tenant context, host accountability, secure media transport, screen-share policy, external guests, recording consent, encryption, retention, access audit, large-meeting review, caption language support, computer-vision assist policy, and cross-tenant denial.
- AI meeting agents for captioning, summaries, moderation, and action tracking across runtimes such as Codex, Claude Code, OpenCode, and Pi.
- Bytewax enforcement for batch video-meeting mutations.
- Dependency-light API helpers, UI view models, package manifest, semantic model, and release evidence.

## Runtime Shape

The generated runtime is `service.VidcService`. It is deterministic and in-memory so generated applications can exercise the video lifecycle without WebRTC servers, media SFUs, object stores, transcription engines, or databases.

Primary methods:

- `create_room(...)`
- `start_meeting(...)`
- `add_participant(...)`
- `create_recording(...)`
- `generate_captions(...)`
- `register_meeting_agent(...)`
- `end_meeting(...)`
- `dashboard_summary(...)`

## Configuration And Rules

`capability_contract.py` is the source of truth for:

- configuration defaults
- configuration schema
- deterministic rules
- UI route contracts
- theme tokens
- APG adapter map

The rule engine returns `allow`, `require_review`, or `deny` decisions with matched rules and required actions.

## UI Surfaces

The package exposes route contracts for:

- dashboard
- meetings
- rooms
- participants
- recordings
- captions
- agents
- analytics
- audit
- settings

`views.py` provides dependency-light models for these screens.

## How To Use

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
```

Use `register_capability()` to expose the full APG registration payload to the composition engine.

## Verification

Focused verification for this packet should use:

```bash
./.venv/bin/python -m py_compile capabilities/common/vidc/__init__.py capabilities/common/vidc/capability_contract.py capabilities/common/vidc/video_runtime.py capabilities/common/vidc/service.py capabilities/common/vidc/api.py capabilities/common/vidc/views.py capabilities/common/vidc/app.py capabilities/common/vidc/test_capability_contract.py capabilities/common/vidc/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/vidc/test_capability_contract.py capabilities/common/vidc/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/vidc --json
./.venv/bin/apg capabilities publish-plan capabilities/common/vidc --json
```
