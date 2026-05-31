# Collaboration Tools Capability

`colb` provides APG's common capability for tenant-scoped collaborative workspaces. It composes chat, notifications, authentication, realtime protocols, governed shared artifacts, annotations, decision records, presence, and AI collaborators into a generated-application packet that can run without live collaboration infrastructure.

## What It Provides

- Collaborative workspaces with owners, members, external participants, retention policies, and large-workspace review.
- Realtime collaboration sessions with secure transport, protocol health, event-bus evidence, recording-retention checks, and participant membership controls.
- Shared artifacts with artifact policy, version history, DLP checks for external sharing, duplicate-ID protection, annotations, and auditable decisions.
- Presence state for active sessions and collaborator cursors.
- First-class AI-agent composition for registered, scoped, owned, purpose-bound, disclosed collaboration agents across runtimes such as Codex, Claude Code, OpenCode, and Pi.
- Bytewax lifecycle batch validation for workspace, session, artifact, annotation, decision, presence, protocol, guest-access, and collaboration-agent mutations.
- Protocol adapter metadata for WebSocket, WebRTC, MQTT, gRPC, production services, and Bytewax event streams.
- Dependency-light generated runtime, API helpers, UI view models, package manifest, semantic model, and release evidence.

## Runtime Shape

The generated runtime is `collaboration_runtime.CollaborationRuntime`. It is deterministic and in-memory so generated applications can exercise the collaboration lifecycle without external databases, WebSocket servers, or protocol daemons.

Primary methods:

- `create_workspace(...)`
- `approve_workspace(...)`
- `start_session(...)`
- `join_session(...)`
- `share_artifact(...)`
- `add_annotation(...)`
- `record_decision(...)`
- `update_presence(...)`
- `register_collaboration_agent(...)`
- `validate_colb_lifecycle_batch(...)`
- `dashboard_summary(...)`

Production integration files such as `production_app.py`, `service.py`, `api.py`, `views.py`, and the protocol managers remain available for heavier deployments.

## Configuration And Rules

`capability_contract.py` is the source of truth for:

- configuration defaults
- configuration schema
- deterministic rules
- UI route contracts
- theme tokens
- APG adapter map

The event stream adapter is Bytewax. Batch collaboration mutations must use Bytewax; broker-specific queue is intentionally not part of the packet.

## Agent And Lifecycle Composition

COLB treats AI collaborators as first-class application citizens. The top-level `agents` manifest defines:

- supported runtimes: `codex`, `claude_code`, `opencode`, and `pi`
- supported roles for workspace, session, artifact, annotation, decision, presence, protocol, guest-access, lifecycle, and collaboration-steward review
- privileged roles that require human approval evidence before activation
- a provider-neutral adapter contract: `aicr_provider_neutral_collaboration_agent_adapter`

The top-level `streaming` manifest defines the `colb.lifecycle` stream, `event_time` watermark, required `bytewax` processor, supported lifecycle operations, and topics. `validate_colb_lifecycle_batch(...)` rejects empty, unsupported, or non-Bytewax lifecycle batches before generated applications treat them as accepted collaboration state.

## UI Surfaces

The generated package exposes route contracts for:

- dashboard
- workspaces
- sessions
- presence
- artifacts
- annotations
- decisions
- agents
- lifecycle
- protocols
- analytics
- audit
- settings

`view_models.py` provides dependency-light models for these screens.

## How To Use

```python
from capabilities.common.colb.collaboration_runtime import CollaborationRuntime

runtime = CollaborationRuntime()
workspace = runtime.create_workspace(
    "tenant-1",
    "workspace-1",
    "Finance Close",
    "owner",
    ["owner", "analyst"],
    "retain-180-days",
)
session = runtime.start_session(
    "tenant-1",
    "session-1",
    workspace["id"],
    "owner",
)
agent = runtime.register_collaboration_agent(
    "tenant-1",
    "agent-1",
    "Workspace Steward",
    "codex",
    "collaboration_steward",
    "workspace:workspace-1",
    "owner",
    "review collaboration lifecycle",
    human_approval_required=True,
)
batch = runtime.validate_colb_lifecycle_batch(
    "tenant-1",
    "bytewax",
    2,
    "collaboration_agent_batch",
)
```

Use `register_capability()` to expose the full APG registration payload to the composition engine.

## Verification

Focused verification for this packet should use:

```bash
./.venv/bin/python -m py_compile capabilities/common/colb/__init__.py capabilities/common/colb/capability_contract.py capabilities/common/colb/collaboration_runtime.py capabilities/common/colb/package_api.py capabilities/common/colb/view_models.py capabilities/common/colb/app.py capabilities/common/colb/test_capability_contract.py capabilities/common/colb/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/colb/test_capability_contract.py capabilities/common/colb/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/colb --json
./.venv/bin/apg capabilities publish-plan capabilities/common/colb --json
```
