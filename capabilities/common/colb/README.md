# Collaboration Tools Capability

`colb` provides APG's common capability for tenant-scoped collaborative workspaces. It composes chat, notifications, authentication, realtime protocols, governed shared artifacts, annotations, decision records, presence, and AI collaborators into a generated-application packet that can run without live collaboration infrastructure.

## What It Provides

- Collaborative workspaces with owners, members, external participants, retention policies, and large-workspace review.
- Realtime collaboration sessions with secure transport, protocol health, event-bus evidence, recording-retention checks, and participant membership controls.
- Shared artifacts with artifact policy, version history, DLP checks for external sharing, duplicate-ID protection, annotations, and auditable decisions.
- Presence state for active sessions and collaborator cursors.
- AI collaborator guardrails for registered, scoped, disclosed contributions across runtimes such as Codex, Claude Code, OpenCode, and Pi.
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

The event stream adapter is Bytewax. Batch collaboration mutations must use Bytewax; Kafka is intentionally not part of the packet.

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
