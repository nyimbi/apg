# Collaboration Tools Implementation Plan

## Objectives

- Integrate the existing COLB production surface with the APG generated-application packet pattern.
- Keep heavy protocol/database/WebRTC surfaces available, but make package publishing dependency-light.
- Expand COLB into a coherent lifecycle and guardrail packet for workspaces, sessions, artifacts, annotations, decisions, presence, and AI collaborators.

## Work Items

1. Documentation
   - Add `README.md`.
   - Add `SPECIFICATION.md`.
   - Keep `cap_spec.md` as a compatibility pointer.

2. Contract
   - Expand configuration sections for workspaces, sessions, artifacts, annotations, presence, protocols, AI agents, security, governance, retention, observability, adapters, UI, and theme.
   - Expand deterministic guardrails to cover the complete collaboration lifecycle.
   - Declare Bytewax as the event-stream adapter.

3. Runtime
   - Add a dependency-light generated runtime in `collaboration_runtime.py`.
   - Preserve the existing production service/API/views/protocol files.
   - Move the existing heavy app entrypoint to `production_app.py`.
   - Make `app.py` the generated package entrypoint.

4. API and UI
   - Add `package_api.py` helper functions.
   - Add `view_models.py` for generated app screens.

5. Evidence and tests
   - Generate semantic model, release report, and package manifest from the live contract.
   - Replace stale package-test terminology.
   - Add focused runtime, API, view, tenant-isolation, rule, and package tests.

## Review Checklist

- Tenant IDs are required and runtime storage keys are tenant scoped.
- Public IDs can repeat across tenants.
- External collaboration, artifacts, DLP, decisions, AI collaborators, secure transport, and protocol health have guardrails.
- Package evidence agrees with the live contract.
- No Kafka dependency is introduced.
- Focused verification passes without live infrastructure.
