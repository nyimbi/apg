# Chat and Messaging Implementation Plan

## Objectives

- Expand `chat` from a basic package into a coherent lifecycle and guardrail packet.
- Keep the runtime dependency-light and executable as a generated application.
- Preserve existing service/API/view naming where practical.
- Add enough focused tests to prove contract shape, rule execution, runtime behavior, tenant isolation, package evidence, and view models.

## Work Items

1. Specification and documentation
   - Add `README.md`.
   - Add `SPECIFICATION.md`.
   - Keep `cap_spec.md` as a compatibility pointer.

2. Contract
   - Expand configuration sections for rooms, messaging, presence, moderation, AI agents, security, governance, retention, observability, adapters, UI, and theme.
   - Expand deterministic rules to cover the full lifecycle.
   - Declare Bytewax as the event-stream adapter.

3. Runtime
   - Keep `ChatService` as the generated runtime.
   - Enforce tenant-qualified storage keys.
   - Route room, message, presence, moderation, and AI-agent decisions through the rule engine.
   - Preserve dependency-light operation.

4. API and views
   - Update API helpers for new runtime options.
   - Add view models for agents, analytics, audit, and settings.

5. Package evidence
   - Replace static semantic evidence with contract-derived semantic model output.
   - Refresh `semantic_model.json`, `release_report.json`, and `package_manifest.json`.

6. Verification
   - Run focused `py_compile`.
   - Run focused CHAT pytest only.
   - Run self-test, implementation audit, publish plan, stale-marker scan, and diff check.

## Review Checklist

- Tenant IDs are required and storage keys are tenant scoped.
- Public room and message IDs can repeat across tenants.
- Restricted content, attachments, DLP, duplicate messages, and AI-agent responses have guardrails.
- UI routes, semantic model, package manifest, and release report all derive from or agree with the live contract.
- No Kafka dependency is introduced.
- No live provider or browser requirement is introduced.
