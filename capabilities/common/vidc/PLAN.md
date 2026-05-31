# Video Conferencing Capability Plan

## Objective

Bring `vidc` to the current APG capability-packet standard: readable local documentation, a complete executable contract, a dependency-light runtime, first-class provider-neutral video agents, Bytewax lifecycle governance, UI/theming metadata, deterministic guardrails, generated package evidence, and focused verification.

## Implementation Packets

1. Document the intended lifecycle.
   - Add `README.md`.
   - Add `SPECIFICATION.md`.
   - Keep `cap_spec.md` as a compatibility pointer to the current docs and runtime.

2. Expand the executable contract.
   - Add meeting-agent, first-class video-agent, and adapter metadata.
   - Add observability and Bytewax lifecycle-stream controls.
   - Expand deterministic rules for rooms, meetings, recordings, captions, agents, audit, tenant isolation, and lifecycle-batch mutations.
   - Add agent, lifecycle, and audit UI routes plus theme components.

3. Strengthen runtime behavior.
   - Extend `VidcService` with meeting-agent, video-agent, and lifecycle-batch records.
   - Enforce secure transport, screen-share, recording retention, recording access audit, caption language, and meeting-agent guardrails.
   - Keep all behavior dependency-light and deterministic.

4. Align composition surfaces.
   - Update API helpers for new meeting options, meeting-agent registration, video-agent registration, and lifecycle-batch validation.
   - Update view models for agent, lifecycle, and audit screens.
   - Update registration metadata and permissions.

5. Refresh package evidence.
   - Regenerate `app.py`, `semantic_model.json`, `package_manifest.json`, and `release_report.json` from the current contract.
   - Ensure the runtime semantic model matches the committed JSON artifact.

6. Review and verify.
   - Run focused package compile and tests.
   - Run implementation audit and publish-plan for `capabilities/common/vidc`.
   - Scan for stale generated or marketing language.
   - Run `git diff --check` for the changed package and progress log.

## Test Strategy

- Contract tests prove configuration, rule count, routes, theme, registration, and Bytewax controls.
- Runtime tests prove room, meeting, participant, recording, caption, meeting-agent, end-meeting, and dashboard summary behavior.
- Guardrail tests prove tenant, host, guest policy, secure transport, screen-share policy, recording consent, encryption, retention, agent scope, first-class video-agent controls, and Bytewax denial behavior.
- API/view tests prove generated applications can compose the capability without private implementation knowledge.

## Review Checklist

- No live media infrastructure is started from package import or self-test.
- Tenant IDs are required and records are listed by tenant.
- Guardrails in the contract are enforced by runtime methods where applicable.
- AI meeting agents have registration, scope, runtime, role, and disclosure.
- First-class video agents have runtime, role, scope, owner, purpose, disclosure, and approval guardrails.
- Bytewax is the only configured lifecycle event-stream adapter.
- UI routes and theme components cover every major lifecycle surface.

## Out Of Scope

- Live WebRTC/SFU implementation.
- TURN/STUN allocation.
- Video blob persistence.
- Real transcription or computer-vision inference.
- External AI-agent CLI invocation.
- Full repository test suite.
