# CONS Development Plan

## Objective

Turn CONS into a coherent lifecycle and guardrail packet for executable APG
consent and privacy-management applications.

## Build Steps

1. Specify the packet.
   - Define notice, purpose, consent, preference, privacy request, processing
     decision, privacy-agent, audit, UI, theme, rule, and Bytewax stream
     requirements.
   - Keep live identity, DLP, audit-log, notification, workflow, and marketing
     providers behind adapters.

2. Align the capability contract.
   - Add privacy-agent, governance, observability, adapter, UI, theme, and
     Bytewax stream configuration.
   - Add deterministic rules for the full privacy lifecycle and guardrails.
   - Ensure rule matching supports numeric and inequality suffixes.

3. Complete the executable runtime.
   - Add the `PrivacyAgent` model.
   - Extend `ConsService` with tenant-safe keys for duplicate IDs across
     tenants.
   - Add scoped privacy-agent registration and guarded purpose state changes.
   - Preserve notice, purpose, consent, preference, request, processing, and
     audit flows.

4. Complete composition surfaces.
   - Extend API helpers for agents, purpose state changes, privacy state, and
     status.
   - Extend view models for agents, audit, analytics, settings, and Bytewax
     stream metadata.
   - Update capability registration metadata, permissions, endpoints, optional
     dependencies, and capabilities.

5. Refresh package evidence.
   - Regenerate `app.py`, `semantic_model.json`, `package_manifest.json`, and
     `release_report.json` from the contract.
   - Confirm generated evidence includes Bytewax, privacy agents, routes, and
     expanded rules.

6. Review and verify.
   - Run focused compile checks and CONS package tests.
   - Run package self-test, implementation audit, publish-plan, stale-marker
     search, and `git diff --check`.
   - Fix emergent issues before committing.

## Risks And Controls

- Privacy workflows carry compliance risk. Keep the package deterministic,
  auditable, tenant-scoped, and side-effect free.
- AI privacy agents can obscure accountability. Require registration, scope,
  supported runtime/role, contribution disclosure, policy reference, and audit.
- Cross-tenant state can leak when IDs are reused. Store tenant-qualified keys
  while preserving user-facing IDs in records.
- Consent-gated processing can be bypassed if active consent is not checked.
  Enforce active consent through the service and rule engine.
- Battery constraints limit verification scope. Run focused CONS checks now and
  document broader live-adapter checks as not run.

## Completion Evidence

- Focused compile and pytest checks pass.
- Package self-test passes.
- Generated semantic model confirms:
  - `streaming.processor == "bytewax"`
  - supported privacy-agent runtimes include Codex, Claude Code, OpenCode, Pi
  - `/cons/agents` is exposed
- Implementation audit reports no CONS errors or warnings.
- Publish-plan reports CONS is side-effect free.
- Stale-marker search returns no matches.
- Progress log records the packet and known verification gaps.

