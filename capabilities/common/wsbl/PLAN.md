# WSBL Development Plan

## Goal

Make WSBL a complete APG capability packet that generated applications can
compose for tenant sites, domain validation, page composition, component
governance, publishing, rollback, accessibility, privacy consent, AI-assisted
review, Bytewax events, UI surfaces, visual theming, documentation, and focused
verification.

## Work Items

1. Documentation packet
   - Add `SPECIFICATION.md`.
   - Add `PLAN.md`.
   - Add `README.md`.
   - Replace `cap_spec.md` with the active lifecycle packet summary.

2. Contract expansion
   - Add `wsbl_agents`, `observability`, and `adapters` configuration.
   - Add provides/requires metadata.
   - Add Bytewax streaming manifest and event-stream helper.
   - Add deterministic rules for domain validation, structured sections,
     preview evidence, publish streams, component policy, rollback streams,
     batch publishing, agents, and privileged agent actions.
   - Add `/wsbl/agents` and `/wsbl/policy` UI routes.

3. Runtime expansion
   - Add WSBL agent records and keep audit event details metadata-rich.
   - Extend `WsblService` with agent registration, privileged agent-action
     validation, batch publish validation, Bytewax lifecycle metadata, and
     stronger publish/component/rollback guardrails.
   - Keep production integration behind adapters.

4. API and views
   - Expose agent and batch validation helpers.
   - Add agent workbench and policy center view models.
   - Include streaming metadata and policy guardrails in dashboard,
     publishing, analytics, settings, and status surfaces.

5. Generated evidence
   - Refresh `app.py`, `semantic_model.json`, `package_manifest.json`, and
     `release_report.json` from the expanded contract.
   - Ensure package manifest lists docs, contract, runtime, API, views, and
     tests.

6. Verification
   - Run focused py_compile for WSBL package files.
   - Run focused WSBL tests.
   - Run implementation audit for `capabilities/common/wsbl`.
   - Run publish-plan for `capabilities/common/wsbl`.
   - Run stale-marker and unsupported stream scans on touched WSBL files.

## Review Checklist

- Tenant context is enforced.
- Sites require owners.
- Publishing requires validated domains, structured sections, preview
  evidence, approval, accessibility pass, and Bytewax routing.
- Privacy banners route to consent-policy review.
- Custom components require review and policy attribution.
- Rollbacks require Bytewax routing.
- WSBL agents are first-class and constrained by runtime, role, and human
  approval policy.
- Generated app evidence matches the contract.
