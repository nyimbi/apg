# THEM Development Plan

## Goal

Make THEM a complete APG capability packet that generated applications can
compose for tenant theming, token governance, brand assets, preview evidence,
accessibility gates, publication approvals, AI-assisted review, Bytewax events,
UI surfaces, visual theming, documentation, and focused verification.

## Work Items

1. Documentation packet
   - Add `SPECIFICATION.md`.
   - Add `PLAN.md`.
   - Add `README.md`.
   - Replace `cap_spec.md` with the active lifecycle packet summary.

2. Contract expansion
   - Add `them_agents`, `observability`, and `adapters` configuration.
   - Add provides/requires metadata.
   - Add Bytewax streaming manifest and event-stream helper.
   - Add deterministic rules for agents, Bytewax publication, batch rollout,
     token review, guideline evidence, asset approval, and preview evidence.
   - Add `/them/agents` UI route and policy metadata.

3. Runtime expansion
   - Add THEM agent records and metadata-rich audit events.
   - Extend `ThemService` with agent registration, privileged agent-action
     validation, batch rollout validation, Bytewax publication metadata, and
     stronger guardrails.
   - Keep production integration behind adapters.

4. API and views
   - Expose agent and batch validation helpers.
   - Add agent workbench view model.
   - Include streaming metadata and policy guardrails in dashboard, policies,
     settings, and status surfaces.

5. Generated evidence
   - Refresh `app.py`, `semantic_model.json`, `package_manifest.json`, and
     `release_report.json` from the expanded contract.
   - Ensure package manifest lists docs, contract, runtime, API, views, and
     tests.

6. Verification
   - Run focused py_compile for THEM package files.
   - Run focused THEM tests.
   - Run implementation audit for `capabilities/common/them`.
   - Run publish-plan for `capabilities/common/them`.
   - Run stale-marker and unsupported stream scans on touched THEM files.

## Review Checklist

- Tenant context is enforced.
- Theme owner and guideline evidence are enforced.
- Token updates have reviewer attribution.
- Brand assets require license and approval.
- Publication requires preview, contrast, approval, and Bytewax stream metadata.
- Broad rollouts require review.
- THEM agents are first-class and constrained by runtime, role, and human
  approval policy.
- Generated app evidence matches the contract.
- Documentation explains how to use the capability without external services.
