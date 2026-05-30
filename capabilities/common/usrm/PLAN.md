# USRM Development Plan

## Goal

Make USRM a complete APG capability packet that generated applications can
compose for user identity, profiles, invitations, access review, role
assignment, privacy preference sync, deprovisioning, AI-assisted review,
Bytewax events, UI surfaces, visual theming, documentation, and focused
verification.

## Work Items

1. Documentation packet
   - Add `SPECIFICATION.md`.
   - Add `PLAN.md`.
   - Add `README.md`.
   - Replace `cap_spec.md` with the active lifecycle packet summary.

2. Contract expansion
   - Add `usrm_agents`, `observability`, and `adapters` configuration.
   - Add provides/requires metadata.
   - Add Bytewax streaming manifest and event-stream helper.
   - Add deterministic rules for owner, profile validation, privacy sync,
     role approval, deprovision evidence, Bytewax lifecycle streams, agents,
     and privileged agent actions.
   - Add `/usrm/agents` and `/usrm/policy` UI routes.

3. Runtime expansion
   - Add USRM agent records and metadata-rich audit events.
   - Extend `UsrmService` with agent registration, privileged agent-action
     validation, batch lifecycle validation, Bytewax lifecycle metadata, and
     stronger user/profile/invitation/access/deprovision guardrails.
   - Keep production integration behind adapters.

4. API and views
   - Expose agent and batch validation helpers.
   - Add agent workbench and policy center view models.
   - Include streaming metadata and policy guardrails in dashboard, lifecycle,
     access, deprovisioning, settings, and status surfaces.

5. Generated evidence
   - Refresh `app.py`, `semantic_model.json`, `package_manifest.json`, and
     `release_report.json` from the expanded contract.
   - Ensure package manifest lists docs, contract, runtime, API, views, and
     tests.

6. Verification
   - Run focused py_compile for USRM package files.
   - Run focused USRM tests.
   - Run implementation audit for `capabilities/common/usrm`.
   - Run publish-plan for `capabilities/common/usrm`.
   - Run stale-marker and unsupported stream scans on touched USRM files.

## Review Checklist

- Tenant context is enforced.
- User identity, owner, and profile validation are enforced.
- Invitations require consent and Bytewax lifecycle metadata.
- Profile updates require privacy sync evidence.
- Privileged users and roles require MFA and approval.
- Access reviews have reviewer attribution.
- Deprovisioning requires access revocation, evidence, and Bytewax metadata.
- Bulk lifecycle actions require review and Bytewax coordination.
- USRM agents are first-class and constrained by runtime, role, and human
  approval policy.
- Generated app evidence matches the contract.
