# AUTH Capability Specification

## Identity

- Capability ID: `auth`
- Display name: Authentication & RBAC
- Category: `common`
- Owner: APG Platform Team
- Runtime shell: `apg_python`
- Theme: `auth_trust_fabric`

## Purpose

AUTH is the tenant-scoped identity, session, and authorization control plane for
APG applications. It governs identity registration, tenant membership,
role definition, role assignment approval, role assignment, session risk,
privileged access, privacy-budget analytics, audit evidence, and composable UI
surfaces.

The package must remain usable without the production Flask API,
Flask-AppBuilder views, JWT manager, biometric engines, behavioral ML engines,
post-quantum cryptography libraries, federated mesh infrastructure, or
neuromorphic processors. Those systems remain adapter boundaries. Local package
proof focuses on deterministic identity governance, tenant isolation, approval
state, role/session/access decisions, and composition behavior.

## Users And Outcomes

- Application builders can compose identity, RBAC, session, and privacy-budget
  workflows without importing the production web stack.
- Security owners can require explicit approval before administrative roles are
  assigned.
- Reviewers can approve or reject role-assignment requests with independent
  reviewer evidence.
- Privacy reviewers can approve or reject privacy-budget override requests with
  independent reviewer evidence instead of trusting caller-supplied booleans.
- Runtime services can deny locked accounts, untrusted federation, high-risk
  sessions without step-up, and privileged access without MFA.
- Data and privacy teams can meter privacy-preserving analytics against
  tenant-scoped privacy budgets.
- Security owners can register AI security agents for identity, role, session,
  privacy, and federation review support while preserving runtime, role, scope,
  disclosure, and audit evidence.
- Generated APG applications can compose AUTH with AUDL, MTEN, KEYM, NTFY,
  SECU, MFAU, BIOM, AICR, and downstream ERP capabilities.

## Domain Model

AUTH owns these package-level records:

- `AuthIdentity`: tenant identity with status, memberships, MFA posture,
  trust score, biometric posture, quantum-key posture, and privacy budget.
- `AuthRole`: tenant role definition with permissions, tier, and status.
- `AuthRoleAssignmentApproval`: independent approval request and decision for
  assigning privileged roles.
- `AuthRoleAssignment`: active role assignment after guardrail enforcement.
- `AuthSession`: tenant session with device, federation, MFA, risk, step-up,
  trust, and status evidence.
- `AuthAccessDecision`: deterministic authorization decision with matched rules
  and role evidence.
- `AuthPrivacyQuery`: privacy-budget query decision and remaining budget.
- `AuthPrivacyBudgetApproval`: independent approval request and decision for
  privacy-budget exhaustion overrides.
- `AuthSecurityAgent`: governed AI security-agent registration with tenant,
  runtime, role, owner, purpose, scope, disclosure, human approval, policy, and
  status evidence.
- `AuthAuditEvent`: tenant-scoped governance event for identity, role, session,
  access, privacy, and approval lifecycle changes.

All mutable package-level state must be tenant-qualified so duplicate IDs in
different tenants cannot collide.

## Lifecycle

The focused lifecycle is:

1. Register tenant identities with membership, MFA, risk, and privacy posture.
2. Define tenant roles and permissions.
3. Request role-assignment approval before assigning administrative roles.
4. Approve or reject role-assignment requests with an independent reviewer and
   notes.
5. Assign roles only when the role, identity, tenant, and approval guardrails
   allow it.
6. Start sessions only for valid tenant members, trusted federation issuers,
   unlocked accounts, and acceptable risk posture.
7. Evaluate access using tenant role assignments, session evidence, MFA, risk,
   role permissions, and inferred privileged tiers when callers omit the tier.
8. Request and decide privacy-budget approval before authorizing analytics that
   exceed the current tenant-local budget.
9. Run privacy analytics only when privacy budget exists or an approved review
   authorizes budget exhaustion.
10. Revoke sessions by tenant-local session ID and list current tenant identity,
   role, session, decision,
   privacy, approval, and audit state.
11. Register security agents only when they use a supported runtime, supported
    role, accountable owner, declared purpose, explicit scope, registration
    state, contribution disclosure, and human approval for privileged roles.
12. Validate batch AUTH mutation intent against the Bytewax lifecycle stream.
13. Emit tenant-scoped audit events for identity, role, approval, assignment,
    session, access, privacy, and revocation lifecycle changes.

## Rules And Guardrails

The contract rules are executable guardrails:

- `locked_accounts_denied`: locked or suspended accounts cannot authenticate.
- `privileged_access_requires_mfa`: privileged access requires MFA evidence.
- `high_risk_sessions_require_step_up`: high-risk sessions require step-up.
- `elevated_role_assignment_requires_approval`: administrative role
  assignments require approved approval evidence.
- `role_assignment_approval_requires_independent_reviewer`: approval reviewers
  cannot approve their own role-assignment requests.
- `untrusted_federation_denied`: federated logins require trusted issuers.
- `cross_tenant_access_requires_membership`: cross-tenant access requires
  confirmed tenant membership.
- `privacy_queries_require_budget`: privacy analytics budget exhaustion requires
  review.
- `privacy_budget_approval_requires_independent_reviewer`: privacy-budget
  reviewers cannot approve their own requests.
- `security_agent_requires_registration`: AI security agents must be
  registered before use in AUTH review workflows.
- `security_agent_runtime_supported`: AI security agents must use a supported
  runtime: `codex`, `claude_code`, `opencode`, or `pi`.
- `security_agent_role_supported`: AI security agents must use a supported
  review role.
- `security_agent_requires_scope`: AI security agents must declare explicit
  operating scope.
- `security_agent_requires_disclosure`: AI-assisted contributions must be
  disclosed.
- `security_agent_privileged_role_requires_human_approval`: privileged AI
  security-agent roles require explicit human approval.
- `auth_state_change_requires_audit`: AUTH lifecycle state changes require
  audit evidence.
- `batch_auth_mutation_requires_bytewax`: batch AUTH mutation intent must use
  Bytewax event streams.

Service methods must enforce these rules and expose the same decisions through
API helpers and view models.

Approval, review, and assignment actors must be real tenant actors with the
required `auth:manage_roles`, `auth:approve_roles`, `auth:manage_privacy`, or
`auth:approve_privacy` permissions. The only bootstrap exception is the
platform actor `system`.

## UI And Theme

AUTH exposes route and view-model surfaces for:

- login;
- trust dashboard;
- role workbench;
- role approval queue;
- privacy approval queue;
- session center;
- access decision console;
- biometric assurance;
- quantum key posture;
- behavioral analysis;
- privacy analytics;
- federation console;
- security-agent panel;
- audit trail;
- analytics dashboard;
- metrics and audit evidence.

The `auth_trust_fabric` theme must provide semantic tokens and component
metadata for identity signal cards, risk posture meters, role assignment
timelines, approval queues, session trust badges, access decision panels, and
privacy budget meters.

## Security-Agent Composition

AUTH supports first-class AI security-agent composition without directly
invoking agent CLIs or provider SDKs. The dependency-light package records the
agent registration and validates guardrails. Production adapters can later bind
these registrations to Codex, Claude Code, OpenCode, Pi, or other approved
runtimes through platform orchestration.

Supported roles are:

- `identity_reviewer`;
- `role_reviewer`;
- `session_reviewer`;
- `privacy_reviewer`;
- `federation_reviewer`.

Generated applications must display agent scope and contribution disclosure in
review surfaces so human approvers can distinguish agent-assisted summaries
from direct reviewer decisions.

The capability contract also publishes an `agents` manifest with
`first_class: true`, supported runtimes, supported roles, privileged roles,
composition points, and guardrails. This is the APG composition surface for
Codex, Claude Code, OpenCode, Pi, and future approved agent runtimes.

## Streaming

AUTH publishes Bytewax lifecycle stream metadata through the capability
contract and generated semantic model:

- engine and processor: `bytewax`;
- topic: `apg.auth.lifecycle`;
- topics: `auth.identities`, `auth.roles`, `auth.sessions`, `auth.privacy`,
  and `auth.agents`;
- state collections: identities, roles, role approvals, assignments, sessions,
  access decisions, privacy queries, privacy approvals, security agents, and
  audit events;
- events: identity registration, role definition, approval decisions, role
  assignment, session start/revocation, access evaluation, privacy decisions,
  and security-agent registration;
- batch mutation guardrail: `batch_auth_mutation_requires_bytewax`.

## Adapter Boundaries

These integrations remain replaceable:

- production Flask and Flask-AppBuilder stacks;
- JWT token issuers and session stores;
- behavioral ML engines and biometric fusion engines;
- post-quantum cryptography and zero-knowledge proof providers;
- federated identity mesh adapters;
- neuromorphic decision processors;
- audit, notification, key management, multi-tenancy, and security services.
- live Bytewax topology execution.

Local package tests must not require those systems.

## Acceptance Gates

- Contract validation passes.
- The dependency-light service runs the full identity, role-approval, role
  assignment, session, access, privacy, and audit lifecycle.
- Administrative role assignment fails without an approved matching approval.
- Approval decisions fail when the reviewer is missing, notes are missing, or
  the reviewer is the requester.
- Privacy-budget exhaustion cannot be bypassed by caller-supplied booleans; it
  requires approved matching approval evidence.
- Privacy-budget mutation requires a tenant-local identity budget, even when a
  user has cross-tenant membership for non-budgeted access flows.
- Privileged permission checks infer MFA requirements from admin permissions
  and assigned privileged roles when callers omit `requested_permission_tier`.
- Cross-tenant privacy queries fail unless a tenant-local privacy-budget
  identity exists for the user.
- Tenant-local duplicate session IDs can be revoked by tenant without affecting
  the same session ID in another tenant.
- Tenant-qualified state allows duplicate IDs across tenants without collision.
- API helpers and view models expose the same lifecycle state.
- Security agents can be registered with supported runtime, supported role,
  owner, purpose, scope, disclosure, human approval, and policy evidence.
- Privileged security-agent roles fail closed when human approval is not
  required.
- Unsupported security-agent runtime, missing scope, or undisclosed agent
  contribution fails closed.
- Batch AUTH mutation validation accepts Bytewax and denies other stream
  providers.
- Generated semantic model exposes the current login/dashboard route names,
  security-agent route, provides/requires metadata, and Bytewax stream metadata.
- Publish-plan and implementation-audit checks pass.
- Legacy generated-package naming is removed from package tests.

## Focused Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/auth/models.py capabilities/common/auth/service.py capabilities/common/auth/api_helpers.py capabilities/common/auth/view_models.py capabilities/common/auth/capability_contract.py capabilities/common/auth/app.py capabilities/common/auth/tests/test_capability_contract.py capabilities/common/auth/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/auth/tests/test_capability_contract.py capabilities/common/auth/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.auth import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/auth --json
./.venv/bin/apg capabilities publish-plan capabilities/common/auth --json
```
