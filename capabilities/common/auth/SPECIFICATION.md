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
11. Emit tenant-scoped audit events for identity, role, approval, assignment,
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
- metrics and audit evidence.

The `auth_trust_fabric` theme must provide semantic tokens and component
metadata for identity signal cards, risk posture meters, role assignment
timelines, approval queues, session trust badges, access decision panels, and
privacy budget meters.

## Adapter Boundaries

These integrations remain replaceable:

- production Flask and Flask-AppBuilder stacks;
- JWT token issuers and session stores;
- behavioral ML engines and biometric fusion engines;
- post-quantum cryptography and zero-knowledge proof providers;
- federated identity mesh adapters;
- neuromorphic decision processors;
- audit, notification, key management, multi-tenancy, and security services.

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
- Publish-plan and implementation-audit checks pass.
- Legacy generated-package naming is removed from package tests.
