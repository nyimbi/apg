# AUTH Capability Development Plan

## Current State

AUTH has a rich production-oriented authentication module with advanced
behavioral, biometric, quantum-safe, federation, privacy, Flask API, and
Flask-AppBuilder surfaces. It also has a dependency-light `AuthService`,
capability contract, package evidence, and contract tests.

The package-level composition gap is that privileged role assignment and
privacy-budget exhaustion can be represented as caller-supplied booleans instead
of first-class approval state, session revocation is not tenant-addressable, and
the dependency-light state stores are not fully tenant-qualified. Generated APG
applications need a fail-closed, composable lifecycle for identity, role
approval, privacy approval, assignment, session, access, privacy, view models,
and audit evidence.

## Packet 1: Governed Role, Privacy, And Access Lifecycle

Deliver a focused lifecycle packet:

- add package-level role-assignment approval state;
- add package-level privacy-budget approval state;
- key mutable service stores by tenant plus record ID;
- request role-assignment approval with requester and justification evidence;
- approve or reject role-assignment requests with independent reviewer notes;
- assign administrative roles only with approved matching approval evidence;
- request and decide privacy-budget approvals with independent reviewer notes;
- complete budget-exhausted privacy queries only with matching approved
  privacy-budget approval evidence;
- require real tenant actors with role-backed permissions before requesting,
  deciding, or applying approval-governed changes;
- infer privileged MFA requirements from admin permissions and assigned
  privileged roles when access callers omit a permission tier;
- mutate privacy budgets only on tenant-local identity records;
- revoke tenant-local sessions by tenant without impacting duplicate session IDs
  in other tenants;
- preserve identity, role, session, access, privacy, and audit lifecycles;
- add API-helper and view-model surfaces for generated APG applications;
- update contract routes, rules, theme metadata, semantic evidence, and release
  evidence;
- rename generated-package tests to package contract naming;
- update package documentation and progress evidence.

## Implementation Steps

1. Extend `models.py` with `AuthRoleAssignmentApproval` and
   `AuthPrivacyBudgetApproval`.
2. Update `service.py` so identities, roles, approvals, assignments, sessions,
   decisions, privacy queries, privacy approvals, and audit events are
   tenant-qualified.
3. Add role approval request/decision methods and enforce approved evidence in
   `assign_role`.
4. Add privacy-budget approval request/decision methods and enforce approved
   evidence in `run_privacy_query`.
5. Add tenant-scoped session revocation for duplicate tenant-local session IDs.
6. Add actor permission checks for role approval, role assignment,
   privacy-budget approval, and privacy override decisions.
7. Infer privileged access tier for admin permissions and assigned privileged
   roles.
8. Restrict privacy-budget mutation to tenant-local identity records.
9. Add `api_helpers.py` for dependency-light generated application calls.
10. Add `view_models.py` for trust dashboard, role workbench, approval queue,
   session center, access console, privacy center, and audit surfaces.
11. Update `capability_contract.py` with role-approval and privacy-approval
   routes, independent reviewer rules, and theme components.
12. Update registration metadata with role-approval and privacy-review
   capabilities, endpoints, and permissions.
13. Replace stale embedded semantic evidence in `app.py` with contract-derived
   evidence.
14. Extend package tests with positive identity-role-approval-assignment-session
   access-privacy approval coverage and negative direct-admin-assignment,
   rejected approval, self-approval, missing notes, tenant-mismatch,
   cross-tenant privacy, raw boolean bypass, duplicate-ID isolation,
   API-helper, and view-model coverage.
15. Rename generated-package tests to package contract naming.
16. Update `cap_spec.md` with the current executable lifecycle and proof
    commands.
17. Run focused package proof, implementation audit, publish-plan, review, and
    diff checks.

## Review Checklist

- Identity, role, approval, assignment, session, decision, privacy, and audit
  state is tenant-qualified.
- Administrative role assignment requires approved matching approval evidence.
- Approval reviewers cannot approve their own requests.
- Approval decisions require reviewer identity and notes.
- Locked users, missing MFA, high-risk sessions, untrusted federation, and
  tenant mismatches fail closed.
- Privacy budget exhaustion creates review-required state unless review is
  approved with matching privacy-budget approval evidence.
- Caller-supplied booleans do not bypass privileged role assignment or
  budget-exhausted privacy queries.
- Approval and assignment actors must exist and hold the required tenant
  permissions.
- Admin permissions require MFA even when callers omit `requested_permission_tier`.
- Cross-tenant privacy queries fail unless a tenant-local budget identity
  exists.
- Tenant-local session revocation does not affect duplicate session IDs in
  other tenants.
- API helpers expose the same behavior as service methods.
- View models expose dashboard, role, approval, session, access, privacy,
  federation, theme, and audit state.
- Production JWT, biometric, behavioral, quantum, federation, and web-server
  stacks remain adapter boundaries.
