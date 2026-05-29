# User Management Capability Specification

- **Capability Name**: User Management
- **Capability ID**: `usrm`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package provides the executable APG runtime for `usrm`.
It gives composed applications a deterministic user lifecycle surface for
unique identity creation, profile and privacy preference management, consented
invitations, role assignment, privileged MFA enforcement, access reviews,
deprovisioning with access-revocation evidence, bulk-action review, UI route
metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `user_directory`
- `profile_management`
- `consented_invitations`
- `role_assignment_governance`
- `access_review_workflows`
- `deprovisioning_governance`
- `user_audit_events`

## Required Services

- `tenant_context`
- `authentication_rbac`
- `multi_factor_authentication`
- `consent_management`
- `audit_sink`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `user_requires_identity`
- `invite_requires_consent_notice`
- `privileged_user_requires_mfa`
- `deprovision_requires_access_revocation`
- `bulk_user_action_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

The view helpers provide dashboard, user directory, profile manager, lifecycle
queue, access review, privacy preferences, deprovisioning, and settings models.

## Theme

The package uses the `usrm_user_lifecycle` APG theme contract.

## Runtime Behavior

`UsrmService` is intentionally dependency-light so it can run inside generated
applications, tests, and publish-plan probes without external infrastructure.
It supports:

- `create_user()` for tenant-scoped users with unique identity, owner,
  validated profile, privileged-user, MFA, and manager metadata.
- `update_profile()` for profile attributes, privacy preferences, and consent
  notice references.
- `invite_user()` for consented user invitations.
- `assign_role()` for audited role assignments with privileged-MFA guardrails.
- `record_access_review()` for periodic access review evidence.
- `deprovision_user()` for access-revocation-backed user deprovisioning.
- `bulk_suspend_users()` for bulk lifecycle actions with review thresholds.
- `dashboard_summary()` and list helpers for API and UI composition.

## Adapter Boundaries

The in-package runtime stores records in memory by design. Production adapters
are expected to bind durable identity stores, RBAC/MFA providers, consent
registries, access-review workflow engines, deprovisioning automation, and audit
sinks at the APG composition layer without changing the deterministic package
contract.

## Focused Verification

- `./.venv/bin/python -m py_compile capabilities/common/usrm/__init__.py capabilities/common/usrm/models.py capabilities/common/usrm/user_runtime.py capabilities/common/usrm/service.py capabilities/common/usrm/api.py capabilities/common/usrm/views.py capabilities/common/usrm/capability_contract.py capabilities/common/usrm/app.py capabilities/common/usrm/test_capability_contract.py capabilities/common/usrm/tests/test_package_contract.py`
- `./.venv/bin/pytest -q capabilities/common/usrm/test_capability_contract.py capabilities/common/usrm/tests/test_package_contract.py`
- `./.venv/bin/apg capabilities implementation-audit --root capabilities/common/usrm --json`
- `./.venv/bin/apg capabilities publish-plan capabilities/common/usrm --json`
