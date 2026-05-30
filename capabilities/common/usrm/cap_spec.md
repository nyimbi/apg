# User Management Capability Packet

- Capability Name: User Management
- Capability ID: `usrm`
- Category: common
- Version: 1.0.0

## Purpose

USRM provides executable APG user lifecycle, profile, invitation, access,
privacy, deprovisioning, agent-review, audit, and Bytewax stream behavior. It
lets generated applications compose user directories, profile management,
consented invitations, role assignment governance, access reviews, privileged
MFA checks, deprovision evidence, bulk lifecycle review, and first-class
AI-assisted review lanes.

## Provides

- `user_directory`
- `profile_management`
- `consented_invitations`
- `role_assignment_governance`
- `access_review_workflows`
- `deprovisioning_governance`
- `user_audit_events`
- `usrm_agents`

## Requires

- `auth`
- `mfau`
- `cons`
- `audl`
- `idfd`

## Configuration Areas

USRM configuration is defined by `capability_contract.py` and covers:

- tenant context;
- user identity, owner, profile validation, and status history;
- invitation consent, deprovision evidence, manager approval, and bulk-action review;
- privileged MFA, privileged role approval, periodic access review, and least privilege;
- first-class user-management agent runtimes, roles, and human approval;
- audit, privacy preference sync, and identity federation governance;
- Bytewax lifecycle-stream observability;
- adapter boundaries for identity, authorization, MFA, consent, audit, and event streaming;
- UI route toggles and theme tokens.

## Lifecycle

USRM supports the following lifecycle:

1. Create a tenant user with unique identity, owner, and profile validation.
2. Update profile attributes and privacy preferences with consent and sync evidence.
3. Invite the user through a consented channel and Bytewax event metadata.
4. Assign roles with privileged MFA and approval guardrails.
5. Record periodic access review evidence.
6. Deprovision users only after access revocation, evidence, and Bytewax stream checks.
7. Coordinate bulk lifecycle actions with review and Bytewax stream gates.
8. Register governed AI agents that review identity, lifecycle, access, privacy, entitlement, and deprovision evidence.

## Deterministic Rules

- `tenant_context_required`
- `user_requires_identity`
- `user_requires_owner`
- `user_requires_profile_validation`
- `invite_requires_consent_notice`
- `invite_requires_bytewax_stream`
- `profile_requires_privacy_sync`
- `privileged_user_requires_mfa`
- `privileged_role_requires_approval`
- `access_review_requires_reviewer`
- `deprovision_requires_access_revocation`
- `deprovision_requires_evidence`
- `deprovision_requires_bytewax_stream`
- `bulk_user_action_requires_review`
- `bulk_user_action_requires_bytewax`
- `usrm_agent_runtime_supported`
- `usrm_agent_role_supported`
- `privileged_agent_user_action_requires_human_approval`

## UI

USRM exposes APG Python view models for dashboard, user directory, profile
manager, lifecycle queue, access review, privacy preferences, deprovisioning,
agent workbench, policy center, and settings.

## Theme

USRM uses the `usrm_user_lifecycle` theme with compact density, user cards,
status pills, access bands, lifecycle approval lists, stage chips, entitlement
matrices, MFA chips, consent chips, review lanes, and guardrail chips.

## Streaming

USRM lifecycle events are described by the Bytewax stream manifest:

- processor: `bytewax`
- stream: `apg.usrm.lifecycle`
- key: `tenant_id`
- events: `user_created`, `profile_updated`, `user_invited`, `role_assigned`,
  `access_review_recorded`, `user_deprovisioned`, `bulk_suspend_users`,
  `usrm_agent_registered`

## Adapter Boundaries

The in-package service is dependency-light and stores records in memory for
generated apps, tests, and publish-plan probes. Production deployments should
bind durable identity stores, RBAC providers, MFA providers, consent
registries, access-review workflows, deprovisioning automation, audit sinks,
and Bytewax workers through APG adapters without weakening the deterministic
contract.

## Focused Verification

- `./.venv/bin/python -m py_compile capabilities/common/usrm/__init__.py capabilities/common/usrm/models.py capabilities/common/usrm/user_runtime.py capabilities/common/usrm/service.py capabilities/common/usrm/api.py capabilities/common/usrm/views.py capabilities/common/usrm/capability_contract.py capabilities/common/usrm/app.py capabilities/common/usrm/test_capability_contract.py capabilities/common/usrm/tests/test_package_contract.py`
- `./.venv/bin/pytest -q capabilities/common/usrm/test_capability_contract.py capabilities/common/usrm/tests/test_package_contract.py`
- `./.venv/bin/apg capabilities implementation-audit --root capabilities/common/usrm --json`
- `./.venv/bin/apg capabilities publish-plan capabilities/common/usrm --json`
