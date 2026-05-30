# USRM Capability Specification

## Identity

- Capability name: User Management
- Capability ID: `usrm`
- Category: common
- Runtime target: APG Python capability package

## Mission

USRM gives generated APG applications a governed user lifecycle. It coordinates
tenant user records, profile data, consented invitations, role assignments,
privileged MFA, access reviews, privacy preference sync, deprovision evidence,
bulk lifecycle actions, AI-assisted review, audit events, and Bytewax lifecycle
streaming.

## Functional Scope

USRM owns the executable lifecycle for:

- unique tenant user identity creation;
- user owner and profile validation evidence;
- profile attributes and privacy preferences;
- consent-backed invitations;
- role assignment and privileged role approval;
- privileged MFA enforcement;
- periodic access reviews;
- access-revocation-backed deprovisioning;
- bulk lifecycle action review;
- first-class user-management agents for identity, lifecycle, access,
  deprovision, privacy, and entitlement review;
- user lifecycle audit and dashboard evidence.

## Configuration Contract

The configuration schema requires:

- `tenant_id`
- `users`
- `lifecycle`
- `access`
- `usrm_agents`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

USRM must expose these through `get_capability_contract()`, generated semantic
model evidence, and package registration metadata.

## Domain Records

### User

A user contains tenant, identity, display name, email, owner, status, profile
validation, privileged flag, MFA flag, manager, and timestamps.

### Profile

A profile contains tenant, user, attributes, privacy preferences, consent notice
reference, updater, and timestamp.

### Invitation

An invitation contains tenant, user, channel, consent notice reference, inviter,
status, and timestamp.

### Role Assignment

A role assignment contains tenant, user, role, scope, privileged flag, approver,
status, and timestamp.

### Access Review

An access review contains tenant, user, reviewer, decision, findings, status,
and timestamp.

### Deprovision Record

A deprovision record contains tenant, user, actor, access revocation evidence,
status, matched rules, required actions, and timestamp.

### Bulk User Action

A bulk user action contains tenant, action, actor, affected users, status,
matched rules, required actions, and timestamp.

### USRM Agent

A USRM agent is a first-class composition element with tenant, name, runtime,
role, scope, owner, status, and human approval policy.

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `identity_reviewer`
- `lifecycle_reviewer`
- `access_reviewer`
- `deprovision_reviewer`
- `privacy_reviewer`
- `entitlement_reviewer`

## Lifecycle States

User states:

- `active`
- `invited`
- `suspended`
- `review_required`
- `deprovisioned`

Lifecycle stream states:

- `active`
- `invited`
- `suspended`
- `review_required`
- `deprovisioned`
- `blocked`

## Rules

The deterministic rule engine must enforce:

- tenant context on all executable operations;
- unique identity on user creation;
- owner on user creation;
- profile validation on user creation;
- consent notice on invitation;
- Bytewax stream metadata on invitation;
- privacy preference sync on profile updates;
- MFA for privileged users and privileged roles;
- approval for privileged role assignment;
- reviewer attribution on access reviews;
- access revocation and evidence for deprovisioning;
- Bytewax stream metadata for deprovisioning;
- review for bulk user actions above threshold;
- Bytewax stream coordination for bulk lifecycle actions;
- approved USRM-agent runtimes;
- approved USRM-agent roles;
- human approval for privileged agent user actions.

## Service Requirements

`UsrmService` must provide:

- `describe()`
- `evaluate()`
- `create_user()`
- `update_profile()`
- `invite_user()`
- `assign_role()`
- `record_access_review()`
- `deprovision_user()`
- `bulk_suspend_users()`
- `register_usrm_agent()`
- `validate_agent_user_action()`
- `validate_batch_user_lifecycle()`
- list helpers for every record type;
- `dashboard_summary()`.

## API Requirements

`api.py` must expose payload-oriented helpers for status, user creation,
profile updates, invitations, role assignment, access review, deprovisioning,
bulk actions, agents, agent-action validation, batch lifecycle validation,
compatibility record creation, and system listing.

## UI Requirements

USRM exposes APG Python view models for:

- `/usrm/dashboard`
- `/usrm/users`
- `/usrm/profiles`
- `/usrm/lifecycle`
- `/usrm/access`
- `/usrm/privacy`
- `/usrm/deprovisioning`
- `/usrm/agents`
- `/usrm/policy`
- `/usrm/settings`

The UI contract must expose rules, summaries, agent policy, Bytewax streaming,
and visual theme tokens.

## Visual Theming

The default visual theme is `usrm_user_lifecycle`. It defines compact density,
status pills, access bands, approval lists, stage chips, entitlement matrices,
MFA chips, consent chips, review lanes, and guardrail chips.

## Streaming

USRM lifecycle events use Bytewax:

- processor: `bytewax`
- stream: `apg.usrm.lifecycle`
- key: `tenant_id`

Events:

- `user_created`
- `profile_updated`
- `user_invited`
- `role_assigned`
- `access_review_recorded`
- `user_deprovisioned`
- `bulk_suspend_users`
- `usrm_agent_registered`

## Adapter Boundaries

The in-package runtime must stay dependency-light. Production deployments bind
identity stores, RBAC providers, MFA providers, consent registries,
access-review workflow systems, deprovisioning automation, audit sinks, and
Bytewax workers through adapters.

## Acceptance Criteria

- README, specification, plan, and capability summary exist.
- Contract shape validates.
- Generated app evidence is refreshed from the contract.
- Tests cover contract, rules, service, API, views, agent guardrails, and
  Bytewax guardrails.
- Focused package tests pass.
- Implementation audit reports domain-specific behavior with no warnings.
- Publish plan reports side-effect-free output with no warnings.
