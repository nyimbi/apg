# Access Control Integration Hub

## Overview

The Access Control Integration Hub provides unified identity, policy, grant, and session management for the APG composition layer. It federates multiple identity providers (local, OIDC, SAML, LDAP, API key, JWT) behind a single policy engine, enforcing fine-grained resource access across all tenant composition boundaries.

Business value lies in eliminating per-capability access silos: every downstream capability delegates its authorization decisions to this hub, creating a single audit trail, a single revocation point, and a consistent risk-scoring model for adaptive step-up authentication. Privileged grants are subject to separation-of-duties and mandatory expiry, making over-entitlement visible and time-bounded.

## Capability ID

`composition_access`  Version: see `package_manifest.json`

## Provides

| Service | Description |
|---------|-------------|
| identity_provider_composition | Register, validate, and activate external and local identity providers |
| resource_access_registry | Catalogue protected resources with scopes and ownership |
| policy_orchestration | Create, simulate, and activate access policies with condition gates |
| grant_lifecycle | Issue, approve, and revoke access grants with justification and expiry |
| session_risk_control | Continuously score session risk and trigger adaptive step-up auth |
| access_decision_audit | Record every authZ decision and emit it to the Bytewax event stream |
| access_agents | AI agent workbench with approved runtimes and human approval gates |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Bootstrap identity for the hub's own API and admin users |
| audl | Persist immutable audit records outside the hub's own store |
| ntfy | Send approval request and decision notifications |
| conf | Read runtime configuration and secret references |
| registry | Register this capability in the global catalog |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant scope for all operations |
| identity_providers.supported_types | list | ["local","oidc","saml","ldap","api_key","jwt"] | Allowed provider types |
| identity_providers.provider_owner_required | bool | true | Provider must have an accountable owner |
| identity_providers.secret_reference_required | bool | true | External providers must use vault references |
| sessions.max_risk_without_review | int | 74 | Risk score above which step-up auth is mandatory |
| sessions.risk_scoring_enabled | bool | true | Enable continuous session risk scoring |
| sessions.adaptive_step_up_enabled | bool | true | Trigger step-up on high-risk sessions |
| access_agents.max_autonomous_scope | string | "read_and_recommend" | Ceiling on autonomous agent actions |
| governance.privileged_action_review | bool | true | Require human approval for privileged agent actions |
| observability.event_stream | string | "apg.composition.access.lifecycle" | Bytewax stream name |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /composition-access/dashboard | GET | composition_access:view | Overview |
| providers | /composition-access/providers | GET/POST | composition_access:admin | Identity |
| resources | /composition-access/resources | GET/POST | composition_access:govern | Resources |
| policies | /composition-access/policies | GET/POST | composition_access:govern | Policy |
| grants | /composition-access/grants | GET/POST | composition_access:grant | Access |
| decisions | /composition-access/decisions | GET | composition_access:view | Operations |
| sessions | /composition-access/sessions | GET | composition_access:operate | Operations |
| agents | /composition-access/agents | GET/POST | composition_access:admin | Automation |
| audit | /composition-access/audit | GET | composition_access:audit | Governance |
| settings | /composition-access/settings | GET/PUT | composition_access:admin | Administration |

REST API prefix: `/composition-access/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny |
| provider_requires_owner | register_provider without owner | deny |
| provider_requires_metadata_evidence | activate_provider without validated metadata | deny |
| provider_requires_secret_reference | activate external provider without secret ref | deny |
| resource_requires_owner | register_resource without owner | deny |
| resource_requires_scope | register_resource without scope | deny |
| policy_requires_owner | create_policy without owner | deny |
| sensitive_policy_requires_conditions | create_policy on sensitive resource without conditions | deny |
| high_risk_policy_requires_simulation | activate_policy at risk_level=high without simulation evidence | require_review |
| privileged_grant_requires_approval | create privileged grant without approval | deny |
| privileged_grant_requires_expiry | create privileged grant without expiry | deny |
| grant_requires_separation_of_duties | requester self-approves | deny |
| grant_requires_justification | create_grant without justification | deny |
| high_risk_session_requires_step_up | session risk_score > 74 without step-up completed | deny |
| decision_requires_bytewax_stream | record_decision not via bytewax | deny |
| batch_grant_requires_bytewax | batch_grant not via bytewax | deny |
| access_agent_runtime_supported | register_access_agent with unsupported runtime | deny |
| access_agent_role_supported | register_access_agent with unsupported role | deny |
| privileged_agent_action_requires_human_approval | agent proposes privileged action without human approval | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| AccessProviderRecord | id, tenant_id, name, provider_type, owner_id, status, external, metadata_validated, secret_reference |
| AccessResourceRecord | id, tenant_id, resource_key, display_name, owner_id, scopes, capability_id, sensitive, status |
| AccessPolicyRecord | id, tenant_id, name, resource_id, owner_id, effect, conditions, risk_level, status, simulation_evidence |
| AccessGrantRecord | id, tenant_id, subject_id, resource_id, scopes, requested_by, justification, privileged, approved_by, expires_at |
| AccessSessionRecord | id, tenant_id, subject_id, provider_id, risk_score, status, step_up_completed, evaluated_at |
| AccessDecisionRecord | id, tenant_id, subject_id, resource_id, action, decision, reason, policy_ids, event_stream |
| AccessAgentRecord | id, tenant_id, name, runtime, role, instructions, status |
| AccessAuditEventRecord | id, tenant_id, event_type, entity_id, actor_id, created_at |

All models are lightweight dataclasses (no ORM dependency at the record layer) with deterministic `stable_id` generation via SHA-256 truncated to 16 hex chars.

## Streaming Events

Events emitted to the composition event stream via Bytewax (`apg.composition.access.lifecycle`).

| Event | Trigger |
|-------|---------|
| provider_registered | Identity provider record created |
| provider_activated | Provider moves to active status |
| resource_registered | Protected resource added to registry |
| policy_created | Access policy record created |
| policy_activated | Policy moves from draft to active |
| grant_created | Access grant issued |
| grant_revoked | Grant explicitly revoked |
| session_evaluated | Session risk score computed |
| access_decision_recorded | AuthZ decision emitted |
| access_agent_registered | New access-control agent registered |

Stream states: `draft → active → review_required → approved → denied → revoked → blocked`

## Edge Cases Handled

- A requester cannot self-approve their own privileged grant (`grant_requires_separation_of_duties`); the rule checks `separation_of_duties_passed: False` in the evaluation context, so the requester must explicitly pass this flag as true after selecting an independent approver.
- High-risk policies block activation until simulation evidence is attached, preventing untested deny-all rules from locking out tenants before the policy is understood.
- Session risk scores above `max_risk_without_review` (default 74, configurable per-tenant) block further operations until step-up authentication completes, not just trigger a warning.
- Batch grant operations must route through Bytewax; this prevents silent bulk privilege escalations that would bypass the event audit trail entirely.
- External identity providers that lack a vault/secret-manager reference cannot be activated, preventing plaintext credential storage at the provider record level.
- The `_matches` rule evaluator supports `_lte`, `_lt`, `_gte`, `_gt`, `_ne` key suffixes for numeric and inequality comparisons, allowing threshold-based rules without additional code.

## Composability

- **Upstream**: `auth` (bootstrap identity), `conf` (secrets and runtime config), `registry` (self-registration at startup)
- **Downstream**: All other composition capabilities (`config`, `events`, `gateway`, `orchestration`, `registry`) delegate their authorization checks here; every write operation in those capabilities requires a policy attached through this hub
- **Peer**: `audl` (receives decision audit records for long-term retention), `ntfy` (sends approval notifications for grants and agent actions)

## Development Notes

- The rule engine is fully deterministic and side-effect-free; `evaluate_capability_rules` can be called in tests with arbitrary context dicts without touching the database.
- Record models use `stable_id(prefix, *parts)` for reproducible IDs on replay; this is intentional for idempotent event sourcing, not a security primitive.
- The `max_autonomous_scope` for access agents is set to `"read_and_recommend"` — more conservative than other capabilities — because access-control mistakes have the broadest blast radius across the composition layer.
- Key files: `capability_contract.py` (executable contract and rule engine), `models.py` (dataclass records), `service.py` (lifecycle operations), `api.py` (API helpers), `views.py` (UI model helpers), `app.py` (package self-test and semantic model).
