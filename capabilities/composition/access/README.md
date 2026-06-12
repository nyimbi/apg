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
| grant_lifecycle | Issue, approve, suspend, reinstate, and revoke access grants with justification and expiry |
| session_risk_control | Continuously score session risk and trigger adaptive step-up auth |
| access_decision_audit | Record every authZ decision and emit it to the Bytewax+NATS event stream |
| access_agents | AI agent workbench with approved runtimes and human approval gates |
| permission_matrix_export | Point-in-time permission snapshot across all subjects and resources |
| policy_simulation | Sandbox policy changes against historical decisions before activation |
| jit_privileged_access | Just-In-Time bounded-window privileged grants with auto-expiry |
| rbac_with_inheritance | Role-based access with parent-role scope inheritance |
| self_service_access_requests | Traceable request → approve/deny → grant workflow |
| grant_expiry_reaper | Automated sweep of expired grants with audit trail |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Bootstrap identity for the hub's own API and admin users |
| audl | Persist immutable audit records outside the hub's own store |
| ntfy | Send approval request and decision notifications |
| conf | Read runtime configuration and secret references |
| registry | Register this capability in the global catalog |
| composition_events | Emit lifecycle events to Bytewax+NATS pipeline |
| moni | Expose service health and latency metrics |

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
| grants.max_grant_duration_days | int | 90 | Hard ceiling on grant lifetime |
| grants.periodic_review_required | bool | true | Quarterly access review requirement |
| access_agents.max_autonomous_scope | string | "read_and_recommend" | Ceiling on autonomous agent actions |
| governance.privileged_action_review | bool | true | Require human approval for privileged agent actions |
| circuit_breaker.enabled | bool | true | Enable circuit breaker state machine |
| circuit_breaker.failure_threshold | int | 5 | Failures before tripping the breaker |
| observability.event_stream | string | "apg.composition.access.lifecycle" | Bytewax+NATS stream name |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /composition-access/dashboard | GET | composition_access:view | Overview |
| providers | /composition-access/providers | GET/POST | composition_access:admin | Identity |
| resources | /composition-access/resources | GET/POST | composition_access:govern | Resources |
| policies | /composition-access/policies | GET/POST | composition_access:govern | Policy |
| policy_simulate | /composition-access/policies/{id}/simulate | POST | composition_access:govern | Policy |
| grants | /composition-access/grants | GET/POST | composition_access:grant | Access |
| jit_grants | /composition-access/grants/jit | POST | composition_access:grant | Access |
| access_requests | /composition-access/access-requests | GET/POST | composition_access:view | Access |
| roles | /composition-access/roles | GET/POST | composition_access:admin | RBAC |
| decisions | /composition-access/decisions | GET | composition_access:view | Operations |
| sessions | /composition-access/sessions | GET | composition_access:operate | Operations |
| permission_matrix | /composition-access/permission-matrix | GET | composition_access:audit | Governance |
| agents | /composition-access/agents | GET/POST | composition_access:admin | Automation |
| audit | /composition-access/audit | GET | composition_access:audit | Governance |
| settings | /composition-access/settings | GET/PUT | composition_access:admin | Administration |

REST API prefix: `/composition-access/api/v1`

## Service Methods

### Core Lifecycle (sync)

| Method | Description |
|--------|-------------|
| `register_provider(...)` | Create an identity provider record in draft status |
| `activate_provider(...)` | Promote provider to active after metadata validation |
| `register_resource(...)` | Catalogue a protected resource with scopes |
| `create_policy(...)` | Create an access policy tied to a resource |
| `activate_policy(...)` | Promote policy to active after simulation gate |
| `create_grant(...)` | Issue an access grant to a subject |
| `revoke_grant(...)` | Permanently revoke a grant |
| `evaluate_session(...)` | Score session risk and compute status |
| `record_decision(...)` | Record and stream an authZ decision |
| `register_access_agent(...)` | Register an AI agent with an approved runtime and role |
| `validate_agent_access_action(...)` | Gate an agent action against privilege rules |
| `validate_batch_grant(...)` | Validate a batch grant operation via Bytewax stream |

### Extended Async Methods

| Method | Description |
|--------|-------------|
| `rotate_secret(provider_id, actor_id, new_secret_reference)` | Rotate vault secret reference on an active provider |
| `suspend_grant(grant_id, actor_id, reason)` | Temporarily suspend a grant without revoking it |
| `reinstate_grant(grant_id, actor_id)` | Reinstate a suspended grant |
| `bulk_revoke_grants(grant_ids, actor_id, reason)` | Revoke multiple grants atomically |
| `check_access(tenant_id, subject_id, resource_id, action, scope)` | Real-time access check against active grants |
| `export_access_log(tenant_id, format)` | Export decision log as JSON or CSV |
| `access_analytics(tenant_id, period)` | Allow/deny rates and top-subject analytics |
| `health_check(tenant_id)` | Service health and count summary |
| `access_compliance_report(tenant_id, standard)` | Compliance report (ISO27001, SOC2) |
| `export_permission_matrix(tenant_id, format)` | Full subject→resource→scope matrix export |
| `simulate_policy(policy_id, sample_decisions, actor_id)` | Sandbox policy against historical decisions |
| `request_jit_grant(...)` | Create a Just-In-Time privileged grant pending approval |
| `approve_jit_grant(grant_id, approver_id)` | Approve and activate a JIT grant with auto-expiry |
| `create_role(...)` | Create an RBAC role with optional parent-role inheritance |
| `assign_role(...)` | Assign a role to a subject with approval gate |
| `resolve_effective_scopes(tenant_id, subject_id, resource_id)` | Compute effective scopes from direct grants + role tree |
| `reap_expired_grants(tenant_id)` | Mark all expired grants and emit audit records |
| `submit_access_request(...)` | Self-service access request (pending approval) |
| `approve_access_request(request_id, approver_id, comment)` | Approve request and auto-create grant |
| `deny_access_request(request_id, approver_id, reason)` | Deny request with mandatory reason |

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
| grant_exceeding_max_duration_blocked | grant duration > 90 days | deny |
| periodic_access_review_required | grant review overdue | require_review |
| high_risk_session_requires_step_up | session risk_score > 74 without step-up | deny |
| concurrent_session_limit_enforced | concurrent sessions > 5 | deny |
| decision_requires_bytewax_stream | record_decision not via bytewax | deny |
| batch_grant_requires_bytewax | batch_grant not via bytewax | deny |
| circuit_breaker_open_blocks_requests | circuit breaker open | deny |
| circuit_breaker_half_open_limits_throughput | half-open budget exhausted | deny |
| privilege_escalation_blocked | grant scope exceeds current maximum | deny |
| cross_tenant_access_blocked | cross-tenant resource access attempted | deny |
| access_agent_runtime_supported | unsupported agent runtime | deny |
| access_agent_role_supported | unsupported agent role | deny |
| privileged_agent_action_requires_human_approval | agent privileged action without human approval | deny |
| service_mesh_identity_required | intra-mesh call without verified identity | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| AccessProviderRecord | id, tenant_id, name, provider_type, owner_id, status, external, metadata_validated, secret_reference |
| AccessResourceRecord | id, tenant_id, resource_key, display_name, owner_id, scopes, capability_id, sensitive, status |
| AccessPolicyRecord | id, tenant_id, name, resource_id, owner_id, effect, conditions, risk_level, status, simulation_evidence |
| AccessGrantRecord | id, tenant_id, subject_id, resource_id, scopes, requested_by, justification, privileged, approved_by, expires_at, status |
| AccessSessionRecord | id, tenant_id, subject_id, provider_id, risk_score, status, step_up_completed, evaluated_at |
| AccessDecisionRecord | id, tenant_id, subject_id, resource_id, action, decision, reason, policy_ids, event_stream |
| AccessAgentRecord | id, tenant_id, name, runtime, role, instructions, status |
| AccessAuditEventRecord | id, tenant_id, event_type, entity_id, actor_id, created_at |

All models are lightweight dataclasses (no ORM dependency at the record layer) with deterministic `stable_id` generation via SHA-256 truncated to 16 hex chars.

## Streaming Events

Events emitted to the composition event stream via Bytewax+NATS (`apg.composition.access.lifecycle`).

| Event | Trigger |
|-------|---------|
| provider_registered | Identity provider record created |
| provider_activated | Provider moves to active status |
| provider_secret_rotated | Secret reference rotated on a provider |
| resource_registered | Protected resource added to registry |
| policy_created | Access policy record created |
| policy_activated | Policy moves from draft to active |
| policy_simulation_completed | Policy sandbox simulation run |
| grant_created | Access grant issued |
| grant_suspended | Grant temporarily suspended |
| grant_reinstated | Suspended grant reactivated |
| grant_revoked | Grant explicitly revoked |
| grant_expired | Grant swept by expiry reaper |
| jit_grant_requested | JIT privileged grant submitted |
| jit_grant_approved | JIT grant activated with expiry |
| role_created | RBAC role defined |
| role_assigned | Role assigned to subject |
| access_request_submitted | Self-service access request created |
| access_request_approved | Request approved; grant auto-created |
| access_request_denied | Request denied with reason |
| permission_matrix_exported | Point-in-time permission snapshot taken |
| session_evaluated | Session risk score computed |
| access_decision_recorded | AuthZ decision emitted |
| access_agent_registered | New access-control agent registered |
| access_compliance_report_generated | Compliance report generated |

Stream states: `draft → active → review_required → approved → denied → revoked → blocked → expired → quarantined`

## NATS Subject Routing

| Subject | Purpose |
|---------|---------|
| `apg.composition.access.lifecycle` | Primary lifecycle event stream (Bytewax processor) |
| `apg.access.jit.requests` | JIT grant request/approval notifications |
| `apg.access.reviews.pending` | Periodic access review dispatch |
| `apg.access.requests.*` | Self-service access request state changes |
| `apg.access.audit.events` | Immutable audit event off-load (JetStream persistent) |
| `apg.risk.signals.>` | Inbound real-time risk signal feed |

## Edge Cases Handled

- A requester cannot self-approve their own privileged grant or access request.
- High-risk policies block activation until `simulate_policy` produces evidence.
- Session risk scores above 74 block further operations until step-up completes.
- Batch grant operations must route through Bytewax; silent bulk escalations are impossible.
- External identity providers without a vault reference cannot be activated.
- JIT grants are created in `pending_jit_approval` status; scope is only active after an independent approver calls `approve_jit_grant`.
- `resolve_effective_scopes` walks the role inheritance chain up to depth 5 to prevent infinite loops from misconfigured parent cycles.
- `reap_expired_grants` is idempotent — already-expired grants are not re-processed.
- The `_matches` rule evaluator supports `_lte`, `_lt`, `_gte`, `_gt`, `_ne` key suffixes for numeric and inequality comparisons.

## Composability

- **Upstream**: `auth` (bootstrap identity), `conf` (secrets and runtime config), `registry` (self-registration at startup)
- **Downstream**: All other composition capabilities (`config`, `events`, `gateway`, `orchestration`, `registry`) delegate their authorization checks here via `authorize_capability_action`; every write operation in those capabilities requires a policy attached through this hub
- **Peer**: `audl` (receives decision audit records for long-term retention), `ntfy` (sends approval notifications for grants and agent actions), `moni` (receives health metrics)

## Development Notes

- The rule engine is fully deterministic and side-effect-free; `evaluate_capability_rules` can be called in tests with arbitrary context dicts without touching any state.
- Record models use `stable_id(prefix, *parts)` for reproducible IDs on replay; intentional for idempotent event sourcing, not a security primitive.
- The `max_autonomous_scope` for access agents is set to `"read_and_recommend"` — more conservative than other capabilities — because access-control mistakes have the broadest blast radius across the composition layer.
- Key files: `capability_contract.py` (executable contract and rule engine), `models.py` (dataclass records), `service.py` (lifecycle operations), `api.py` (API helpers), `views.py` (UI model helpers), `app.py` (package self-test and semantic model).
- See `WORLD_CLASS_IMPROVEMENTS.md` for the 15 prioritised enhancement tracks.

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Continuous Grant Expiry Reaper** [Lifecycle Automation]
- **I2. Attribute-Based Access Control (ABAC) Policy Engine** [Policy Richness]
- **I3. Policy-as-Code Version Control** [Governance]
- **I4. Just-In-Time (JIT) Privileged Access** [Zero-Trust]
- **I5. Real-Time Risk Signal Aggregation via NATS** [Streaming / Zero-Trust]
- **I6. Permission Matrix Snapshot Export** [Observability / Compliance]
- **I7. Periodic Access Review Scheduler** [Governance / Compliance]
- **I8. Circuit Breaker State Machine with NATS Event Propagation** [Resilience]
- **I9. Delegated Authorization Chains** [Composability]
- **I10. Behavioural Anomaly Scoring** [Intelligence / Zero-Trust]
- **I11. Policy Simulation Sandbox** [Developer Experience]
- **I12. Cross-Capability Authorization Middleware** [Composability]
- **I13. Immutable Audit Log with NATS-Backed Off-Load** [Compliance / Durability]
- **I14. Self-Service Access Request Portal API** [Developer Experience / Usability]
- **I15. RBAC Permission Matrix with Role Inheritance** [Policy Richness / Composability]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
