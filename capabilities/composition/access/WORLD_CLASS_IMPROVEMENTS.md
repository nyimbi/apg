# Composition Access Control — World-Class Improvements

15 targeted improvements to move from "solid capability" to a reference-grade access control hub. Each maps to a concrete implementation path and a real competitor that has shipped a version of the idea.

---

### I1. Continuous Grant Expiry Reaper
**Category**: Lifecycle Automation
**Justification**: Expired grants are silently queryable; an active reaper converts time-bounded entitlements into an enforceable runtime guarantee rather than a paper promise — eliminating the "forgotten privileged token" class of incidents that plague manual revocation workflows.
**Implementation**: Background `asyncio.Task` started at service init; sweeps `_grants` every N seconds, marks overdue entries `status="expired"`, emits `grant_expired` to the Bytewax+NATS `apg.composition.access.lifecycle` stream, and appends an audit record. Configurable sweep interval via `grants.expiry_reaper_interval_seconds`.
**Competitor**: HashiCorp Vault (dynamic secrets with TTL-enforced auto-revocation)

---

### I2. Attribute-Based Access Control (ABAC) Policy Engine
**Category**: Policy Richness
**Justification**: The current flat rule engine handles subject/resource/action triples; ABAC extends matching to arbitrary subject and environment attributes (department, clearance, time-of-day, geo-fence), enabling fine-grained policies impossible with pure RBAC without role explosion.
**Implementation**: `ABACPolicyRecord` with a `conditions: dict[str, Any]` expression tree. Extend `evaluate_capability_rules` with an `abac_evaluate(subject_attrs, resource_attrs, env_attrs, conditions)` function using a safe recursive evaluator (no `eval`). Expose via `async def evaluate_abac(...)` on the service.
**Competitor**: AWS IAM condition keys / Google IAP CEL expressions

---

### I3. Policy-as-Code Version Control
**Category**: Governance
**Justification**: Policies that are mutable dicts lose causality — who changed what, when, and why. Append-only policy versions with structured changelogs give rollback semantics, diff-based review, and the ability to pin a policy version in the mesh identity rule.
**Implementation**: `AccessPolicyVersionRecord(policy_id, version_int, snapshot_json, changed_by, change_reason, created_at)`. `create_policy` writes version 1; `update_policy` increments. `async def get_policy_version(policy_id, version)` and `async def rollback_policy(policy_id, target_version, actor_id)` on service.
**Competitor**: OPA (Open Policy Agent) bundle versioning + Styra DAS changelogs

---

### I4. Just-In-Time (JIT) Privileged Access
**Category**: Zero-Trust
**Justification**: Standing privileged grants are the highest-risk entitlement pattern; JIT grants are issued only for the duration of an approved work window, then auto-expired — reducing the blast radius of credential compromise by orders of magnitude.
**Implementation**: `async def request_jit_grant(tenant_id, subject_id, resource_id, scopes, justification, duration_minutes, approver_id)` creates a grant with `expires_at = now + duration`, status `pending_approval`. `async def approve_jit_grant(grant_id, approver_id)` flips to `active`. Reaper (I1) handles expiry. NATS subject: `apg.access.jit.requests`.
**Competitor**: CyberArk Alero / Azure AD PIM (Privileged Identity Management)

---

### I5. Real-Time Risk Signal Aggregation via NATS
**Category**: Streaming / Zero-Trust
**Justification**: Risk scores today are point-in-time inputs; subscribing to a NATS subject for live threat-intel signals (anomalous geo, device posture changes, concurrent logins) allows the session engine to re-score in flight and trigger step-up without waiting for the next explicit `evaluate_session` call.
**Implementation**: `async def subscribe_risk_signals(subject: str = "apg.risk.signals.>")` uses `nats.py` async client; on each message calls `_recompute_session_risk(session_id, signal)` which updates `risk_score` and may flip status to `blocked`. Integrates with Bytewax pipeline via `apg.composition.access.lifecycle`.
**Competitor**: Okta ThreatInsight / Cloudflare Zero Trust (real-time risk enrichment)

---

### I6. Permission Matrix Snapshot Export
**Category**: Observability / Compliance
**Justification**: Auditors require a point-in-time snapshot of who has what access to which resources — a "permission matrix". Today this requires manual query assembly; a dedicated export method produces a structured matrix consumable by SOC 2 auditors, IaC tools, and access-review workflows.
**Implementation**: `async def export_permission_matrix(tenant_id, format="json")` joins `_grants`, `_resources`, `_policies`, and `_providers` in memory; builds a `{subject_id: {resource_id: [scopes]}}` nested dict. Supports `json`, `csv`, and `html` formats. Emits `permission_matrix_exported` audit event.
**Competitor**: Ermetic / Vanta (cloud permissions matrix reporting)

---

### I7. Periodic Access Review Scheduler
**Category**: Governance / Compliance
**Justification**: Quarterly access reviews are mandated by ISO 27001 A.9.2.5 and SOC 2 CC6.3, but the current model has no scheduler — `review_overdue` must be set externally. An internal scheduler marks overdue grants and dispatches review tasks via NATS, closing the loop without depending on an external cron.
**Implementation**: Background `asyncio.Task`; sweep interval = 1 hour. Evaluates each active grant against `granted_at + review_cadence`. Overdue grants get `async def trigger_access_review(grant_id, reviewer_id)` which creates an `AccessReviewRecord` and publishes to `apg.access.reviews.pending` NATS subject.
**Competitor**: Sailpoint IIQ / Saviynt (automated access certification campaigns)

---

### I8. Circuit Breaker State Machine with NATS Event Propagation
**Category**: Resilience
**Justification**: The capability contract defines circuit breaker rules but the service has no state machine backing them. A first-class `CircuitBreakerRecord` with state transitions (`closed→open→half_open→closed`) and NATS event propagation lets downstream capabilities react to access-hub degradation without polling.
**Implementation**: `CircuitBreakerRecord(id, tenant_id, state, failure_count, last_tripped_at, recovery_at)`. `async def trip_circuit_breaker(tenant_id, reason, actor_id)` and `async def reset_circuit_breaker(tenant_id, actor_id)`. State transitions emit to `apg.composition.access.lifecycle` via Bytewax. `_enforce_context` checks active circuit breaker state before each operation.
**Competitor**: Istio / Envoy proxy circuit breaker (translated to application-layer access control)

---

### I9. Delegated Authorization Chains
**Category**: Composability
**Justification**: Modern service meshes require token delegation (OAuth 2.0 token exchange, impersonation) — agent A acts on behalf of user B within a scope ceiling. Without first-class delegation tracking, agent impersonation bypasses the grant lifecycle entirely and leaves no audit trail.
**Implementation**: `AccessDelegationRecord(id, tenant_id, delegator_id, delegate_id, resource_id, scopes, max_depth, expires_at)`. `async def create_delegation(...)` enforces that `scopes ⊆ delegator_scopes` and `max_depth ≤ 3`. `async def resolve_delegation_chain(delegation_id)` returns the full principal chain for audit.
**Competitor**: Google Workspace domain-wide delegation / OAuth 2.0 RFC 8693 token exchange

---

### I10. Behavioural Anomaly Scoring
**Category**: Intelligence / Zero-Trust
**Justification**: Static risk scores miss temporal patterns — a subject accessing 50 resources in 60 seconds at 3 AM is a different risk profile than the same access spread over a day. Lightweight sliding-window anomaly detection raises the risk score dynamically without requiring an ML pipeline.
**Implementation**: `_access_windows: dict[str, deque]` keyed by `subject_id`; each entry is a `(timestamp, resource_id)` tuple. `async def score_access_anomaly(tenant_id, subject_id, resource_id, action)` computes rate, time-of-day deviation, and resource entropy over a 5-minute window; returns an anomaly delta added to the session risk score. Configurable thresholds in `sessions.anomaly_*`.
**Competitor**: Securonix / Exabeam (UEBA — User and Entity Behaviour Analytics)

---

### I11. Policy Simulation Sandbox
**Category**: Developer Experience
**Justification**: Policy authors currently must activate and test against live data; a simulation sandbox runs a proposed policy against historical decision records to estimate allow/deny ratio before activation, eliminating the "surprise deny-all" lockout class of incidents.
**Implementation**: `async def simulate_policy(policy_id, sample_decisions: list[dict])` replays each sample decision against the proposed policy conditions and the existing rule engine; returns `{allow_count, deny_count, changed_decisions}`. Output stored as `simulation_evidence` on the policy record, satisfying the `high_risk_policy_requires_simulation` rule gate.
**Competitor**: OPA `opa test` / Cedar policy playground (Amazon Verified Permissions)

---

### I12. Cross-Capability Authorization Middleware
**Category**: Composability
**Justification**: Each downstream capability (events, gateway, orchestration, registry) currently implements its own check; a callable middleware function `async def authorize_capability_action(caller_capability, tenant_id, subject_id, resource_key, action)` provides a single integration point — removing N ad-hoc integration contracts and ensuring every call goes through the same rule engine and audit path.
**Implementation**: Thin `async def authorize_capability_action(...)` on `CompositionAccessService` that calls `check_access` + `record_decision` in one atomic operation. Publishes result to `apg.composition.access.decisions` NATS subject so downstream capabilities can subscribe without polling.
**Competitor**: Netflix Zuul / Spring Security central `AuthorizationManager`

---

### I13. Immutable Audit Log with NATS-Backed Off-Load
**Category**: Compliance / Durability
**Justification**: The in-memory `_audit_events` list is lost on restart; there is no persistence path. NATS JetStream subjects with `DiscardNew` retention and a Bytewax consumer writing to PostgreSQL give the audit log durability and queryability required for SOC 2 CC7.2 and ISO 27001 A.12.4.
**Implementation**: `async def emit_audit_event(event: AccessAuditEventRecord)` publishes to NATS subject `apg.access.audit.events` (JetStream persistent). Bytewax pipeline `audit_consumer.py` subscribes and batch-inserts into `ac_audit_events` PostgreSQL table (defined in `database/schema.sql`). `async def query_audit_log(tenant_id, start, end, event_types)` queries PostgreSQL via asyncpg.
**Competitor**: AWS CloudTrail / Datadog Audit Trail (immutable, queryable audit with stream-backed persistence)

---

### I14. Self-Service Access Request Portal API
**Category**: Developer Experience / Usability
**Justification**: Access requests today require direct service method calls; a structured request lifecycle (`pending → approved/denied → active`) with in-band notifications removes the informal Slack-approval antipattern and gives the grant workflow a traceable, cancellable paper trail consumable by compliance tools.
**Implementation**: `AccessRequestRecord(id, tenant_id, requester_id, resource_id, scopes, justification, status, approver_id, decided_at, expires_at)`. `async def submit_access_request(...)`, `async def approve_access_request(request_id, approver_id, comment)`, `async def deny_access_request(request_id, approver_id, reason)`. Approved requests auto-create grants. NATS subject `apg.access.requests.*` drives notifications.
**Competitor**: Indent.com / Lumos (self-service access request portals)

---

### I15. RBAC Permission Matrix with Role Inheritance
**Category**: Policy Richness / Composability
**Justification**: The current model grants subject→resource→scope triples; there is no role abstraction, so adding a new team member requires N individual grants. Role inheritance (admin > operator > viewer) with hierarchical scope inheritance enables "hire one, grant role" workflows and reduces over-entitlement from copy-paste grant patterns.
**Implementation**: `AccessRoleRecord(id, tenant_id, name, scopes, parent_role_id, description)`. `async def create_role(...)`, `async def assign_role(subject_id, role_id, tenant_id, approver_id, expires_at)`, `async def resolve_effective_scopes(subject_id, resource_id, tenant_id)` walks the role inheritance tree and unions scopes from all ancestor roles. Role assignments go through the same privileged-grant rules when the role includes `admin` or `privileged` scopes.
**Competitor**: Okta roles / Google Cloud IAM predefined roles with inheritance
