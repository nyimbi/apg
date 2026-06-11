# Vendor Management

## Overview

Vendor Management is the APG capability for the full supplier lifecycle — from initial prospecting through qualification, onboarding, active relationship management, and eventual offboarding. It consolidates vendor master records, performance tracking, risk governance, compliance evidence, contract management, communication logging, self-service portal access, and AI-generated scorecards into a single coherent service boundary.

The capability is designed as a dependency-light executable packet. Web frameworks, databases, procurement systems, contract repositories, document stores, risk policy providers, and notification services all attach through APG composition adapters at deployment time. This means the core lifecycle logic is testable and portable without any external infrastructure.

## Capability ID

`scm_ven`  Version: 2.1.0

## Provides

| Service | Description |
|---------|-------------|
| vendor_profile_lifecycle | Create, update, and archive vendor master records with ownership, classification, and AI-scored attributes |
| vendor_onboarding_workflow | Checklist-driven onboarding tracking from prospect to active status |
| vendor_qualification_lifecycle | Criteria capture, reviewer assignment, scoring, and threshold-gated review |
| vendor_performance_lifecycle | Multi-dimension scoring (quality, delivery, cost, service, sustainability, innovation) per measurement period |
| vendor_risk_lifecycle | Risk record creation across low/medium/high/critical tiers with mandatory ownership for elevated tiers |
| vendor_contract_lifecycle | Approved commercial contract records with value, currency, date range, and SLA attachment |
| vendor_compliance_lifecycle | Framework evidence capture with status tracking and mandatory review on non-compliance or expiry |
| vendor_communication_lifecycle | Communication log with channel, sentiment scoring, and owner escalation on negative sentiment |
| vendor_portal_lifecycle | External self-service portal user provisioning with approval gate and MFA enforcement |
| vendor_scorecard_service | Composite scorecard generation from performance, risk, and compliance records |
| vendor_sourcing_integration | Bridge to upstream sourcing events for seamless vendor creation from awarded sourcing activities |
| vendor_agents | AI review agents (Codex, Claude Code, OpenCode, Pi) for onboarding, risk, performance, compliance, contract, and supplier query review — all human-approval-gated |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Identity and permission enforcement for all write operations |
| audl | Immutable audit trail for all state-changing vendor operations |
| ntfy | Notification dispatch for lifecycle events and escalations |
| composition_events | APG event bus for inter-capability coordination |
| composition_config | Runtime configuration injection and tenant overrides |
| workflow | Approval workflow orchestration for contracts, portal users, and agent recommendations |
| procurement_requisition_lifecycle | Feeds vendor context into purchase requisition flows |
| sourcing_event_lifecycle | Upstream sourcing awards that trigger vendor creation or promotion |
| contract_lifecycle | Stores and versions the underlying contract documents referenced by vendor contract records |
| document_management | Stores compliance evidence, onboarding checklists, and contract attachments |
| risk_policy | External risk policy rules applied during risk tier assessment |
| supplier_master_data | Canonical supplier identifiers shared across SCM capabilities |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant isolation key |
| vendors.code_required | bool | true | Enforce unique vendor code on creation |
| vendors.name_required | bool | true | Enforce vendor name on creation |
| vendors.category_required | bool | true | Enforce category classification |
| vendors.country_required | bool | true | Enforce country of registration |
| vendors.owner_required | bool | true | Enforce owner assignment |
| qualification.minimum_score | int | 70 | Score below which review is triggered |
| performance.review_required_below | int | 60 | Performance score threshold requiring review |
| performance.score_range | [int, int] | [0, 100] | Valid score bounds |
| risk.owner_required_for_high_or_critical | bool | true | Mandatory owner for elevated risk tiers |
| compliance.evidence_required | bool | true | Evidence documents required on compliance records |
| compliance.review_required_for_noncompliance | bool | true | Force review on non-compliant or expired status |
| contracts.approval_required | bool | true | Approval workflow required before contract activation |
| communications.owner_required_for_negative_sentiment | bool | true | Escalation owner required for negative-sentiment comms |
| portal.approval_required | bool | true | Approval required before portal user activation |
| scorecards.performance_required | bool | true | Performance record required for scorecard generation |
| scorecards.risk_required | bool | true | Risk record required for scorecard generation |
| vendor_agents.enabled | bool | true | Enable AI agent workbench |
| vendor_agents.human_approval_required | bool | true | Gate all agent privileged actions behind human approval |
| vendor_agents.max_autonomous_scope | string | "inspect_prepare_and_recommend" | Maximum autonomy level for agents |
| governance.require_tenant_context | bool | true | Reject operations lacking tenant context |
| governance.audit_state_changes | bool | true | Emit audit records for every state change |
| governance.segregation_of_duties | bool | true | Enforce SoD controls on sensitive operations |
| observability.event_stream | string | "apg.scm.ven.lifecycle" | Bytewax event stream identifier |
| theme.default_theme | string | "vendor_control" | Default UI theme |
| theme.allow_tenant_overrides | bool | true | Allow per-tenant theme overrides |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /scm/vendors/dashboard | GET | scm_ven:view | Overview |
| vendors | /scm/vendors | GET/POST | scm_ven:manage_vendors | Master Data |
| qualification | /scm/vendors/qualification | GET/POST | scm_ven:qualify | Lifecycle |
| onboarding | /scm/vendors/onboarding | GET/POST | scm_ven:onboard | Lifecycle |
| performance | /scm/vendors/performance | GET/POST | scm_ven:score | Performance |
| risk | /scm/vendors/risk | GET/POST | scm_ven:govern | Governance |
| compliance | /scm/vendors/compliance | GET/POST | scm_ven:govern | Governance |
| contracts | /scm/vendors/contracts | GET/POST | scm_ven:contract | Commercial |
| communications | /scm/vendors/communications | GET/POST | scm_ven:communicate | Engagement |
| portal | /scm/vendors/portal | GET/POST | scm_ven:portal | Engagement |
| scorecards | /scm/vendors/scorecards | GET/POST | scm_ven:score | Performance |
| agents | /scm/vendors/agents | GET/POST | scm_ven:agent_manage | Automation |
| rules | /scm/vendors/rules | GET | scm_ven:govern | Governance |
| settings | /scm/vendors/settings | GET/POST | scm_ven:admin | Administration |

API prefix: `/scm/vendors/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | Operation has no tenant context | deny — attach_tenant_context |
| operation_policy_required | Write operation has no policy attached | deny — attach_operation_policy |
| vendor_code_required | create_vendor with no code | deny — provide_vendor_code |
| vendor_name_required | create_vendor with no name | deny — provide_vendor_name |
| vendor_type_supported | create_vendor with unsupported type | deny — choose_supported_vendor_type |
| vendor_category_required | create_vendor with no category | deny — provide_category |
| vendor_country_required | create_vendor with no country | deny — provide_country |
| vendor_owner_required | create_vendor with no owner | deny — assign_owner |
| qualification_criteria_required | qualify_vendor with no criteria | deny — attach_criteria |
| qualification_actor_required | qualify_vendor with no reviewer | deny — record_reviewer |
| qualification_score_threshold | qualify_vendor score below 70 and no review recorded | require_review — record_qualification_review |
| onboarding_checklist_required | onboard_vendor with no checklist | deny — attach_checklist |
| performance_dimensions_supported | record_performance with unsupported dimensions | deny — choose_supported_dimensions |
| performance_scores_in_range | record_performance with score outside 0–100 | deny — correct_scores |
| performance_low_score_review | record_performance score below 60 and no review | require_review — record_performance_review |
| risk_owner_required | record_risk tier high/critical and no owner | deny — assign_risk_owner |
| risk_tier_supported | record_risk with unsupported tier | deny — choose_supported_risk_tier |
| compliance_evidence_required | record_compliance with no evidence | deny — attach_evidence |
| compliance_review_required | record_compliance non-compliant/expired and no review | require_review — record_compliance_review |
| contract_approval_required | create_contract with no approval record | deny — record_contract_approval |
| contract_dates_required | create_contract missing start or end date | deny — provide_dates |
| negative_sentiment_owner_required | record_communication negative sentiment and no owner | deny — assign_owner |
| portal_approval_required | create_portal_user with no approval | deny — record_approval |
| scorecard_performance_required | create_scorecard with no performance record | deny — attach_performance |
| scorecard_risk_required | create_scorecard with no risk record | deny — attach_risk |
| bytewax_event_stream_required | vendor_batch routed to queue instead of Bytewax | deny — route_to_bytewax_stream |
| agent_runtime_supported | register_vendor_agent with unsupported runtime | deny — choose_supported_runtime |
| agent_scope_limited | agent_action is privileged and lacks human approval | require_review — record_human_approval |
| audit_required_for_state_change | Write operation with audit disabled | deny — enable_audit |

## Data Models

| Model | Key Fields |
|-------|-----------|
| VMVendor | id, tenant_id, vendor_code, name, vendor_type, category, country, status, lifecycle_stage, intelligence_score, performance_score, risk_score, relationship_score, strategic_importance, preferred_vendor, shared_vendor |
| VMPerformance | id, tenant_id, vendor_id, measurement_period, overall_score, quality_score, delivery_score, cost_score, service_score, innovation_score, on_time_delivery_rate, total_spend |
| VMRisk | id, tenant_id, vendor_id, risk_type, risk_category, severity, probability, impact, overall_risk_score, mitigation_status, assigned_to |
| VMContract | id, tenant_id, vendor_id, contract_number, contract_type, effective_date, expiration_date, contract_value, currency, status, auto_renewal |
| VMCommunication | id, tenant_id, vendor_id, communication_type, direction, subject, communication_date, sentiment_score, action_items |
| VMCompliance | id, tenant_id, vendor_id, framework, requirement, status, compliance_score, evidence_documents, next_review_date |
| VMPortalUser | id, tenant_id, vendor_id, email, role, status, mfa_enabled, permissions, session_timeout_minutes |
| VMPortalSession | id, user_id, vendor_id, session_token, expires_at, last_activity, ip_address, security_context |
| VMIntelligence | id, tenant_id, vendor_id, model_version, confidence_score, behavior_patterns, predictive_insights, performance_forecasts, valid_until |
| VMBenchmark | id, tenant_id, vendor_id, benchmark_type, vendor_value, benchmark_value, percentile_rank, performance_gap |
| VMAuditLog | id, tenant_id, event_type, resource_type, resource_id, user_id, old_values, new_values, event_timestamp |
| VendorAIDecision | decision_id, vendor_id, decision_type, recommendation, confidence_score, autonomous_approved, reasoning |
| VendorOptimizationPlan | vendor_id, optimization_objectives, recommended_actions, predicted_outcomes, success_metrics |

## Streaming Events

Events emitted to the `apg.scm.ven.lifecycle` event stream via Bytewax. Delivery guarantee: at-least-once. Ordering key: `tenant_id`.

| Event | Trigger |
|-------|---------|
| vendor_created | New vendor master record persisted |
| vendor_qualified | Qualification record accepted (score at or above threshold, or review completed) |
| vendor_onboarded | Onboarding checklist completed and stage advanced to active |
| vendor_performance_recorded | Performance scores recorded for a measurement period |
| vendor_risk_recorded | Risk record created or updated |
| vendor_compliance_recorded | Compliance status recorded with evidence |
| vendor_contract_created | Contract record created with approval |
| vendor_communication_recorded | Communication log entry created |
| vendor_portal_user_created | Portal user provisioned and approved |
| vendor_scorecard_created | Composite scorecard generated |
| vendor_agent_registered | AI agent registered to the vendor agent workbench |

## Edge Cases Handled

- Qualification scores below the minimum threshold (70) do not hard-deny — they emit a `require_review` effect, allowing the workflow to proceed once a reviewer records their assessment. This prevents bottlenecks for borderline vendors while maintaining auditability.
- High and critical risk records without an owner are hard-denied at the rule layer, not deferred to the workflow engine. This ensures elevated risks cannot exist in an unowned state even briefly.
- Negative-sentiment communications require an owner assignment at record time, not as a post-creation task, so escalation accountability is guaranteed in the immutable audit trail.
- Bytewax event stream routing is enforced as a hard rule — batch operations that attempt to use a generic queue are rejected at the rules layer, preventing silent loss of lifecycle event metadata.
- Agent privileged actions (any state change beyond inspect/prepare/recommend) require explicit human approval recorded before execution. The `max_autonomous_scope` config key codifies this boundary, and the rule engine enforces it independently of the agent runtime.
- Vendor codes are normalised to uppercase at validation time (`VMVendor.validate_vendor_code`) and must be alphanumeric with optional hyphens/underscores, preventing lookup divergence between case variants.
- Portal user email addresses are lowercased at validation time; uniqueness is enforced at the database level to avoid duplicate accounts across different email capitalisation forms.
- Contract renewal dates earlier than the expiration date are rejected by `validate_contract_dates`, preventing nonsensical auto-renewal configurations.
- Shared vendors (`shared_vendor=True`) maintain an explicit `sharing_tenants` list, preserving tenant isolation semantics even for cross-tenant vendor records.

## Composability

- **Upstream**: `sourcing_event_lifecycle` — sourcing award events trigger vendor creation or lifecycle promotion; `supplier_master_data` — canonical supplier identifiers flow in as the authoritative source of vendor codes.
- **Downstream**: `procurement_requisition_lifecycle` — active qualified vendors are available as approved supplier options on purchase requisitions; `contract_lifecycle` — vendor contract records reference contract documents managed by the contract capability.
- **Peer**: `scm_srm` (Supplier Relationship Management) — shares the vendor entity as the anchor object; `scm_prc` (Procurement) — consumes vendor qualification and performance data for supplier selection scoring; `scm_rrl` (Risk and Regulatory) — exchanges risk tier and compliance status for enterprise risk roll-up.

## Development Notes

- The local `VendorManagementLifecycleService` in `service.py` holds state in-memory for executable packaging and testing. Production deployments attach a PostgreSQL store and integration adapters via the `adapters` config section.
- All batch operations must route through the Bytewax event stream. Attempting to route through a generic queue will be rejected by the `bytewax_event_stream_required` rule — this is intentional and must not be worked around.
- The `models.py` file uses `uuid_extensions.uuid7str` for ID generation. If that package is unavailable, substitute with the `uuid6` package shim documented in the project CLAUDE.md.
- Pydantic v1-style `@validator` and `@root_validator` decorators are present in `models.py`. Migration to Pydantic v2 `@field_validator` / `@model_validator` is a pending cleanup item.
- The `VMVendor` model carries AI-scored fields (`intelligence_score`, `performance_score`, `risk_score`, `relationship_score`) directly on the vendor record for fast dashboard queries. These are denormalised from the detailed `VMIntelligence` and `VMPerformance` records and should be kept in sync by the service layer.
- UI theme tokens are defined in `THEME` within `capability_contract.py` under the `vendor_control` theme. Tenant overrides are applied at runtime via the `theme.allow_tenant_overrides` config flag.
- The six AI agent roles (`vendor_onboarding_reviewer`, `risk_reviewer`, `performance_reviewer`, `compliance_reviewer`, `contract_reviewer`, `supplier_query_reviewer`) map to distinct review scopes. Registering an agent to the wrong role will pass the rules engine (role is validated) but will receive incorrect context payloads from the intelligence service.

## New Async Methods (v2.2.0)

Eight async methods added to `VendorManagementService` covering the highest-value
gaps identified in `WORLD_CLASS_IMPROVEMENTS.md`:

| Method | Description |
|--------|-------------|
| `contract_expiry_alerts(days_ahead, tenant_id)` | Surface contracts expiring within a window; auto-renew `auto_renew=True` contracts |
| `spend_concentration_risk(period, category_threshold_pct, total_threshold_pct, tenant_id)` | Flag single-source spend dependency by category and total share |
| `bulk_onboard_vendors(vendors, tenant_id)` | Concurrent fan-out onboarding with per-row success/error results |
| `compliance_expiry_scan(as_of_date, expiry_warning_days, tenant_id)` | Scan compliance records for expired or soon-to-expire certifications |
| `sla_breach_scan(vendor_id, tenant_id)` | Cross-reference SLA terms against performance scores; auto-create risk records on breach |
| `vendor_reinstatement(vendor_id, rationale, approved_by, tenant_id)` | Complete the suspension lifecycle — reinstate a suspended vendor |
| `compare_vendors(vendor_ids, tenant_id)` | Head-to-head multi-vendor comparison matrix for data-driven selection |
| `ai_early_warning_digest(tenant_id)` | Portfolio-level ML-powered digest of at-risk vendors; falls back to rule-based tiers |
| `vendor_health_score(vendor_id, tenant_id)` | Composite 0–100 health score (performance 40 %, compliance 25 %, risk 25 %, engagement 10 %) |

All new methods are `async def` and safe to call with `await` from any async
web framework or APG composition layer.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/scm/ven/__init__.py capabilities/scm/ven/capability_contract.py capabilities/scm/ven/service.py capabilities/scm/ven/api.py capabilities/scm/ven/views.py capabilities/scm/ven/app.py
./.venv/bin/pytest -q capabilities/scm/ven/tests/test_package_contract.py
./.venv/bin/apg capabilities publish-plan capabilities/scm/ven --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/scm/ven --json
```
