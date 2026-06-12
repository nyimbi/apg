# Sustainability and ESG Management

## Overview

The Sustainability and ESG Management capability provides end-to-end lifecycle management for Environmental, Social, and Governance programs within the APG platform. It covers the full data chain from tenant profile setup through framework selection, metric definition, measurement recording, target tracking, supplier assessment, initiative management, risk governance, report generation, and stakeholder engagement. Every write operation is gated by deterministic business rules enforced at the capability boundary, with all state transitions emitted to a Bytewax event stream for real-time observability.

The capability is designed for multi-tenant deployment and is dependency-light at import time. Authorisation, audit, notifications, workflows, document storage, carbon data feeds, supplier master data, regulatory content, and the event bus all attach through APG composition adapters. This means the ESG core can be tested and run in isolation, and downstream consumers receive a stable contract regardless of which adapter implementations are wired in.

Version 2.2.0 adds eight world-class async methods covering SBTi target validation, EU Digital Product Passport PCF calculation, carbon budget ledger accounting, Biodiversity Net Gain (BNG) per Defra statutory metric, internal carbon price allocation, CSRD ESRS gap analysis, SFDR PAI aggregation, continuous GHG assurance (ISO 14064-3), and EXIOBASE spend-based Scope 3 calculation.

## Capability ID

`ecd_esg`  Version: 2.2.0

## Provides

| Service | Description |
|---------|-------------|
| esg_profile_lifecycle | Create, update, and retire tenant ESG profiles that anchor all other ESG objects |
| esg_framework_lifecycle | Manage reporting framework registrations (GRI, SASB, TCFD, ISSB, CSRD, SEC Climate, local regulatory) per profile |
| esg_metric_lifecycle | Define and maintain the metric catalogue across Environmental, Social, and Governance pillars |
| esg_measurement_lifecycle | Record time-series measurements against metrics with evidence attachment and review gates |
| esg_target_lifecycle | Set and track absolute, intensity, reduction, and compliance targets with milestone support |
| esg_supplier_assessment_lifecycle | Score and track supply-chain ESG performance with graded assessments and improvement plans |
| esg_initiative_lifecycle | Log and track sustainability projects, programs, policies, and investments with ROI measurement |
| esg_risk_lifecycle | Maintain an ESG risk register with probability/impact scoring, mitigation tracking, and AI-driven trend analysis |
| esg_report_workflow | Orchestrate draft-to-approved-to-published report packages aligned to registered frameworks |
| esg_stakeholder_lifecycle | Register and maintain consented stakeholders with communication preferences and influence scoring |
| esg_engagement_lifecycle | Record stakeholder engagements with sentiment analysis and negative-engagement owner escalation |
| esg_dashboard_service | Aggregate cross-domain ESG health metrics for real-time dashboard consumption |
| esg_agents | Register and run Codex, Claude Code, OpenCode, and Pi review agents scoped to inspect-prepare-and-recommend with mandatory human approval |
| sbti_validation | Validate reduction targets against SBTi 1.5°C pathway using IPCC AR6 sector trajectories |
| product_carbon_footprint | Calculate per-SKU PCF per ISO 14067, producing EU DPP-compatible output |
| carbon_budget_ledger | Track cumulative emissions drawdown against a finite science-aligned carbon budget |
| biodiversity_net_gain | Calculate statutory BNG units per Defra metric (UK Environment Act 2021) |
| internal_carbon_price | Allocate shadow carbon price charges across cost centres by headcount/floor-area/revenue |
| csrd_esrs_gap_analysis | Cross-reference materiality assessment against ESRS E1-E5, S1-S4, G1 disclosure requirements |
| sfdr_pai_aggregate | Produce weighted SFDR Annex I PAI indicator table across portfolio holdings |
| continuous_assurance | Run ISO 14064-3 completeness, consistency, and source-chain tests on every measurement |
| scope3_spend_based | Calculate Scope 3 emissions via EXIOBASE 3.8 MRIO spend-based approach |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Identity and permission enforcement for all ESG operations |
| audl | Immutable audit trail for every state change (mandatory for write operations) |
| ntfy | Notification delivery for review requests, approvals, and escalations |
| composition_events | APG event bus through which Bytewax stream metadata is routed |
| composition_config | Runtime configuration injection and tenant override resolution |
| workflow | Approval workflow engine for measurements, reports, and agent actions |
| document_management | Evidence attachment storage for measurements and supplier assessments |
| supplier_master_data | Canonical supplier identity data consumed by supplier assessments |
| carbon_data_provider | External carbon factor data used for scope emissions calculations |
| regulatory_content | Up-to-date regulatory framework content for CSRD, SEC, and local compliance |
| risk_policy | Enterprise risk policy definitions that govern ESG risk tier thresholds |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant scope for all ESG operations |
| profiles.name_required | bool | true | Enforce name on profile creation |
| profiles.industry_required | bool | true | Enforce industry on profile creation |
| profiles.country_required | bool | true | Enforce country (ISO 3166-1 alpha-3) on profile creation |
| profiles.reporting_year_required | bool | true | Enforce reporting year on profile creation |
| profiles.owner_required | bool | true | Enforce owner assignment on profile creation |
| frameworks.supported_frameworks | list[str] | gri, sasb, tcfd, issb, csrd, sec_climate, local_regulatory | Allowlist of accepted framework codes |
| metrics.supported_pillars | list[str] | environmental, social, governance | Accepted ESG pillar values |
| metrics.supported_types | list[str] | emissions, energy, water, waste, labor, safety, diversity, ethics, board, supply_chain | Accepted metric type values |
| metrics.supported_units | list[str] | tco2e, kwh, m3, tonnes, percent, count, score, currency | Accepted measurement units |
| measurements.evidence_required | bool | true | Measurements must have an evidence document attached |
| measurements.review_required_for_calculation_or_supplier | bool | true | Calculated or supplier-sourced measurements require a review record |
| supplier_assessments.score_range | [int, int] | [0, 100] | Valid score range for supplier ESG assessments |
| targets.supported_types | list[str] | absolute, intensity, reduction, compliance | Accepted target type values |
| reports.approval_required | bool | true | Reports must be approved before publication |
| esg_agents.max_autonomous_scope | string | "inspect_prepare_and_recommend" | Maximum autonomous action scope for ESG agents |
| esg_agents.human_approval_required | bool | true | Agent privileged actions require a human approval record |
| governance.require_tenant_context | bool | true | All operations must carry a tenant context |
| governance.audit_state_changes | bool | true | All state changes are forwarded to the audit adapter |
| governance.segregation_of_duties | bool | true | Enforce role separation for approval and execution |
| observability.stream_processor | string | "bytewax" | Streaming processor for ESG lifecycle events |
| theme.default_theme | string | "esg_control" | Default UI theme token set |
| theme.allow_tenant_overrides | bool | true | Tenants may supply theme token overrides |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /ecd/esg/dashboard | GET | ecd_esg:view | Overview |
| profiles | /ecd/esg/profiles | GET/POST | ecd_esg:manage_profiles | Setup |
| frameworks | /ecd/esg/frameworks | GET/POST | ecd_esg:manage_frameworks | Setup |
| metrics | /ecd/esg/metrics | GET/POST | ecd_esg:manage_metrics | Data |
| measurements | /ecd/esg/measurements | GET/POST | ecd_esg:record_data | Data |
| targets | /ecd/esg/targets | GET/POST | ecd_esg:manage_targets | Planning |
| suppliers | /ecd/esg/suppliers | GET/POST | ecd_esg:assess_suppliers | Supply Chain |
| initiatives | /ecd/esg/initiatives | GET/POST | ecd_esg:manage_initiatives | Planning |
| risks | /ecd/esg/risks | GET/POST | ecd_esg:govern | Governance |
| reports | /ecd/esg/reports | GET/POST | ecd_esg:report | Reporting |
| stakeholders | /ecd/esg/stakeholders | GET/POST | ecd_esg:engage | Engagement |
| agents | /ecd/esg/agents | GET/POST | ecd_esg:agent_manage | Automation |
| rules | /ecd/esg/rules | GET | ecd_esg:govern | Governance |
| settings | /ecd/esg/settings | GET/POST | ecd_esg:admin | Administration |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present is False | deny — attach_tenant_context |
| operation_policy_required | write operation without policy attached | deny — attach_operation_policy |
| profile_name_required | create_esg_profile without name | deny — provide_name |
| profile_industry_required | create_esg_profile without industry | deny — provide_industry |
| profile_country_required | create_esg_profile without country | deny — provide_country |
| profile_year_required | create_esg_profile without reporting year | deny — provide_reporting_year |
| profile_owner_required | create_esg_profile without owner | deny — assign_owner |
| framework_profile_required | add_framework without profile | deny — select_profile |
| framework_supported | add_framework with unsupported code | deny — choose_supported_framework |
| framework_version_required | add_framework without version | deny — provide_version |
| framework_owner_required | add_framework without owner | deny — assign_owner |
| metric_profile_required | define_metric without profile | deny — select_profile |
| metric_pillar_supported | define_metric with unsupported pillar | deny — choose_supported_pillar |
| metric_type_supported | define_metric with unsupported type | deny — choose_supported_metric_type |
| metric_unit_supported | define_metric with unsupported unit | deny — choose_supported_unit |
| metric_name_required | define_metric without name | deny — provide_metric_name |
| metric_owner_required | define_metric without owner | deny — assign_owner |
| measurement_metric_required | record_measurement without metric | deny — select_metric |
| measurement_period_required | record_measurement without period | deny — provide_period |
| measurement_value_required | record_measurement without value | deny — provide_value |
| measurement_source_supported | record_measurement with unsupported source | deny — choose_supported_source |
| measurement_evidence_required | record_measurement without evidence | deny — attach_evidence |
| measurement_review_required | calculation or supplier source without review | require_review — record_measurement_review |
| target_metric_required | set_target without metric | deny — select_metric |
| target_type_supported | set_target with unsupported type | deny — choose_supported_target_type |
| target_baseline_required | set_target without baseline | deny — provide_baseline |
| target_value_required | set_target without target value | deny — provide_target |
| target_due_date_required | set_target without due date | deny — provide_due_date |
| target_owner_required | set_target without owner | deny — assign_owner |
| supplier_required | record_supplier_assessment without supplier | deny — select_supplier |
| supplier_period_required | record_supplier_assessment without period | deny — provide_period |
| supplier_score_range | supplier score outside 0–100 | deny — correct_score |
| supplier_evidence_required | record_supplier_assessment without evidence | deny — attach_evidence |
| supplier_owner_required | high-risk supplier assessment without owner | deny — assign_owner |
| initiative_profile_required | record_initiative without profile | deny — select_profile |
| initiative_name_required | record_initiative without name | deny — provide_name |
| initiative_pillar_supported | record_initiative with unsupported pillar | deny — choose_supported_pillar |
| initiative_owner_required | record_initiative without owner | deny — assign_owner |
| initiative_impact_required | record_initiative without expected impact | deny — provide_expected_impact |
| risk_profile_required | record_risk without profile | deny — select_profile |
| risk_tier_supported | record_risk with unsupported tier | deny — choose_supported_tier |
| risk_description_required | record_risk without description | deny — provide_description |
| risk_owner_required | high or critical risk without owner | deny — assign_owner |
| report_profile_required | create_report without profile | deny — select_profile |
| report_type_supported | create_report with unsupported type | deny — choose_supported_report_type |
| report_frameworks_required | create_report without frameworks | deny — attach_frameworks |
| report_measurements_required | create_report without measurements | deny — attach_measurements |
| report_approval_required | create_report without approval | deny — record_report_approval |
| stakeholder_profile_required | register_stakeholder without profile | deny — select_profile |
| stakeholder_name_required | register_stakeholder without name | deny — provide_name |
| stakeholder_consent_required | register_stakeholder without consent | deny — record_consent |
| engagement_stakeholder_required | record_engagement without stakeholder | deny — select_stakeholder |
| engagement_topic_required | record_engagement without topic | deny — provide_topic |
| negative_engagement_owner_required | negative sentiment engagement without owner | deny — assign_owner |
| bytewax_event_stream_required | esg_batch routed to queue instead of Bytewax | deny — route_to_bytewax_stream |
| agent_runtime_supported | register_esg_agent with unsupported runtime | deny — choose_supported_runtime |
| agent_role_supported | register_esg_agent with unsupported role | deny — choose_supported_role |
| agent_scope_limited | agent privileged action without human approval | require_review — record_human_approval |
| audit_required_for_state_change | write operation with audit disabled | deny — enable_audit |

## Data Models

| Model | Key Fields |
|-------|-----------|
| ESGTenant | id, name, slug, industry, headquarters_country, employee_count, annual_revenue, esg_frameworks, ai_enabled, subscription_tier, is_active |
| ESGFramework | id, tenant_id, name, code, framework_type, version, categories, standards, indicators, is_mandatory, is_active |
| ESGMetric | id, tenant_id, framework_id, name, code, metric_type, category, unit, current_value, target_value, baseline_value, is_kpi, is_automated, data_quality_score |
| ESGMeasurement | id, tenant_id, metric_id, value, measurement_date, period_start, period_end, data_source, collection_method, is_validated, is_approved, anomaly_score |
| ESGTarget | id, tenant_id, metric_id, name, target_value, baseline_value, current_progress, start_date, target_date, status, achievement_probability, owner_id |
| ESGMilestone | id, tenant_id, target_id, name, milestone_value, milestone_date, achieved_value, achieved_date, is_achieved, is_critical |
| ESGStakeholder | id, tenant_id, name, organization, stakeholder_type, email, country, engagement_score, sentiment_score, influence_score, portal_access, is_active |
| ESGCommunication | id, tenant_id, stakeholder_id, subject, communication_type, channel, sent_at, response_sentiment, effectiveness_score, status |
| ESGSupplier | id, tenant_id, name, country, industry_sector, overall_esg_score, environmental_score, social_score, governance_score, risk_level, criticality_level |
| ESGSupplierAssessment | id, tenant_id, supplier_id, assessment_type, overall_score, grade, risk_rating, strengths, weaknesses, action_items, improvement_plan |
| ESGInitiative | id, tenant_id, name, description, category, initiative_type, status, progress_percentage, budget_allocated, budget_spent, success_probability, project_manager |
| ESGReport | id, tenant_id, name, report_type, framework, period_start, period_end, reporting_year, status, auto_generated, published_at, file_format |
| ESGRisk | id, tenant_id, name, risk_category, probability, impact_severity, risk_score, risk_level, time_horizon, mitigation_status, risk_owner |

## Streaming Events

Events emitted to the `apg.ecd.esg.lifecycle` event stream via Bytewax/NATS. Delivery guarantee: at-least-once. Ordering key: `tenant_id`.

| Event | Trigger |
|-------|---------|
| esg_profile_created | New ESG profile successfully persisted |
| esg_framework_added | Framework registered against a profile |
| esg_metric_defined | New metric added to the metric catalogue |
| esg_measurement_recorded | Measurement value accepted (after any required review) |
| esg_target_set | New sustainability target created |
| esg_supplier_assessed | Supplier assessment completed and scored |
| esg_initiative_recorded | New sustainability initiative logged |
| esg_risk_recorded | New ESG risk entered into the risk register |
| esg_report_created | Report package created and approved |
| esg_stakeholder_registered | Stakeholder registered with consent recorded |
| esg_engagement_recorded | Stakeholder engagement interaction logged |
| esg_agent_registered | ESG AI agent enrolled in the agent roster |
| sbti_target_validated | SBTi 1.5°C pathway validation completed for a reduction target |
| product_carbon_footprint_calculated | ISO 14067 PCF calculation completed; DPP payload ready |
| bng_calculated | Biodiversity Net Gain units computed per Defra BNG Metric 4.0 |
| carbon_offset_retired | Carbon offset retirement recorded against a registry |
| continuous_assurance_completed | ISO 14064-3 assurance check result published to tenant NATS subject |

NATS subjects used for assurance push: `apg.ecd.esg.assurance.<tenant_id>`

## Edge Cases Handled

- Supplier and calculated measurements are soft-blocked with `require_review` rather than outright denied, allowing the value to be staged while a review record is collected before the measurement is promoted to confirmed.
- High-risk supplier assessments and high-or-critical ESG risks enforce owner assignment at rule evaluation time, not at database constraint time, so the error surfaces with an actionable `required_action` before any persistence attempt.
- ESG agents are restricted to `inspect_prepare_and_recommend` autonomous scope; any privileged state-change action triggers `require_review` with `record_human_approval` as the required action, preventing autonomous data corruption even if the agent runtime misbehaves.
- The `bytewax_event_stream_required` rule blocks ESG batch operations that are mistakenly routed to a generic queue rather than the Bytewax stream, preventing loss of ordering guarantees.
- Tenant override of the Bytewax event stream name is not permitted through configuration; the stream name `apg.ecd.esg.lifecycle` is a constant in the contract, ensuring downstream consumers have a stable subscription address.
- Negative-sentiment stakeholder engagements require an owner to be assigned, ensuring no adverse stakeholder signal goes unacknowledged.
- The `ESGMetric.progress_to_target` hybrid property handles the case where both `baseline_value` and `target_value` are present by computing progress relative to the baseline rather than absolute value, which is the correct interpretation for reduction targets.
- `ESGMeasurement` anomaly and validation scores are stored separately, allowing a measurement to be validated (passes format and range checks) but still flagged as anomalous (statistical outlier), without conflating the two quality dimensions.
- Report creation is denied if frameworks or measurements are absent; this prevents the generation of empty or framework-unaligned reports that would fail regulatory submission.

## Composability

- **Upstream**: `supplier_master_data` provides canonical supplier identities consumed by `esg_supplier_assessment_lifecycle`. `carbon_data_provider` supplies emission factors used in scope calculation measurements. `regulatory_content` feeds current framework requirements into `esg_framework_lifecycle`. `risk_policy` provides enterprise risk tier thresholds that ESG risk classification is normalised against.
- **Downstream**: `esg_report_workflow` outputs approved report packages that can be consumed by document management and investor-relations capabilities. The Bytewax event stream (`apg.ecd.esg.lifecycle`) is the primary integration point for analytics, real-time dashboards, and audit pipelines. `esg_dashboard_service` aggregates across all ESG sub-domains and is typically consumed by the APG shell overview panel.
- **Peer**: Commonly deployed alongside `auth` (mandatory), `audl` (mandatory), and `ntfy` for a fully operational ESG control environment. In regulated deployments, `workflow` and `document_management` are also required to meet evidence and approval chain requirements for CSRD and SEC climate rule submissions.

## Development Notes

- All ESG models inherit `ESGAuditMixin` which adds `created_at`, `updated_at`, `created_by`, `updated_by`, `version`, `is_deleted`, `deleted_at`, `deleted_by`. Soft-delete is the expected pattern; hard deletes should be avoided to preserve audit continuity.
- Model prefix convention: `esg_` for all table names (e.g. `esg_tenants`, `esg_metrics`). The two-character capability prefix is `es` but the full prefix `esg_` is used for readability given the domain specificity.
- `uuid7str` is imported from `uuid_extensions` in the current codebase; the project standard shim via `uuid6` should be used if `uuid_extensions` is not available in the target environment.
- The `evaluate_capability_rules` function is a pure deterministic function with no I/O. It can be used in tests, pre-flight checks, and policy-as-code pipelines without any infrastructure dependencies.
- The `get_capability_contract` function performs a deep merge of `DEFAULT_CONFIGURATION` with tenant overrides, meaning partial overrides are safe — only the specified keys are replaced.
- Pydantic view models (`ESGTenantView`, `ESGMetricView`, etc.) use `extra='forbid'` to prevent unknown fields leaking through the API boundary.
- The `ESGReport` model stores `ai_insights` as a JSON column, allowing the AI-generated narrative and data cross-references to be attached without schema migration as the AI output structure evolves.
- Theme tokens are intentionally compact (`density: compact`) to maximise data density in the ESG dashboard, which typically displays a large number of metric values simultaneously.

### Quick Verification

```bash
./.venv/bin/python -m py_compile capabilities/ecd/esg/__init__.py capabilities/ecd/esg/capability_contract.py capabilities/ecd/esg/service.py capabilities/ecd/esg/api.py capabilities/ecd/esg/views.py
./.venv/bin/pytest -q capabilities/ecd/esg/tests/test_package_contract.py
./.venv/bin/apg capabilities publish-plan capabilities/ecd/esg --json
```

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Real-Time Scope 3 Value-Chain Emission Tracing via NATS** [Streaming Architecture]
- **I2. Science-Based Target Validation Engine (SBTi Protocol)** [Domain Intelligence]
- **I3. Digital Product Passport (DPP) Carbon Footprint Embedding** [Engineering Design Integration]
- **I4. Parametric Insurance Trigger for Physical Climate Risk** [Risk Finance Integration]
- **I5. CSRD ESRS Double Materiality Automated Gap Analysis** [Regulatory Compliance]
- **I6. Embedded LLM Sustainability Narrative Generator (Local Ollama)** [AI / Reporting]
- **I7. Carbon Budget Accounting with Remaining Budget Drawdown** [Carbon Accounting]
- **I8. Biodiversity Net Gain Calculator (BNG Units per UK/TNFD)** [Nature & Biodiversity]
- **I9. NATS-Driven Real-Time Regulatory Alert Subscription** [Regulatory Intelligence / Streaming]
- **I10. Automated Internal Carbon Pricing (ICP) Allocation Engine** [Carbon Economics]
- **I11. Supply Chain Forced Labour Risk Screening (UFLPA / LkSG)** [Social Compliance / Supply Chain]
- **I12. Scope 3 Category Mapping via Spend-Based MRIO Model** [Carbon Accounting]
- **I13. ESG-Linked KPI Vesting Schedule Validator (Exec Compensation)** [Governance]
- **I14. Portfolio-Level SFDR PAI Indicator Aggregation** [Financial Regulation / Reporting]
- **I15. Continuous Assurance Stream for GHG Verification (ISO 14064-3)** [Data Integrity / Assurance]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
