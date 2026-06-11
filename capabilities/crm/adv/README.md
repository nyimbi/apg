# Advanced CRM Analytics

## Overview

Advanced CRM Analytics (`crm.adv`) is the full-lifecycle customer relationship management capability for the APG platform. It provides a governed, multi-tenant surface covering account management, contact relationship mapping, lead scoring and assignment, sales pipeline tracking, activity timelines, campaign governance, and forecast analytics — all wired to the APG event bus via Bytewax for real-time state propagation.

The capability is designed as an opinionated control plane: every write operation is policy-gated, every state transition is audited, and every agent action that crosses a privileged boundary requires explicit human approval. This makes it suitable for regulated sales environments where data quality, consent evidence, and forecast integrity are non-negotiable.

## Capability ID

`crm_adv`  Version: 1

## Provides

| Service | Description |
|---------|-------------|
| account_lifecycle | Full CRUD and hierarchy management for accounts, with owner and segment enforcement |
| contact_relationship_management | Contact profiling, relationship mapping, consent tracking, and outreach gating |
| lead_scoring_and_assignment | Automated lead scoring, territory-aware assignment with policy attachment |
| sales_pipeline_management | Opportunity staging, amount validation, close-date tracking, and win probability |
| activity_timeline | Chronological activity log across calls, emails, meetings, tasks, demos, and proposals |
| campaign_governance | Audience-gated campaign management with consent evidence and budget review thresholds |
| forecast_analytics | Evidence-backed forecasting with confidence bounds and pipeline health metrics |
| crm_agents | AI agent workbench for pipeline analysis, lead review, forecasting, and campaign governance with human-approval guardrails |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | User authentication and permission resolution for all CRM operations |
| audl | Audit logging adapter — every state change is recorded via this dependency |
| ntfy | Notification adapter for activity reminders, assignment alerts, and approval requests |
| composition_events | Event bus integration for cross-capability event routing |
| composition_config | Runtime configuration injection and tenant-level overrides |
| common_mdm | Master data management for shared reference data (territories, segments, industries) |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | `"default"` | Multi-tenant context identifier; required for all operations |
| accounts.owner_required | bool | `true` | Block account creation without an assigned owner |
| accounts.segment_required | bool | `true` | Block account creation without a segment |
| accounts.territory_supported | bool | `true` | Enable territory assignment on accounts |
| contacts.consent_required_for_outreach | bool | `true` | Require consent evidence before outreach contact creation |
| contacts.relationship_mapping_supported | bool | `true` | Enable contact-to-contact relationship graph |
| leads.source_required | bool | `true` | Block lead creation without a declared source |
| leads.score_required_for_assignment | bool | `true` | Block lead assignment until a score is present |
| leads.assignment_policy_required | bool | `true` | Require assignment policy attachment before lead assignment |
| opportunities.account_required | bool | `true` | Block opportunity creation without a linked account |
| opportunities.stage_required | bool | `true` | Block opportunity creation without a sales stage |
| opportunities.amount_required | bool | `true` | Block opportunity creation without a positive amount |
| opportunities.close_date_required | bool | `true` | Block opportunity creation without a close date |
| activities.owner_required | bool | `true` | Block activity creation without an assigned owner |
| activities.next_step_required_for_open_pipeline | bool | `true` | Flag activities on open pipeline with no next step for review |
| analytics.forecast_evidence_required | bool | `true` | Block forecast recording without supporting evidence |
| analytics.confidence_required | bool | `true` | Block forecast recording without a confidence value in [0, 1] |
| analytics.pipeline_health_supported | bool | `true` | Enable pipeline health scoring |
| campaigns.audience_required | bool | `true` | Block campaign launch without a defined audience |
| campaigns.consent_required | bool | `true` | Block campaign launch without consent evidence |
| campaigns.budget_review_required | bool | `true` | Trigger budget review for campaigns exceeding $50,000 |
| crm_agents.enabled | bool | `true` | Enable the CRM agent workbench |
| crm_agents.human_approval_required | bool | `true` | Gate all privileged agent actions on human approval |
| crm_agents.max_autonomous_scope | string | `"recommend_validate_and_prepare"` | Maximum autonomous scope for CRM agents |
| governance.require_tenant_context | bool | `true` | Deny all operations without tenant context |
| governance.audit_state_changes | bool | `true` | Emit audit events on every state change |
| governance.policy_attached_for_writes | bool | `true` | Require policy attachment on all write operations |
| governance.privacy_review_for_bulk_outreach | bool | `true` | Require privacy review before bulk outreach campaigns |
| observability.event_stream | string | `"apg.crm.adv.lifecycle"` | Bytewax stream name for CRM lifecycle events |
| ui.enable_dashboard | bool | `true` | Render the CRM analytics dashboard |
| theme.default_theme | string | `"crm_adv_control"` | Default visual theme; tenant overrides permitted |
| max_records_per_page | int | `100` | Pagination cap (10–1000) |
| cache_ttl_seconds | int | `300` | Cache TTL in seconds (60–3600) |
| background_job_timeout | int | `3600` | Max background job runtime in seconds (300–7200) |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | `/crm-adv/dashboard` | GET | `crm_adv:view` | Overview |
| accounts | `/crm-adv/accounts` | GET/POST/PUT/DELETE | `crm_adv:manage_accounts` | Accounts |
| contacts | `/crm-adv/contacts` | GET/POST/PUT/DELETE | `crm_adv:manage_contacts` | Contacts |
| leads | `/crm-adv/leads` | GET/POST/PUT/DELETE | `crm_adv:manage_leads` | Pipeline |
| pipeline | `/crm-adv/pipeline` | GET/POST/PUT/DELETE | `crm_adv:manage_pipeline` | Pipeline |
| activities | `/crm-adv/activities` | GET/POST/PUT/DELETE | `crm_adv:manage_activities` | Engagement |
| campaigns | `/crm-adv/campaigns` | GET/POST/PUT/DELETE | `crm_adv:manage_campaigns` | Engagement |
| forecasts | `/crm-adv/forecasts` | GET/POST/PUT/DELETE | `crm_adv:forecast` | Analytics |
| agents | `/crm-adv/agents` | GET/POST/PUT/DELETE | `crm_adv:admin` | Automation |
| settings | `/crm-adv/settings` | GET/PUT | `crm_adv:admin` | Administration |

REST API prefix: `/crm-adv/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny |
| crm_write_requires_policy | Write operation without policy attachment | deny |
| account_requires_owner | `create_account` with no owner assigned | deny |
| account_requires_segment | `create_account` with no segment | deny |
| contact_outreach_requires_consent | `create_contact` with outreach enabled but no consent | deny |
| lead_requires_source | `create_lead` with no source | deny |
| lead_assignment_requires_score | `assign_lead` with no score present | deny |
| lead_assignment_requires_policy | `assign_lead` with no assignment policy | deny |
| opportunity_requires_account | `create_opportunity` with no account linked | deny |
| opportunity_requires_stage | `create_opportunity` with no sales stage | deny |
| opportunity_requires_amount | `create_opportunity` with no amount | deny |
| opportunity_amount_must_be_positive | `create_opportunity` with amount <= 0 | deny |
| opportunity_requires_close_date | `create_opportunity` with no close date | deny |
| activity_requires_owner | `record_activity` with no owner | deny |
| open_pipeline_requires_next_step | `record_activity` on open pipeline with no next step | require_review |
| forecast_requires_evidence | `record_forecast` with no supporting evidence | deny |
| forecast_requires_confidence | `record_forecast` with no confidence value | deny |
| forecast_confidence_minimum | Forecast confidence < 0 | deny |
| forecast_confidence_maximum | Forecast confidence > 1 | deny |
| campaign_requires_audience | `launch_campaign` with no audience defined | deny |
| campaign_requires_consent | `launch_campaign` with no consent evidence | deny |
| bulk_outreach_requires_privacy_review | Bulk outreach campaign without privacy review | require_review |
| large_campaign_requires_budget_review | Campaign budget > $50,000 without budget review | require_review |
| crm_batch_import_requires_bytewax | Batch import not routed through Bytewax | deny |
| crm_event_requires_bytewax | CRM lifecycle event not routed through Bytewax | deny |
| crm_agent_runtime_supported | Agent registration with unsupported runtime | deny |
| crm_agent_role_supported | Agent registration with unsupported role | deny |
| privileged_agent_crm_action_requires_human_approval | Privileged agent action without human approval recorded | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| CRMCapabilityConfig | `default_lead_score_threshold`, `default_opportunity_probability`, `customer_health_score_enabled`, `ai_recommendations_enabled`, `max_records_per_page`, `cache_ttl_seconds` |
| BaseAuditModel | `id` (UUID7), `tenant_id`, `created_at`, `updated_at`, `created_by`, `updated_by`, `version`, `status` |
| Address | `street_address`, `city`, `state_province`, `postal_code`, `country`, `address_type`, `is_primary` |
| PhoneNumber | `number`, `type`, `country_code`, `is_primary` |
| CRMContact | `first_name`, `last_name`, `email`, `phone`, `job_title`, `account_id`, `contact_type`, `lead_source`, `lead_score`, `customer_health_score`, `addresses`, `phone_numbers`, `tags` |
| CRMAccount | `account_name`, `account_type`, `industry`, `annual_revenue`, `employee_count`, `parent_account_id`, `account_owner_id`, `account_health_score`, `addresses`, `tags` |
| CRMLead | `first_name`, `last_name`, `company`, `email`, `lead_source`, `lead_status`, `lead_score`, `budget`, `owner_id`, `is_converted`, `converted_contact_id`, `converted_account_id`, `converted_opportunity_id` |
| CRMOpportunity | `opportunity_name`, `amount`, `probability`, `expected_revenue` (auto-calculated), `close_date`, `stage`, `is_closed`, `is_won`, `account_id`, `owner_id`, `win_probability_ai` |
| CRMActivity | `subject`, `activity_type`, `start_datetime`, `end_datetime`, `status`, `priority`, `is_completed`, `related_to_type`, `related_to_id`, `assigned_to_id` |
| CRMCampaign | `campaign_name`, `campaign_type`, `start_date`, `end_date`, `budget`, `actual_cost`, `expected_leads`, `actual_leads`, `status`, `is_active` |

**Enums**: `RecordStatus` (active/inactive/archived/deleted), `ContactType`, `AccountType`, `LeadSource` (10 variants), `LeadStatus` (7 stages), `OpportunityStage` (7 stages), `ActivityType` (9 types), `ActivityStatus`, `Priority`

## New Capabilities (v1.1)

### AI Sales Copilot
`async copilot_query(prompt, context_ids, tenant_id)` — Natural-language interface to CRM data. Assembles an account/opportunity/activity context bundle and routes the query to a local Ollama model. Tokens stream to NATS subject `crm.adv.copilot.{tenant_id}`. Falls back to rule-based stub when `OLLAMA_BASE_URL` is unset.

### 360-Degree Customer View
`async get_360_view(account_id, tenant_id)` — Returns a single aggregated view of account, contacts, open/closed opportunities, activity timeline, journey touchpoints, health index, churn probability, and an AI-generated account summary.

### Next-Best-Action Engine
`async next_best_action(entity_id, entity_type, tenant_id)` — Returns ranked action recommendations (`nurture_email`, `schedule_demo`, `exec_sponsor_call`, etc.) for any lead, opportunity, or account. Ollama-backed with heuristic fallback.

### Proactive Deal Risk Assessment
`async compute_deal_risk(opportunity_id, tenant_id)` — Composite 0–1 risk score based on inactivity days, close-date proximity, and win probability. Publishes `deal_at_risk` to NATS when score >= 0.65.
`async run_deal_risk_scan(tenant_id, risk_threshold)` — Batch scan of all open opportunities; returns ranked at-risk list. Schedule via APG cron every 6 hours.

### Conversation Intelligence
`async analyze_call_transcript(activity_id, transcript, tenant_id)` — Extracts action items, competitor mentions, objections, topics, sentiment score, and talk-time ratio from call transcripts. Uses local Whisper + Ollama with keyword-heuristic fallback.

### Multi-Touch Attribution
`async compute_multi_touch_attribution(opportunity_id, model_type, tenant_id)` — Allocates deal credit across touchpoints. Models: `first_touch`, `last_touch`, `linear`, `time_decay`, `data_driven` (Shapley approximation via Ollama).

### Account-Based Marketing Target List
`async build_abm_target_list(icp_definition, limit, tenant_id)` — Scores all tenant accounts against an ICP definition (industry, ARR range, employee count, segment, geography) and returns a ranked target list.

### ARR Waterfall
`async arr_waterfall(period, tenant_id)` — Computes new ARR, expansion ARR, churn ARR, and net ARR for a period from win/loss event history. Accurate monthly revenue waterfall without external BI tools.

## Streaming Events

Events emitted to the `apg.crm.adv.lifecycle` stream via Bytewax+NATS. Stream key: `tenant_id`.

| Event | Trigger |
|-------|---------|
| `account_created` | A new account passes all creation rules and is persisted |
| `contact_created` | A new contact is created (consent verified if outreach-enabled) |
| `lead_created` | A lead with a valid source is registered |
| `lead_assigned` | A scored lead is assigned to an owner under an assignment policy |
| `opportunity_created` | An opportunity with account, stage, amount, and close date is persisted |
| `activity_recorded` | An owned activity is logged against a related record |
| `campaign_launched` | A consent- and audience-verified campaign transitions to active |
| `forecast_recorded` | An evidence-backed forecast with confidence in [0, 1] is recorded |
| `crm_agent_registered` | A CRM agent with an approved runtime and role is registered |
| `deal_at_risk` | Deal risk scan detects composite risk score >= threshold (default 0.65) |
| `call_analyzed` | A call transcript is processed and structured intelligence extracted |
| `attribution_computed` | Multi-touch attribution is computed for an opportunity |
| `nba_generated` | Next-best-action recommendations produced for an entity |
| `360_view_generated` | A 360-degree account view is assembled |
| `abm_list_built` | An ABM target list is scored and ranked |
| `copilot_queried` | A copilot query is processed |

**Lifecycle states**: `draft` -> `active` -> `qualified` -> `assigned` -> `open` -> `won` / `lost` -> `archived`

**Streaming guardrails enforced at stream level**: `crm_batch_import_requires_bytewax`, `crm_event_requires_bytewax`, `privileged_agent_crm_action_requires_human_approval`

## Edge Cases Handled

- **Zero-amount opportunities**: Rejected at rule evaluation — `opportunity_amount_must_be_positive` fires before persistence, preventing silent $0 pipeline inflation.
- **Forecast confidence drift**: Both lower (< 0) and upper (> 1) bounds are checked as separate rules, so a misconfigured float like `1.001` is caught even though the "out of range" reason is the same.
- **Lead assignment race**: `lead_assignment_requires_score` and `lead_assignment_requires_policy` are independent rules — both must pass, so partial readiness (score present, policy missing) still blocks assignment.
- **Bulk outreach vs. large budget**: `bulk_outreach_requires_privacy_review` and `large_campaign_requires_budget_review` resolve to `require_review`, not `deny`, allowing a human to override rather than hard-blocking legitimate large campaigns.
- **Converted lead lineage**: `CRMLead` stores `converted_contact_id`, `converted_account_id`, and `converted_opportunity_id` as separate nullable fields rather than a single polymorphic reference, preserving traceability for partial conversions.
- **Account hierarchy cycles**: `parent_account_id` references the same model — callers must validate DAG integrity at the service layer; the model intentionally does not enforce this to avoid recursive validation overhead.
- **Agent privileged scope**: Only actions flagged `privileged_scope=True` require human approval; recommendation-only agent actions (`recommend_validate_and_prepare`) pass without a human gate, keeping low-risk automation frictionless.
- **Tenant context on every operation**: The `tenant_context_required` rule is evaluated first in rule order, short-circuiting all other rules with a `deny` if context is absent, preventing cross-tenant data leakage via misconfigured requests.

## Composability

- **Upstream**: `common_mdm` provides reference data (territories, segments, industry codes) consumed by account and lead management. `auth` resolves user identities used in owner and assignment fields. `composition_config` injects tenant-level configuration overrides at runtime.
- **Downstream**: `intel.alerts` and `intel.correlation` subscribe to the `apg.crm.adv.lifecycle` stream for anomaly detection on pipeline changes and lead scoring drift. `fintech.terminal` can consume opportunity and forecast data for revenue projection models. Notification-driven capabilities (`ntfy`) receive activity reminders and approval-request events.
- **Peer**: `crm.adv` is commonly deployed alongside `crm.ord` (order management), `crm.quo` (quoting), and `crm.pri` (pricing) as a sales-cycle stack. Campaign capabilities in `crm.mkt` share the consent model and audience definitions produced here.

## Development Notes

- **Rule engine is purely deterministic**: `evaluate_capability_rules` iterates all rules without short-circuiting on first `deny`. Multiple violations are surfaced in a single call — intentional for UX. The final `decision` reflects the worst-case across all matched rules.
- **`_deep_merge` is destructive toward source values**: Tenant config overrides applied via `get_capability_contract(overrides=...)` replace scalars in the base config but recursively merge nested dicts. Callers should not pass partial nested objects expecting base defaults to fill gaps at that level.
- **`CRMOpportunity.expected_revenue` auto-calculation**: The `@validator` runs `always=True`, so it fires even when `expected_revenue` is `None`. It will silently produce `None` if `amount` or `probability` is missing from the `values` dict (which happens when those fields fail their own validators). Tests should cover partial-failure combinations.
- **Pydantic v1 validators in models.py**: `models.py` uses `@validator` and `root_validator` (Pydantic v1 API) despite the `ConfigDict` import from Pydantic v2. This is a migration in progress — new models should use `@field_validator` and `model_validator` from Pydantic v2.
- **`uuid_extensions` shim needed**: `models.py` imports `from uuid_extensions import uuid7str`. The `uuid_extensions` package is not on PyPI; replace with the project-local shim wrapping `uuid6.uuid7`. See root `CLAUDE.md` for the canonical shim location.
- **Supported agent runtimes**: `codex`, `claude_code`, `opencode`, `pi`. Adding a new runtime requires updating `SUPPORTED_CRM_AGENT_RUNTIMES` in `capability_contract.py` and redeploying the contract; the rule `crm_agent_runtime_supported` reads from this list at evaluation time.
- **Theme tokens**: The `crm_adv_control` theme uses compact density. UI components map entity types to specific visual primitives (score-lanes for leads, stage-board for pipeline, forecast-grid for analytics) — override these in `THEME["components"]` per tenant as needed.
