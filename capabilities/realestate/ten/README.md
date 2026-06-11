# Tenant Management

## Overview

Full tenant lifecycle — prospect registration through onboarding (10-step workflow with mandatory-step gating), deposit protection, rent arrears tracking with escalation ladders, service request management with SLA enforcement, multi-channel communication portal, satisfaction surveying with automatic review triggers, tenant scoring and credit grading, escalation management, break clause workflows, predictive churn scoring, relationship health scoring, guarantor management, lease incentive recording, and compliance calendar generation.

## Capability ID

`realestate_ten`

## Provides

- `tenant_onboarding_workflow`: 10-step onboarding with mandatory-step prerequisite enforcement
- `tenant_communication_portal`: 7-channel portal (email, SMS, WhatsApp, letter, in-person)
- `service_request_management`: Typed requests with SLA deadlines, breach escalation, and performance reporting
- `tenant_scoring_engine`: 5-model scoring (payment history, lease compliance, satisfaction)
- `satisfaction_tracking`: Multi-dimension survey with low-score automatic review trigger
- `tenant_document_management`: 9 document types with access logging
- `tenant_event_timeline`: Full audit timeline of tenant interactions
- `escalation_management`: 6 escalation types including anti-social and unauthorised subletting
- `tenant_performance_reporting`: Retention rate, satisfaction trend, scoring distribution
- `tenant_retention_analytics`: At-risk identification using score thresholds
- `deposit_protection_management`: Full deposit lifecycle — registration, interest, deductions, return
- `rent_arrears_tracking`: Per-period arrears with 4-stage escalation ladder
- `compliance_calendar`: Forward-looking obligation schedule with urgency flagging
- `break_clause_management`: Lease break registration, condition checking, eligibility verdict
- `relationship_health_scoring`: Composite 4-dimension health score with tier classification
- `predictive_churn_scoring`: 0–1 probability from 5 behavioural signals
- `guarantor_management`: Guarantor registration and coverage validation
- `lease_incentive_tracking`: Rent-free, fit-out, stepped rent with daily amortisation
- `sla_performance_reporting`: Per-type SLA compliance metrics with P50/P95 resolution times

## Requires

| Capability | Reason |
|-----------|--------|
| `auth` | Blacklist and activation authority |
| `audl` | Data access always logged |
| `mten` | Multi-tenant isolation |
| `conf` | Onboarding step configuration |
| `ntfy` | SLA breach, low satisfaction, retention risk alerts |
| `wflo` | Activation and escalation workflows |
| `nlpc` | Service request classification |
| `mqeb` | Publish tenant lifecycle events |
| `schd` | Survey scheduling |

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `onboarding.mandatory_steps` | referencing, credit_check, deposit_registration | Steps required before activation |
| `service_requests.sla_response_hours.maintenance_request` | 4 | Hours to respond |
| `satisfaction.low_score_threshold` | 3 | Score triggering automatic review |
| `retention.risk_score_threshold` | 40 | Tenant score below which retention risk is flagged |
| `arrears.reminder_days` | 7 | Days overdue before first arrears reminder |
| `arrears.formal_notice_days` | 14 | Days overdue before formal notice |
| `arrears.legal_referral_days` | 28 | Days overdue before legal referral flag |
| `compliance_calendar.default_lookahead_days` | 90 | Default forward window for compliance calendar |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/ten/dashboard` | GET | Portfolio summary | `view` |
| `/realestate/ten/tenants` | GET/POST | List/register tenants | `tenants` |
| `/realestate/ten/tenants/<id>` | GET/PUT | Fetch/update tenant | `tenants` |
| `/realestate/ten/tenants/<id>/activate` | POST | Activate tenant | `tenants` |
| `/realestate/ten/tenants/<id>/blacklist` | POST | Blacklist tenant | `tenants` |
| `/realestate/ten/tenants/<id>/grade` | POST | Assign credit grade | `scoring` |
| `/realestate/ten/onboarding/<id>` | GET | Onboarding progress | `onboarding` |
| `/realestate/ten/onboarding` | POST | Complete step | `onboarding` |
| `/realestate/ten/service-requests` | GET/POST | Service requests | `service_requests` |
| `/realestate/ten/service-requests/<id>` | GET/PUT | Fetch/update request | `service_requests` |
| `/realestate/ten/service-requests/<id>/resolve` | POST | Resolve request | `service_requests` |
| `/realestate/ten/communications` | GET/POST | Communications | `communications` |
| `/realestate/ten/satisfaction` | GET/POST | Surveys | `satisfaction` |
| `/realestate/ten/satisfaction/<id>/trend` | GET | Trend analysis | `satisfaction` |
| `/realestate/ten/scoring` | POST | Calculate score | `scoring` |
| `/realestate/ten/escalations` | GET/POST | Escalations | `escalations` |
| `/realestate/ten/escalations/<id>/resolve` | POST | Resolve escalation | `escalations` |
| `/realestate/ten/retention/at-risk` | GET | At-risk tenants | `retention` |

## Key Service Methods

### Core Tenant Lifecycle
- `register_tenant()` — Register prospect with type, email, contact
- `get_tenant()` / `list_tenants()` / `update_tenant()`
- `activate_tenant()` — Gates on mandatory onboarding steps
- `blacklist_tenant()` — Blocks future activation
- `assign_credit_grade()` — A/B/C/D/F grading

### Onboarding
- `complete_onboarding_step()` — Record step completion with document evidence
- `get_onboarding_progress()` — Step completion breakdown with percentage
- `tenant_onboarding_checklist()` — Full checklist with per-step descriptions

### Service Requests
- `raise_service_request()` / `service_request()` — Typed requests with SLA
- `get_service_request()` / `list_service_requests()`
- `update_service_request()` / `resolve_service_request()`
- `get_sla_performance_report()` — Per-type SLA compliance and resolution metrics

### Communications
- `send_communication()` — Directional messages across 7 channels
- `list_communications()`
- `welcome_communication()` — Structured onboarding welcome pack

### Satisfaction
- `record_satisfaction_survey()` / `satisfaction_survey()`
- `list_satisfaction_surveys()`
- `get_satisfaction_trend()` — Improving/stable/declining trend analysis

### Scoring and Analytics
- `calculate_tenant_score()` — Model-based 0–100 score
- `compute_relationship_health_score()` — 4-dimension composite with tier
- `predict_churn_probability()` — 5-signal 0–1 probability with recommended actions
- `tenant_analytics()` — Full portfolio analytics for a period
- `get_tenant_summary()` — High-level portfolio snapshot
- `get_retention_at_risk()` — Score-threshold at-risk list

### Escalations
- `raise_escalation()` / `resolve_escalation()` / `list_escalations()`

### Deposit Management
- `register_deposit()` — Scheme registration with certificate reference
- `get_deposit()` — Fetch active deposit record
- `process_deposit_return()` — Apply deductions, compute net return

### Rent Arrears
- `track_rent_arrears()` — Per-period tracking with 4-stage escalation ladder
- `get_arrears_summary()` — Total balance and worst escalation stage

### Compliance and Lease Terms
- `get_compliance_calendar()` — Forward obligation schedule with urgency flagging
- `lease_covenant_compliance()` — Point-in-time covenant compliance record
- `rent_review_notification()` — Notify tenant of proposed rent change
- `renewal_negotiation()` — Record offer/counter-offer and outcome
- `register_break_clause()` — Clause registration with conditions
- `check_break_clause_eligibility()` — Per-condition eligibility verdict
- `record_lease_incentive()` — Rent-free, fit-out, stepped rent with amortisation

### Guarantors
- `register_guarantor()` — Limited or unlimited guarantee registration
- `validate_guarantor_coverage()` — Coverage vs arrears gap analysis

### Portal and Lifecycle
- `tenant_portal_access()` — Enable/disable/reset portal access
- `vacating_notice_processing()` — Checkout workflow initiation

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| `blacklisted_tenant_activation_denied` | blacklisted | deny |
| `activation_requires_completed_onboarding` | mandatory steps missing | deny |
| `sla_breach_requires_escalation` | SLA breached, not escalated | deny |
| `satisfaction_rating_valid` | not 1–5 | deny |
| `low_score_triggers_review` | score < 3, no review | deny |
| `tenant_data_access_always_logged` | access not logged | deny |
| `onboarding_step_sequence_enforced` | prereqs not met | deny |
| `deposit_deductions_cannot_exceed_deposit` | deductions > deposit | raise ValueError |
| `rent_review_minimum_notice` | effective_date < today + notice_period | raise ValueError |
| `break_clause_conditions_enforced` | conditions not met | eligible=False |

## Data Models

- `TenantEntityCreate/Response` — tenant with type, status, credit grade, portal access
- `OnboardingStepRecord/Response` — step completion with document evidence
- `ServiceRequestCreate/Response` — typed request with SLA deadline and breach flag
- `CommunicationCreate/Response` — directional message with channel and delivery status
- `SatisfactionSurveyCreate/Response` — multi-dimension ratings with average and threshold flag
- `TenantScoreCreate/Response` — model-specific score with retention risk flag
- `TenantEscalationCreate/Response` — typed escalation with resolution tracking

## Streaming Events

- `tenant_registered`, `tenant_onboarded`, `tenant_activated`, `tenant_vacated`, `tenant_blacklisted`
- `onboarding_step_completed`, `service_request_raised`, `service_request_resolved`
- `satisfaction_survey_completed`, `tenant_score_updated`
- `escalation_raised`, `escalation_resolved`
- `retention_risk_flagged`, `communication_sent`
- `deposit_registered`, `deposit_returned`
- `rent_arrears_tracked`, `break_clause_registered`
- `guarantor_registered`, `lease_incentive_recorded`

## Edge Cases Handled

- Blacklisted tenants blocked at activation regardless of onboarding completion
- Three mandatory steps checked as a set (referencing, credit_check, deposit_registration)
- SLA response deadline calculated per request type at creation time; critical priority overrides to 1h
- Low satisfaction score triggers `review_triggered=True` automatically in same transaction
- Tenant data access logging is a hard gate, not advisory
- Retention risk uses tenant score < 40 (configurable) as threshold
- Deposit deductions validated against gross deposit before return processing
- Rent review effective_date enforced to be at least notice_period_days in the future
- Break clause eligibility checks live arrears and escalation state at check_date
- Churn probability capped at 1.0 regardless of signal accumulation

## Composability Notes

- Tenant entities link to `realestate_ren` tenancies (tenant_entity_id)
- Service requests may generate `realestate_mai` work orders
- Communications sync with `realestate_ren` notice management
- Satisfaction data feeds landlord relationship management workflows
- Deposit records link to `realestate_lea` lease financial schedules
- Arrears data feeds `realestate_acc` accounts receivable module
- Guarantor records link to `realestate_lea` lease credit support schedules
