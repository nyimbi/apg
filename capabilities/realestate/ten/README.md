# Tenant Management

## Overview
Full tenant lifecycle from prospect registration through onboarding (10-step workflow with mandatory-step gating), service request management with SLA enforcement, multi-channel communication portal, satisfaction surveying with automatic review triggers, tenant scoring and credit grading, escalation management, and retention risk analytics.

## Capability ID
`realestate_ten`

## Provides
- `tenant_onboarding_workflow`: 10-step onboarding with mandatory-step prerequisite enforcement
- `tenant_communication_portal`: 7-channel portal (email, SMS, WhatsApp, letter, in-person)
- `service_request_management`: Typed requests with SLA deadlines and breach escalation
- `tenant_scoring_engine`: 5-model scoring (payment history, lease compliance, satisfaction)
- `satisfaction_tracking`: Multi-dimension survey with low-score automatic review trigger
- `tenant_document_management`: 9 document types with access logging
- `tenant_event_timeline`: Full audit timeline of tenant interactions
- `escalation_management`: 6 escalation types including anti-social and unauthorised subletting
- `tenant_performance_reporting`: Retention rate, satisfaction trend, scoring distribution
- `tenant_retention_analytics`: At-risk identification using score thresholds

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

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/ten/tenants` | GET/POST | List/register tenants | `tenants` |
| `/realestate/ten/tenants/<id>/activate` | POST | Activate tenant | `tenants` |
| `/realestate/ten/tenants/<id>/blacklist` | POST | Blacklist tenant | `tenants` |
| `/realestate/ten/onboarding/<id>` | GET | Onboarding progress | `onboarding` |
| `/realestate/ten/onboarding` | POST | Complete step | `onboarding` |
| `/realestate/ten/service-requests` | GET/POST | Service requests | `service_requests` |
| `/realestate/ten/service-requests/<id>/resolve` | POST | Resolve request | `service_requests` |
| `/realestate/ten/communications` | GET/POST | Communications | `communications` |
| `/realestate/ten/satisfaction` | GET/POST | Surveys | `satisfaction` |
| `/realestate/ten/satisfaction/<id>/trend` | GET | Trend analysis | `satisfaction` |
| `/realestate/ten/scoring` | POST | Calculate score | `scoring` |
| `/realestate/ten/escalations` | GET/POST | Escalations | `escalations` |
| `/realestate/ten/retention/at-risk` | GET | At-risk tenants | `retention` |

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

## Edge Cases Handled
- Blacklisted tenants blocked at activation regardless of onboarding completion
- Three mandatory steps checked as a set (referencing, credit_check, deposit_registration)
- SLA response deadline calculated per request type at creation time
- Low satisfaction score triggers `review_triggered=True` automatically in same transaction
- Tenant data access logging is a hard gate, not advisory
- Retention risk uses tenant score < 40 (configurable) as threshold

## Composability Notes
- Tenant entities link to `realestate_ren` tenancies (tenant_entity_id)
- Service requests may generate `realestate_mai` work orders
- Communications sync with `realestate_ren` notice management
- Satisfaction data feeds landlord relationship management workflows
