# Tenant Management — User Guide

**Capability**: `realestate_ten` | **Domain**: `realestate` | **Version**: `1.1.0`
**© 2025 Datacraft** | www.datacraft.co.ke

---

## Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Tenant Lifecycle](#tenant-lifecycle)
5. [Onboarding Workflow](#onboarding-workflow)
6. [Deposit Management](#deposit-management)
7. [Rent Arrears Tracking](#rent-arrears-tracking)
8. [Service Request Management](#service-request-management)
9. [Communications](#communications)
10. [Satisfaction Surveys](#satisfaction-surveys)
11. [Scoring and Analytics](#scoring-and-analytics)
12. [Compliance Calendar](#compliance-calendar)
13. [Break Clause Management](#break-clause-management)
14. [Guarantor Management](#guarantor-management)
15. [Lease Incentives](#lease-incentives)
16. [Escalation Management](#escalation-management)
17. [Retention and Churn Analytics](#retention-and-churn-analytics)
18. [REST API Reference](#rest-api-reference)
19. [Configuration Reference](#configuration-reference)
20. [Composability](#composability)

---

## Overview

`realestate_ten` covers the complete tenant relationship lifecycle for commercial and residential portfolios:

- **Onboarding**: 10-step workflow with mandatory-step gating. A tenant cannot be activated until referencing, credit check, and deposit registration are recorded.
- **Deposit protection**: Full lifecycle from scheme registration through deduction claims to return processing.
- **Rent arrears**: Per-period tracking with a configurable 4-stage escalation ladder (reminder, formal notice, legal referral).
- **Service requests**: Typed requests with per-type SLA deadlines, breach detection, and performance reporting.
- **Communications**: Multi-channel (email, SMS, WhatsApp, portal, letter, phone, in-person) with delivery tracking.
- **Satisfaction**: Multi-dimension surveys, trend analysis, and automatic review triggering on low scores.
- **Scoring**: Model-based 0–100 tenant score plus composite 4-dimension relationship health score with Platinum/Gold/Silver/Standard tier.
- **Churn prediction**: 5-signal probabilistic model producing a 0–1 churn probability with recommended actions.
- **Break clauses**: Registration, condition checking, and eligibility verdicts for both tenant and landlord breaks.
- **Guarantors**: Limited and unlimited guarantee registration with coverage gap validation.
- **Lease incentives**: Rent-free periods, fit-out contributions, and stepped rent with daily amortisation rates.
- **Compliance calendar**: Forward-looking obligation schedule across covenant reviews, rent reviews, and vacating notices.

---

## Installation

```bash
pip install apg-realestate-ten
```

Or in a `pyproject.toml`:

```toml
[project.dependencies]
apg-realestate-ten = ">=1.1.0"
```

---

## Quick Start

```python
import asyncio
from decimal import Decimal
from datetime import date, timedelta
from capabilities.realestate.ten.service import TenService
from capabilities.realestate.ten.models import (
    TenantEntityCreate, TenantType,
    OnboardingStepRecord, OnboardingStep,
)

svc = TenService(tenant_id="acme", actor_id="ops-agent")

async def main():
    # Register tenant
    tenant = await svc.register_tenant(TenantEntityCreate(
        tenant_id="acme",
        name="Widget Corp",
        tenant_type=TenantType.corporate,
        email="facilities@widgetcorp.com",
        created_by="ops-agent",
    ))
    print(f"Tenant registered: {tenant.id} status={tenant.status.value}")

    # Complete mandatory onboarding steps
    for step in [OnboardingStep.referencing, OnboardingStep.credit_check, OnboardingStep.deposit_registration]:
        await svc.complete_onboarding_step(OnboardingStepRecord(
            tenant_id="acme",
            tenant_entity_id=tenant.id,
            step=step,
            completed_by="ops-agent",
        ))

    # Activate
    active = await svc.activate_tenant(tenant.id, "acme")
    print(f"Tenant active: {active.status.value}")

    # Register deposit
    deposit = await svc.register_deposit(
        tenant_entity_id=tenant.id,
        tenant_id="acme",
        unit_id="U-101",
        deposit_amount=Decimal("12000.00"),
        scheme_name="DPS Custodial",
        certificate_reference="DPS-2025-00123",
    )
    print(f"Deposit registered: {deposit['id']}")

asyncio.run(main())
```

---

## Tenant Lifecycle

### Statuses

| Status | Description |
|--------|-------------|
| `prospect` | Initial registration |
| `applicant` | Formal application submitted |
| `approved` | Application approved, awaiting lease signing |
| `active` | Fully onboarded, live tenancy |
| `notice_served` | Vacating notice received |
| `vacating` | In checkout process |
| `former` | Tenancy ended |
| `blacklisted` | Permanently blocked from activation |

### Registering a Tenant

```python
tenant = await svc.register_tenant(TenantEntityCreate(
    tenant_id="acme",
    name="Acme Ltd",
    tenant_type=TenantType.sme,
    email="contact@acme.com",
    phone="+254700000000",
    created_by="agent-1",
))
```

### Activation

Activation is gated. The following three onboarding steps must be completed first:

- `referencing`
- `credit_check`
- `deposit_registration`

If any are missing, `activate_tenant()` raises `ValueError: rule_denied`.

### Blacklisting

```python
await svc.blacklist_tenant(tenant_id_entity, tenant_id, reason="Persistent rent arrears and property damage")
```

Blacklisted tenants cannot be activated regardless of onboarding completion.

---

## Onboarding Workflow

### Steps (in order)

| Step | Mandatory |
|------|-----------|
| `application_received` | No |
| `referencing` | **Yes** |
| `credit_check` | **Yes** |
| `right_to_rent` | No |
| `lease_negotiation` | No |
| `lease_signing` | No |
| `deposit_registration` | **Yes** |
| `key_handover` | No |
| `welcome_pack_sent` | No |
| `portal_activated` | No |

### Completing a Step

```python
from capabilities.realestate.ten.models import OnboardingStepRecord, OnboardingStep

await svc.complete_onboarding_step(OnboardingStepRecord(
    tenant_id="acme",
    tenant_entity_id=tenant_id,
    step=OnboardingStep.right_to_rent,
    completed_by="agent-1",
    notes="Passport and BRP verified",
    document_ids=["doc-001", "doc-002"],
))
```

### Checking Progress

```python
progress = await svc.get_onboarding_progress(tenant_id_entity, tenant_id)
# Returns: completed_steps, remaining_steps, mandatory_complete, portal_active, completion_pct
```

### Generating a Checklist

```python
checklist = await svc.tenant_onboarding_checklist(
    tenant_id_entity=tenant_id,
    unit_id="U-101",
    tenant_id="acme",
)
for item in checklist["checklist"]:
    status = "DONE" if item["completed"] else ("REQUIRED" if item["mandatory"] else "optional")
    print(f"  [{status}] {item['step']}: {item['description']}")
```

---

## Deposit Management

UK and Kenyan residential tenancies require deposit protection. This module tracks the full lifecycle.

### Registering a Deposit

```python
deposit = await svc.register_deposit(
    tenant_entity_id=tenant_id,
    tenant_id="acme",
    unit_id="U-101",
    deposit_amount=Decimal("15000.00"),
    scheme_name="MyDeposits",
    certificate_reference="MYD-2025-44567",
    registered_date=date.today(),
)
```

### Processing a Return at Checkout

```python
result = await svc.process_deposit_return(
    tenant_entity_id=tenant_id,
    tenant_id="acme",
    deductions=[
        {"reason": "Carpet replacement", "amount": "2500.00", "evidence_reference": "photo-checkout-001"},
        {"reason": "Professional clean", "amount": "450.00", "evidence_reference": "invoice-clean-001"},
    ],
    return_method="bank_transfer",
)
print(f"Net return: {result['return_amount']}")
```

**Rule**: Total deductions cannot exceed the gross deposit amount. `ValueError` is raised if they do.

---

## Rent Arrears Tracking

### Recording a Payment Period

```python
arrears = await svc.track_rent_arrears(
    tenant_entity_id=tenant_id,
    tenant_id="acme",
    period="2025-11",
    amount_due=Decimal("85000.00"),
    amount_paid=Decimal("70000.00"),
    due_date=date(2025, 11, 1),
    unit_id="U-101",
    payment_reference="PMT-2025-11-001",
)
print(f"Arrears balance: {arrears['arrears_balance']}, Stage: {arrears['escalation_stage']}")
```

### Escalation Stages

| Days Overdue | Stage |
|-------------|-------|
| 0–6 | `monitoring` |
| 7–13 | `reminder` |
| 14–27 | `formal_notice` |
| 28+ | `legal_referral` |

### Portfolio Arrears Summary

```python
summary = await svc.get_arrears_summary(tenant_id="acme")
print(f"Total arrears: {summary['total_arrears_balance']}, Worst stage: {summary['worst_escalation_stage']}")
```

---

## Service Request Management

### Raising a Request

```python
from capabilities.realestate.ten.models import ServiceRequestCreate, ServiceRequestType, CommunicationChannel

req = await svc.raise_service_request(ServiceRequestCreate(
    tenant_id="acme",
    tenant_entity_id=tenant_id,
    property_id="PROP-001",
    unit_id="U-101",
    request_type=ServiceRequestType.maintenance_request,
    subject="Heating unit failure",
    description="Boiler stopped working on 2025-11-15",
    preferred_channel=CommunicationChannel.portal,
    created_by="tenant-portal",
))
print(f"Ref: {req.ref}, SLA deadline: {req.sla_response_deadline}")
```

### SLA Deadlines by Request Type

| Type | Hours |
|------|-------|
| `noise_complaint` | 2 |
| `maintenance_request` | 4 |
| `access_request` | 8 |
| `general_enquiry` | 24 |
| Critical priority (any) | 1 |

### SLA Performance Report

```python
report = await svc.get_sla_performance_report(tenant_id="acme", period="2025-11")
for row in report["by_request_type"]:
    print(f"  {row['request_type']}: compliance={row['compliance_pct']}%, avg_resolution={row['avg_resolution_hours']}h")
```

---

## Communications

### Sending a Communication

```python
from capabilities.realestate.ten.models import CommunicationCreate, CommunicationChannel

comm = await svc.send_communication(CommunicationCreate(
    tenant_id="acme",
    tenant_entity_id=tenant_id,
    channel=CommunicationChannel.email,
    subject="Scheduled building maintenance — 20 Nov 2025",
    body="Please be advised that routine maintenance will be carried out on 20 November...",
    sent_by="property-manager",
    direction="outbound",
    created_by="property-manager",
))
```

### Sending a Welcome Pack

```python
welcome = await svc.welcome_communication(
    tenant_id_entity=tenant_id,
    tenant_id="acme",
    channel="email",
    unit_id="U-101",
    property_name="Capital Square",
)
```

---

## Satisfaction Surveys

### Recording Survey Responses

```python
from capabilities.realestate.ten.models import SatisfactionSurveyCreate

survey = await svc.record_satisfaction_survey(SatisfactionSurveyCreate(
    tenant_id="acme",
    tenant_entity_id=tenant_id,
    property_id="PROP-001",
    survey_period="2025-Q3",
    ratings={
        "overall_satisfaction": 4,
        "maintenance_response": 3,
        "communication": 5,
        "value_for_money": 3,
    },
    comments="Response times have improved but value for money could be better.",
    created_by="survey-system",
))
print(f"Average: {survey.average_score}, Below threshold: {survey.score_below_threshold}")
```

Scores below 3.0 automatically set `review_triggered=True` and log a warning.

### Trend Analysis

```python
trend = await svc.get_satisfaction_trend(tenant_id="acme", tenant_entity_id=tenant_id)
# Returns: surveys count, average_score, trend ('improving' | 'stable' | 'declining')
```

---

## Scoring and Analytics

### Computing Relationship Health Score

The health score combines four dimensions:

| Dimension | Weight | Inputs |
|-----------|--------|--------|
| Financial Health | 35% | Credit grade, tenant score, arrears balance |
| Operational Health | 25% | SLA breach rate, open request count |
| Engagement | 25% | Survey count, communications count |
| Compliance | 15% | Onboarding completion, covenant compliance |

```python
health = await svc.compute_relationship_health_score(
    tenant_entity_id=tenant_id,
    tenant_id="acme",
)
print(f"Score: {health['composite_score']}, Tier: {health['tier']}")
for rec in health["recommendations"]:
    print(f"  - {rec}")
```

### Tenant Tiers

| Score Range | Tier |
|-------------|------|
| 85–100 | Platinum |
| 70–84 | Gold |
| 50–69 | Silver |
| 0–49 | Standard |

---

## Compliance Calendar

The compliance calendar generates a forward-looking schedule of all obligations across covenant reviews, rent reviews, and vacating notices.

```python
calendar = await svc.get_compliance_calendar(tenant_id="acme", lookahead_days=90)
print(f"Total obligations: {calendar['total_obligations']}, Urgent: {calendar['urgent_count']}")
for item in calendar["calendar"]:
    flag = "OVERDUE" if item["overdue"] else ("URGENT" if item["urgent"] else "")
    print(f"  {flag} {item['obligation_type']} — deadline: {item['deadline']} ({item['days_to_deadline']}d)")
```

---

## Break Clause Management

### Registering a Break Clause

```python
from datetime import date

clause = await svc.register_break_clause(
    tenant_entity_id=tenant_id,
    tenant_id="acme",
    unit_id="U-101",
    break_date=date(2027, 3, 25),
    notice_period_days=180,
    break_type="tenant",
    conditions=["no_rent_arrears", "no_open_escalations"],
    lease_id="LEASE-001",
)
print(f"Notice deadline: {clause['notice_deadline']}")
```

### Checking Eligibility

```python
eligibility = await svc.check_break_clause_eligibility(
    clause_id=clause["id"],
    tenant_id="acme",
)
print(f"Eligible: {eligibility['eligible']}")
for cond, met in eligibility["condition_results"].items():
    print(f"  {cond}: {'PASS' if met else 'FAIL'}")
```

---

## Guarantor Management

### Registering a Guarantor

```python
from decimal import Decimal

guarantor = await svc.register_guarantor(
    tenant_entity_id=tenant_id,
    tenant_id="acme",
    guarantor_name="John Doe",
    guarantor_email="john.doe@example.com",
    guarantee_type="limited",
    guarantee_amount=Decimal("120000.00"),
    credit_check_reference="CRA-2025-00456",
    signed_deed_reference="DEED-2025-00789",
)
```

### Validating Coverage

```python
coverage = await svc.validate_guarantor_coverage(
    tenant_entity_id=tenant_id,
    tenant_id="acme",
)
print(f"Coverage sufficient: {coverage['coverage_sufficient']}")
if not coverage['coverage_sufficient']:
    print(f"  Gap: {coverage['coverage_gap']}")
```

---

## Lease Incentives

Record non-standard economic terms for accurate financial reporting and IFRS 16 amortisation.

```python
incentive = await svc.record_lease_incentive(
    tenant_entity_id=tenant_id,
    tenant_id="acme",
    unit_id="U-101",
    incentive_type="rent_free",
    value=Decimal("255000.00"),
    start_date=date(2025, 11, 1),
    end_date=date(2026, 1, 31),
    lease_id="LEASE-001",
    description="3-month rent-free fit-out period",
)
print(f"Daily amortisation: {incentive['daily_amortisation']}")
```

Supported types: `rent_free`, `fitout_contribution`, `stepped_rent`, `rent_cap`, `cash_incentive`.

---

## Escalation Management

### Raising an Escalation

```python
from capabilities.realestate.ten.models import TenantEscalationCreate, EscalationType

esc = await svc.raise_escalation(TenantEscalationCreate(
    tenant_id="acme",
    tenant_entity_id=tenant_id,
    escalation_type=EscalationType.rent_arrears,
    description="Tenant 60 days in arrears, legal referral stage reached",
    severity="high",
    created_by="property-manager",
))
```

Escalation types: `noise_complaint`, `rent_arrears`, `lease_breach`, `property_damage`, `anti_social_behaviour`, `subletting_unauthorised`.

---

## Retention and Churn Analytics

### Predictive Churn Probability

```python
churn = await svc.predict_churn_probability(
    tenant_entity_id=tenant_id,
    tenant_id="acme",
    lease_expiry_date=date(2026, 3, 31),
)
print(f"Churn probability: {churn['churn_probability']}, Risk level: {churn['risk_level']}")
for action in churn["recommended_actions"]:
    print(f"  -> {action}")
```

Risk levels: `low` (< 0.30), `medium` (0.30–0.49), `high` (0.50–0.69), `critical` (>= 0.70).

### At-Risk Portfolio

```python
at_risk = await svc.get_retention_at_risk(tenant_id="acme")
print(f"{len(at_risk)} tenants at retention risk (score < 40)")
```

### Portfolio Analytics

```python
analytics = await svc.tenant_analytics(period="2025-Q4", tenant_id="acme")
print(f"Active: {analytics['active_tenants']}, SLA compliance: {analytics['sla_compliance_pct']}%")
```

---

## REST API Reference

All routes require `X-Tenant-ID` header.

```
GET  /realestate/ten/dashboard
GET  /realestate/ten/tenants
POST /realestate/ten/tenants
GET  /realestate/ten/tenants/<id>
PUT  /realestate/ten/tenants/<id>
POST /realestate/ten/tenants/<id>/activate
POST /realestate/ten/tenants/<id>/blacklist
POST /realestate/ten/tenants/<id>/grade
GET  /realestate/ten/onboarding/<id>
POST /realestate/ten/onboarding
GET  /realestate/ten/service-requests
POST /realestate/ten/service-requests
GET  /realestate/ten/service-requests/<id>
PUT  /realestate/ten/service-requests/<id>
POST /realestate/ten/service-requests/<id>/resolve
GET  /realestate/ten/communications
POST /realestate/ten/communications
GET  /realestate/ten/satisfaction
POST /realestate/ten/satisfaction
GET  /realestate/ten/satisfaction/<id>/trend
POST /realestate/ten/scoring
GET  /realestate/ten/escalations
POST /realestate/ten/escalations
POST /realestate/ten/escalations/<id>/resolve
GET  /realestate/ten/retention/at-risk
```

All responses follow:

```json
{"status": "ok", "data": {...}}
{"status": "error", "message": "..."}
```

---

## Configuration Reference

Set via `conf` capability or environment variables prefixed `REALESTATE_TEN_`:

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `onboarding.mandatory_steps` | `referencing,credit_check,deposit_registration` | list | Steps that must complete before activation |
| `sla.maintenance_request_hours` | `4` | int | Response SLA for maintenance requests |
| `sla.noise_complaint_hours` | `2` | int | Response SLA for noise complaints |
| `sla.access_request_hours` | `8` | int | Response SLA for access requests |
| `sla.general_enquiry_hours` | `24` | int | Response SLA for general enquiries |
| `sla.default_hours` | `12` | int | Default SLA when type not listed |
| `satisfaction.low_score_threshold` | `3` | Decimal | Average score below which review triggers |
| `retention.risk_score_threshold` | `40` | Decimal | Tenant score below which retention risk flags |
| `arrears.reminder_days` | `7` | int | Days overdue for reminder stage |
| `arrears.formal_notice_days` | `14` | int | Days overdue for formal notice stage |
| `arrears.legal_referral_days` | `28` | int | Days overdue for legal referral flag |
| `compliance_calendar.default_lookahead_days` | `90` | int | Default forward window in days |

---

## Composability

Reference in `.apg` source files:

```apg
use realestate_ten;
```

Integration points:

| External Capability | Link Field | Use Case |
|--------------------|-----------|----------|
| `realestate_ren` | `tenant_entity_id` | Tenancy agreement management |
| `realestate_mai` | `service_request_id` | Work order generation from service requests |
| `realestate_lea` | `lease_id` | Lease financial schedule linkage |
| `realestate_acc` | `tenant_entity_id` | Accounts receivable arrears sync |
| `ntfy` | internal | SLA breach, low satisfaction, retention risk notifications |
| `audl` | internal | All data access events |
| `mqeb` | internal | Tenant lifecycle event publication |
