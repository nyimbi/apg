# Employee Self-Service — User Guide

## Overview

The Employee Self-Service (ESS) capability empowers employees to manage their own HR interactions without HR staff intervention for routine tasks. It covers leave management, payslip access, personal data, expense claims, benefits, training, TOIL, statutory reporting, and notifications.

## Leave Management

### Submitting a Leave Request

POST `/api/hcm/ess/leave-requests` with:

```json
{
  "employee_id": "EMP001",
  "leave_type": "annual",
  "start_date": "2026-07-01",
  "end_date": "2026-07-05",
  "reason": "Family holiday",
  "handover_to": "EMP002"
}
```

Supported leave types: `annual`, `sick`, `maternity`, `paternity`, `compassionate`, `unpaid`, `study`.

Leave balances are automatically deducted on approval.

### Clash Detection

Before submitting, call `check_leave_conflicts` to surface any overlapping approved or pending leave:

```
GET /api/hcm/ess/leave-conflicts?employee_id=EMP001&start_date=2026-07-01&end_date=2026-07-05
```

Returns a list of conflicting request IDs. An empty list means the dates are clear.

### Leave Accrual

HR runs a monthly accrual to credit earned leave days:

```
POST /api/hcm/ess/leave-accrual
{
  "employee_id": "EMP001",
  "period_end_date": "2026-06-30",
  "proration_factor": 1.0
}
```

Default rates: annual = 1.75 days/month, sick = 0.833 days/month, study = 0.417 days/month. Override via `accrual_rates`.

For bulk end-of-month runs, use `POST /api/hcm/ess/leave-accrual/batch` with a list of employee IDs. Failures are reported per-employee without aborting other accruals.

### TOIL (Time-Off-In-Lieu)

Record overtime worked to earn compensatory leave:

```
POST /api/hcm/ess/toil
{
  "employee_id": "EMP001",
  "overtime_date": "2026-06-07",
  "hours_worked": 4,
  "rate": 1.5,
  "notes": "Saturday deployment support"
}
```

TOIL days credited = (hours × rate) / 8. Typical rates: 1.5 for weekend, 2.0 for public holidays. View balance at `GET /api/hcm/ess/toil/{employee_id}`.

## Payslip Access

### Listing and Downloading

```
GET /api/hcm/ess/payslips?employee_id=EMP001&year=2026
```

Payslips are sorted most-recent first. PAYE, NSSF, and NHIF are computed automatically when `deductions_breakdown` is omitted.

### Year-to-Date Summary

```
GET /api/hcm/ess/payslips/ytd?employee_id=EMP001&year=2026
```

Returns gross pay, total deductions, net pay, and per-type deduction aggregates for the year-to-date.

## Personal Data

Update contact, banking, or statutory registration details:

```
PATCH /api/hcm/ess/personal-data/EMP001
{
  "phone": "+254712345678",
  "bank_account_number": "01234567890",
  "bank_name": "KCB",
  "kra_pin": "A001234567B"
}
```

Allowed fields: `phone`, `emergency_contact_name`, `emergency_contact_phone`, `address_line1`, `address_line2`, `city`, `county`, `country`, `bank_account_number`, `bank_name`, `bank_branch`, `nssf_number`, `nhif_number`, `kra_pin`.

## Expense Claims

### Claim Lifecycle

```
draft → submitted → approved → paid
```

Only draft claims can be edited or deleted.

### Expense Policy

HR configures per-category limits:

```
PUT /api/hcm/ess/expense-policy
{
  "limits": {"meals": 1500, "travel": 8000, "accommodation": 12000}
}
```

Validate before submitting:

```
POST /api/hcm/ess/expense-policy/validate
{"category": "meals", "amount": 2500}
```

Returns `{"compliant": false, "limit": 1500, "violation": "..."}`.

### Bulk Submit

Submit all your draft claims in one call:

```
POST /api/hcm/ess/expense-claims/bulk-submit
{
  "employee_id": "EMP001",
  "claim_ids": ["ec-abc123", "ec-def456"],
  "enforce_policy": true
}
```

Non-compliant claims are returned in `policy_violations` without blocking the rest.

## Benefits Enrolment

Coverage tiers: `individual`, `spouse`, `family`. Employee and employer contributions are computed from the tier. Enrol:

```
POST /api/hcm/ess/benefit-enrolments
{
  "employee_id": "EMP001",
  "benefit_plan_id": "MEDICAL-2026",
  "benefit_type": "medical",
  "coverage_tier": "family",
  "effective_date": "2026-01-01"
}
```

Terminate with `PUT /api/hcm/ess/benefit-enrolments/{id}/terminate`.

## Training Registration

Register, track approval, and record completion:

```
POST /api/hcm/ess/training-registrations
{
  "employee_id": "EMP001",
  "course_id": "PY-ADV-01",
  "course_name": "Advanced Python",
  "training_type": "online",
  "start_date": "2026-08-01",
  "end_date": "2026-08-31",
  "cost": 15000
}
```

On completion: `PUT /api/hcm/ess/training-registrations/{id}/complete` with `completion_date`, optional `score`, and `certificate_url`.

### Certificate Store

After completing a course, certificates are stored as first-class records:

```
POST /api/hcm/ess/certificates
{
  "employee_id": "EMP001",
  "course_name": "Advanced Python",
  "issuer": "Datacamp",
  "issued_date": "2026-09-01",
  "certificate_url": "https://certs.example.com/abc",
  "expiry_date": "2029-09-01",
  "cpd_credits": 10
}
```

List current certificates: `GET /api/hcm/ess/certificates/EMP001?include_expired=false`.

## Statutory Reporting

Generate PAYE/NSSF/NHIF statutory schedules for a pay period:

```
GET /api/hcm/ess/statutory-report?period_month=6&period_year=2026
```

Returns per-employee rows and grand totals (gross, PAYE, NSSF, NHIF, net) ready for KRA iTax, NSSF, and NHIF portal uploads.

## Notifications

### Setting Preferences

```
PUT /api/hcm/ess/notification-prefs/EMP001
{
  "email_enabled": true,
  "sms_enabled": false,
  "in_app_enabled": true,
  "event_subscriptions": ["leave_approved", "payslip_ready", "expense_claim_approved"]
}
```

`event_subscriptions: null` subscribes the employee to all ESS events.

### Sending Notifications

```
POST /api/hcm/ess/notifications/send
{
  "employee_id": "EMP001",
  "event_type": "leave_approved",
  "context": {"leave_id": "lv-abc123", "approved_by": "MGR001", "days": 5}
}
```

Actual delivery (email/SMS/push) is handled by a downstream worker that consumes `ess_notification_requested` audit events.

## API Quick Reference

All endpoints accept `tenant_id` as a query parameter (default: `"default"`).

```
GET  /api/hcm/ess/health
GET  /api/hcm/ess/describe
GET  /api/hcm/ess/dashboard
GET  /api/hcm/ess/audit-events

POST /api/hcm/ess/leave-requests
GET  /api/hcm/ess/leave-conflicts
POST /api/hcm/ess/leave-accrual
POST /api/hcm/ess/leave-accrual/batch

GET  /api/hcm/ess/payslips?employee_id=<id>
GET  /api/hcm/ess/payslips/ytd?employee_id=<id>&year=<year>
POST /api/hcm/ess/payslips

GET  /api/hcm/ess/personal-data/<employee_id>
PATCH /api/hcm/ess/personal-data/<employee_id>

POST /api/hcm/ess/expense-claims
POST /api/hcm/ess/expense-claims/bulk-submit
PUT  /api/hcm/ess/expense-policy
POST /api/hcm/ess/expense-policy/validate

POST /api/hcm/ess/benefit-enrolments

POST /api/hcm/ess/training-registrations
POST /api/hcm/ess/certificates
GET  /api/hcm/ess/certificates/<employee_id>

POST /api/hcm/ess/toil
GET  /api/hcm/ess/toil/<employee_id>

GET  /api/hcm/ess/statutory-report

PUT  /api/hcm/ess/notification-prefs/<employee_id>
POST /api/hcm/ess/notifications/send
```
