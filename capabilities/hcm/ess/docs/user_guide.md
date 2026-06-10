# Employee Self-Service — User Guide

## Overview

The Employee Self-Service (ESS) capability empowers employees to manage their own HR interactions without requiring HR staff intervention for routine tasks.

## Use Cases

- Submit and track leave requests
- Download payslips for any period
- Update personal contact and banking details
- Submit expense claims with receipts
- Enrol in benefit plans (medical, pension, etc.)
- Register for training courses and track completions

## Leave Management

Submit a leave request via `POST /api/hcm/ess/leave-requests`. Supported leave types: `annual`, `sick`, `maternity`, `paternity`, `compassionate`, `unpaid`, `study`. Leave balances are automatically updated on approval.

## Payslip Access

Payslips are available via `GET /api/hcm/ess/payslips?employee_id=<id>`. Deductions (PAYE, NSSF, NHIF) are computed automatically if not provided.

## Expense Claims

Claims flow: `draft → submitted → approved → paid`. Only draft claims can be edited or deleted.

## Benefits Enrolment

Coverage tiers: `individual`, `spouse`, `family`. Employee and employer contributions are computed from the tier.

## Training Registration

Register for training via `POST /api/hcm/ess/training-registrations`. On completion, attach a certificate URL and score via `complete_training`.

## API Quick Reference

All endpoints accept `tenant_id` as a query parameter (default: `"default"`).

```
GET  /api/hcm/ess/health
GET  /api/hcm/ess/describe
POST /api/hcm/ess/leave-requests
GET  /api/hcm/ess/payslips?employee_id=<id>
POST /api/hcm/ess/expense-claims
POST /api/hcm/ess/benefit-enrolments
POST /api/hcm/ess/training-registrations
GET  /api/hcm/ess/dashboard
```
