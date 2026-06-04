# Tax Administration — User Guide

© 2025 Datacraft | Author: Nyimbi Odero

---

## Overview

The APG Tax Administration capability provides a complete, end-to-end revenue authority platform covering taxpayer registration, return filing, assessment, debt collection, audit management, refunds, objections, appeals, and exchange of information.

It is designed for national and sub-national revenue authorities managing multiple tax types simultaneously, with full multi-tenancy for regional deployments.

---

## Getting Started

### 1. Dashboard

Navigate to `/tax/dashboard`. The dashboard shows:

| KPI | Description |
|-----|-------------|
| Registered Taxpayers | Total registered (all statuses) |
| Active Taxpayers | Status = active |
| Returns Filed YTD | Year-to-date filings |
| Returns Overdue | Filed but past due date |
| Assessments Pending | Status = issued, awaiting payment |
| Total Tax Assessed | Sum of all assessed amounts |
| Total Tax Collected | Sum of confirmed payments |
| Outstanding Debt | Sum of unpaid balances |
| Open Objections | Submitted + under review |
| Open Audits | Planned + in progress |
| Compliance Rate | Active taxpayers who filed / total active |
| Collection Rate | Collected / assessed |

---

## Taxpayer Registration

### Register a New Taxpayer

1. Navigate to `/tax/taxpayers/register`
2. Fill in:
   - **Taxpayer Name** (legal name, required)
   - **Taxpayer Type**: individual / company / partnership / trust / government_entity / ngo / foreign_entity
   - **National ID** (for individuals) or **Business Registration Number** (for companies)
   - **Email**, **Phone**, **Physical Address** (optional but recommended)
   - **Evidence Reference** — document reference for audit trail
3. Click **Register**. The system generates a KRA-format PIN automatically.

### Taxpayer Statuses

| Status | Meaning |
|--------|---------|
| pending | Registered, awaiting verification |
| active | Verified, fully operational |
| suspended | Temporarily restricted |
| deregistered | Ceased operations |
| under_investigation | Subject to investigation |
| blocked | Access blocked by authority |

### Searching for Taxpayers

Use the search bar at `/tax/taxpayers` to search by:
- **Name** (partial match)
- **PIN** (exact)
- **National ID** (exact)
- **Phone** (exact)

### TIN Verification

API: `GET /api/v1/tax/taxpayers/{tin}/verify?country=KE`

Returns format validity + existence check. Useful for third-party integrations (banks, procurement portals).

---

## Return Filing

### Supported Return Types

| Return Type | Tax | Frequency |
|-------------|-----|-----------|
| monthly_vat | VAT | Monthly |
| annual_income | Income Tax | Annual |
| quarterly_advance | Advance Tax | Quarterly |
| withholding_tax_return | WHT | Monthly |
| corporate_annual | Corporate Tax | Annual |
| customs_entry | Customs Duty | Per transaction |
| turnover_tax_monthly | Turnover Tax | Monthly |
| capital_gains | CGT | Per transaction |

### Filing a Return

1. Navigate to `/tax/returns/file`
2. Enter:
   - **TIN** (taxpayer PIN)
   - **Tax Type** and **Period** (e.g. "2025-01" for Jan 2025, "2025" for annual)
   - Financial figures: gross income, deductions, tax liability, credits, tax paid
3. Submit. The system validates consistency: `taxable_income = gross_income - deductions`

### Nil Returns

File a nil return for periods with no activity: `POST /api/v1/tax/returns/nil`

Required: `tax_pin`, `tax_type`, `period`

### Amending Returns

File an amendment referencing the original return ID. The original is marked `amended`; the new record carries `is_amended=true` and `original_return_id`.

---

## Assessments

### Assessment Types

| Type | When Used |
|------|-----------|
| self_assessment | Taxpayer self-declares |
| amended_assessment | Correction to prior assessment |
| best_judgement | No return filed; authority estimates |
| audit_assessment | Result of formal audit |
| estimated_assessment | Based on industry benchmarks |
| agency_assessment | Third-party agency data |

### Penalty & Interest Calculation

After issuing an assessment, use `POST /api/v1/tax/assessments/{id}/penalty-interest` with `payment_date` to calculate:

- **Late filing penalty**: 5% of tax due (minimum KES 2,000)
- **Late payment interest**: 1% per month (commenced) on outstanding balance

---

## Debt Collection Workflow

1. **Issue Demand Notice** — `POST /api/v1/tax/debts/demand-notice`
   - Required before any collection action
   - Sets a formal deadline
2. **Collection Action** — `POST /api/v1/tax/debts/collection-action`
   - Types: payment_plan, garnishment, bank_levy, salary_attachment, asset_seizure, legal_proceedings, write_off
3. **Payment** — `POST /api/v1/tax/payments`
4. **Allocate Payment** — `POST /api/v1/tax/payments/{id}/allocate`
   - Automatically allocates to oldest debts first (FIFO by due date)

### Debt Aging Report

`GET /api/v1/tax/reports/delinquency?as_of=2025-12-31`

Buckets: 0–30 days, 31–90, 91–180, 180+

---

## Audit Case Management

### Audit Types

| Type | Description |
|------|-------------|
| desk_audit | Document review at office |
| field_audit | On-site inspection |
| it_audit | IT systems review |
| transfer_pricing | TP documentation review |
| vat_refund_audit | Validate refund claims |
| forensic_audit | Fraud investigation |
| compliance_audit | General compliance check |
| sector_audit | Industry-wide programme |

### Audit Lifecycle

1. **Open** (`POST /api/v1/tax/audits`) → status: `planned`
2. **Record Findings** (`POST /api/v1/tax/audits/{id}/findings`) → status: `in_progress`
3. **Close** (`POST /api/v1/tax/audits/{id}/close`) → status: `finalised`
   - If `final_tax_due > 0`, an audit assessment is automatically created

---

## Objections & Appeals

### Objection Rules

- Must be filed within **30 days** of assessment date
- Grounds must be substantive
- Status flow: `submitted` → `under_review` → `upheld` / `partially_upheld` / `dismissed`

### Appeal Rules

- Available only after `dismissed` or `partially_upheld` objection
- Tribunal: Tax Appeals Tribunal (default)
- Status flow: `submitted` → `registered` → `hearing_scheduled` → `heard` → `decided`

---

## Refunds

### Supported Refund Types

- `overpayment` — excess tax paid
- `input_vat_credit` — input VAT exceeds output VAT
- `withholding_tax_credit` — WHT credits exceed liability

### Refund Lifecycle

1. **Apply** (`POST /api/v1/tax/refunds`) → status: `claimed`
2. **Review** (`POST /api/v1/tax/refunds/{id}/review`) → status: `under_review`
3. **Approve** (`POST /api/v1/tax/refunds/{id}/approve`) → status: `approved`
4. **Process** → status: `paid` or `offset` (against outstanding debt)

---

## Tax Clearance Certificates

A TCC confirms a taxpayer has no outstanding tax obligations.

**Requirements for issuance:**
- Taxpayer must be `active`
- Zero outstanding debts (any balance > 0 blocks issuance)
- All required returns filed

**Verify a TCC:** `GET /api/v1/tax/clearances/verify/{certificate_number}`

Default validity: 6 months (configurable via `validity_days`).

---

## Exchange of Information (EOI)

For treaty-based information exchange (FATCA, CRS, OECD BEPS):

`POST /api/v1/tax/eoi`

```json
{
  "treaty_partner": "GB",
  "tax_pin": "A000000001X",
  "information_requested": "account_balances",
  "urgency": "routine"
}
```

Urgency levels: `routine` (90-day deadline), `urgent` (30-day), `spontaneous` (immediate).

---

## Reports

| Report | Endpoint |
|--------|----------|
| Dashboard KPIs | `GET /api/v1/tax/reports/dashboard` |
| Revenue Collection | `GET /api/v1/tax/reports/revenue?period=2025` |
| Compliance Rate | `GET /api/v1/tax/reports/compliance?period=2025` |
| Debt Aging | `GET /api/v1/tax/reports/delinquency?as_of=2025-12-31` |
| Audit Pipeline | `GET /api/v1/tax/reports/audits?period=2025` |
| Refund Analytics | `GET /api/v1/tax/reports/refunds?period=2025` |

---

## Common Workflows

### Complete VAT Return-to-Payment

```
Register taxpayer → File monthly VAT return → Issue assessment (if underpaid)
→ Calculate penalty/interest → Process payment → Allocate to debt
```

### Audit-to-Collection

```
Open audit case → Record findings → Close audit (creates assessment)
→ Issue demand notice → Initiate collection action
```

### Refund Processing

```
File return (net_tax_payable < 0) → Apply for refund → Review → Approve → Pay
```

---

## FAQ

**Q: Can I file returns for past periods?**
A: Yes. Any period string is accepted: "2020", "2020-01", "Q1-2020", "2020-01-01/2020-03-31".

**Q: What happens if I file a return after the due date?**
A: The system calculates late filing penalty (5%) and late payment interest (1%/month) automatically via the penalty-interest endpoint.

**Q: Can a TCC be revoked?**
A: Yes. Update the certificate status to `revoked` if circumstances change after issuance.

**Q: How are payments allocated?**
A: FIFO by due date. Oldest debts are settled first from each payment.

**Q: What PIN format is used?**
A: Kenya KRA format: `A` (individual) or `P` (company) + 9 digits + check letter. Example: `A123456789B`.
