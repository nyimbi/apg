# Check-off Management — User Guide

## What is check-off?

Check-off is a payroll deduction arrangement where an employer deducts loan repayments and savings contributions from employee salaries before payment and remits the total to the SACCO.  It is the dominant collection mechanism for SACCO loans in East Africa.

## Monthly Workflow

### 1. Register the employer (once)

Register each employer that has signed a check-off agreement with the SACCO.

```
POST /api/fintech/sacco/ckf/employers
X-Tenant-ID: my_sacco

{
  "name": "Equity Bank",
  "registration_number": "C.12345/2020",
  "payroll_contact": "Jane Mwangi <payroll@equitybank.co.ke>",
  "remittance_account": "0100123456789",
  "check_off_agreement_date": "2025-01-15",
  "deduction_frequency": "monthly",
  "email": "payroll@equitybank.co.ke",
  "phone": "+254700123456"
}
```

### 2. Link members to their employer

Each SACCO member working at a check-off employer needs a link record.

```
POST /api/fintech/sacco/ckf/links
{
  "member_id": "mem-0042",
  "employer_id": "<employer_id>",
  "employee_number": "EQ-00234",
  "basic_salary": "95000.00",
  "effective_date": "2025-02-01",
  "member_name": "John Kamau"
}
```

A member can only have one active employer link at a time.  Adding a new link automatically ends the previous one.

### 3. Generate the monthly schedule

At the start of each payroll month, generate the deduction schedule for each employer.  This calculates:
- All due loan installments (principal + interest + penalties)
- Contractual savings contributions
- Any arrears from previous short payments

```
POST /api/fintech/sacco/ckf/employers/<employer_id>/schedule
{ "payroll_month": 6, "payroll_year": 2026 }
```

Send the returned schedule to the employer's payroll department.

### 4. Receive and upload the deduction file

When the employer processes payroll, they return a file showing what was actually deducted per employee.

```
POST /api/fintech/sacco/ckf/employers/<employer_id>/upload
{
  "payroll_month": 6, "payroll_year": 2026,
  "deductions": [
    {
      "member_id": "mem-0042",
      "amount_received": "7500.00",
      "loan_deductions": "5500.00",
      "savings_deductions": "2000.00"
    }
  ]
}
```

### 5. Reconcile

Compare what was expected with what was received.

```
POST /api/fintech/sacco/ckf/employers/<employer_id>/reconcile
{ "payroll_month": 6, "payroll_year": 2026 }
```

The response includes:
- `status`: `reconciled` | `short_paid` | `over_paid`
- `demand_notice_required`: true if employer under-remitted
- `excess_to_savings`: amount to credit to member savings if over-paid
- Per-member variance breakdown

### 6. Post GL receipts

After reconciliation, post the accounting entries.  This is **idempotent** — safe to call multiple times.

```
POST /api/fintech/sacco/ckf/employers/<employer_id>/post
{ "payroll_month": 6, "payroll_year": 2026 }
```

GL entries created:
- DR Check-off Receivable (1310) / CR Loan Ledger (1410) — for loan repayments
- DR Check-off Receivable (1310) / CR Savings Ledger (2110) — for savings contributions

## Handling Short Payments

If an employer remits less than expected:

1. Reconciliation sets `demand_notice_required: true` and status `short_paid`.
2. Call `POST /employers/<id>/remind` to log a reminder (integrate with your notification system).
3. If the employer still does not pay, call `POST /employers/<id>/default` to flag the period as defaulted.
4. The shortage automatically appears as `arrears` in subsequent months' schedules.

## Metrics Dashboard

```
GET /api/fintech/sacco/ckf/metrics?month=6&year=2026
```

Returns:
- Collection rate % (total received / total expected)
- Compliance rate % (employers fully paid / all employers)
- Counts of short-paying and over-paying employers
- Total outstanding amount

## Member View

Members can see their own check-off deductions:

```
GET /api/fintech/sacco/ckf/members/<member_id>/deductions
GET /api/fintech/sacco/ckf/members/<member_id>/history?months=12
```
