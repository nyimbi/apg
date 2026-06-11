# EOD/BOD Processing Engine — User Guide

## Overview

The APG EOD/BOD Processing Engine automates the nightly batch lifecycle for financial institutions. It runs a fixed sequence of 10 jobs every evening, handles month-end processing automatically, and guarantees that re-running the same date is always safe (no duplicate postings).

## Concepts

### EOD Run
A single nightly execution for one tenant on one calendar date. An EOD run has:
- A unique `run_id`
- A `status`: `completed`, `partial`, `failed`, or `cancelled`
- Per-job results with counts, amounts, and error details

### BOD Run
Morning (begin-of-day) processing:
- If it's the 1st of the month: opens the new GL period
- Every morning: clears overnight float (uncleared cheques, EFT settlements)

### Idempotency
Re-running EOD for the same `(tenant_id, eod_date)` always returns the same result. There is no risk of double-posting interest, fees, or repayments.

### Dry Run
Pass `dry_run=true` to validate the full sequence without committing any postings. Useful before scheduled go-live and for auditors.

---

## Running EOD via API

### 1. Pre-flight check (recommended)

```bash
curl -H "X-Tenant-Id: my_bank" \
     http://localhost:5000/api/fin/eod/prerequisites/2026-06-11
```

Response:
```json
{
  "ok": true,
  "data": {
    "ready": true,
    "blockers": [],
    "warnings": ["EOD for 2026-06-11 already completed — will return cached result"]
  }
}
```

Blockers that prevent EOD from running:
- Non-zero suspense account balance
- EOD already in progress for this date
- Processing date is in the future

### 2. Run EOD

```bash
curl -X POST \
     -H "X-Tenant-Id: my_bank" \
     -H "Content-Type: application/json" \
     -d '{"eod_date": "2026-06-11", "dry_run": false}' \
     http://localhost:5000/api/fin/eod/run
```

### 3. Check status

```bash
curl -H "X-Tenant-Id: my_bank" \
     http://localhost:5000/api/fin/eod/status/2026-06-11
```

### 4. View full report

```bash
curl -H "X-Tenant-Id: my_bank" \
     http://localhost:5000/api/fin/eod/report/2026-06-11
```

---

## Handling failures

### View exceptions

```bash
curl -H "X-Tenant-Id: my_bank" \
     http://localhost:5000/api/fin/eod/exceptions/2026-06-11
```

### Retry a single failed job

```bash
curl -X POST \
     -H "X-Tenant-Id: my_bank" \
     http://localhost:5000/api/fin/eod/jobs/2026-06-11/fee_posting_batch/retry
```

### Resolve an exception

```bash
curl -X POST \
     -H "X-Tenant-Id: my_bank" \
     -H "Content-Type: application/json" \
     -d '{"resolution": "Corrected GL mapping", "resolved_by": "ops@bank.com"}' \
     http://localhost:5000/api/fin/eod/exceptions/{exception_id}/resolve
```

### Emergency stop

```bash
curl -X POST \
     -H "X-Tenant-Id: my_bank" \
     -H "Content-Type: application/json" \
     -d '{"reason": "FX rates not loaded"}' \
     http://localhost:5000/api/fin/eod/cancel/2026-06-11
```

---

## Job reference

| Job name | What it does | Month-end only |
|----------|--------------|----------------|
| `pre_eod_validations` | Suspense check, unposted entry count | No |
| `interest_accrual_batch` | Post daily interest (actual/365) | No |
| `fee_posting_batch` | Post monthly/quarterly fees due today | No |
| `dormancy_check_batch` | Flag accounts inactive >365 days | No |
| `term_deposit_maturity_batch` | Payout or auto-renew maturing TDs | No |
| `loan_repayment_batch` | Collect due instalments, update arrears | No |
| `standing_order_batch` | Execute standing transfers | No |
| `fx_revaluation` | Restate FCY at closing rates | **Yes** |
| `period_close` | Lock period, roll retained earnings | **Yes** |
| `eod_reports_generation` | Generate management reports | No |

---

## Monitoring

```bash
# Processing trends (last 30 days)
curl -H "X-Tenant-Id: my_bank" \
     "http://localhost:5000/api/fin/eod/metrics?days=30"

# History for a date range
curl -H "X-Tenant-Id: my_bank" \
     "http://localhost:5000/api/fin/eod/history?from_date=2026-05-01&to_date=2026-06-11"

# Health
curl http://localhost:5000/api/fin/eod/health
```

---

## Common workflows

### Month-end close
EOD automatically detects month-end dates. On `2026-06-30`, FX revaluation and period close run after the standard jobs. No special flag needed.

### Year-end
`is_year_end=True` is set in the EOD result for December 31. Additional year-end steps (audit provisions, annual report) can be triggered by checking this flag in downstream systems.

### Re-processing after correction
1. Resolve the exception via API
2. Retry the specific failed job
3. Re-generate the EOD report

The idempotency layer ensures no double-posting.
