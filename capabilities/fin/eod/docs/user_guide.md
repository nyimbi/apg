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

---

## New capabilities (v2)

### Penalty interest accrual

Penalty interest accrues on overdue loan instalments at the contractual penalty rate (default: 2× the loan rate, minimum 5 % p.a.). It is computed daily and posted to the `Penalty Interest Income` GL.

```bash
# Python SDK
result = await svc.compute_penalty_interest("my_bank", "2026-06-11")
print(result["total_penalty_accrued"])
```

To call via a custom batch trigger instead of the default EOD sequence, use `run_job` with the method directly or wire it into a custom adapter.

---

### IFRS 9 ECL staging

Each loan is classified into Stage 1, 2, or 3 daily. Stage 2 is triggered by:
- Days Past Due > 30
- Credit rating downgrade
- Watchlist flag

Stage 3 (credit-impaired): DPD ≥ 90 or legal default event.

Provision deltas are posted to `Impairment Expense` (DR) / `Loan Loss Reserve` (CR) or reversed on improvement.

```bash
result = await svc.run_ifrs9_ecl_staging("my_bank", "2026-06-30")
# {"stage_counts": {"stage_1": 4200, "stage_2": 87, "stage_3": 12}, "provision_delta": "340000.00"}
```

---

### Liquidity Coverage Ratio (LCR)

Basel III requires daily LCR monitoring. The engine computes:

```
LCR = HQLA / Net Stressed Outflows (30-day horizon)
```

HQLA tiers:

| Tier | Haircut | Examples |
|------|---------|----------|
| Level 1 | 0 % | Sovereign bonds, central bank reserves |
| Level 2A | 15 % | GSE bonds, qualifying covered bonds |
| Level 2B | 25–50 % | Non-financial equities, qualifying RMBS |

A CRITICAL exception is raised if LCR < 100 %. A warning is logged if LCR < 105 % (early-warning threshold).

```bash
result = await svc.compute_liquidity_coverage_ratio("my_bank", "2026-06-11")
# {"lcr_ratio": "1.1523", "status": "compliant"}
```

---

### Nostro reconciliation

Matches GL nostro entries against SWIFT MT940 / ISO 20022 camt.053 statement lines:

1. **Exact match** — same amount + value date + reference → auto-cleared
2. **Near match** — same amount + ±1 day value date → queued for manual review
3. **Unmatched** — CRITICAL exception raised; escalated if value > threshold

```bash
result = await svc.run_nostro_reconciliation("my_bank", "2026-06-11")
# {"matched": 1420, "near_match": 3, "unmatched": 1, "unmatched_value": "45000.00"}
```

Unmatched exceptions appear in `GET /api/fin/eod/exceptions/{date}`.

---

### ZBA sweep execution

Zero-Balance Accounting runs after all other EOD postings. For each sweep group:

- Sub-account balance > `target_balance` → surplus swept to master
- Sub-account balance < `minimum_balance` → funded from master (if master has surplus and `notional_only=false`)

```bash
result = await svc.run_zba_sweeps("my_bank", "2026-06-11")
# {"groups_processed": 12, "sweeps_up": 34, "total_swept": "8500000.00"}
```

---

### NPA classification

Loans with DPD ≥ 90 (configurable) are promoted to NPA. On promotion:

1. Account status set to `NPA`
2. Future interest redirected from P&L to `Sundry Payable`
3. Uncollected accrued interest reversed from P&L
4. 100 % provision posted

On cure (DPD drops below threshold):

1. Status restored to `Sub-standard` or `Performing`
2. Interest accrual resumed to P&L
3. Provision released per IFRS 9

```bash
result = await svc.classify_npa_accounts("my_bank", "2026-06-11", dpd_threshold=90)
# {"newly_classified": 2, "cured": 1, "provision_posted": "150000.00"}
```

---

### SLA monitoring

Configure the processing window (default 360 minutes / 6 hours). The engine evaluates:

- **In-progress**: if > 70 % of window consumed and < 50 % of jobs complete → `at_risk` status + exception
- **Completed**: actual duration vs window → `met` or `breached`

```bash
result = await svc.check_sla_compliance("my_bank", "2026-06-11", sla_window_minutes=360)
# {"status": "met", "elapsed_seconds": 187.4, "sla_breach": false}
```

Integrate with your alerting stack by subscribing to the `SLA_AT_RISK` exception code.

---

### Regulatory return generation

Returns are generated automatically based on the processing date:

| Trigger | Returns |
|---------|---------|
| Every month-end | BSL02 Balance Sheet, BSL03 Credit Exposure |
| Every quarter-end | Capital Adequacy (CAR), Large Exposure Return |
| Year-end | Annual Supervisory Return, AML Statistical Report |

```bash
result = await svc.generate_regulatory_returns("my_bank", "2026-06-30")
# {"returns_generated": ["BSL02_BALANCE_SHEET", "BSL03_CREDIT_EXPOSURE"], "validation_failures": []}
```

Validation failures appear as `REGULATORY_VALIDATION_FAILURE` exceptions and must be resolved before submission.

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
