# APG EOD/BOD Processing Engine (`fin/eod`)

Nightly end-of-day and morning begin-of-day batch processor for APG Financial Management.

## What it does

| Job | Trigger | Description |
|-----|---------|-------------|
| `pre_eod_validations` | Every EOD | Suspense balance check, unposted entry count, GL period availability |
| `interest_accrual_batch` | Every EOD | Daily interest accrual (actual/365) for all active interest-bearing accounts |
| `fee_posting_batch` | Every EOD | Monthly/quarterly fees due today |
| `dormancy_check_batch` | Every EOD | Flag accounts inactive beyond regulatory threshold |
| `term_deposit_maturity_batch` | Every EOD | Process maturities: payout or auto-renew |
| `loan_repayment_batch` | Every EOD | Collect instalments due, classify arrears, trigger IFRS9 ECL update |
| `standing_order_batch` | Every EOD | Execute standing transfers due today |
| `fx_revaluation` | **Month-end only** | Restate FCY balances at closing rates, post FX gain/loss |
| `period_close` | **Month-end only** | Lock GL period, close P&L to retained earnings, open next period |
| `eod_reports_generation` | Every EOD | Daily Balance Sheet, P&L flash, Liquidity ratio |

BOD jobs: open GL period (month-start only) + clear overnight float.

## Idempotency guarantee

Every batch job is decorated with `@idempotent` keyed on `(tenant_id, processing_date)`.
Re-running EOD for the same date returns the cached result — no double-posting.

```python
import asyncio
from capabilities.fin.eod import EODService

svc    = EODService()
result = asyncio.run(svc.run_eod("my_bank", "2026-06-11"))
print(result.status, result.jobs_completed)  # completed  10

# Safe to call again — returns cached result
result2 = asyncio.run(svc.run_eod("my_bank", "2026-06-11"))
assert result.run_id == result2.run_id       # same run_id
```

## REST API

```
POST  /api/fin/eod/run                         Run full EOD
POST  /api/fin/eod/bod                         Run BOD
POST  /api/fin/eod/jobs/{job_name}             Run single job
GET   /api/fin/eod/status/{date}               EOD run status
GET   /api/fin/eod/jobs/{date}/{job_name}      Single job result
POST  /api/fin/eod/jobs/{date}/{job_name}/retry Retry failed job
GET   /api/fin/eod/history                     Run history
GET   /api/fin/eod/exceptions/{date}           Processing exceptions
POST  /api/fin/eod/exceptions/{id}/resolve     Resolve exception
GET   /api/fin/eod/pending                     Pending items
POST  /api/fin/eod/schedule                    Schedule job
GET   /api/fin/eod/report/{date}               Full EOD report
GET   /api/fin/eod/prerequisites/{date}        Pre-flight check
GET   /api/fin/eod/running                     Currently running jobs
POST  /api/fin/eod/cancel/{date}               Emergency stop
GET   /api/fin/eod/metrics?days=30             Processing metrics
GET   /api/fin/eod/health                      Health check
```

All endpoints except `/health` require `X-Tenant-Id` header.

## Production adapter wiring

```python
from capabilities.fin.eod.service import EODService
from myapp.adapters import PostgresGLAdapter, PostgresAccountAdapter

svc = EODService()
# Override default null adapters:
svc._gl_adapter       = PostgresGLAdapter(db_session)
svc._account_adapter  = PostgresAccountAdapter(db_session)
```

See `domain/adapters.py` for the full Protocol definitions.

## New Features (v2)

### Penalty Interest on Overdue Loans
Accrues contractual penalty interest (2× normal rate, min 5 % p.a.) on overdue instalments daily. Posts to `Penalty Interest Income` GL.

```python
result = asyncio.run(svc.compute_penalty_interest("my_bank", "2026-06-11"))
print(result["total_penalty_accrued"])   # "1234.56"
```

### IFRS 9 ECL Staging
Classifies every loan into Stage 1 / 2 / 3 and posts provision deltas to `Impairment Expense / Loan Loss Reserve`. Triggered as part of the loan repayment batch.

```python
result = asyncio.run(svc.run_ifrs9_ecl_staging("my_bank", "2026-06-30"))
print(result["stage_counts"], result["provision_delta"])
```

### Basel III LCR Computation
Computes the Liquidity Coverage Ratio daily. Emits a CRITICAL exception if LCR < 100 % and a warning if < 105 %.

```python
result = asyncio.run(svc.compute_liquidity_coverage_ratio("my_bank", "2026-06-11"))
print(result["lcr_ratio"], result["status"])   # "1.1523"  "compliant"
```

### Nostro Reconciliation
Matches GL nostro entries against SWIFT MT940/camt.053 statement lines. Unmatched items exceeding the configured threshold are raised as CRITICAL exceptions.

```python
result = asyncio.run(svc.run_nostro_reconciliation("my_bank", "2026-06-11"))
print(result["unmatched"], result["unmatched_value"])
```

### ZBA Sweeps (Zero-Balance Accounting)
Concentrates cash from sub-accounts into a master account after all EOD postings. Funds sub-accounts from master when below their minimum balance.

```python
result = asyncio.run(svc.run_zba_sweeps("my_bank", "2026-06-11"))
print(result["groups_processed"], result["total_swept"])
```

### NPA Classification
Promotes loans with DPD ≥ 90 to Non-Performing Asset status. Suspends P&L accrual, reverses uncollected interest to sundry, and posts 100 % provision.

```python
result = asyncio.run(svc.classify_npa_accounts("my_bank", "2026-06-11", dpd_threshold=90))
print(result["newly_classified"], result["provision_posted"])
```

### SLA Compliance Monitoring
Evaluates whether EOD completed within the configured processing window. Raises `SLA_AT_RISK` exception if > 70 % of the window is consumed with < 50 % of jobs complete.

```python
result = asyncio.run(svc.check_sla_compliance("my_bank", "2026-06-11", sla_window_minutes=360))
print(result["status"], result["elapsed_seconds"])   # "met"  187.4
```

### Regulatory Returns Generation
Renders CBK BSL02/BSL03 (month-end), Capital Adequacy / Large Exposure (quarter-end), and Annual Supervisory Return / AML Report (year-end). Validates totals before filing.

```python
result = asyncio.run(svc.generate_regulatory_returns("my_bank", "2026-06-30"))
print(result["returns_generated"])   # ["BSL02_BALANCE_SHEET", "BSL03_CREDIT_EXPOSURE", ...]
```

## Running tests

```bash
python -m pytest capabilities/fin/eod/tests/ -v
```

---

## World-Class Enhancements (v2.0)

- **I1.** EOD Processing Engine — World-Class Improvements
- **I2.** Penalty Interest on Overdue Loans | Credit Risk | Missed instalments should accr
- **I3.** IFRS 9 ECL Staging and Provision Posting | Regulatory / IFRS | IFRS 9 requires c
- **I4.** Liquidity Coverage Ratio (LCR) Computation | Regulatory Liquidity | Basel III ma
- **I5.** Intraday Nostro Reconciliation | Operational Risk | Unreconciled nostro entries 
- **I6.** Tiered Interest Rate Bands | Product Management | Most deposit products tier rat
- **I7.** Cheque Clearance and Float Management | Operations | Deposited cheques sit in fl
- **I8.** Automated Regulatory Return Generation | Compliance | CBK, RBA, and similar regu
- **I9.** Non-Working Day Roll Logic | Calendar Management | Financial contracts use day-c
- **I10.** Parallel Job Execution with Dependency Graph | Performance | 10 sequential jobs 
- **I11.** Audit Trail with Immutable Journal Entries | Audit / Compliance | All financial 
- **I12.** Configurable Dormancy Rules per Product Type | Product / Regulatory | Dormancy t
- **I13.** NPA (Non-Performing Asset) Classification | Credit Risk / Regulatory | RBI/CBK r
- **I14.** Multi-Currency Balance Sheet Generation | Reporting / FX | A bank reporting in U
- **I15.** Sweep Optimisation and Zero-Balance Accounting | Treasury | Corporate clients wi

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
