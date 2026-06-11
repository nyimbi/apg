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

## Running tests

```bash
python -m pytest capabilities/fin/eod/tests/ -v
```
