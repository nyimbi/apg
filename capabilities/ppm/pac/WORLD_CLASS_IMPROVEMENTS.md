# Project Accounting (ppm_pac) — World-Class Improvements

## Current State Assessment

The service is functional but operates at a commodity level: in-memory storage, thin EVM approximations, stub analytics, and no time-series awareness. The 15 improvements below target correctness gaps, analytical depth, and production-grade operations.

---

## Improvement 1: Real EVM with Configurable Progress Sources

**Problem**: `earned_value_analysis` hard-codes PV as 60% of BAC and EV as 50% BAC when no revenue recognitions exist. These magic constants produce nonsensical SPI/CPI on any real project.

**Fix**: Accept an explicit `percent_complete` parameter (0–100 float) supplied by the schedule integration (ppm_pbl). Derive PV from a time-phased spend curve stored per cost code. EVM metrics become meaningful and auditable.

**Impact**: SPI/CPI are the primary early-warning signals in project controls. Garbage inputs produce false confidence or false panic.

---

## Improvement 2: Persistent Store Abstraction via Repository Pattern

**Problem**: All state is in-memory dicts on the service instance. Restart = total data loss. The `store` constructor argument is accepted but never used.

**Fix**: Introduce a `ProjectAccountingRepository` protocol with `save`, `get`, `list`, and `delete` methods. Provide an `InMemoryRepo` (current behaviour) and a `PostgresRepo` backed by asyncpg. The service delegates all persistence to the repo, making it stateless and horizontally scalable.

**Impact**: Required for any production deployment. Currently impossible to persist data without rebuilding the entire service.

---

## Improvement 3: Strict Pydantic v2 Input Validation

**Problem**: Input validation is scattered `assert` statements with unhelpful error messages. No schema enforcement at API boundaries — callers learn about missing fields only at runtime explosions deep in service logic.

**Fix**: Replace all method signatures with Pydantic v2 `model_validate`-able input models in `views.py`. Use `Annotated[float, AfterValidator(lambda v: v > 0)]` for positive-float constraints. Generate OpenAPI schemas for free.

**Impact**: Eliminates an entire class of runtime errors and enables auto-generated API docs.

---

## Improvement 4: Idempotent Operations via Upsert Semantics

**Problem**: `create_account` silently overwrites if called twice with the same `account_id`. `cost_code_create` raises `AssertionError` on duplicate — but only if the key format is consistent. Neither is documented.

**Fix**: Implement explicit `upsert_account` (returns existing + flag `created=False`) and make `cost_code_create` return `(record, created: bool)`. Add an idempotency key parameter to cost transactions for safe retries from event streams.

**Impact**: Bytewax at-least-once delivery currently causes silent double-counting. This is a financial correctness bug.

---

## Improvement 5: Cash Flow Forecasting from Cost Commitments

**Problem**: README lists `cash_flow_forecasting` as a provided service, but no such method exists.

**Fix**: Implement `async cash_flow_forecast(project_id, periods)` that projects monthly outflows from: open purchase orders (committed but unpaid), estimated labour from remaining budget, and milestone invoice schedules. Return a `{period: str, inflow: float, outflow: float, net: float, cumulative: float}[]` series.

**Impact**: This is the #1 feature requested by project controllers — knowing when cash runs out before it does.

---

## Improvement 6: Multi-Currency with Live Rate Conversion

**Problem**: `SUPPORTED_CURRENCIES` includes USD, EUR, GBP, KES, etc., but all arithmetic happens in the account's native currency with no conversion logic. Cross-currency costs (e.g., USD PO on a KES project) are silently added as same-currency amounts.

**Fix**: Add `async convert_to_base(amount, from_currency, to_currency, rate_date)` using a pluggable `ExchangeRateProvider` protocol. All cost aggregations normalise to the account's base currency before summing. Store original currency + converted amount on every transaction.

**Impact**: Any multi-currency project currently produces incorrect totals. Silent financial error.

---

## Improvement 7: Budget Utilisation Alerts with Configurable Thresholds

**Problem**: README lists `cost_variance_alerts` as a provided service; no alert-generation method exists. Overruns are detectable only by manually reading variance reports.

**Fix**: Implement `async check_budget_thresholds(project_id, warn_pct=80, critical_pct=95)` that scans cost codes, computes utilisation, and emits structured `BudgetAlert` events to the notify adapter. Integrate into every cost-recording path as a post-write side effect.

**Impact**: Passive alerting converts reactive cost management to proactive. Current design requires a human to pull a report to notice an overrun.

---

## Improvement 8: Proper ProjectAccount Model with `account_id` Field

**Problem**: `ProjectAccount` dataclass uses `id` as the field name, but `create_account` receives `account_id` and constructs `ProjectAccount(account_id, ...)` — relying on positional argument order. Throughout service.py, there are inconsistent attempts to read `account.id`, `account.account_id`, and `account.budget` (which doesn't exist — the real field is `budget_amount`). These silent AttributeErrors return 0.0 budgets on many paths.

**Fix**: Rename the field to `account_id` or add a `@property account_id` alias. Fix all attribute lookups. Add regression tests covering each access pattern.

**Impact**: `budget_vs_actual` currently returns `budget: 0.0` for every project because `account.budget` is `None`, not `account.budget_amount`. This is an active data-integrity bug.

---

## Improvement 9: Period-Aware Revenue Recognition with IFRS 15 / ASC 606 Controls

**Problem**: `revenue_recognition_project` posts revenue immediately with no period-close controls. Under IFRS 15, revenue can only be recognised in open accounting periods, and re-recognition in a closed period must be blocked.

**Fix**: Add a `PeriodCalendar` component with `open_period`, `close_period(period)`, and `is_period_open(period)` methods. Revenue recognition methods must assert the target period is open. Implement `reopen_period` requiring two-approver sign-off (controller + CFO proxy).

**Impact**: Without period controls, the system cannot produce auditable financial statements. Current implementation allows backdated revenue manipulation.

---

## Improvement 10: Earned Value Trend History and Forecasting

**Problem**: EV snapshots are stored (`_ev_snapshots`) but never queried for trend analysis. CPI and SPI trends over time are the most important leading indicators in EVM — a single-period snapshot is nearly useless.

**Fix**: Implement `async ev_trend_analysis(project_id, periods)` that returns a time series of `{period, bac, pv, ev, ac, spi, cpi, eac, etc}` dicts, plus derived trend indicators: CPI trend direction (improving/degrading/stable), forecast at completion confidence band.

**Impact**: EVM without trend context is like a speedometer without an odometer.

---

## Improvement 11: Cost Accruals for Period-End Close

**Problem**: No accrual mechanism exists. Labour costs incurred but not yet invoiced, and goods received but not yet invoiced (GRNI), are invisible to period-end financial statements.

**Fix**: Implement `async post_accrual(project_id, cost_code, amount, accrual_type, period, reversal_period)` that creates a time-bounded accrual entry which auto-reverses in the next period. Track accrual status (`open`, `reversed`, `cleared`).

**Impact**: Without accruals, period-end P&L is materially misstated for any project with timing differences between cost incurrence and invoicing.

---

## Improvement 12: Intercompany Cost Recharging

**Problem**: No mechanism exists to recharge costs from one entity/tenant to another. Cross-entity project work (common in professional services) requires transfer pricing logic.

**Fix**: Implement `async create_intercompany_recharge(from_project_id, to_project_id, cost_code, amount, markup_pct, evidence_reference)` that creates matching debit/credit entries across the two tenant-scoped accounts with an audit trail linking both entries.

**Impact**: Intercompany eliminations are required for consolidated financial reporting. Currently impossible.

---

## Improvement 13: Automated Variance Root-Cause Classification

**Problem**: `project_cost_report` flags codes as `over_budget` or `on_track` but provides no root-cause analysis. Controllers manually investigate every red line.

**Fix**: Implement `async classify_variance(project_id, cost_code)` that uses heuristic rules to classify variances as: `scope_creep` (cost grew without scope-change order), `rate_variance` (hours on budget but rate higher), `volume_variance` (rate on budget but more units), `timing_variance` (costs earlier than planned, will self-correct), or `true_overrun`. Return classification with supporting evidence (rate comparison, hours comparison, baseline schedule delta).

**Impact**: Transforms a list of numbers into actionable intelligence. Dramatically reduces time-to-decision for controllers.

---

## Improvement 14: Three-Point Cost Estimation (PERT) for EAC

**Problem**: EAC is computed as `BAC / CPI` — the single-point deterministic formula. This understates risk on projects with high CPI volatility and provides no confidence interval.

**Fix**: Implement `async three_point_eac(project_id, optimistic_cpi, pessimistic_cpi)` that computes PERT-weighted EAC: `EAC = (O + 4M + P) / 6` where M is the current CPI-derived EAC. Return `{p50: float, p80: float, p90: float, expected: float, std_dev: float}`.

**Impact**: Risk-adjusted cost forecasts enable better contingency management. Standard practice in any PMI/PRINCE2 project controls regime.

---

## Improvement 15: Reconciliation Engine for External Ledger Sync

**Problem**: No reconciliation capability. When the ERP (SAP, Xero, Sage) posts actuals, there is no mechanism to compare and reconcile against APG's project ledger. Silent divergence between systems is undetectable.

**Fix**: Implement `async reconcile_with_external_ledger(project_id, external_entries: list[dict], tolerance_pct=0.5)` that matches external line items against internal cost transactions by period, cost code, and amount (within tolerance). Return `{matched, unmatched_internal, unmatched_external, total_variance}` with line-level match results. Emit `reconciliation_complete` audit event.

**Impact**: Without reconciliation, the system cannot be the system of record — it's just another silo. This is the foundation of financial close confidence.
