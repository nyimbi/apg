# Multi-Currency Management — User Guide

**Capability ID**: `loc_mcy` | **Domain**: `loc` | **Version**: `1.1.0`
**Copyright**: Datacraft © 2025 | **Author**: Nyimbi Odero

---

## Overview

Multi-Currency Management (MCY) provides full lifecycle management of currencies, exchange rates, FX revaluation, currency translation, and FX gain/loss reporting. It is designed for organisations operating across multiple currencies and regulatory jurisdictions.

Version 1.1.0 adds eight new analytical methods covering: stale rate detection, bulk rate upload with idempotency, rate history, batch conversion, spread/volatility analysis, multi-entity consolidated exposure, period-close automation, rate matrix generation, and FX impact projection under hypothetical scenarios.

---

## Installation

```bash
pip install apg-loc-mcy
```

---

## Quick Start

```python
import asyncio
from datetime import date
from apg_loc_mcy.service import MultiCurrencyManagementService
from apg_loc_mcy.models import (
	CurrencyConfigCreate, ExchangeRateCreate, FxAccountCreate, RevaluationCreate
)

svc = MultiCurrencyManagementService()
TENANT = "acme"

async def main():
	# 1. Configure currencies
	usd = await svc.configure_currency(CurrencyConfigCreate(
		tenant_id=TENANT, code="USD", name="US Dollar", symbol="$",
		decimal_places=2, is_functional=True,
	))
	kes = await svc.configure_currency(CurrencyConfigCreate(
		tenant_id=TENANT, code="KES", name="Kenyan Shilling", symbol="KSh",
		decimal_places=2,
	))

	# 2. Record an exchange rate
	rate = await svc.record_exchange_rate(ExchangeRateCreate(
		tenant_id=TENANT, from_currency="KES", to_currency="USD",
		rate=0.00775, rate_type="spot", rate_source="cbk",
		effective_date=date(2025, 6, 1),
	))

	# 3. Convert an amount
	result = await svc.convert_amount(TENANT, 100_000, "KES", "USD", date(2025, 6, 1))
	print(result["converted_amount"])  # ~775.0

asyncio.run(main())
```

---

## Core Features

### Currency Configuration

Configure currencies with ISO 4217 codes, decimal precision, and rounding modes. Mark one currency as the functional currency per entity and one as the presentation currency.

```python
currency = await svc.configure_currency(CurrencyConfigCreate(
	tenant_id="acme",
	code="EUR",
	name="Euro",
	symbol="€",
	decimal_places=2,
	rounding_mode="round_half_even",
	is_functional=False,
	is_presentation=True,
))
```

Supported rounding modes: `round_half_even`, `round_half_up`, `round_down`, `round_up`.

### Exchange Rate Management

Record spot, forward, average, and closing rates from multiple sources.

```python
rate = await svc.record_exchange_rate(ExchangeRateCreate(
	tenant_id="acme",
	from_currency="EUR",
	to_currency="USD",
	rate=1.0823,
	rate_type="spot",          # spot | forward | average | closing | budget
	rate_source="ecb",         # ecb | cbk | bloomberg | reuters | xe | manual | custom_api
	effective_date=date(2025, 6, 1),
	expiry_date=date(2025, 6, 30),
))
```

Manual rates require an `approval_reference`. Backdated rates require a `backdating_override` justification.

### Currency Conversion

Convert amounts between any configured currency pair. If a direct rate is unavailable, the inverse is automatically tried.

```python
result = await svc.convert_amount("acme", 50_000, "EUR", "KES", date(2025, 6, 1))
# result = {"converted_amount": ..., "rate": ..., "from_currency": "EUR", "to_currency": "KES", ...}
```

---

## New in v1.1.0

### 1. Stale Rate Detection

```python
stale = await svc.detect_stale_rates(tenant_id="acme", staleness_days=3)
# stale["stale_count"] — number of stale rates
# stale["stale_rates"] — list with rate_id, days_stale, reason
```

Surfaces expired rates (past `expiry_date`) and rates with no expiry that are older than `staleness_days` business days. Wire the result to the `ntfy` capability to push alerts to treasury staff before period-end.

### 2. Bulk Rate Upload with Idempotency

```python
result = await svc.bulk_record_exchange_rates(
	tenant_id="acme",
	payloads=[rate1, rate2, rate3, ...],  # list[ExchangeRateCreate]
	upload_batch_id="ecb-2025-06-01",
	actor_id="rate_feed_agent",
)
# result["created"], result["skipped_duplicate"], result["rejected"]
```

Safe to call multiple times with the same `upload_batch_id`. Duplicates (same pair, date, type) are skipped rather than raising. Rejected items include the failure reason.

### 3. Rate History

```python
history = await svc.get_rate_history(
	tenant_id="acme",
	from_currency="KES",
	to_currency="USD",
	rate_type="spot",
	limit=90,
)
# Returns list of dicts sorted oldest-first: rate, effective_date, is_active, created_by
```

Returns both active and superseded rates, enabling full audit reconstruction. `is_active=False` records were deactivated by subsequent updates.

### 4. Batch Currency Conversion

```python
results = await svc.multi_currency_convert_batch(
	tenant_id="acme",
	conversions=[
		{"amount": 100_000, "from_currency": "KES", "to_currency": "USD"},
		{"amount": 50_000, "from_currency": "EUR", "to_currency": "GBP"},
	],
	as_of=date(2025, 6, 1),
)
# Each result has "status": "ok" or "error", plus conversion fields
```

Processes all conversions in a single call. Failed items include `"error"` key rather than raising, allowing partial success.

### 5. Spread and Volatility Analysis

```python
analysis = await svc.currency_pair_spread_analysis(
	tenant_id="acme",
	from_currency="KES",
	to_currency="USD",
	lookback_days=30,
)
# analysis["mean_rate"], ["std_dev"], ["min_rate"], ["max_rate"]
# analysis["coefficient_of_variation_pct"]
# analysis["is_volatile"]  — True if CoV > 2%
```

Requires at least one rate in the lookback window. If insufficient data, returns `"message": "insufficient_data"` without raising.

### 6. Consolidated Multi-Entity Exposure

```python
report = await svc.consolidated_exposure_summary(
	tenant_id="acme",
	entity_ids=["entity-ke", "entity-ug", "entity-tz"],
	consolidation_currency="USD",
	as_of=date(2025, 6, 30),
)
# report["group_total_exposure"] — all entities' exposures translated to USD
# report["entities"] — per-entity breakdown
```

Translates each entity's per-currency FX account balances to the consolidation currency using closing rates as of `as_of`. Entities with no registered FX accounts contribute 0. Missing rates are excluded from the total with no error raised.

### 7. Period-Close Checklist

```python
checklist = await svc.period_close_checklist(
	tenant_id="acme",
	period_start=date(2025, 6, 1),
	period_end=date(2025, 6, 30),
	actor_id="controller",
)
# checklist["overall_pass"] — True if all steps pass
# checklist["blocking_issues"] — list of issues preventing close
# checklist["checklist"] — per-step pass/fail detail
```

Runs four checks in sequence:

| Step | Check |
|------|-------|
| 1 | All active rates non-stale as of period end |
| 2 | No revaluations in draft/pending_approval for the period |
| 3 | No translations in draft/pending_approval for the period |
| 4 | FX gain/loss report generates without error |

Emits `period_close_checked` audit event. Integrate into a scheduled job to produce a close-readiness notification every morning during period-end week.

### 8. Exchange Rate Matrix

```python
matrix = await svc.rate_matrix(
	tenant_id="acme",
	currencies=["USD", "EUR", "KES", "GBP"],
	as_of=date(2025, 6, 1),
	rate_type="spot",
)
# matrix["matrix"]["USD"]["EUR"] = 0.9234
# matrix["coverage_pct"] = 83.3  (% of possible pairs with available rates)
```

Builds an N×N matrix with direct and inverse lookups. Diagonal is always 1.0. Missing pairs are `null`. `coverage_pct` shows what fraction of non-diagonal cells have rates — useful for detecting gaps in the rate feed before batch conversion runs.

### 9. FX Impact Projection

```python
projection = await svc.fx_impact_projection(
	tenant_id="acme",
	open_positions=[
		{"currency": "EUR", "amount": 1_000_000},
		{"currency": "KES", "amount": -5_000_000},  # short KES
	],
	scenario_rates={"EUR/USD": 1.15, "KES/USD": 0.0072},
	base_currency="USD",
)
# projection["net_fx_impact"] — scenario P&L vs. current rates
# projection["net_fx_impact_direction"] — "gain" | "loss" | "neutral"
# projection["positions"] — per-position impact breakdown
```

Projects the P&L impact of open FX positions under a hypothetical rate scenario. Long positions are positive amounts; short positions negative. Scenario rates override current registered rates for the projection only — no rates are modified.

---

## FX Revaluation Workflow

```
create_revaluation() -> approve_revaluation() -> post_revaluation() -> (optional) reverse_revaluation()
```

```python
# Create
rev = await svc.create_revaluation(RevaluationCreate(
	tenant_id="acme", entity_id="ke-entity",
	period_start=date(2025, 6, 1), period_end=date(2025, 6, 30),
	revaluation_method="period_end_rate",
	functional_currency="KES",
	fx_gain_account_id=gain_acct.id,
	fx_loss_account_id=loss_acct.id,
))

# Approve (moves status to approved)
rev = await svc.approve_revaluation("acme", rev.id, "cfo@acme.com", "REF-2025-001")

# Post (moves status to posted, sets posted_date)
rev = await svc.post_revaluation("acme", rev.id, actor_id="cfo@acme.com")
```

FX gain/loss accounts must be registered before creating a revaluation.

---

## Currency Translation Workflow

```
create_translation() -> approve_translation() -> post_translation()
```

Supports IFRS-compliant `current_rate` and `temporal` methods. The translation reserve account accumulates OCI differences.

---

## FX Gain/Loss Reporting

```python
report = await svc.generate_fx_report(
	tenant_id="acme",
	period_start=date(2025, 6, 1),
	period_end=date(2025, 6, 30),
	entity_id="ke-entity",  # optional — omit for all entities
)
# report.total_realised_gain, report.total_realised_loss, report.net_fx_impact
```

Only `posted` revaluations contribute to the report. Draft and pending runs are excluded.

---

## Dashboard

```python
summary = await svc.dashboard_summary(tenant_id="acme")
# Returns: currency_count, active_currency_count, exchange_rate_count,
#          pending_revaluation_count, pending_translation_count, etc.
```

---

## Business Rules Summary

| Rule | Effect |
|------|--------|
| `tenant_context_required` | All operations require a non-empty tenant_id |
| `write_requires_policy` | Write operations fail without a policy context |
| `rate_value_positive` | Exchange rates must be > 0 |
| `manual_rate_approval_required` | Manual rates require an `approval_reference` |
| `rate_backdating_restricted` | Backdated rates require `backdating_override` |
| `unapproved_revaluation_posting_denied` | Only approved revaluations can be posted |
| `revaluation_reversal_requires_posted_status` | Only posted revaluations can be reversed |
| `unapproved_translation_posting_denied` | Only approved translations can be posted |
| `fx_gain_loss_account_bypass_denied` | FX accounts must exist before creating a revaluation |

---

## Composability

```apg
use loc_mcy;
```

| Upstream capability | What it provides |
|--------------------|-----------------|
| `mco` (Multi-Country Operations) | Functional currency assignments per entity |
| `schd` (Scheduler) | Triggers periodic rate ingestion and period-close checklist |

| Downstream capability | What MCY feeds |
|----------------------|----------------|
| `fin` (General Ledger) | Revaluation and translation journal entries |
| `grc` (Governance/Risk) | FX exposure data for treasury risk reporting |
| `ntfy` (Notifications) | Stale rate alerts, pending approval reminders |
| `mqeb` (Event Bus) | `apg.loc.mcy.lifecycle` stream for all lifecycle events |

---

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `exchange_rates.approval_required_for_manual` | bool | `true` | Require approval for manually entered rates |
| `revaluation.approval_required` | bool | `true` | Require approval before posting revaluation |
| `translation.approval_required` | bool | `true` | Require approval before posting translation |
| `rounding.default_mode` | string | `round_half_even` | Default rounding mode |
| `staleness.threshold_days` | int | `3` | Days before a rate without expiry is considered stale |

All configuration keys are tenant-scoped. Set via the `conf` capability or `LOC_MCY_` prefixed environment variables.

---

## Further Reading

- `service.py` — All service methods with inline docstrings
- `models.py` — Pydantic v2 data models with validation
- `api.py` — REST API endpoints (Flask-AppBuilder blueprints)
- `capability_contract.py` — Policy rules and supported constants
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 deep improvements
- `tests/test_service.py` — Comprehensive service-layer tests
