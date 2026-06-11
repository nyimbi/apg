# FinTech Risk Management

**Capability ID**: `fintech_risk` | **Domain**: `fintech` | **Version**: `1.2.0`

## Description

FinTech Risk Management provides the enterprise risk framework for the APG platform: risk appetite registration across credit, market, liquidity, operational, fraud, compliance, model, and third-party domains; tenant-scoped risk profiles for customers, merchants, wallets, accounts, portfolios, loans, agents, and counterparties; exposure tracking with limit enforcement and human-approval-gated overrides; control assurance with effectiveness scoring; stress scenario modeling; limit breach recording; risk event management; and governance reviews.

## Installation

```bash
pip install apg-fintech-risk
```

## Provides

- `risk_appetite_workflow`
- `risk_profile_workflow`
- `risk_exposure_workflow`
- `risk_control_workflow`
- `risk_stress_testing_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-risk/dashboard` | `fintech_risk:view` | Overview |
| `/fintech-risk/appetite` | `fintech_risk:appetite` | Governance |
| `/fintech-risk/profiles` | `fintech_risk:profiles` | Risk |
| `/fintech-risk/exposures` | `fintech_risk:exposures` | Risk |
| `/fintech-risk/controls` | `fintech_risk:controls` | Controls |
| `/fintech-risk/stress-tests` | `fintech_risk:stress` | Analytics |
| `/fintech-risk/breaches` | `fintech_risk:breaches` | Issues |
| `/fintech-risk/events` | `fintech_risk:events` | Issues |

## Key Service Methods

### Core Workflow Methods (synchronous)

- `describe()` — Return the capability contract for a tenant
- `evaluate(context)` — Evaluate capability rules against a context dict
- `register_appetite(...)` — Register a risk appetite threshold for a domain
- `create_profile(...)` — Create a risk profile for a subject (customer, account, portfolio, etc.)
- `record_exposure(...)` — Record an exposure with limit enforcement and human-approval gate
- `evaluate_control(...)` — Evaluate a risk control with effectiveness scoring
- `run_stress_scenario(...)` — Record a stress scenario with impact, probability, and mitigation
- `record_limit_breach(...)` — Record a limit breach with severity and remediation owner
- `open_risk_event(...)` — Open a risk event (operational, model drift, loss event, etc.)
- `record_review(...)` — Record a governance review
- `dashboard_summary(tenant_id)` — Aggregate count summary for the tenant

### Async Analytics Methods

- `credit_risk_assessment(customer_id)` — Composite credit score with PD/LGD/EL
- `market_risk_var(portfolio_id, confidence_level)` — Parametric Value-at-Risk
- `var_backtest(portfolio_id, confidence_level, window)` — Kupiec POF VaR backtest; emits `model_drift` event on failure
- `liquidity_risk_report(period)` — LCR and NSFR compliance report
- `intraday_liquidity_monitor(correspondent_bank_id, ...)` — BCBS 248 intraday settlement position tracker
- `operational_risk_register()` — Structured operational risk event register
- `concentration_risk(portfolio_id)` — HHI-based concentration measurement with type breakdown
- `stress_test_portfolio(scenario)` — Forward stress test: apply a shock and count breaches
- `reverse_stress_test(threshold_type, threshold_value, portfolio_id)` — Bisection search for minimum shock that breaches CAR/LCR/VaR
- `risk_appetite_monitoring()` — Compare exposures against appetite thresholds; surface warnings and breaches
- `capital_adequacy_check()` — Basel III CAR check from controls and exposures
- `regulatory_capital_report(period)` — Basel IV SA-CR full capital report: CET1/AT1/T2 + RWA breakdown
- `raroc_calculation(portfolio_id, net_revenue, allocated_opex, hurdle_rate_pct)` — RAROC vs hurdle rate
- `ifrs9_stage_migration(profile_id, macro_scenario, macro_multiplier)` — IFRS 9 SICR detection + macro overlay
- `ecl_computation(profile_id)` — Expected Credit Loss under IFRS 9
- `basel_iii_compliance(period)` — Three-pillar Basel III compliance summary
- `sanctions_screening(subject_name, subject_id, country_code)` — Fuzzy-match against OFAC/EU/UN/CBK watchlists
- `psi_model_stability(model_id, baseline_distribution, current_distribution)` — PSI; emits `model_drift` event when PSI > 0.25
- `risk_report_summary(period)` — Board-ready concurrent RAG report across all risk domains
- `aml_transaction_monitoring(transactions)` — Batch AML rule engine over a transaction list
- `fraud_typology_detection(transaction)` — FATF typology detection (structuring, TBML, layering)
- `country_risk_assessment(country_code)` — FATF/CBK country risk classification
- `portfolio_credit_metrics(tenant_id)` — Portfolio EL, UL, EAD, PD, LGD
- `exposure_heatmap()` — Exposure breakdown by domain and subject type
- `risk_scoring_model_run(subject_reference, features)` — Rule-based credit scoring from features
- `push_return_observation(portfolio_id, daily_return)` — Append daily return to VaR series
- `escalate_breach(breach_id, escalation_reason, escalated_to)` — Escalate an open breach
- `close_risk_event(event_id, resolution, reviewer_id)` — Close a risk event
- `health_check()` — Service liveness probe
- `bulk_create_profiles(profiles)` — Bulk import risk profiles
- `model_validation_report(model_id, validation_type)` — Basic model accuracy report
- `counterparty_risk_limit(counterparty_id, exposure_amount, limit_amount)` — Counterparty limit check
- `risk_appetite_statement(entity_id, period)` — Board-level appetite statement
- `export_risk_data(fmt)` — Export risk data in JSON/CSV/Excel format

_(See `service.py` for complete signatures.)_

## Usage Examples

### VaR Backtesting (Kupiec POF)

```python
import asyncio
from capabilities.fintech.risk.service import FintechRiskService

svc = FintechRiskService(tenant_id="acme", actor_id="risk_officer")

# Feed 252 days of observed daily returns
for r in observed_returns:
    asyncio.run(svc.push_return_observation("portfolio-acme", r))

result = asyncio.run(svc.var_backtest("portfolio-acme", confidence_level=0.99))
# result["model_valid"] == True  -> VaR model passes Kupiec test
# result["kupiec_lr_stat"]       -> chi-squared statistic (critical: 3.841)
```

### Reverse Stress Test

```python
result = asyncio.run(svc.reverse_stress_test(
    threshold_type="car",      # find shock that breaks capital adequacy
    threshold_value=8.0,       # Basel IV minimum CET1 ratio
    portfolio_id="all",
))
# result["critical_shock_bps"]  -> e.g. 3450 (34.5% loss breaks the CAR threshold)
# result["critical_shock_pct"]  -> 34.5
```

### RAROC Calculation

```python
result = asyncio.run(svc.raroc_calculation(
    portfolio_id="portfolio-acme",
    net_revenue=5_000_000.0,
    allocated_opex=800_000.0,
    hurdle_rate_pct=15.0,     # Kenya market default
))
# result["raroc_pct"]      -> e.g. 18.7
# result["above_hurdle"]   -> True
```

### Intraday Liquidity Monitoring (BCBS 248)

```python
# Record an outflow against a correspondent bank intraday limit
result = asyncio.run(svc.intraday_liquidity_monitor(
    correspondent_bank_id="KCBKEN",
    settlement_amount_minor=5_000_000_00,   # KES 5M in minor units
    direction="outflow",
    intraday_limit_minor=50_000_000_00,     # KES 50M limit
))
# result["alert_level"]         -> "normal" | "warning" | "breach"
# result["utilisation_pct"]     -> 10.0
# result["bcbs248_compliant"]   -> True
```

### IFRS 9 Stage Migration with Macro Overlay

```python
result = asyncio.run(svc.ifrs9_stage_migration(
    profile_id="prof-001",
    macro_scenario="adverse",
    macro_multiplier=1.4,     # 40% macro stress multiplier
))
# result["current_stage"]       -> "stage_1"
# result["migration_stage"]     -> "stage_2"  (upgraded by macro overlay)
# result["sicr_triggered"]      -> True/False
# result["ecl_12m_adjusted"]    -> forward-looking ECL
```

### Basel IV SA-CR Regulatory Capital Report

```python
result = asyncio.run(svc.regulatory_capital_report("2026-Q2"))
# result["cet1_ratio_pct"]      -> calculated CET1 ratio
# result["total_car_pct"]       -> total capital adequacy ratio
# result["compliant"]           -> True if cet1 >= 4.5%, T1 >= 6%, CAR >= 8%
# result["approach"]            -> "Basel_IV_SA_CR"
```

### Sanctions Screening

```python
result = asyncio.run(svc.sanctions_screening(
    subject_name="Al Shabaab Finance",
    subject_id="cust-9912",
    country_code="SO",
))
# result["hit"]                     -> True
# result["matches"][0]["list_source"] -> "UN_CONSOLIDATED"
# result["recommended_action"]      -> "block"
```

### PSI Model Stability

```python
result = asyncio.run(svc.psi_model_stability(
    model_id="credit-score-v2",
    baseline_score_distribution=baseline_scores,   # list[float], >= 10 items
    current_score_distribution=current_scores,
))
# result["psi"]                  -> e.g. 0.32 (major shift)
# result["stability_status"]     -> "major_shift"
# result["recommended_action"]   -> "revalidate_model"
# PSI > 0.25 automatically emits a model_drift risk event
```

### Board-Ready Risk Report Summary

```python
result = asyncio.run(svc.risk_report_summary("2026-Q2"))
# result["overall_rag"]          -> "green" | "amber" | "red"
# result["domain_rag"]           -> per-domain RAG scores
# result["capital"]              -> full regulatory capital report
# result["liquidity"]            -> LCR/NSFR report
# result["market_risk_var"]      -> VaR result
# result["risk_appetite"]        -> appetite monitoring result
# result["operational_risk"]     -> operational risk register
# result["concentration"]        -> HHI concentration result
```

## Interoperability

`fintech_risk` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_risk;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_RISK_`.

| Key | Default | Description |
|-----|---------|-------------|
| `FINTECH_RISK_HURDLE_RATE_PCT` | `15.0` | RAROC hurdle rate for Kenya market |
| `FINTECH_RISK_VAR_WINDOW` | `252` | Rolling window (days) for VaR and backtest |
| `FINTECH_RISK_PSI_MINOR_THRESHOLD` | `0.10` | PSI minor shift threshold |
| `FINTECH_RISK_PSI_MAJOR_THRESHOLD` | `0.25` | PSI major shift threshold (triggers model_drift event) |
| `FINTECH_RISK_SANCTIONS_JW_THRESHOLD` | `0.92` | Jaro-Winkler similarity threshold for sanctions matching |
| `FINTECH_RISK_INTRADAY_WARNING_PCT` | `80.0` | Intraday utilisation % that triggers a warning alert |

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 improvement proposals for roadmap planning
