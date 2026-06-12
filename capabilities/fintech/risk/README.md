# FinTech Risk Management

## Overview

Enterprise risk framework for the APG platform: risk appetite registration across credit, market, liquidity, operational, fraud, compliance, model, and third-party domains; tenant-scoped risk profiles for customers, merchants, wallets, accounts, portfolios, loans, agents, and counterparties; exposure tracking with limit enforcement and human-approval-gated overrides; control assurance with effectiveness scoring; stress scenario modeling; limit breach recording; risk event management; and governance reviews.

Limit overrides require human approval — exceeding a limit without approval is a hard deny. Control effectiveness scores must be in 0–100. Stress scenario probabilities are in basis points (0–10000). All risk events stream to `apg.fintech.risk.lifecycle` via Bytewax.

**Version**: 2.0.0 | **Capability ID**: `fintech_risk`

## Capability ID
`fintech_risk`

## Provides
| Service | Description |
|---------|-------------|
| risk_appetite_workflow | Register risk appetite thresholds by domain with owner and evidence |
| risk_profile_workflow | Create risk profiles for customers, merchants, accounts, portfolios, and counterparties |
| risk_exposure_workflow | Record exposures with limit references and human-approval-gated overrides |
| risk_control_workflow | Evaluate controls with type, owner, evidence, and effectiveness scoring |
| risk_stress_testing_workflow | Record stress scenarios with impact, probability, and mitigation evidence |
| risk_limit_breach_workflow | Record limit breaches with severity, exposure, and remediation owner |
| risk_event_workflow | Open risk events with type, severity, profile, and evidence |
| risk_review_workflow | Governance reviews for appetite changes, exposures, and breaches |
| risk_agent_workflow | Register AI agents for exposure monitoring, stress testing, and control assurance |
| var_backtest_workflow | Kupiec POF backtesting of VaR models with automatic model-drift event emission |
| reverse_stress_test_workflow | Bisection search to find minimum shock that breaches CAR, LCR, or VaR thresholds |
| raroc_workflow | Risk-Adjusted Return on Capital computation against configurable hurdle rate |
| intraday_liquidity_workflow | BCBS 248 intraday settlement position tracking per correspondent bank |
| ifrs9_migration_workflow | IFRS 9 stage migration assessment with forward-looking macro scenario overlays |
| regulatory_capital_workflow | Basel IV SA-CR capital report: CET1/AT1/T2 stack with credit, market, and OpRWA |
| sanctions_screening_workflow | Fuzzy-match screening against OFAC SDN, EU, UN, and CBK watchlists |
| psi_model_stability_workflow | Population Stability Index computation with automatic model-drift event emission |
| risk_report_summary_workflow | Board-ready concurrent RAG risk report across all domains |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Risk officer notifications |
| nlpc | NLP for risk narrative |
| keym | Key management |
| fintech_payments | Payment risk context |
| fintech_wallets | Wallet risk context |
| fintech_kyc | Customer identity for risk profiles |
| fintech_aml | AML risk signals |
| fintech_fraud | Fraud risk signals |
| bia | Risk analytics |
| fin_rpt | Risk reporting |

## Quick Start

```python
from apg_fintech_risk import RiskManagementService

svc = RiskManagementService(tenant_id="acme_bank", db_url="postgresql+asyncpg://...")

# Register a risk appetite threshold
await svc.register_risk_appetite(
    domain="credit",
    threshold_amount=50_000_000,
    currency="KES",
    owner_id="cro@acme.com",
    evidence_reference="board-resolution-2026-01",
)

# Create a risk profile for a customer
await svc.create_profile(
    subject_reference="cust_001",
    subject_type="customer",
    kyc_reference="kyc_cust_001",
    exposure_amount=500_000,
    currency="KES",
    score=42,
)

# Run a VaR backtest
result = await svc.var_backtest("portfolio_001", confidence_level=0.99, window=252)
# result["model_valid"] == True / False; model_drift event emitted on failure

# Compute RAROC
result = await svc.raroc_calculation(
    portfolio_id="portfolio_001",
    net_revenue=12_000_000,
    allocated_opex=2_000_000,
    hurdle_rate_pct=15.0,
)
# result["raroc_pct"], result["above_hurdle"]
```

## World-Class Enhancements (v2.0)

1. **Historical VaR Backtesting (Kupiec POF)** — Kupiec proportion-of-failures test + Christoffersen interval-forecast test; emits `var_backtest_exception` risk event when POF p-value < 0.05. Satisfies BCBS 239 model risk governance.

2. **Monte Carlo CVaR / Expected Shortfall** — 10,000-path Monte Carlo with Cholesky-decomposed correlated asset returns; ES at 97.5% per Basel IV / FRTB. Offloaded via `asyncio.to_thread`.

3. **Dynamic PD Calibration (Merton Structural Model)** — Derives probability of default from asset value, asset volatility, and debt face value via Black-Scholes. Falls back to Altman Z-score for unlisted counterparties. Outputs point-in-time and through-the-cycle PD for IFRS 9 accuracy.

4. **IFRS 9 Three-Stage Bucket Migration Engine** — Snapshot-based stage migration with SICR detection (30-day past due, credit watch, macro threshold breaches) and probability-weighted macro scenarios (base/adverse/optimistic) per CBK/PG/01. Full audit trail for examiners.

5. **Intraday Liquidity Monitoring (BCBS 248)** — Real-time settlement position ledger per correspondent bank; tracks peak intraday usage; early-warning alert at 80% of intraday limit. Satisfies BCBS 248 and CBK supervision expectations.

6. **Regulatory Capital Optimizer (Basel IV SA-CR)** — Full SA-CR risk-weight table (sovereigns, banks, corporates, retail, SME, real estate by LTV band, defaulted) plus FRTB SBA market risk capital. Outputs CET1/AT1/T2 capital stack with credit, market, and operational RWA.

7. **Concentration Risk via DRC Granularity Adjustment (FRTB)** — Default Risk Charge with JTD per issuer, net-long/net-short netting, and DRC add-on. Supplemented by GICS-sector HHI and country CR3/CR5. FRTB-compliant capital allocation for trading books.

8. **Real-Time AML Graph Analytics (Entity Resolution)** — In-memory transaction graph (networkx DiGraph); PageRank and betweenness centrality on 90-day rolling window; connected-component structuring ring detection; FATF R.16 wire transfer alerts for missing originator data.

9. **Behavioral Scoring with LSTM Anomaly Detection** — LSTM autoencoder (PyTorch / ONNX Runtime CPU inference) trained on normal transaction sequences per customer cohort. Reconstruction error triggers behavioral anomaly alert with SHAP explainability values per CBK Consumer Protection guidelines.

10. **Reverse Stress Test Engine** — Bisection search over [0, 10000] bps (20 iterations) to find the minimum shock breaching CAR, LCR, or VaR threshold. Board-level ICAAP/ILAAP tool; identifies binding constraint and generates scenario narrative.

11. **Watchlist & Sanctions Screening** — Fuzzy name matching (Jaro-Winkler ≥ 0.92) against OFAC SDN, EU Consolidated List, UN Consolidated List, and CBK Designated Entities. 24-hour list cache TTL. Returns match confidence, matched list/entry, and recommended action (block/hold/EDD). Pre-condition check on `create_profile`.

12. **RAROC Calculator** — RAROC = (Net Revenue − Expected Loss − Allocated OpEx) / Economic Capital; Economic Capital = UL × 2.33 (99% confidence). Per product, portfolio, and customer segment with configurable hurdle rate (default 15% for Kenya market).

13. **Automated Regulatory Report Generation (CBK/CMA)** — CBK Prudential Returns PR1–PR4 (capital adequacy, large exposures, liquidity, asset quality) and CMA periodic risk disclosure. JSON/CSV/Excel output with mandatory-field validation, cross-schedule consistency checks, and digital signature hash.

14. **Model Risk Management Framework (SR 11-7 / SS1/23)** — Full MRM lifecycle: model inventory, pre-deployment validation gate (Gini ≥ 0.35, KS ≥ 0.25, HL p > 0.05, PSI < 0.10), monthly PSI/CSI monitoring, annual revalidation triggers, and automatic `model_drift` risk event emission when PSI > 0.10.

15. **Integrated Risk Appetite Dashboard with RAG Status** — Hierarchical RAG aggregation from transaction to board level (Green < 70%, Amber 70–90%, Red > 90%); board-ready PDF with ARIMA-extrapolated trend projections, sparklines per domain, breach history, and automated narrative via local Ollama LLM (mistral/llama3).

## New Methods

### `var_backtest` — Kupiec POF VaR Validation
```python
result = await svc.var_backtest(
    portfolio_id="port_fx_book",
    confidence_level=0.99,
    window=252,           # trading days
)
# {model_valid: bool, kupiec_lr_stat: float, p_value_approx: float,
#  exceedances: int, expected_exceedances: float, var_amount: float}
# Automatically opens a model_drift risk event if model_valid is False.
```

### `reverse_stress_test` — Tipping-Point Shock Finder
```python
result = await svc.reverse_stress_test(
    threshold_type="car",       # "car" | "lcr" | "var_pct"
    threshold_value=8.0,        # minimum CAR % before breach
    portfolio_id="all",
)
# {critical_shock_bps: int, critical_shock_pct: float, binding_constraint: str}
# Uses 20-iteration bisection search over [0, 10000] bps.
```

### `raroc_calculation` — Risk-Adjusted Return on Capital
```python
result = await svc.raroc_calculation(
    portfolio_id="sme_loans",
    net_revenue=12_000_000.0,
    allocated_opex=2_000_000.0,
    hurdle_rate_pct=15.0,
)
# {raroc_pct: float, above_hurdle: bool, economic_capital: float,
#  expected_loss: float, risk_adjusted_income: float}
```

### `ifrs9_stage_migration` — Macro-Overlay Stage Assessment
```python
result = await svc.ifrs9_stage_migration(
    profile_id="prof_cust_001",
    macro_scenario="adverse",
    macro_multiplier=1.4,       # 40% uplift under adverse macro
)
# {current_stage: str, migration_stage: str, stage_upgraded: bool,
#  sicr_triggered: bool, ecl_12m_adjusted: float, ecl_lifetime_adjusted: float}
```

### `intraday_liquidity_monitor` — BCBS 248 Settlement Tracking
```python
result = await svc.intraday_liquidity_monitor(
    correspondent_bank_id="KCBKENA",
    settlement_amount_minor=5_000_000_00,   # in minor currency units
    direction="outflow",
    intraday_limit_minor=100_000_000_00,
)
# {utilisation_pct: float, alert_level: "normal"|"warning"|"breach",
#  peak_outflow_minor: int, bcbs248_compliant: bool}
# alert_level "warning" fires at > 80% utilisation.
```

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| profiles.supported_subject_types | list | customer, merchant, wallet, account, portfolio, loan, agent, counterparty | Risk profile subjects |
| appetite.supported_domains | list | credit, market, liquidity, operational, fraud, compliance, model, third_party | Risk domains |
| exposures.supported_types | list | credit_limit, settlement, liquidity, fx, market_value, operational_loss, fraud_loss | Exposure categories |
| controls.supported_types | list | preventive, detective, corrective, compensating, automated, manual | Control types |
| events.supported_types | list | limit_breach, control_failure, loss_event, model_drift, policy_exception, third_party_issue | Risk event types |
| breaches.supported_severities | list | low, medium, high, critical | Breach severity levels |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-risk/dashboard | GET | fintech_risk:view | Overview |
| appetite | /fintech-risk/appetite | GET/POST | fintech_risk:appetite | Governance |
| profiles | /fintech-risk/profiles | GET/POST | fintech_risk:profiles | Risk |
| exposures | /fintech-risk/exposures | GET/POST | fintech_risk:exposures | Risk |
| controls | /fintech-risk/controls | GET/POST | fintech_risk:controls | Controls |
| stress_tests | /fintech-risk/stress-tests | GET/POST | fintech_risk:stress | Analytics |
| breaches | /fintech-risk/breaches | GET/POST | fintech_risk:breaches | Issues |
| events | /fintech-risk/events | GET/POST | fintech_risk:events | Issues |
| reviews | /fintech-risk/reviews | GET/POST | fintech_risk:reviews | Governance |
| agents | /fintech-risk/agents | GET/POST | fintech_risk:admin | Automation |
| settings | /fintech-risk/settings | GET/POST | fintech_risk:admin | Administration |
| var_backtest | /fintech-risk/analytics/var-backtest | GET | fintech_risk:analytics | Analytics |
| reverse_stress | /fintech-risk/analytics/reverse-stress | POST | fintech_risk:stress | Analytics |
| raroc | /fintech-risk/analytics/raroc | POST | fintech_risk:analytics | Analytics |
| intraday_liquidity | /fintech-risk/liquidity/intraday | GET/POST | fintech_risk:liquidity | Liquidity |
| ifrs9_migration | /fintech-risk/credit/ifrs9-migration | GET | fintech_risk:credit | Credit |
| regulatory_capital | /fintech-risk/capital/regulatory | GET | fintech_risk:capital | Capital |
| sanctions_screening | /fintech-risk/aml/sanctions | POST | fintech_risk:aml | AML |
| psi_stability | /fintech-risk/models/psi | POST | fintech_risk:models | Model Risk |
| risk_report | /fintech-risk/reports/summary | GET | fintech_risk:reports | Reporting |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| appetite_threshold_required | Risk appetite with zero or negative threshold | deny |
| profile_kyc_required | Risk profile without KYC evidence | deny |
| profile_score_range | Score outside valid range | deny |
| exposure_amount_positive | Zero or negative exposure amount | deny |
| exposure_limit_required | Exposure without positive limit | deny |
| limit_override_requires_human_approval | Exposure over limit without approval | deny |
| control_effectiveness_required | Effectiveness score out of valid range | deny |
| scenario_probability_valid | Stress scenario probability out of range | deny |
| scenario_mitigation_required | Stress scenario without mitigation | deny |
| breach_evidence_required | Limit breach without evidence | deny |
| breach_owner_required | Limit breach without remediation owner | deny |
| risk_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_risk_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| RiskAppetite | id, domain, threshold_amount, currency, owner_id, evidence_reference |
| RiskProfile | id, subject_reference, subject_type, kyc_reference, exposure_amount, currency, score, source_reference |
| RiskExposure | id, profile_id, exposure_type, amount, currency, limit_amount, source_reference |
| ControlEvaluation | id, profile_id, control_type, owner_id, evidence_reference, effectiveness_score |
| StressScenario | id, profile_id, scenario_type, impact_amount, probability_bps, mitigation_reference |
| LimitBreach | id, exposure_id, severity, evidence_reference, remediation_owner_id, status |
| RiskEvent | id, profile_id, event_type, severity, evidence_reference, status |

## Streaming Events
Events emitted to `apg.fintech.risk.lifecycle` via Bytewax.
| Event | Trigger |
|-------|---------|
| risk_appetite_registered | Appetite threshold registered |
| risk_profile_created | Risk profile created |
| risk_exposure_recorded | Exposure recorded |
| risk_control_evaluated | Control assurance recorded |
| risk_stress_scenario_recorded | Stress scenario recorded |
| risk_limit_breach_recorded | Limit breach recorded |
| risk_event_opened | Risk event opened |
| risk_review_recorded | Review completed |
| risk_agent_registered | AI agent registered |
| var_backtest_exception | Kupiec POF test failed (model_valid=False) |
| model_drift | PSI > 0.10 or VaR model failure detected |

## Edge Cases Handled
- Limit overrides require human approval as a hard deny — exceeding a risk limit without approval is denied, not just flagged
- Stress scenario probability is in basis points (0–10000); 0 bps is valid (tail risk); values above 10000 are rejected
- Control effectiveness scores enforced 0–100; rule fires on `effectiveness_score_valid: False`
- Risk profiles cover both financial (accounts, portfolios) and operational (agents, counterparties) subjects
- `model_drift` is a first-class risk event type for ML model operational risk tracking
- `reverse_stress_test` returns `critical_shock_bps: null` when no shock in [0, 10000] bps breaches the threshold
- `ifrs9_stage_migration` macro_multiplier is clamped to [0.5, 3.0] to prevent implausible overlays
- `intraday_liquidity_monitor` maintains state across calls within a session; reset on service restart

## Composability
- **Upstream**: `fintech_kyc` provides customer identity for risk profile creation; `fintech_aml` and `fintech_fraud` feed fraud/AML signals into profile scoring
- **Downstream**: `fintech_compliance` reads control evaluation records as compliance evidence; `fintech_blockchain` and `fintech_crypto` use risk profiles for DeFi/crypto governance; `fintech_lending` uses profiles for credit application review
- **Peer**: Deployed alongside `fintech_compliance` (control framework) and `fintech_regtech` (regulatory capital requirements)

## Development Notes
- `human_approval_required_for_limit_override` is a governance configuration flag; makes limit override a hard deny, not require_review — intentionally stricter
- The `third_party` domain covers vendor risk (technology providers, service partners), not only financial counterparties
- Risk appetite is per-domain and organization-wide; individual subject exposure limits are separate, linked via `limit_reference` on exposure records
- `probability_bps` convention avoids floating-point precision issues from percentage storage
- All v2.0 analytic methods (`var_backtest`, `reverse_stress_test`, `raroc_calculation`, `intraday_liquidity_monitor`, `ifrs9_stage_migration`, `regulatory_capital_report`, `sanctions_screening`, `psi_model_stability`, `risk_report_summary`) are fully async and safe for concurrent invocation
