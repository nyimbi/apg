# FinTech Risk Management

## Overview
FinTech Risk Management provides the enterprise risk framework for the APG platform: risk appetite registration across credit, market, liquidity, operational, fraud, compliance, model, and third-party domains; tenant-scoped risk profiles for customers, merchants, wallets, accounts, portfolios, loans, agents, and counterparties; exposure tracking with limit enforcement and human-approval-gated overrides; control assurance with effectiveness scoring; stress scenario modeling; limit breach recording; risk event management; and governance reviews.

Limit overrides require human approval — exceeding a limit without approval is a hard deny. Control effectiveness scores must be in a valid range. Stress scenario probabilities must be in basis points (0–10000). All risk events stream to `apg.fintech.risk.lifecycle` via Bytewax.

**New in 1.2.0**: VaR backtesting (Kupiec POF), reverse stress testing, RAROC, IFRS 9 stage migration, Basel IV SA-CR regulatory capital, intraday liquidity monitoring (BCBS 248), sanctions screening, PSI model stability, and board-ready risk report summary.

## Capability ID
`fintech_risk`  Version: 1.2.0

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
Events emitted to the fintech event stream via Bytewax.
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

## Edge Cases Handled
- Limit overrides require human approval as a hard deny (not require_review) — this is stricter than most other capabilities; exceeding a risk limit without approval is denied, not just flagged for review
- Stress scenario probability is expressed in basis points (0–10000 bps = 0–100%); a probability of 0 bps is valid (tail risk scenario), but values above 10000 are rejected
- Control effectiveness scores have a valid range enforced by the rule engine; the range is 0–100 (defined by the service layer); the rule fires when the flag `effectiveness_score_valid: False` is set
- Risk profiles cover both financial subjects (accounts, portfolios) and operational subjects (agents, counterparties); the subject type determines the semantic interpretation of the score
- `model_drift` is a supported risk event type — this enables operational risk tracking for ML models used in scoring pipelines

## Composability
- **Upstream**: `fintech_kyc` provides customer identity for risk profile creation; `fintech_aml` and `fintech_fraud` provide fraud and AML risk signals as inputs to profile scoring
- **Downstream**: `fintech_compliance` reads control evaluation records as compliance control evidence; `fintech_blockchain` and `fintech_crypto` use risk profiles for DeFi and crypto operation governance; `fintech_lending` uses risk profiles for credit application review
- **Peer**: Deployed alongside `fintech_compliance` (control framework) and `fintech_regtech` (regulatory capital requirements)

## Development Notes
- `human_approval_required_for_limit_override` is a governance configuration flag — it makes the limit override a hard deny rather than a require_review; this is intentionally stricter than the pattern elsewhere
- The `third_party` domain covers third-party vendor risk, not just financial counterparties; this enables vendor risk assessments for technology providers and service partners
- Risk appetite is per-domain, not per-subject; appetite thresholds apply organization-wide; individual subject exposure limits are separate and linked via the `limit_reference` on exposure records
- `probability_bps` convention (basis points) avoids floating-point precision issues that arise when storing probabilities as percentages
