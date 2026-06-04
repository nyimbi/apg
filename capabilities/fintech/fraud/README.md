# Fraud Detection

## Overview
Fraud Detection provides real-time transaction risk scoring, multi-factor decision making (approve, step-up, hold, block, review), account takeover detection, device risk assessment, chargeback evidence management, and fraud case investigation. It acts as the cross-cutting fraud control layer across all payment-generating capabilities — every financial operation that carries a monetary amount requires a fraud signal before authorization can proceed.

Fraud signals require KYC linkage (no anonymous fraud scoring). Hold and block decisions require both a reason and human approval. Step-up decisions require an auth challenge reference. All fraud lifecycle events stream to `apg.fintech.fraud.lifecycle` via Bytewax.

## Capability ID
`fintech_fraud`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| fraud_signal_scoring | Score payment, wallet, login, device, refund, and chargeback signals with 0–100 risk score |
| transaction_risk_decisioning | Record approve/step-up/hold/block/review decisions with evidence gates |
| account_takeover_detection | Flag and review account takeover indicators |
| device_risk_detection | Score device anomalies and bind device risk to transactions |
| chargeback_evidence_workflow | Capture and manage chargeback evidence records |
| fraud_case_management | Open, investigate, and resolve fraud cases with disposition and reviewer |
| fraud_agent_workflow | Register AI agents for transaction risk analysis, chargeback review, and investigation |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Risk analyst notifications |
| nlpc | NLP processing |
| keym | Key management |
| fintech_payments | Payment transaction source |
| fintech_wallets | Wallet transfer source |
| fintech_kyc | KYC profile linking (mandatory per fraud signal) |
| fintech_aml | AML alert cross-reference |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| scoring.review_threshold | number | 45 | Score triggering review flag |
| scoring.step_up_threshold | number | 60 | Score triggering step-up |
| scoring.hold_threshold | number | 75 | Score triggering hold |
| scoring.block_threshold | number | 90 | Score triggering block |
| scoring.max_score | number | 100 | Maximum fraud risk score |
| decisions.supported_decisions | list | approve, step_up, hold, block, review | Valid decision values |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-fraud/dashboard | GET | fintech_fraud:view | Overview |
| signals | /fintech-fraud/signals | GET/POST | fintech_fraud:score | Signals |
| decisions | /fintech-fraud/decisions | GET/POST | fintech_fraud:decide | Decisions |
| cases | /fintech-fraud/cases | GET/POST | fintech_fraud:investigate | Cases |
| chargebacks | /fintech-fraud/chargebacks | GET/POST | fintech_fraud:chargebacks | Evidence |
| devices | /fintech-fraud/devices | GET/POST | fintech_fraud:devices | Signals |
| agents | /fintech-fraud/agents | GET/POST | fintech_fraud:admin | Automation |
| settings | /fintech-fraud/settings | GET/POST | fintech_fraud:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| signal_requires_kyc_link | Signal without KYC profile | deny |
| money_amount_positive | Money-bearing signal with non-positive amount | deny |
| risk_score_range | Score outside 0–100 | deny |
| high_risk_score_requires_review | Score > review_threshold without review | require_review |
| velocity_requires_review | Velocity indicator without review | require_review |
| device_anomaly_requires_review | Device anomaly without review | require_review |
| aml_alert_requires_review | Signal linked to AML alert without review | require_review |
| chargeback_requires_evidence | Chargeback signal without evidence | deny |
| step_up_requires_challenge | Step-up decision without challenge reference | deny |
| hold_or_block_requires_reason | Hold or block without reason | deny |
| hold_or_block_requires_human_approval | Hold or block without human approval | deny |
| case_resolution_requires_disposition | Case resolved without disposition | deny |
| fraud_batch_requires_bytewax | Batch without Bytewax | deny |
| fraud_event_requires_bytewax | Event without Bytewax | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| FraudSignal | id, subject_reference, kyc_profile_id, signal_type, channel, source_reference, amount, currency, risk_score, status |
| FraudDecision | id, signal_id, decision, reason, challenge_reference, human_approval_reference, reviewer_id |
| FraudCase | id, signal_id, case_type, investigator_id, evidence_references, status, disposition |
| ChargebackEvidence | id, transaction_reference, evidence_references |
| DeviceRisk | id, signal_id, device_reference, anomaly_flags, risk_score |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| fraud_signal_scored | Signal scored and stored |
| fraud_decision_recorded | Risk decision recorded |
| fraud_case_opened | Investigation case opened |
| fraud_case_resolved | Case resolved with disposition |
| fraud_agent_registered | AI agent registered |

## Edge Cases Handled
- KYC linkage is mandatory on every signal — anonymous fraud scoring is architecturally blocked; this prevents scoring bypass via identity-free payment paths
- Hold and block decisions require BOTH a reason AND human approval — either missing component produces a deny; reason alone or approval alone is insufficient
- Step-up decisions require an auth challenge reference — a step-up without a challenge is a logic error and is rejected to prevent phantom step-up records
- Chargeback signals specifically require evidence references at signal time — chargebacks without evidence cannot be scored, preventing unsupported chargeback abuse
- The score range 0–100 is enforced; a caller passing a score of 101 is rejected even if the decision would be `block` regardless

## Composability
- **Upstream**: `fintech_kyc` provides the KYC profile linkage required per signal; `fintech_aml` alert presence is checked as an additional fraud indicator; `fintech_payments` and `fintech_wallets` are primary signal sources
- **Downstream**: `fintech_cards` reads fraud decisions as authorization gates; `fintech_remittance` reads fraud decisions as transfer gates; `fintech_bnpl` requires fraud evidence at checkout
- **Peer**: Deployed alongside `fintech_aml` (complementary financial crime detection) and `fintech_kyc` (identity foundation)

## Development Notes
- Score thresholds (45/60/75/90) are configured defaults; they are not enforced by the rule engine directly — the rule engine checks the `high_risk_score`, `velocity_indicator`, etc. flags set by the caller based on these thresholds
- Both batch operations (`fraud_batch`) and individual events (`fraud_event`) require Bytewax routing — two separate `_ne` guard rules
- `SUPPORTED_SIGNAL_TYPES` includes `agent_review` for signals generated by AI agents acting as fraud reviewers
- Case resolution requires disposition; valid dispositions are service-layer defined, not constrained by the rule engine to a fixed list
