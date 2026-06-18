# Fraud Detection

## Overview
Fraud Detection provides real-time transaction risk scoring, multi-factor decision making (approve, step-up, hold, block, review), account takeover detection, device risk assessment, chargeback evidence management, and fraud case investigation. It acts as the cross-cutting fraud control layer across all payment-generating capabilities — every financial operation that carries a monetary amount requires a fraud signal before authorization can proceed.

Fraud signals require KYC linkage (no anonymous fraud scoring). Hold and block decisions require both a reason and human approval. Step-up decisions require an auth challenge reference. All fraud lifecycle events stream to `apg.fintech.fraud.lifecycle` via Bytewax.

## Capability ID
`fintech_fraud`  Version: 2.0.0

## Features
- Real-time transaction fraud scoring with ML + rule engine fusion
- Async-first API — all scoring, detection, and analytics methods are `async`
- Velocity checks with configurable rolling windows (5 min / 1 hr / 24 hr / 7 day)
- Device fingerprint binding and multi-customer device anomaly detection
- Behavioral anomaly detection with z-score baselines
- Account takeover (ATO) detection from login event signals
- Synthetic identity fraud heuristics
- Mobile money (M-Pesa/USSD), agency banking, and card-specific fraud checks
- Network graph analysis for fraud ring detection
- AML typology pattern detection (structuring, rapid movement)
- Watchlist screening (internal/external)
- Chargeback fraud analytics
- Merchant fraud risk indexing
- Bulk/batch transaction scoring via `asyncio.gather`
- False-positive feedback loop for behavioral baseline adjustment
- Regulatory reporting: `fraud_report`, `false_positive_rate_report`, `export_fraud_data`
- Real-time alert queue with `acknowledge_alert` workflow
- Full audit trail on every operation

## Provides
| Service | Description |
|---------|-------------|
| fraud_signal_scoring | Score payment, wallet, login, device, refund, and chargeback signals with 0–100 risk score |
| transaction_risk_decisioning | Record approve/step-up/hold/block/review decisions with evidence gates |
| account_takeover_detection | Flag and review account takeover indicators from login events |
| device_risk_detection | Score device anomalies and bind device risk to transactions |
| chargeback_evidence_workflow | Capture and manage chargeback evidence records |
| fraud_case_management | Open, investigate, and resolve fraud cases with disposition and reviewer |
| fraud_agent_workflow | Register AI agents for transaction risk analysis, chargeback review, and investigation |
| ml_fraud_scoring | Ollama-backed ML scoring with deterministic rule-based fallback |
| behavioral_analytics | Customer behavioral baseline tracking and anomaly detection |
| network_graph_analysis | Shared-device graph for fraud ring detection |
| regulatory_reporting | Fraud statistics, false-positive rates, and data export |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication + step-up challenge management |
| audl | Audit trail |
| ntfy | Risk analyst notifications + real-time alert push |
| nlpc | NLP processing |
| keym | Key management |
| fintech_payments | Payment transaction source |
| fintech_wallets | Wallet transfer source |
| fintech_kyc | KYC profile linking (mandatory per fraud signal) |
| fintech_aml | AML alert cross-reference |

## Quick Start

```python
from capabilities.fintech.fraud.service import FraudDetectionService

svc = FraudDetectionService(tenant_id="acme", actor_id="api-gateway")

# Score a single transaction
result = await svc.detect_transaction_fraud({
    "transaction_id": "txn-001",
    "customer_id": "cust-42",
    "amount": 150_000,
    "currency": "KES",
    "channel": "mobile",
    "device_id": "dev-abc",
    "country": "KE",
})
print(result["recommended_decision"])  # "review" | "block" | "approve" ...

# Legacy synchronous scoring (v1 compat)
svc.score_signal(
    signal_id="sig-001",
    tenant_id="acme",
    subject_reference="cust-42",
    kyc_profile_id="kyc-99",
    signal_type="payment",
    channel="mobile",
    source_reference="txn-001",
    amount=150_000,
    currency="KES",
    risk_score=62,
)
```

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
| transaction_fraud_assessed | Full real-time transaction assessment complete |
| account_takeover_assessed | ATO detection run on login event |
| real_time_alert_raised | High-score alert pushed to queue |

## New Methods

### `detect_transaction_fraud(txn)` — full real-time pipeline
Runs velocity check, device fingerprint check, ML scoring, and rule evaluation concurrently. Single call for end-to-end transaction assessment.

```python
result = await svc.detect_transaction_fraud({
    "transaction_id": "txn-789",
    "customer_id": "cust-42",
    "amount": 85_000,
    "currency": "KES",
    "channel": "mobile",
    "device_id": "dev-xyz",
    "country": "NG",           # non-EAC → geo_anomaly flag
    "merchant_category": "crypto",  # → high_risk_merchant flag
})
# result["fraud_score"], result["recommended_decision"], result["velocity"], result["device"]
```

### `ml_fraud_score(features)` — Ollama-backed scoring with fallback
Uses `OLLAMA_BASE_URL` if set; falls back to deterministic rule-based scorer for offline/test use.

```python
score = await svc.ml_fraud_score({
    "amount": 500_000,
    "velocity_flag": True,
    "device_anomaly": False,
    "geo_anomaly": True,
    "cross_border": True,
    "high_risk_merchant": False,
    "night_transaction": False,
})
# score["score"], score["risk_band"], score["top_contributing_features"]
```

### `account_takeover_detection(login_event)` — ATO from login signals
Combines brute-force attempt count, MFA bypass, password change recency, device anomaly, and geo anomaly into a composite ATO score. Fires a real-time alert when score >= 65.

```python
ato = await svc.account_takeover_detection({
    "customer_id": "cust-42",
    "device_id": "dev-new",
    "ip_address": "197.x.x.x",
    "country": "CN",
    "failed_attempts": 6,
    "mfa_bypassed": True,
    "password_changed_recently": True,
})
# ato["is_suspected_takeover"], ato["signals_detected"], ato["recommended_action"]
```

### `bulk_score_signals(signals)` — concurrent batch ML scoring
Fans out `ml_fraud_score` over a list of feature dicts via `asyncio.gather`.

```python
results = await svc.bulk_score_signals([
    {"amount": 10_000, "velocity_flag": False},
    {"amount": 900_000, "velocity_flag": True, "geo_anomaly": True},
    {"amount": 250_000, "device_anomaly": True},
])
# list of score dicts, one per input
```

### `network_graph_analysis(customer_ids)` — fraud ring detection
Builds a shared-device graph across a customer cohort. `ring_suspected` fires when 3+ shared-device edges exist.

```python
graph = await svc.network_graph_analysis(["cust-1", "cust-2", "cust-3", "cust-4"])
# graph["shared_device_edges"], graph["ring_suspected"], graph["edges"]
```

## World-Class Enhancements (v2.0)

1. **Graph-Based Ring-Fraud Detection** — upgrade `network_graph_analysis` to a proper in-process graph (networkx/igraph) with PageRank, betweenness centrality, and community IDs. Catches organised rings that distribute risk below per-transaction thresholds.

2. **Streaming Feature Store with Windowed Aggregates** — replace the in-process `_velocity` dict with a pluggable feature store (Redis Streams default, in-memory fallback). Pre-computed rolling windows at 5 min / 1 hr / 24 hr / 7 day per customer, device, and merchant. Eliminates the #1 cause of training/serving feature drift.

3. **Calibrated Probability Outputs with Confidence Intervals** — add isotonic regression calibration so scores are true probabilities P(fraud|features), with `score_lower_95` and `score_upper_95` bounds. CI width > 20 points forces `review` regardless of point estimate — reduces false positives 15–30%.

4. **Explainability Layer (SHAP-Style Feature Attribution)** — `explain_score(signal_id)` returns per-feature Shapley value approximations with direction tags. Persists explanations in evidence store for GDPR Article 22 / CBK compliance.

5. **Challenge-Response Step-Up Orchestration** — extend `record_decision` for `step_up` to generate an OTP, register it with the `auth` capability, set a TTL, and add `verify_step_up(challenge_id, otp)` to resolve and re-score. Closes the gap between decision and enforcement.

6. **Multi-Jurisdiction Rule Packs** — `RulePack` abstraction loaded per tenant: amount thresholds in local currency, trusted geo-sets, channel risk weights, MCC risk tables. Built-in packs for KE, NG, ZA, GH, TZ with `load_rule_pack(tenant_id, pack_id)`.

7. **Adaptive Threshold Auto-Tuning** — `recalibrate_thresholds` applies recommendations via EMA-tracked precision/recall per risk band. When 7-day precision drops below target, tightens block threshold by 5 points and emits `threshold_adjusted` audit event. Gated by `auto_tune_enabled` flag.

8. **SLA-Tracked Case Management** — `case_sla_status(case_id)` returns time-in-state, SLA target, breach flag, and escalation chain. Integrates with `ntfy` for Slack/SMS escalation on SLA breach.

9. **Streaming Pipeline Integration (Bytewax / Bytewax)** — `FraudEventEmitter` adapter serialises `FraudSignal`, `FraudDecision`, and `FraudCase` lifecycle events as CloudEvents and publishes to `apg.fintech.fraud.lifecycle`. `stream_signal` and `stream_decision` async entry points.

10. **Chargeback Representment Workflow** — state machine `received → evidence_gathering → representment_filed → won | lost` with `file_representment(case_id, evidence_bundle)` and `record_representment_outcome`. Win/loss rates feed back to ML feature store as long-term fraud labels.

11. **Cross-Capability Fraud Evidence Bus** — `InboundFraudSignal` event schema for `fintech_aml`, `fintech_kyc`, and `fintech_payments` to push asynchronously via `ingest_external_signal(source_cap, payload)`. Replaces point-to-point polling with event-driven intake.

12. **Behavioural Biometrics Signals** — extend `behavioral_anomaly` with a `biometrics` sub-dict: typing cadence z-score, swipe pattern delta, session duration deviation, tap pressure anomaly. Composite biometric score separate from amount z-score. Primary ATO signal after MFA bypass.

13. **Regulatory Reporting Automation** — `generate_str(case_id)` produces Suspicious Transaction Reports in CBK/FRC standard XML. `generate_fraud_registry_submission(period)` for monthly registry reports. Reports stored as evidence items with document references.

14. **Model Versioning and A/B Shadow Scoring** — `ModelRegistry` tracks active model versions. `ml_fraud_score` runs primary and logs shadow score + delta asynchronously. `promote_shadow_model(model_id)` when shadow consistently outperforms. Zero-downtime model promotion.

15. **Privacy-Preserving Federated Score Aggregation** — `federated_score_aggregate(peer_scores)` accepts encrypted partial scores from consortium members and computes aggregate fraud signal via additive homomorphic encryption or secure multiparty computation. Enables shared negative lists (40–60% first-party fraud reduction) without exposing which institution flagged a customer.

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
- `FintechFraudService` is an alias for `FraudDetectionService` for backward compatibility
- Set `OLLAMA_BASE_URL` to enable Ollama-backed ML scoring; omit for deterministic rule-based fallback (CI-safe)
