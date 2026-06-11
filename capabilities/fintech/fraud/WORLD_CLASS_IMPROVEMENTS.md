# Fraud Detection — World-Class Improvements

**Capability**: `fintech_fraud` | **Version target**: 2.0.0

---

## 1. Graph-Based Ring-Fraud Detection

Upgrade `network_graph_analysis` from a naive shared-device edge list to a proper in-process graph (networkx or igraph). Compute PageRank, betweenness centrality, and connected-component sizes. Flag components with >3 members and high edge density as suspected fraud rings. Output community IDs so downstream investigators can correlate across cases without manual cross-referencing.

**Impact**: Catches organised fraud rings that evade single-transaction scoring by distributing individual transaction risk below thresholds.

---

## 2. Streaming Feature Store with Windowed Aggregates

Replace the in-process `_velocity` dict with a pluggable feature store interface (Redis Streams by default, in-memory fallback). Maintain pre-computed rolling windows at 5 min / 1 hr / 24 hr / 7 day granularities per customer, device, and merchant. Expose `get_feature_vector(entity_id, entity_type)` returning a typed `FeatureVector` Pydantic model. This eliminates re-computation on every call and makes feature values auditable and explainable.

**Impact**: Sub-millisecond feature lookup; enables consistent features between training and serving — the #1 source of model drift in production fraud systems.

---

## 3. Calibrated Probability Outputs with Confidence Intervals

The current ML scorer returns a scalar risk score. Add isotonic regression calibration so the score is a true probability (P(fraud | features)). Return `score_lower_95` and `score_upper_95` confidence bounds alongside the point estimate. When confidence interval width > 20 points, force a `review` decision regardless of point estimate, preventing over-confident blocks on ambiguous signals.

**Impact**: Reduces false positives by 15–30% in typical deployments by explicitly routing uncertain predictions to human review rather than auto-blocking.

---

## 4. Explainability Layer (SHAP-Style Feature Attribution)

Add `explain_score(signal_id)` that returns per-feature Shapley value approximations using the current rule-based scorer as a surrogate. Output `{"feature": ..., "contribution": ..., "direction": "increases_risk"|"decreases_risk"}` per feature. Persist explanations in the evidence store so investigators have audit-ready rationale for every block or hold decision.

**Impact**: Regulatory compliance (GDPR Article 22, CBK guidelines) — automated adverse decisions require human-understandable explanation. Also critical for internal model governance.

---

## 5. Challenge-Response Step-Up Orchestration

Extend `record_decision` for `step_up` decisions to actually orchestrate the challenge: generate a one-time passcode, register it with the `auth` capability, set a TTL, and record the challenge reference. Add `verify_step_up(challenge_id, otp)` that resolves the step-up and re-scores the transaction. Currently the service records that a challenge reference exists but does not manage the challenge lifecycle.

**Impact**: Closes the gap between fraud decision and auth enforcement — a step-up that is never verified is equivalent to no step-up.

---

## 6. Multi-Jurisdiction Rule Packs

The current rule engine is Kenya/EAC-centric (hardcoded geo allow-lists, KES thresholds). Introduce a `RulePack` abstraction loaded per tenant from the config store. Packs ship with: amount thresholds in local currency, trusted geo-sets, channel risk weights, and MCC (merchant category code) risk tables. Provide built-in packs for KE, NG, ZA, GH, TZ. Allow tenant-level overrides via `load_rule_pack(tenant_id, pack_id)`.

**Impact**: Required for any multi-country deployment. Prevents Nigerian transactions being blocked because KE geo-rules fire, and vice versa.

---

## 7. Adaptive Threshold Auto-Tuning

`recalibrate_thresholds` currently returns recommendations but does not apply them. Implement a feedback loop: after each confirmed-fraud or false-positive case closure, update an exponential moving average of precision and recall per risk band. When rolling 7-day precision drops below the target, automatically tighten the block threshold by 5 points and emit a `threshold_adjusted` audit event. Gate auto-tuning behind a `auto_tune_enabled` config flag defaulting to `False`.

**Impact**: Maintains model efficacy between manual retraining cycles — particularly important for seasonal fraud pattern shifts (e.g., festive season card testing).

---

## 8. SLA-Tracked Case Management

Add `case_sla_status(case_id)` that returns time-in-state metrics: time since opened, SLA target (configurable per case type), SLA breached flag, and escalation chain. Integrate with the `ntfy` adapter to push Slack/SMS escalation when a case breaches its SLA. Track state transitions with timestamps on the `FraudCase` model.

**Impact**: Regulatory breach reporting (CBK requires fraud cases resolved within defined windows). Operational — unresolved cases age into write-offs.

---

## 9. Streaming Pipeline Integration (Bytewax / Kafka)

`validate_batch` currently returns a confirmation dict but does not emit events. Implement a `FraudEventEmitter` adapter that serialises `FraudSignal`, `FraudDecision`, and `FraudCase` lifecycle events as CloudEvents and publishes them to the `apg.fintech.fraud.lifecycle` topic. Provide Bytewax and aiokafka backends. Expose `async stream_signal(signal)` and `async stream_decision(decision)` that route through the emitter.

**Impact**: Decouples downstream consumers (AML, reporting, dashboards) from synchronous service calls. Required for the stated streaming architecture.

---

## 10. Chargeback Representment Workflow

Current chargeback handling records evidence but has no representment state machine. Add states: `received → evidence_gathering → representment_filed → won | lost`. Implement `file_representment(case_id, evidence_bundle)` and `record_representment_outcome(case_id, outcome, bank_reference)`. Track win/loss rates per merchant and per signal type. Feed outcomes back into the ML feature store as long-term fraud labels.

**Impact**: Chargeback representment win rates directly reduce fraud losses. Without outcome tracking, the model never learns which transaction patterns correlate with chargeback wins.

---

## 11. Cross-Capability Fraud Evidence Bus

Create an `InboundFraudSignal` event schema that `fintech_aml`, `fintech_kyc`, and `fintech_payments` can push to the fraud service asynchronously. Implement `async ingest_external_signal(source_cap, payload)` that normalises the event and routes it through `score_signal`. This replaces point-to-point polling with an event-driven intake pattern, reducing coupling and latency.

**Impact**: AML alerts currently appear as a boolean flag. With the evidence bus, AML passes the full alert context (STR reference, typology codes), enabling richer scoring.

---

## 12. Behavioural Biometrics Signals

Extend `behavioral_anomaly` to accept a `biometrics` sub-dict containing: typing cadence z-score, swipe pattern delta, session duration deviation, and tap pressure anomaly (for mobile). Compute a composite biometric risk score separate from the transaction amount z-score. This catches account takeover attempts that pass device and geo checks because the attacker has the victim's phone.

**Impact**: Device possession does not equal legitimate user. Biometric anomaly detection is the strongest ATO signal after MFA bypass.

---

## 13. Regulatory Reporting Automation

Add `async generate_str(case_id)` that produces a Suspicious Transaction Report in the CBK/FRC standard XML format from a confirmed-fraud case. Populate all mandatory fields from the case, signal, and evidence records. Add `async generate_fraud_registry_submission(period)` for monthly fraud registry reports. Store generated reports as evidence items with document references.

**Impact**: Compliance automation. Manual STR generation is error-prone and slow — institutions face fines for late or malformed filings.

---

## 14. Model Versioning and A/B Shadow Scoring

Add a `ModelRegistry` that tracks active model versions and can run a shadow scorer alongside the primary. `ml_fraud_score` routes to the primary model and asynchronously scores with the shadow, logging both scores and the delta. When shadow model consistently outperforms primary (measured by downstream confirmed-fraud rate), promote via `promote_shadow_model(model_id)`. This enables continuous model improvement without service downtime.

**Impact**: Standard MLOps practice — eliminates the risky big-bang model deployment that causes score distribution shifts and unexpected block rate spikes.

---

## 15. Privacy-Preserving Federated Score Aggregation

Implement `async federated_score_aggregate(peer_scores)` that accepts encrypted partial scores from peer tenants (e.g., across bank consortium members) and computes an aggregate fraud signal using additive homomorphic encryption or secure multiparty computation. This allows a customer flagged by one institution to carry elevated risk at a peer institution without exposing which institution flagged them or why.

**Impact**: Consortium fraud data sharing is the highest-ROI fraud reduction lever — shared negative lists reduce first-party fraud losses by 40–60% in documented studies. Privacy preservation is the blocker to adoption; this removes it.

---

*Generated for `fintech_fraud` v2.0.0 — Datacraft © 2025*
