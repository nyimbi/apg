# KYC Service — World-Class Improvement Proposals

## Overview
Analysis of the current `KYCService` implementation against global KYC/AML best practices,
FATF Recommendations, Basel III, ISO 20022, and Africa-specific regulatory requirements.

---

## 1. AI-Powered Document Fraud Detection with Local Vision Models

**Gap**: `verify_document_authenticity` uses a deterministic stub with a hardcoded
`synthetic_fraud_score`. Real fraud patterns (screen replay, morphed ID, UV feature
simulation) are not detectable without a computer-vision model.

**Improvement**: Route via an Ollama-served vision model (e.g. `llava:13b` or `moondream`)
that classifies: security hologram presence, microprint integrity, font consistency,
background gradient continuity, and edge artifact detection. Add a new async method
`ai_document_fraud_analysis(document_id, model="llava")` returning a structured
`FraudSignal` with per-feature scores. No cloud API dependency.

---

## 2. Continuous Transaction-Behavioural Risk Re-Scoring

**Gap**: `calculate_risk_score` is point-in-time. KYC risk drifts with customer behaviour
(velocity, counterparty risk, channel changes) but the service has no feedback loop from
the payment/wallet layer.

**Improvement**: Add `recompute_behavioral_risk(customer_id, transaction_summary)` that
accepts an aggregated transaction summary from `fintech_aml` and recalculates the risk
score using a rolling 90-day behavioural window. Emit a `kyc_behavioral_risk_updated`
event downstream and update `RiskProfile.score_breakdown` with a `behavioral` sub-score.

---

## 3. Federated Identity with SSI / Verifiable Credentials

**Gap**: Identity verification is fully centralised. Global best practice and emerging
African digital-identity infrastructure (Kenya NIIMS, Nigeria NIMC e-ID, Ghana GhanaCard
digital) increasingly support W3C Decentralised Identifiers (DIDs) and Verifiable
Credentials (VCs).

**Improvement**: Add `verify_verifiable_credential(vc_jwt, issuer_did)` that validates
W3C VC format, checks issuer DID resolution, verifies the JWT signature, and asserts
credential schema compliance. This enables wallet-based identity presentation without
re-uploading raw documents.

---

## 4. Encrypted-At-Rest Audit Trail with Tamper-Evidence

**Gap**: `_emit` writes audit events as plain JSON to the store. Any database admin can
modify or delete records, violating compliance integrity requirements.

**Improvement**: Chain each audit event to the previous via SHA-256 of the prior
event's ID, creating an append-only hash chain. Add `verify_audit_integrity(from_dt, to_dt)`
that re-derives the chain and flags any gaps or mutations. This satisfies ISAE 3402 /
SOC 2 Type II audit trail requirements.

---

## 5. Multi-Jurisdiction Regulatory Rule Engine

**Gap**: Business rules in `domain/rules.py` are static. Regulatory thresholds differ
by jurisdiction (e.g. CBK Kenya requires EDD for PEP at risk_score > 60 vs FATF default
of 75; CBN Nigeria mandates BVN linkage for all accounts).

**Improvement**: Replace hardcoded thresholds with a `JurisdictionRuleSet` registry keyed
by `country_code`. Add `load_jurisdiction_rules(country_code)` and
`evaluate_jurisdiction_compliance(application_id)` that applies the correct rule set.
This enables a single service deployment to serve KE, NG, GH, ZA, TZ concurrently with
correct local thresholds.

---

## 6. Async Batch Document OCR with Progress Tracking

**Gap**: `extract_document_data` is a single-document synchronous stub. Processing a
backlog of 10,000 documents requires individual API calls with no progress visibility.

**Improvement**: Add `bulk_document_ocr(application_ids, concurrency=10)` that spawns
`asyncio.Semaphore`-bounded concurrent `extract_document_data` calls, persists progress
in a batch job record, and emits `kyc_bulk_ocr_progress` events. Returns a `job_id`
for status polling via `get_batch_job_status(job_id)`.

---

## 7. Real-Time Sanctions List Delta Sync

**Gap**: `_DEFAULT_SANCTIONS_LISTS` is a static list. OFAC, UN, and EU lists update
daily/weekly but the service has no mechanism to pull fresh delta updates.

**Improvement**: Add `sync_sanctions_lists(lists)` that calls each list's update API
(or reads from a local mirror), computes a diff against the previous snapshot, and
re-screens all `approved` customers against changed entries only. Delta screening is
O(delta × customers) vs O(full_list × customers). Emit `kyc_sanctions_list_synced`
with `additions`, `removals`, and `customers_rescreened` counts.

---

## 8. Multi-Language Name Transliteration for Screening

**Gap**: `name_match_score` operates on Latin-script names only. A significant portion
of KYC subjects in Ethiopia (Ge'ez/Amharic), North Africa (Arabic), and East Africa
(Swahili diacritics) are screened with the wrong character set, causing false negatives.

**Improvement**: Add `transliterate_and_screen(name, source_script, aliases)` that
uses a local transliteration model (e.g. `Helsinki-NLP/opus-mt-*` via Ollama or a
lightweight phonetic transliterator) to produce Latin-script variants before running
PEP/sanctions screening. Store all variants in `PEPCheck.aliases_checked`.

---

## 9. Liveness Check with Challenge-Response (Active Liveness)

**Gap**: `perform_liveness_check` uses passive frame-count heuristics. A 5-frame
sequence can be spoofed with a video replay. NIST FIDO2 / ISO 30107-3 require
challenge-response active liveness for PAD Level 2.

**Improvement**: Add `active_liveness_challenge(session_id)` that generates a
randomised challenge sequence (blink twice, turn left, smile) and returns a
`ChallengeToken`. Add `verify_liveness_challenge_response(session_id, challenge_token,
response_frames)` that validates the response frames match the issued challenge. This
reaches ISO 30107-3 PAD Level 2 compliance.

---

## 10. Geographic Risk Layering with Cell-Tower / IP Signals

**Gap**: `is_high_risk_country` is a binary country-level flag. Intra-country risk
varies enormously (e.g. North-Eastern Kenya border vs Nairobi CBD; conflict zones in
Ethiopia vs Addis Ababa).

**Improvement**: Add `geo_risk_score(latitude, longitude, radius_km)` that queries a
local risk raster (ACLED conflict events, FSI fragility index, UNODC drug-route
proximity) and returns a sub-national geographic risk score (0.0–1.0). Integrate into
`calculate_risk_score` as a `geographic_risk` factor with configurable weight.

---

## 11. KYC Data Portability — Customer-Controlled Export

**Gap**: There is no mechanism for customers to export their own KYC data, which is
required by Kenya's Data Protection Act 2019, Nigeria's NDPR, GDPR (Art. 20), and
South Africa's POPIA.

**Improvement**: Add `export_customer_kyc_data(customer_id, format)` that packages
all KYC data (application, documents metadata, screening results, risk history) into
a structured JSON or PDF export, encrypted with the customer's public key if provided.
Add `delete_customer_kyc_data(customer_id, reason)` for right-to-erasure compliance
where regulatory retention periods allow.

---

## 12. Automated KYC Tier Upgrade via ML Scoring

**Gap**: `kyc_tier` is set at application creation and never automatically upgraded.
Customers who demonstrate low-risk behaviour over time are stuck in `standard` tier
even when they qualify for `simplified` or could be upgraded to `premium` service.

**Improvement**: Add `evaluate_tier_upgrade(customer_id)` that uses a trained gradient
boosting model (served via Ollama or local scikit-learn) to evaluate whether a customer
qualifies for tier upgrade based on: transaction history, document quality scores,
screening clearance streak, and account tenure. Emit `kyc_tier_upgrade_recommended`
for ops approval.

---

## 13. Cross-Capability Composability Event Bus

**Gap**: Service emits domain events but callers (`fintech_aml`, `fintech_fraud`,
`fintech_payments`) must poll the KYC store directly for status changes, creating
tight coupling.

**Improvement**: Add a structured event schema `KYCLifecycleEvent` (Pydantic model)
and replace raw `_emit` calls with a typed publish to a Bytewax topic with guaranteed
delivery semantics. Add `subscribe_to_kyc_events(event_types, handler)` for peer
capabilities to register interest. This decouples KYC from its consumers and enables
the event-driven architecture described in `cap_spec.md`.

---

## 14. UBO Corporate Ownership Tree Traversal

**Gap**: `beneficial_ownership_verification` only inspects direct UBO records at
exactly one layer of ownership. Modern corporate structures use multi-layer holding
companies to obscure beneficial ownership (common in NG/KE corporate structures).

**Improvement**: Add `resolve_ownership_tree(entity_id, max_depth=5)` that recursively
resolves corporate shareholders through registered business KYC records, building a
directed graph of ownership. Flag circular ownership patterns (common in shell
structures) and identify UBOs at any depth who exceed the threshold. Regulatory
reference: FATF R.24, EU 6AMLD.

---

## 15. KYC Health Dashboard with SLA Metrics

**Gap**: There is no operational telemetry. Ops teams cannot see: average time-to-approve,
document OCR failure rate, screening false-positive rate, or SLA breach risk.

**Improvement**: Add `kyc_health_metrics(period)` that computes: `p50/p95/p99`
processing times by step, `document_verification_pass_rate`, `screening_hit_rate`,
`edd_escalation_rate`, `sla_breach_count` (configurable SLA per tier), and
`onboarding_funnel_conversion`. Expose via a Flask-AppBuilder blueprint endpoint
at `/fintech-kyc/health` for ops dashboards. This enables proactive SLA management
and regulator-facing operational reporting.
