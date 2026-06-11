# Telecom Billing — World-Class Improvement Roadmap

**Capability:** `telecom_bil` (Billing & Revenue Management)
**Author:** Nyimbi Odero — Datacraft © 2025
**Date:** 2026-06-11

---

## 1. Real-Time Streaming Rating via Bytewax Pipelines

Current rating methods are request-response.  High-volume telcos process millions of CDRs/hour; batching them into synchronous calls creates back-pressure.  Replace the in-process rating loop with Bytewax dataflow pipelines that consume from a Kafka/Redpanda topic, apply tariff rules, and emit charged events downstream — enabling sub-second CDR-to-charge latency at carrier scale.

**Impact:** Scales to 10 M+ CDRs/hour; decouples rating from invoice generation; enables per-event fraud holds.

---

## 2. Policy-as-Code Tariff Engine with Hot-Reload

Hard-coded rate tables (`_DATA_TIERS`, `_PAYG_RATE`) require code deployments to change.  Introduce an OPA-backed (Open Policy Agent) tariff engine where operators upload JSON/Rego tariff policies that are evaluated at rating time and reloaded without restart.  Policy versions are stored, diffed, and rolled back independently of application code.

**Impact:** Eliminates change-freeze windows for tariff adjustments; enables A/B tariff experiments per subscriber cohort.

---

## 3. Multi-Currency & Real-Time FX Settlement

`currency` defaults to "KES" everywhere.  For MVNO/roaming settlement, charges must be expressed in the originating network's currency and converted at the settlement rate locked at billing run time.  Integrate with an ECB/CBK FX feed, store daily rates in the store, and apply them during invoice generation and interconnect reconciliation.

**Impact:** Supports East Africa multi-currency deployments (KES, UGX, TZS, RWF, USD) without manual rate updates.

---

## 4. Mediation-Grade CDR Deduplication with Bloom Filters

`record_cdr` accepts duplicate CDR IDs silently.  Production mediation requires probabilistic deduplication at ingestion: a Bloom filter (or Redis HyperLogLog) to flag probable duplicates before expensive DB writes, followed by a deterministic hash check on (msisdn, duration, timestamp) for confirmation.  Rejected duplicates are quarantined, not dropped, for audit.

**Impact:** Eliminates double-billing incidents; reduces dispute volume by 30-40% based on industry benchmarks.

---

## 5. Convergent Real-Time Notification Engine

The `_NoopNotify` default means no notifications in dev/test.  Replace with an async notification multiplexer that supports SMS (Africa's Talking / Twilio), email (Resend), WhatsApp (WABA), push (FCM), and USSD callback — all configurable per subscriber preference stored in the customer profile.  Notifications include rich templates with itemised bill previews.

**Impact:** Reduces inbound billing inquiry calls by 20-25%; improves payment conversion on dunning reminders.

---

## 6. AI-Powered Fraud Scoring on CDR Ingestion

Revenue leakage detection runs as a batch report.  Integrate an Ollama-served anomaly model (`telecom_fraud_cdr`) that scores each CDR at mediation time: impossible location velocity, SIM swap after high-value top-up, international bulk SMS blasting.  Suspicious CDRs are quarantined pending human review rather than rated and billed.

**Impact:** Catches SIMBox fraud and subscription fraud before revenue loss is realised, not after.

---

## 7. Hierarchical Account Groups for Corporate & MVNE Billing

`BilConvergentAccount` models a flat master/member structure.  Enterprise customers need multi-tier hierarchies: holding company → subsidiary → cost centre → individual line, with shared caps, cross-subsidisation rules, and per-node invoice suppression.  Implement a tree-structured account model with recursive charge rollup.

**Impact:** Unlocks enterprise MVNE contract wins; single invoice across 10,000-line corporate accounts.

---

## 8. Automated Revenue Assurance Reconciliation Loop

`revenue_leakage_detection` generates a point-in-time report.  Productionise it as a continuous reconciliation loop: every billing cycle, compare rated revenue against switch/mediation feed totals, flag variances above configurable thresholds, auto-raise internal disputes, and track resolution SLAs.  Integrated with the dunning escalation path for unpaid interconnect settlements.

**Impact:** Closes the gap between CDR generation and revenue recognition; reduces leakage from typical 3-7% to under 1%.

---

## 9. Prepaid Balance Reservation (CAMEL/Diameter Emulation)

`real_time_balance_check` checks but does not reserve.  Implement a two-phase commit pattern: `reserve_balance` locks the amount for the call duration, `commit_charge` finalises on call completion, `rollback_reservation` releases on call failure.  Mirrors CAMEL/Diameter Gy interface semantics for OCS integration.

**Impact:** Prevents over-spend on prepaid accounts; enables real-time credit control for 5G slices.

---

## 10. Regulatory Levy & Tax Engine (VAT, Excise, USF)

Tax is uniformly calculated at 16% VAT.  African regulators impose stacked levies: VAT (16%), Excise Duty on airtime (15% in Kenya), Universal Service Fund contributions (0.5%), and county-level levies in some jurisdictions.  Build a pluggable tax engine with levy stacking, exemption certificates, and automated regulatory filing exports (iTax XML format).

**Impact:** Compliance with KRA requirements; eliminates manual tax reconciliation; supports exemptions for NGO/government accounts.

---

## 11. Dispute SLA Management & Escalation

`raise_billing_dispute` sets `sla_deadline: None`.  Populate SLA deadlines per dispute tier (standard: 14 days, high-value: 5 days, regulatory: 2 days), track breach status, auto-escalate via the dunning workflow, and generate regulator-ready resolution reports (CA dispute register format).

**Impact:** Avoids regulatory penalties for SLA breaches; reduces cost-per-dispute through automated triage.

---

## 12. Idempotent Charge & Invoice API

`record_charge` and `generate_invoice` silently overwrite on duplicate IDs.  Add idempotency keys (client-supplied or server-generated) at the API layer: repeated calls with the same key within a TTL window return the original result without side effects.  Store idempotency records in the BillingStore with expiry.

**Impact:** Safe for client retries on network failures; eliminates duplicate invoice generation during bill run restarts.

---

## 13. Event-Sourced Audit Trail with CQRS Read Models

`audit_events` is an in-memory list.  Replace with full event sourcing: every state mutation appends an immutable event to a Postgres event table (using TimescaleDB for time-series efficiency).  Project read models (invoice summary, subscriber balance history, dispute timeline) from the event log.  Enables full temporal queries ("what was the balance at 14:32 yesterday?").

**Impact:** Meets GDPR/DPA audit requirements; enables forensic billing investigations; simplifies compliance reporting.

---

## 14. Bundle Lifecycle Management (Activation, Expiry, Auto-Renewal)

Bundles have `status` and `remaining_units` but no expiry or auto-renewal.  Add `valid_from`, `valid_to`, `auto_renew`, `renewal_price`, and `expiry_action` (expire vs rollover).  A scheduler (backed by APG `schd` capability) triggers expiry notifications at T-3 days and T-0, auto-debits for renewal, and rolls over unused units per configured policy.

**Impact:** Reduces manual bundle management overhead; improves subscriber retention through seamless auto-renewal.

---

## 15. Convergent Invoice Presentation with PDF Generation

`view_bill` returns a JSON dict.  Subscribers and enterprise finance teams need print-quality PDF invoices with Datacraft branding, itemised CDR listings, tax breakdowns, QR code for M-Pesa payment, and regulatory-compliant layout (VAT invoice number, PIN/TIN, supplier details).  Generate via WeasyPrint/reportlab, store in object storage, return a signed URL.

**Impact:** Eliminates manual invoice PDF production; enables self-service download from customer portal; satisfies KRA VAT invoice requirements.
