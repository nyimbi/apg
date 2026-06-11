# World-Class Improvements: Government Contracts & Procurement (government_con)

## Overview

This document catalogues 15 high-impact improvements to elevate the `government_con` capability
from a functional prototype to a production-grade public procurement platform. Each improvement
is scoped to a concrete engineering outcome with measurable impact on correctness, compliance,
or operational efficiency.

---

## 1. Full Async Service Layer

**Problem**: All service methods are synchronous, blocking the event loop when DB I/O, external PPDA
notifications, or AI risk scoring is in flight.

**Fix**: Convert every public method to `async def`, replace in-memory dicts with `asyncpg`/`SQLAlchemy
asyncio` queries, and introduce an `AsyncProcurementService` base that wraps a repository interface.

**Impact**: 10-50x throughput under concurrent load; enables integration with async event buses
(bytewax, Kafka) without thread-pool bridging.

---

## 2. Persistent PostgreSQL-Backed Repository

**Problem**: All state lives in instance-level Python dicts — data vanishes on process restart;
no tenant isolation at the database level.

**Fix**: Introduce a `ProcurementRepository` using SQLAlchemy 2.x async ORM with row-level security
(RLS) policies keyed on `tenant_id`. Map every in-memory dict to a corresponding table with proper
indexes and foreign key constraints.

**Impact**: Durable state, multi-process scale-out, tenant data isolation enforced at DB layer.

---

## 3. Structured Audit Trail with Immutable Event Store

**Problem**: `audit_events` is a plain list appended in-memory — no persistence, no tamper
evidence, no query capability.

**Fix**: Persist audit events to an append-only `con_audit_log` table (no UPDATE/DELETE RLS policy).
Include `actor_id`, `ip_address`, `session_id`, `prev_hash` (SHA-256 chain), and
`correlation_id`. Expose `async def audit_trail(contract_id)` for forensic queries.

**Impact**: Satisfies PPDA Act § 66 record-keeping requirements; enables anti-corruption audit
queries by the `intel` capability.

---

## 4. AI-Powered Bid Collusion Detection

**Problem**: No mechanism to flag statistically abnormal bidding patterns (price clustering,
vendor rotation, suspiciously identical bids).

**Fix**: Add `async def detect_bid_collusion(tender_id)` that calls a locally-hosted Ollama
embedding model to cluster bid vectors by amount, timing, and document fingerprint. Score
clusters above a threshold as `suspected_collusion`, emit a `BidCollusionAlert` cloud event
to the `intel` capability.

**Impact**: Automated red-flagging of cartel behaviour — a top PPDA enforcement priority.

---

## 5. Real-Time PPDA Notification Gateway

**Problem**: PPDA notification references are string placeholders; no actual transmission occurs.

**Fix**: Add `async def submit_ppda_notification(award_id, notification_type)` that POST-s a
structured payload to the PPDA e-Government portal REST API (or mocks it in test via
`pytest-httpserver`). Record HTTP response code, timestamp, and acknowledgement reference.

**Impact**: Closes compliance gap — non-submission of award notices is a criminal offence
under PPDA Act § 45.

---

## 6. Conflict-of-Interest Declaration Workflow

**Problem**: No mechanism to capture or enforce committee-member conflict-of-interest (COI)
declarations before evaluation begins.

**Fix**: Add `async def declare_conflict_of_interest(tender_id, member_id, has_conflict, details)`
and block evaluation scoring by any member with an undeclared or positive COI. Store declarations
in `con_coi_declarations` table with a digital signature field.

**Impact**: Directly addresses PPDA § 43 (corrupt practices) and satisfies donor-funded project
procurement requirements (World Bank, AfDB).

---

## 7. Multi-Criteria Weighted Scoring Engine

**Problem**: `evaluate_bid` accepts a single scalar `score` — no breakdown by technical, financial,
or capacity criteria. Evaluation is effectively a black box.

**Fix**: Add `async def score_bid_criteria(tender_id, bid_id, criteria_scores: dict[str, float])`
that applies configurable criterion weights from the tender's evaluation matrix, computes a
weighted composite score, and stores per-criterion evidence references.

**Impact**: Auditable, reproducible bid rankings — eliminates subjective single-score manipulation.

---

## 8. Contract Expiry & Renewal Alert Engine

**Problem**: No proactive alerting when contracts approach expiry; renewals are done ad hoc.

**Fix**: Add `async def scan_expiring_contracts(days_ahead: int = 30)` as a background task
(schedulable via APG's cron registry) that queries for contracts expiring within `days_ahead`,
emits `ContractExpiryWarning` events to the `ntfy` capability, and returns an actionable list
with recommended next steps.

**Impact**: Eliminates service-continuity gaps caused by unrenewed contracts — a common audit
finding in public entities.

---

## 9. Vendor Due-Diligence & Sanctions Screening

**Problem**: `vendor_registration` approves vendors immediately with no screening; debarment
check only validates internal debarment register, missing national/international sanctions lists.

**Fix**: Add `async def screen_vendor(vendor_id, vendor_name, registration_number)` that
cross-checks against the PPDA debarment database API, OFAC/UN sanctions (via local cache
refreshed nightly), and KRA PIN validation endpoint. Return a structured risk score and block
registration if any hard stop is triggered.

**Impact**: Prevents sanctioned or tax-non-compliant vendors from entering the supplier register.

---

## 10. E-Procurement Portal Integration (IFMIS/G2B)

**Problem**: `e_procurement_integration` records a sync status string but performs no actual
data exchange with government systems.

**Fix**: Add `async def sync_with_ifmis(sync_mode: str)` and `async def publish_to_g2b_portal(tender_id)`
with retry logic, idempotency tokens, and reconciliation reports. Map local data models to
IFMIS XML/REST schemas. Expose a webhook receiver endpoint for inbound award confirmations.

**Impact**: Closes the last-mile integration gap with national IFMIS; enables straight-through
processing without manual re-keying.

---

## 11. Procurement Plan vs. Actuals Variance Reporting

**Problem**: `procurement_plan` creates a plan but there is no mechanism to compare planned vs.
actual procurement activity.

**Fix**: Add `async def procurement_plan_variance(fiscal_year: str)` that joins the annual
procurement plan against actual tenders issued, awards made, and contract values, computing
variance percentages per line item. Flag items with >20% overshoot for review.

**Impact**: Enables the Accounting Officer to meet PPDA annual procurement plan reporting
obligations and identify budget discipline issues early.

---

## 12. Digital Contract Signing with e-Signature

**Problem**: `record_contract` stores `signed_by` as a string reference; no cryptographic
evidence of actual signing.

**Fix**: Add `async def sign_contract(contract_id, signatory_id, signature_payload: bytes)`
that validates the payload against a PKI certificate, records the signature hash, timestamp,
and certificate fingerprint in `con_signatures`. Reject any subsequent contract modification
without re-signature on amended sections.

**Impact**: Legally binding digital execution; reduces paper-based process delays from days to
minutes.

---

## 13. Anti-Corruption Pattern Intelligence Feed

**Problem**: No outbound feed of procurement anomalies to the `intel` capability for cross-entity
pattern detection.

**Fix**: Add `async def emit_intel_signals(period: str)` that computes single-source rate,
award concentration index (top-3 vendor share), and variation frequency, then pushes structured
`ProcurementIntelSignal` events to the `intel.alerts` capability. Include statistical Z-scores
for each metric relative to the entity's historical baseline.

**Impact**: Enables population-level anti-corruption analytics across all tenants without
exposing confidential bid data.

---

## 14. Contract Deliverable & Invoice Matching

**Problem**: `invoice_approve` approves invoices against a contract ID but does not verify
the invoice amount against approved milestone deliverables or payment schedule.

**Fix**: Add `async def match_invoice_to_deliverable(invoice_ref, contract_id, deliverable_id, amount)`
that enforces three-way matching (purchase order / delivery note / invoice), flags over-billing
above the payment schedule line by >5%, and blocks approval until a goods-receipt record exists.

**Impact**: Eliminates over-payment and fictitious invoice fraud — the most common contract
fraud vector in public procurement.

---

## 15. Procurement Risk Dashboard with Predictive Analytics

**Problem**: `procurement_analytics` returns static counts; no forward-looking risk signals
or exception reporting for management action.

**Fix**: Add `async def risk_dashboard(period: str)` that combines: (a) current metrics from
`procurement_analytics`, (b) AI risk scores from `ml_contract_risk_assess` across all active
contracts, (c) collusion detection signals, (d) expiry alerts, and (e) compliance check pass
rates — into a single structured response consumable by the Flask-AppBuilder dashboard blueprint.
Include a `risk_score` (0-100) and `top_5_actions` list.

**Impact**: Single pane of glass for the Procurement Director; shifts procurement oversight
from reactive audit to proactive risk management.

---

## Implementation Priority Matrix

| # | Improvement | Compliance Impact | Operational Impact | Complexity |
|---|-------------|------------------|--------------------|------------|
| 1 | Full Async Layer | Low | Critical | Medium |
| 2 | PostgreSQL Repository | Medium | Critical | High |
| 3 | Immutable Audit Trail | High | High | Medium |
| 4 | Bid Collusion Detection | High | High | High |
| 5 | PPDA Notification Gateway | Critical | High | Medium |
| 6 | COI Declaration Workflow | High | Medium | Low |
| 7 | Multi-Criteria Scoring | High | High | Medium |
| 8 | Expiry Alert Engine | Low | High | Low |
| 9 | Vendor Sanctions Screening | Critical | High | Medium |
| 10 | IFMIS/G2B Integration | High | Critical | High |
| 11 | Plan vs. Actuals Variance | High | Medium | Low |
| 12 | Digital Contract Signing | Medium | High | High |
| 13 | Intel Anti-Corruption Feed | High | Medium | Medium |
| 14 | Invoice Deliverable Matching | High | Critical | Medium |
| 15 | Risk Dashboard | Medium | High | Medium |

**Recommended first sprint**: Items 1, 3, 6, 7, 8, 11 — all medium-or-low complexity with
high compliance or operational payoff.
