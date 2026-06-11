# World-Class Improvements — scm_ven (Vendor Management)

© 2025 Datacraft (www.datacraft.co.ke)

Fifteen prioritised improvements to elevate `scm_ven` from a capable in-memory
lifecycle service to a production-grade, event-driven vendor management platform.

---

## 1. Async-First Service Layer

**Current state**: All public methods are synchronous. Only `ml_vendor_risk_assess` is
`async`. I/O-bound operations (DB writes, event emission, external API calls) block the
event loop when the service is embedded in an async web framework.

**Improvement**: Convert every public method to `async def`. Use
`asyncio.gather` for fan-out operations (e.g. emit + index + notify in parallel).
Wrap the current synchronous in-memory store operations with `asyncio.to_thread` for
backward-compat during migration.

**Impact**: Eliminates event-loop blocking under load; enables composition with
async capabilities (`ntfy`, `audl`, `workflow`) without sync-to-async bridges.

---

## 2. PostgreSQL Persistence via SQLAlchemy Async

**Current state**: All state lives in in-memory dicts reset on process restart. No
persistence, no concurrent access safety, no horizontal scaling.

**Improvement**: Add a `VendorStore` abstraction with an async SQLAlchemy
(`asyncpg`) backend. Use the existing `alembic/` migration infrastructure already
present in the capability directory. The in-memory dict store stays as the default
for tests; production injects the Postgres store via the `adapters` config section.

**Impact**: Durable storage, concurrent reads, ACID guarantees on state transitions,
and horizontal scaling across multiple service instances.

---

## 3. Structured Event Emission via CloudEvents + Bytewax

**Current state**: `_emit` appends a plain dict to `self._audit_events`. Events
never leave the process. The capability contract declares a Bytewax stream
(`apg.scm.ven.lifecycle`) but nothing actually writes to it.

**Improvement**: Implement a `VendorEventPublisher` that wraps the APG
CloudEvents envelope and publishes to the declared Bytewax stream. Use
`situ_cloudevents` (already in the monorepo) as the envelope format. Provide a
`NullPublisher` for tests.

**Impact**: Enables downstream capabilities (`scm_prc`, `scm_srm`, `ntfy`) to
react to vendor lifecycle events in near-real-time without polling.

---

## 4. Vendor Health Score — Composite KPI

**Current state**: Performance tier (`gold`/`silver`/`bronze`/`at_risk`) is computed
from the average of explicitly recorded performance metrics only. Risk score, compliance
status, and suspension history are not folded in.

**Improvement**: Add `vendor_health_score(vendor_id)` that computes a single
0–100 composite from: performance average (40 %), compliance score (25 %),
risk-adjusted factor (25 %), and relationship engagement (10 %). Persist the
result back onto the vendor record for fast dashboard queries.

**Impact**: Single number for executive dashboards; consistent signal for
`approved_vendor_list` sorting and procurement-routing decisions.

---

## 5. Contract Lifecycle Alerts (Expiry + Auto-Renew)

**Current state**: Contracts are stored but there is no mechanism to surface
upcoming expirations or trigger auto-renewal workflows.

**Improvement**: Add `contract_expiry_alerts(days_ahead, tenant_id)` that
returns all contracts expiring within the window and emits
`vendor_contract_expiry_approaching` events. For `auto_renew=True` contracts,
emit `vendor_contract_auto_renew_triggered` and create a renewal record
automatically.

**Impact**: Eliminates manual contract-expiry tracking; prevents unintended
lapsing of active supplier relationships.

---

## 6. Spend Concentration Risk Detection

**Current state**: `spend_analysis` returns per-vendor and per-category totals
but does not flag concentration risk (single vendor > N % of category or total spend).

**Improvement**: Extend `spend_analysis` with a `concentration_risks` field.
Flag any vendor whose share of category spend exceeds 40 % or total spend exceeds
20 %. Emit `vendor_spend_concentration_risk_detected` events for flagged vendors.

**Impact**: Gives procurement teams and supply chain risk managers an automated
signal for single-source dependency without requiring a separate BI tool.

---

## 7. Bulk Onboarding via Async Batch API

**Current state**: `onboard_vendor` handles one vendor at a time. Importing
50+ vendors from an ERP migration requires 50 sequential calls.

**Improvement**: Add `bulk_onboard_vendors(vendors: list[dict], tenant_id)` that
fans out to `onboard_vendor` concurrently using `asyncio.gather`, collects
`(success, error)` per row, and returns a structured batch result with success
count, failure count, and per-row diagnostics.

**Impact**: Reduces ERP migration time from O(n) sequential to O(1) bounded by
concurrency limit; preserves per-row error visibility.

---

## 8. Compliance Expiry Monitoring

**Current state**: Compliance records are stored with a framework and status but
there is no mechanism to detect records that have passed their review date or
expired certifications.

**Improvement**: Add `compliance_expiry_scan(tenant_id, as_of_date)` that
scans all compliance records, flags `expired` and `expiring_soon` (within 30 days),
emits `vendor_compliance_expiry_detected` events, and returns a structured report.

**Impact**: Eliminates manual certification-expiry tracking; feeds directly into
vendor risk score calculation (Improvement 4).

---

## 9. Vendor Diversity and ESG Tracking

**Current state**: No first-class support for supplier diversity classifications
(MSME, women-owned, minority-owned) or ESG metrics.

**Improvement**: Add `vendor_diversity_profile(vendor_id, diversity_categories,
esg_scores, tenant_id)` that attaches a diversity profile to a vendor and
`diversity_spend_report(period, tenant_id)` that breaks down spend by diversity
category. Add ESG dimensions (`environmental`, `social`, `governance`) to the
performance score model.

**Impact**: Supports mandatory supplier diversity reporting for public-sector and
ESG-mandated procurement programmes.

---

## 10. Vendor Segmentation Engine

**Current state**: Vendors are classified only by `category` and `vendor_type`.
Strategic importance and Kraljic matrix positioning are mentioned in models but
not computed.

**Improvement**: Add `segment_vendors(tenant_id)` that applies a configurable
Kraljic-style segmentation (spend impact × supply risk) to all active vendors,
assigns them to `strategic`, `leverage`, `bottleneck`, or `non_critical` segments,
and persists the segment assignment on the vendor record.

**Impact**: Gives category managers an automated starting point for relationship
strategy without manual spreadsheet analysis.

---

## 11. SLA Breach Detection and Escalation

**Current state**: `contract_management` stores SLA terms as a freeform dict but
there is no logic to detect SLA breaches from performance data.

**Improvement**: Add `sla_breach_scan(vendor_id, tenant_id)` that cross-references
SLA terms in the vendor's active contracts with recorded performance scores. Flag
any dimension where the SLA threshold is breached, emit
`vendor_sla_breach_detected`, and auto-create a high-tier risk record.

**Impact**: Closes the gap between contractual commitments and performance data;
enables automated escalation without manual cross-referencing.

---

## 12. Vendor Reinstatement Workflow

**Current state**: Suspension is a one-way operation. There is no `reinstate_vendor`
method, leaving suspended vendors permanently stuck unless the underlying dict is
mutated directly.

**Improvement**: Add `vendor_reinstatement(vendor_id, rationale, approved_by,
tenant_id)` that validates the suspension record, sets `stage` back to `active`,
archives the suspension record with a resolved timestamp, and emits
`vendor_reinstated`.

**Impact**: Completes the suspension lifecycle; prevents indefinite vendor lockout
from a single suspension event.

---

## 13. Multi-Vendor Comparison Report

**Current state**: No method compares two or more vendors head-to-head on
performance, risk, spend, and compliance dimensions.

**Improvement**: Add `compare_vendors(vendor_ids: list[str], tenant_id)` that
returns a structured comparison matrix: per-vendor scores on all performance
dimensions, risk tier, compliance status, spend total, and preferred/suspended flags.
Identify the "recommended" vendor per dimension.

**Impact**: Supports data-driven vendor selection during sourcing events and
contract renewals without exporting data to spreadsheets.

---

## 14. Immutable Audit Log with Tamper Evidence

**Current state**: Audit events are stored in a mutable list. Any code with access
to `self._audit_events` can delete or modify records, violating the immutability
requirement stated in the capability contract.

**Improvement**: Replace `self._audit_events: list` with an append-only
`AuditLog` class that hashes each entry with a rolling HMAC chain (SHA-256,
keyed with a tenant secret). Expose `audit_verify(tenant_id)` that re-computes
the chain and returns `verified: True/False`. In production the backing store is
a Postgres `audit_log` table with `INSERT`-only grants.

**Impact**: Audit trail becomes legally defensible; chain verification detects any
post-hoc tampering at rest or in transit.

---

## 15. AI-Powered Early Warning Digest

**Current state**: `ml_vendor_risk_assess` classifies a single vendor on demand.
There is no portfolio-level early warning that proactively surfaces at-risk vendors
before they breach thresholds.

**Improvement**: Add `ai_early_warning_digest(tenant_id)` that runs
`ml_vendor_risk_assess` concurrently across all active vendors using
`asyncio.gather`, filters those with `ml_risk_tier in {"high_risk", "critical_risk"}`,
enriches each result with contract expiry and compliance expiry data, and returns a
ranked digest sorted by composite risk. Falls back to rule-based risk tiers when
Ollama is unavailable.

**Impact**: Procurement teams receive a single daily digest that surfaces the
highest-priority vendor interventions without running individual reports.

---

*Generated by Claude Code for APG scm_ven capability.*
