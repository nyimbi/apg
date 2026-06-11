# Rental Operations (realestate_ren) — World-Class Improvements

## Overview
Fifteen targeted improvements to elevate `realestate_ren` from functional to production-grade,
addressing correctness, observability, domain coverage, and composability gaps.

---

## 1. Persistent SQL Store via AsyncPG
**Current**: In-memory `dict` store — loses all state on restart.
**Improvement**: Swap `self._store` for a proper `AsyncPG` connection pool backed by the
`alembic/versions/0001_initial.py` schema already in the repo. Add a `DatabaseStore` abstraction
that mirrors the current dict API so the service layer stays unchanged.
**Impact**: Production-ready durability with zero service-layer refactor.

---

## 2. Rent-Increase Workflow
**Current**: Rent changes are applied immediately via `update_tenancy()` with no notice period,
no statutory notice requirements, and no tenant acknowledgement.
**Improvement**: Add `propose_rent_increase(tenancy_id, new_rent, effective_date, notice_days)`.
Enforce minimum statutory notice (e.g. 1 rental period for monthly tenancies). Generate a
`rent_increase_notice` event. Block application until `effective_date` passes.
**Impact**: Legal compliance, tenant-facing transparency.

---

## 3. Move-In / Move-Out Inspection Workflow
**Current**: `end_tenancy()` accepts freeform `deposit_deductions` dicts with no structured
evidence capture or condition grading.
**Improvement**: Add `record_inspection(tenancy_id, inspection_type, condition_items, photos)`
where `inspection_type ∈ {move_in, mid_term, move_out}`. Link move-out inspection to deposit
deduction approval flow. Require photo evidence IDs per deduction item.
**Impact**: Dispute-proof deposit accounting, defensible deduction records.

---

## 4. Automated Arrears Chasing Schedule
**Current**: Arrears escalation is purely manual. `arrears_management()` records but does not
schedule follow-up actions.
**Improvement**: Add `schedule_arrears_chase(arrears_id, chase_sequence)` where
`chase_sequence` is `[{days_after: 7, method: "email"}, {days_after: 14, method: "letter"}, ...]`.
Integrate with `schd` capability to fire `chase_sent` events.
**Impact**: Removes manual arrears admin; reduces time-to-collect.

---

## 5. Multi-Currency Rent Collection with FX Conversion
**Current**: `currency` field exists on models but FX conversion is ignored — all amounts
treated as nominal.
**Improvement**: Add `FXProvider` adapter. On payment receipt, convert to base currency
(default KES) using a pluggable rate source. Record both original and converted amounts.
Expose `currency_gain_loss` on the payment record.
**Impact**: Required for properties leased in USD/EUR to foreign nationals.

---

## 6. Rent Receipt Generation (PDF)
**Current**: Payments are recorded but no formal receipt is issued. `receipt_number` field
on `RentPaymentResponse` is never populated.
**Improvement**: Add `generate_rent_receipt(payment_id, tenant_id)` that serialises payment
details into a receipt dict (and optionally renders PDF via `reportlab`). Populate
`receipt_number` as a sequential formatted string `REC-{YYYY}-{NNN}`.
**Impact**: Legal requirement in most jurisdictions; reduces tenant queries.

---

## 7. Vacancy Tracking and Void Analysis
**Current**: Rent roll only covers active tenancies. There is no void period tracking.
**Improvement**: Add `record_void_period(unit_id, start_date, end_date, reason, tenant_id)`.
Surface void rate in `rental_analytics()` as `void_rate_pct` (void days / total days).
Add `get_void_report(tenant_id, period)`.
**Impact**: Void periods directly drive net yield calculations.

---

## 8. Statement of Account per Tenancy
**Current**: Payments and arrears are separate lists with no unified statement.
**Improvement**: Add `get_tenancy_statement(tenancy_id, tenant_id, from_date, to_date)`
that returns a chronological ledger of charges, payments, credits, and balance.
Include opening balance, closing balance, and days-in-arrears count.
**Impact**: Self-service portal requires this; reduces finance queries by ~60%.

---

## 9. Guarantor Management
**Current**: `tenant_application()` accepts a freeform `guarantor` dict with no validation
or lifecycle tracking.
**Improvement**: Add structured `GuarantorCreate/Response` models with `guarantee_limit`,
`guarantee_type ∈ {unlimited, capped}`, and expiry date. Add `activate_guarantor()`,
`call_on_guarantor()` methods. Link to arrears escalation flow.
**Impact**: Guarantors are legally binding instruments; freeform dicts are insufficient.

---

## 10. Lease Break Clause Tracking
**Current**: Break clauses are not modelled. `TenancyCreate` has no break clause fields.
**Improvement**: Add `BreakClause` sub-model (dates, conditions, notice_required_days,
exercised_by). Store on tenancy. Add `exercise_break_clause(tenancy_id, break_date)` that
validates conditions are met and transitions tenancy to `vacating`.
**Impact**: Residential and commercial tenancies routinely contain break clauses.

---

## 11. Concurrent Modification Guard (Optimistic Locking)
**Current**: `update_tenancy()` and `_clear_arrears()` mutate store entries by list index
with no concurrency protection.
**Improvement**: Add an `updated_at` version check. On update, compare `expected_version`
(datetime) to stored `updated_at`. If mismatch, raise `ConflictError`. This translates
directly to a `WHERE updated_at = $expected` predicate when the SQL store is added.
**Impact**: Prevents lost-update anomalies under concurrent rent collection workers.

---

## 12. Partial Payment Allocation (FIFO)
**Current**: Short payments create a single arrears record but do not allocate against
previous arrears periods.
**Improvement**: Add `allocate_payment(payment_id, tenant_id)` using FIFO across open
arrears periods. Each allocation reduces the oldest arrears first. Produce an
`AllocationResult` with per-period breakdown.
**Impact**: Correct accounting treatment required for aged-debt ledgers.

---

## 13. Regulatory Compliance Checklist Engine
**Current**: `compliance_audit()` is a stub that always returns `"compliant"`.
**Improvement**: Add a `ComplianceChecklist` model with items driven by tenancy type and
jurisdiction. Items: gas safety cert, EICR, EPC, fire safety, deposit protection.
`run_compliance_check(tenancy_id, tenant_id)` returns per-item pass/fail with
expiry dates and remediation links.
**Impact**: Property managers face personal liability for compliance failures.

---

## 14. Webhook / Event Bus Emission
**Current**: Events (`tenancy_created`, `rent_received`, etc.) are listed in `README.md`
but never emitted. The `mqeb` capability dependency is declared but unused.
**Improvement**: Add an `EventEmitter` adapter injected via `__init__`. Each mutating
method calls `await self._emit(event_type, payload)` post-mutation. Default adapter
is a no-op; production wires in the `mqeb` CloudEvents publisher.
**Impact**: Downstream capabilities (accounting, notifications, portal) cannot react
to rental state changes without events.

---

## 15. Rent Roll Versioned Snapshots
**Current**: `generate_rent_roll()` produces a point-in-time view with no history.
**Improvement**: Add `snapshot_rent_roll(tenant_id, snapshot_date, property_id)` that
stores the generated roll as a named snapshot in `_store["rent_roll_snapshots"]`.
Add `get_rent_roll_snapshot(snapshot_id, tenant_id)` and `compare_rent_rolls(id_a, id_b)`
returning added/removed/changed tenancies. Useful for month-end reconciliation and auditor
evidence packs.
**Impact**: Auditors and finance teams require point-in-time rent roll evidence.
