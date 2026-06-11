# Real Estate Accounting — World-Class Improvements

**Capability**: `realestate_acc` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Waterfall CAM Allocation Engine

**Current gap**: CAM reconciliation records a single variance but cannot distribute that variance back to individual leases proportionally.

**Improvement**: Implement `allocate_cam_to_leases()` that reads each lease's NLA (net lettable area) from `realestate_lea`, computes each tenant's pro-rata share of actual vs. estimated CAM, raises credit/debit adjustments per lease, and emits `cam_tenant_adjustment_raised` events. Support multiple allocation bases: NLA, gross area, fixed percentage, and weighted occupancy days.

---

## 2. IFRS 16 Lease Modification Handling

**Current gap**: `generate_ifrs16_schedule()` creates a fixed amortisation table; lease modifications (rent reviews, term extensions, partial surrenders) require a full remeasurement of the lease liability using the revised discount rate at modification date.

**Improvement**: Add `remeasure_ifrs16_lease()` that accepts a modification date, revised payments, and revised discount rate, computes the new ROU/liability delta, raises a journal entry for the remeasurement gain/loss, and appends a new schedule segment — all in one atomic operation.

---

## 3. Rent-Free Period Amortisation

**Current gap**: Lease incentives such as rent-free periods are captured as metadata but not amortised to the P&L over the lease term using the straight-line method required by IFRS 16 / IFRS 15.

**Improvement**: Add `amortise_lease_incentive()` that calculates the monthly amortisation charge, links it to the IFRS 16 schedule, auto-generates the deferred incentive journal each period, and produces an incentive amortisation schedule report.

---

## 4. Percentage-Rent (Turnover Rent) Recognition

**Current gap**: `RevenueMethod.percentage_rent` exists as an enum value but the recognition engine always uses straight-line division regardless of the selected method.

**Improvement**: Implement `recognise_percentage_rent()` that ingests tenant turnover figures from `realestate_ren`, computes base rent plus the percentage-over-threshold component, performs the breakpoint calculation, and posts the variable component as a separate journal line. Supports natural and artificial breakpoints.

---

## 5. Multi-Currency Revaluation and FX Gains/Losses

**Current gap**: The service hard-codes KES as the default currency with no currency conversion logic. Cross-border portfolios denominated in USD/EUR produce no FX accounting.

**Improvement**: Add `record_fx_revaluation()` that fetches spot rates from a configurable exchange-rate provider, revalues all monetary balances in foreign-currency accounts, posts unrealised FX gain/loss journals to the correct P&L lines per IAS 21, and generates a currency exposure report per property.

---

## 6. Operating Expense Accruals with Auto-Reversal

**Current gap**: Accruals are created as manual journals; there is no mechanism to automatically schedule their reversal at the start of the next period, creating risk of double-counting.

**Improvement**: Add `accrue_operating_expense()` that creates the accrual journal, sets a `reversal_date` field, hooks into the `schd` capability to auto-create the reversing journal on the first day of the next period, and emits `accrual_created` / `accrual_reversed` events.

---

## 7. Sinking Fund (Reserve Fund) Management

**Current gap**: Major capital expenditure planning requires tenants to contribute to a sinking fund over time, but the capability has no reserve fund tracking or contribution schedule.

**Improvement**: Add `create_sinking_fund()` and `record_sinking_fund_contribution()` that maintain a separate reserve fund ledger per property, compute each tenant's contribution based on their NLA share, enforce minimum balance covenants, and produce a fund adequacy report showing projected capex against accumulated reserves.

---

## 8. Audit-Trail Immutable Ledger Integration

**Current gap**: `_log_operation()` logs to Python's logging system; there is no structured, tamper-evident audit event published to the `audl` capability stream.

**Improvement**: Integrate every write operation with a dedicated `AuditEvent` publisher that writes to an append-only store via `audl`. Each event captures: actor_id, operation, entity type/ID, before/after snapshot diff (JSON Patch), timestamp, and IP/session context. Queryable by entity, actor, and date range.

---

## 9. Budget vs. Actual Variance Reporting

**Current gap**: `service_charge_budget()` stores budget line items, but there is no comparison against actual charges or journals.

**Improvement**: Add `budget_variance_report()` that joins budget line items against posted journals and service charges for the same period, computes line-level and total variances (absolute and percentage), flags items exceeding a configurable tolerance threshold, and outputs a structured report consumable by `realestate_rep`.

---

## 10. Withholding Tax (WHT) Compliance Workflow

**Current gap**: `calculate_tax()` returns a tax amount but has no workflow for WHT certificates, remittance scheduling, or KRA reconciliation.

**Improvement**: Implement `generate_wht_certificate()`, `schedule_wht_remittance()`, and `reconcile_wht_with_revenue_authority()` covering the full KRA WHT lifecycle: deduction at source, certificate generation per vendor/tenant, monthly M-payment remittance, and period-end reconciliation against iTax returns.

---

## 11. Property Disposal and Derecognition

**Current gap**: `property_acquisition_cost()` records initial recognition; IAS 40 / IFRS 5 disposal accounting (derecognition, gain/loss on disposal, reclassification to "held for sale") is absent.

**Improvement**: Add `record_property_disposal()` that computes the disposal gain/loss as net proceeds minus carrying amount, removes the asset from the investment property register, closes all related IFRS 16 schedules and revenue schedules, and posts the composite disposal journal in one transaction.

---

## 12. Service Charge Dispute and Credit Note Workflow

**Current gap**: Once a service charge is posted there is no mechanism for a tenant to raise a dispute or for the landlord to issue a credit note without manually reversing journals.

**Improvement**: Add `raise_service_charge_dispute()`, `review_dispute()`, and `issue_credit_note()` implementing a three-state workflow (raised → under_review → resolved/rejected). Credit notes automatically generate matching reversal journals, update the tenant statement, and notify the `ntfy` capability.

---

## 13. Lease Incentive Liability (Lessor Perspective)

**Current gap**: The capability models IFRS 16 from the lessee's perspective. Lessors recognise lease incentive liabilities and deferred income, which is not tracked.

**Improvement**: Add `record_lessor_lease_incentive()` that creates a deferred income liability for rent-free periods granted, amortises the deferred income to P&L over the lease term via `schd`-scheduled journals, and discloses the balance in the IAS 17/IFRS 16 disclosure note template.

---

## 14. Cash Flow Statement Preparation

**Current gap**: The reporting suite covers trial balance and summary financials but does not produce a cash flow statement — a mandatory IAS 7 disclosure.

**Improvement**: Add `prepare_cash_flow_statement()` that classifies all posted journals into operating, investing, and financing activities using the account's `ledger_type` and a configurable classification map, computes net cash movement per section, reconciles to the closing bank balance, and flags unclassified accounts for review.

---

## 15. Automated Period-End Checklist and Close Gating

**Current gap**: `close_period()` enforces dual control but does not verify that all required period-end tasks are complete (all accruals posted, all CAM reconciliations approved, all service charges posted, WHT remitted).

**Improvement**: Add `get_period_close_checklist()` that returns a structured checklist of all period-end items with their completion status, and modify `close_period()` to gate on `checklist_complete: bool` derived from that check — preventing close if any blocking item remains outstanding, with a `force_close` override requiring a third approver.
