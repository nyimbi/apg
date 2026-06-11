# SACCO Guarantor Management — World-Class Improvements

**Capability:** `fintech_sacco_gua`  
**Author:** Nyimbi Odero  
**Date:** 2026-06-11

---

### I1. Partial Guarantee Release on Loan Repayment Progress
**Category:** Business Logic  
**Justification:** When a borrower repays 40 % of the principal, the guarantor's frozen savings should proportionally unfreeze. Holding 100 % throughout the full loan term is punitive and discourages guarantorship in peer-financed SACCOs. Co-operatives Trust Fund Kenya mandates pro-rata release.  
**Implementation:** `partial_release_on_repayment(tenant_id, guarantee_id, repayment_pct)` — recomputes `frozen_amount = guaranteed_amount * (1 - repayment_pct / 100)`, posts a GL credit-reversal for the delta, and emits `guarantee_partially_released`.  
**Competitor Reference:** M-Shwari Okoa Jahazi (partial collateral release), MWALIMU SACCO pro-rata freeze policy.

---

### I2. Multi-Guarantor Pool with Proportional Liability Splitting
**Category:** Business Logic / Product  
**Justification:** Most loans require 2–4 guarantors. Today each guarantee is standalone; there is no concept of a pool. When a call is triggered, the system should apportion the call amount across pool members by their guarantee fraction, sending individual GL entries per guarantor.  
**Implementation:** `create_guarantee_pool(tenant_id, loan_id, guarantor_amounts)` — creates a `GuaranteePool` record, validates total coverage >= loan principal, and links child guarantees. `call_guarantee_pool(pool_id, total_called)` apportions proportionally.  
**Competitor Reference:** Kenya Women Finance Trust group-liability model; KWFT proportional call.

---

### I3. Guarantor Consent Deadline with Automated Expiry
**Category:** Workflow Automation  
**Justification:** Pending consent requests block loan disbursement indefinitely. Industry standard is a 48-to-72-hour window; after which the request auto-expires, allowing the loan officer to source an alternative.  
**Implementation:** `expire_stale_requests(tenant_id, max_age_hours)` — marks `pending` requests older than threshold as `expired`, emits `guarantee_request_expired`, and triggers replacement-search notification to loan officer.  
**Competitor Reference:** Equity Bank guarantor portal (48 h auto-expire); NCBA digital SACCO.

---

### I4. Blackout Period After Guarantee Call
**Category:** Risk / Compliance  
**Justification:** A guarantor whose savings were seized should be barred from guaranteeing new loans for a configurable cooling-off period (e.g. 12 months) to prevent compounding exposure losses.  
**Implementation:** `set_guarantor_blackout(tenant_id, member_id, months, reason)` stores a blackout record; `check_guarantor_eligibility` checks for active blackout before returning `eligible=True`.  
**Competitor Reference:** SACCO Societies Regulatory Authority (SASRA) Prudential Guidelines 2020, Rule 17(c).

---

### I5. Cross-SACCO Exposure Aggregation via Federated Query
**Category:** Risk Intelligence  
**Justification:** Members who belong to multiple SACCOs (common in Kenya) may have guarantees distributed across institutions. A federated exposure query collects cross-tenant obligations to compute a true risk exposure score.  
**Implementation:** `get_federated_exposure(member_id, registry_url)` calls an external registry API (async httpx), aggregates cross-tenant frozen amounts, and returns a `FederatedExposure` summary for the credit risk engine.  
**Competitor Reference:** Credit Reference Bureau Africa (CRB Africa) shared guarantor registry.

---

### I6. Savings-Lien Ledger with Per-Loan Traceability
**Category:** Accounting / Audit  
**Justification:** Current implementation tracks `frozen_amount` as a scalar per guarantee. A proper lien ledger records each freeze/thaw event as a ledger line, enabling point-in-time frozen balance queries for regulatory reporting (SASRA IFRS 9 provisioning).  
**Implementation:** `_post_lien_entry(tenant_id, member_id, guarantee_id, delta, direction)` appends to `_lien_ledger`; `get_lien_balance(tenant_id, member_id, as_of)` sums entries up to the timestamp.  
**Competitor Reference:** KCB Group savings-lien module; ABSA Collateral Manager ledger.

---

### I7. Risk-Tiered Early Warning Notification Cascade
**Category:** Risk Management  
**Justification:** A single notice at DPD 30 is too late. A three-stage cascade (DPD 7 — informational; DPD 21 — formal warning; DPD 45 — pre-call notice) gives guarantors time to resolve defaults before their savings are seized, reducing social friction within the SACCO.  
**Implementation:** `process_early_warning_cascade(tenant_id)` — iterates active guarantees, evaluates DPD bucket, sends the appropriate notice if not already sent at that stage, and records the stage in `notices_sent`.  
**Competitor Reference:** Stanbic SACCO DPD-tiered alert ladder; Co-op Bank guarantor portal (3-stage cascade).

---

### I8. Guarantee Fee Revenue Recognition
**Category:** Revenue / Accounting  
**Justification:** Some SACCOs charge a 0.5–1 % guarantee processing fee, shared between the SACCO and a guarantee insurance fund. The capability should compute, collect, and post this fee at acceptance time, boosting non-interest income reporting.  
**Implementation:** `compute_guarantee_fee(amount, fee_pct)` returns `(sacco_share, insurance_share)`; integrated into `accept_guarantee` which posts two additional GL lines: DR Borrower Fee Payable / CR Guarantee Revenue and CR Guarantee Insurance Fund.  
**Competitor Reference:** SASRA permitted charges schedule; ICEA Lion guarantee bond premium model.

---

### I9. Automated Guarantor Score (GScore)
**Category:** Analytics / Credit Risk  
**Justification:** Credit officers need a single trustworthiness number per prospective guarantor: history of honoring obligations, DPD on own loans, ratio of called to total guarantees, and years of membership. GScore enables instant auto-approval of top-tier guarantors.  
**Implementation:** `compute_guarantor_score(tenant_id, member_id)` weights four sub-scores (call rate, own-loan DPD, tenure, coverage headroom) into a 0–1000 integer, cached with a 24 h TTL, and returned alongside eligibility checks.  
**Competitor Reference:** FICO Guarantor Risk Score; CRB Africa G-Score product (2023).

---

### I10. Guarantee Insurance Integration
**Category:** Risk Transfer  
**Justification:** SACCOs can offload catastrophic default risk by purchasing guarantee insurance on pools above a threshold (e.g. > KES 200,000 aggregate). Integration with an insurance provider API creates a policy record and reduces the effective exposure from the SACCO's books.  
**Implementation:** `insure_guarantee(tenant_id, guarantee_id, insurer_ref, premium, coverage_pct)` creates an `InsurancePolicy` record, adjusts `effective_exposure` on the guarantee to `guaranteed_amount * (1 - coverage_pct)`, and routes eligibility checks through the adjusted figure.  
**Competitor Reference:** Kenya Reinsurance CRI product; ICEA Lion group credit shield.

---

### I11. Dispute and Objection Workflow
**Category:** Compliance / Member Rights  
**Justification:** SASRA member protection rules require that a guarantor who disputes a guarantee call must have a formal objection process with a defined resolution SLA (5 business days). Currently there is no such workflow.  
**Implementation:** `raise_guarantee_dispute(tenant_id, guarantee_id, guarantor_id, reason)` sets guarantee status to `disputed`, blocks further calls on that guarantee, and creates a `DisputeRecord` with `resolution_deadline`. `resolve_dispute(dispute_id, resolution, resolved_by)` re-activates or cancels the guarantee.  
**Competitor Reference:** SASRA Dispute Resolution Guidelines 2022; Co-operative Bank member grievance portal.

---

### I12. Dormancy Detection and Forced Substitution Trigger
**Category:** Operational Resilience  
**Justification:** If a guarantor account goes dormant (no transactions for > 6 months) while their savings are frozen, the SACCO risks an illiquid lien. Detecting this early allows proactive substitution before the loan matures.  
**Implementation:** `detect_dormant_guarantors(tenant_id, dormancy_days)` checks each active guarantor's last transaction date (from `_member_savings` metadata), flags accounts as `dormant_risk`, and triggers substitution requests automatically with a credit-manager notification.  
**Competitor Reference:** CBK Bank Supervision Annual Report 2023 — dormant account policy; SASRA operational risk checklist.

---

### I13. Bulk Guarantee Release Endpoint with Loan-Pool Support
**Category:** Operations Efficiency  
**Justification:** When a SACCO closes an entire batch of seasonal agricultural loans at harvest time, releasing guarantees one-by-one generates hundreds of API calls. A bulk operation reduces latency and GL posting overhead by an order of magnitude.  
**Implementation:** `bulk_release_guarantees(tenant_id, guarantee_ids, reason, released_by)` processes each guarantee in a gather coroutine group, batches GL entries into a single journal, and returns a `BulkReleaseResult` with per-item status and a summary count.  
**Competitor Reference:** Finserve Africa batch settlement engine; Temenos T24 batch guarantee closure.

---

### I14. Consent Re-confirmation on Material Loan Modification
**Category:** Compliance / Consent Management  
**Justification:** If a loan's principal is restructured upward (top-up), existing guarantors gave consent for the original amount only. Re-soliciting consent for the delta is a SASRA and Consumer Protection Act requirement. Currently the capability has no hook for loan modifications.  
**Implementation:** `handle_loan_modification(tenant_id, loan_id, new_amount, changed_by)` computes the delta, checks if it exceeds a materiality threshold (e.g. 10 % of original), voids and re-requests consent from all active guarantors for the incremental amount, and marks old guarantees `pending_reconfirmation`.  
**Competitor Reference:** SASRA Prudential Guideline No. 2, §9.4 — loan restructuring guarantor consent.

---

### I15. Guarantee Obligation Inheritance on Member Death/Incapacitation
**Category:** Legal / Estate Management  
**Justification:** When a guarantor dies, their estate inherits the frozen savings and the guarantee obligation. The capability must record the event, notify the estate administrator, and either transfer the guarantee to a named beneficiary or trigger substitution within a legal grace period.  
**Implementation:** `record_guarantor_incapacitation(tenant_id, member_id, event_type, estate_contact, grace_period_days)` marks all active guarantees for that member as `inherited`, freezes substitution deadline, notifies the SACCO legal team, and schedules a forced-substitution trigger after `grace_period_days`.  
**Competitor Reference:** Kenya Succession Act Cap 160; KWFT cooperative estate transfer protocol.

---

© 2026 Datacraft — Nyimbi Odero
