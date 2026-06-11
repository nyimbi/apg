# leg_bil — World-Class Improvement Roadmap

15 targeted improvements that move `leg_bil` from a competent billing ledger to a revenue-intelligence platform that outpaces Clio, MyCase, and PracticePanther on the features that actually drive realization.

---

### I1. AI-Powered Time Narrative Quality Scoring
**Category**: AI/ML
**Justification**: Attorneys lose 1–2 billable hours per day to vague narrative writing and write-downs. Auto-scoring narratives at entry time and suggesting rewrites captures 10–15% more realized revenue by blocking "vague" entries before they reach invoice review.
**Implementation**: On `create_time_entry`, score the description against a rubric (specificity, ABA code alignment, client-readability) using an Ollama-hosted Mistral 7B model; attach `narrative_score` (0–100) and `narrative_suggestions` list to the returned record.
**Competitive reference**: BrightFlag AI narrative scoring, Smokeball AI narrative assist, Clio Duo draft-from-calendar

---

### I2. LEDES 98B / LEDES XML 2.1 Electronic Invoice Export
**Category**: Compliance
**Justification**: Corporate clients and insurance carriers mandate LEDES-format invoices. Without it, invoices stall in client AP queues and firms are locked out of 60% of corporate legal spend governed by outside-counsel guidelines.
**Implementation**: `export_invoice_ledes98b(tenant_id, invoice_id)` serialises invoice lines into LEDES 98B tab-delimited format (UTBMS codes, line-item amounts, timekeeper classifications) returning `bytes` ready for e-billing portal upload; validates all UTBMS codes before export.
**Competitive reference**: LexisNexis CounselLink, Serengeti Tracker, Brightflag eBilling

---

### I3. Matter Budget vs. Actual Burn with Velocity Projection
**Category**: Performance
**Justification**: Clients demand budget predictability; firms that surface "you are at 75% of agreed budget" proactively reduce invoice-shock write-downs by up to 30% and win repeat mandates. Real-time burn tracking is the #1 requested client-facing analytics feature.
**Implementation**: `set_matter_budget(tenant_id, matter_id, total_budget, phase_budgets)` stores allocation; every `create_time_entry` call checks cumulative fees and emits `budget_alert` event at 80% and 100% thresholds; `matter_budget_status` returns burn%, projected completion cost, and days remaining.
**Competitive reference**: Clio matter budget alerts, TimeSolv budget enforcement, TyMetrix 360

---

### I4. Decimal-Precision Monetary Arithmetic
**Category**: Compliance
**Justification**: Using `float` for money causes silent rounding errors that compound across thousands of transactions — a trust accounting disaster and a KRA audit finding waiting to happen. All monetary fields must use `Decimal` with explicit `ROUND_HALF_UP` quantization.
**Implementation**: Replace all `float` monetary fields with `Decimal`; use `Decimal("0.01")` quantize for KES amounts; store serialized as `str` in the dict layer to avoid JSON precision loss; expose a `to_decimal` helper for external callers.
**Competitive reference**: Standard regulatory requirement (Law Society of Kenya accounts rules, SRA accounts rules)

---

### I5. Recurring / Subscription Fee Schedule Automation
**Category**: Feature
**Justification**: Fixed-fee retainers (monthly corporate advisory, compliance monitoring) are the fastest-growing billing model. Manual monthly invoice creation is error-prone and causes missed billing cycles worth 5–20% of annual retainer revenue.
**Implementation**: `create_invoice_schedule(tenant_id, matter_id, amount, frequency, start_date, end_date)` stores a schedule; `generate_due_invoices(tenant_id)` iterates schedules, creates draft invoices for all periods past `next_due_date`, advances the cursor, and returns a generation summary — callable from a daily cron.
**Competitive reference**: Bill4Time recurring billing, TimeSolv subscription invoicing, Rocket Matter

---

### I6. Write-Off Approval Workflow with Mandatory Reason Codes
**Category**: Compliance
**Justification**: Uncontrolled write-offs mask poor billing hygiene and create partnership equity disputes. A mandatory approval chain with reason codes (client_discount, courtesy, error, non_billable) enables Partner review and accurate profitability reporting.
**Implementation**: `request_write_off(tenant_id, entry_id, reason_code, notes)` creates a write-off request record in `"pending"` state; `approve_write_off(tenant_id, request_id, approved_by)` / `reject_write_off` control state transitions; direct write-offs without an approved request are blocked.
**Competitive reference**: Aderant Expert write-off approval, PracticeEvolve billing governance

---

### I7. Invoice Dispute Resolution Sub-Ledger
**Category**: Feature
**Justification**: Disputed invoices stall collections for months. A structured dispute workflow (raise → respond → negotiate → resolve) cuts average dispute resolution from 45 days to under 10, directly improving DSO. Dispute reason categorisation also feeds billing quality feedback.
**Implementation**: `raise_invoice_dispute(tenant_id, invoice_id, reason_code, details)` creates a `disputes` sub-record and transitions invoice to `"disputed"`; `resolve_dispute(tenant_id, dispute_id, resolution, credit_amount)` applies optional credit note and re-opens the invoice collection cycle.
**Competitive reference**: LexisNexis CounselLink eBilling dispute management, Apperio

---

### I8. Realization Rate Analytics (Worked → Billed → Collected Cascade)
**Category**: Performance
**Justification**: Most billing systems report collected vs. billed but miss the full realization cascade. Per-attorney realization rates reveal who under-records, over-writes-off, or gets discounted — enabling targeted coaching and repricing worth $30K–$150K per attorney annually.
**Implementation**: `realization_report(tenant_id, attorney_id, period_start, period_end)` computes worked_hours × rate, billed_amount, collected_amount, realization_pct, and collection_pct; `firm_realization_report` ranks all attorneys by realization with quartile banding.
**Competitive reference**: Clio Insights realization dashboard, Aderant Expert, Intapp Pricing

---

### I9. Automated Overdue Detection and AR Aging Buckets
**Category**: Feature
**Justification**: Manual overdue checking causes collections to lag 15–30 days. Automated aging (0–30, 31–60, 61–90, 90+ days) with automatic status escalation and configurable dunning triggers closes the gap between invoice send and cash receipt.
**Implementation**: `mark_overdue_invoices(tenant_id)` scans all `"sent"` invoices past `due_date` and flips status to `"overdue"`, emitting a `dunning_due` event per escalation tier (polite/firm/final); `aging_report(tenant_id)` groups outstanding amounts into standard AR aging buckets per client.
**Competitive reference**: Xero Practice Manager AR aging, QuickBooks Legal, Karbon billing

---

### I10. Multi-Currency Support with Exchange Rate Snapshots
**Category**: Feature
**Justification**: East African firms routinely bill USD, EUR, or GBP for international clients while maintaining KES trust accounts. Without exchange rate snapshots at invoice date, multi-currency reconciliation is impossible and creates KRA audit risk.
**Implementation**: `set_exchange_rate(tenant_id, from_currency, to_currency, rate, effective_date)` stores a dated rate snapshot; `create_invoice` auto-converts amounts when invoice currency differs from time-entry currency, recording `exchange_rate_used` and `exchange_rate_date` on each line.
**Competitive reference**: Xero Practice Manager multi-currency, LEAP Legal Software

---

### I11. Partial Payment and Payment Plan Instalment Tracking
**Category**: Feature
**Justification**: 35–40% of legal invoices are partially paid, especially in litigation matters. Without partial payment tracking, firms either close invoices prematurely or maintain manual spreadsheets, leading to revenue leakage and trust account reconciliation errors.
**Implementation**: `record_partial_payment(tenant_id, invoice_id, amount, reference)` reduces `outstanding_amount`, appends to a `payments` list on the invoice, transitions to `"partially_paid"`; `create_payment_plan(tenant_id, invoice_id, instalments)` stores scheduled instalment amounts and due dates with per-instalment status tracking.
**Competitive reference**: MyCase payment plans, Clio Payments instalment billing, LawPay

---

### I12. Bulk Time Entry Import from CSV with Duplicate Detection
**Category**: Integration
**Justification**: Attorneys using external timekeepers (Toggl, Harvest, Outlook calendar) need a frictionless import path. Manual re-entry is the #1 cause of month-end billing backlogs; bulk import with validation and duplicate detection cuts entry overhead by 70%.
**Implementation**: `bulk_import_time_entries(tenant_id, rows)` accepts a list of row dicts, validates each against UTBMS codes and rate cards, deduplicates by `(attorney_id, matter_id, date, hours, description)` hash, creates entries transactionally (all-or-none per batch), and returns an import summary with per-row error detail.
**Competitive reference**: Clio CSV import, TimeSolv batch import, Smokeball bulk entry

---

### I13. Real-Time Billing Timer (Start/Stop Time Capture)
**Category**: UX
**Justification**: Manual time entry after the fact recovers only 70% of actual billable time. Built-in timers running against open matters capture the remaining 30%, increasing annual revenue per attorney by $15K–$50K without requiring behavior change.
**Implementation**: `start_timer(tenant_id, matter_id, attorney_id, activity_code)` creates a `timer` record with `started_at`; `stop_timer(tenant_id, timer_id, description)` computes elapsed seconds, rounds to the nearest 6-minute increment (0.1 hr), and calls `create_time_entry` with the computed hours; supports concurrent timers per attorney.
**Competitive reference**: Clio Timer, MyCase time tracking, Bill4Time stopwatch, Toggl Legal

---

### I14. KRA eTIMS E-Invoice Compliance Submission
**Category**: Compliance
**Justification**: Kenya's eTIMS mandate (effective 2024) requires all VAT-registered businesses to submit e-invoices to KRA in real time. Non-compliance attracts a 5% penalty on transaction value and threatens operating licences — a critical blocker for any law firm billing tool in Kenya.
**Implementation**: `submit_invoice_etims(tenant_id, invoice_id)` serialises the invoice to the KRA eTIMS JSON schema (PIN, invoice_number, line items, VAT breakdown) and POSTs to the eTIMS endpoint; stores `etims_submission_id` and `etims_status` on the invoice record; retries on network failure with exponential backoff.
**Competitive reference**: QuickBooks Kenya eTIMS connector, Sage 300 KRA module, Tally eTIMS

---

### I15. Client Portal Read-Only Invoice Access via Scoped Tokens
**Category**: UX / Security
**Justification**: Emailing PDF invoices is an information-security risk and offers zero client engagement. Scoped read-only tokens let clients view invoice line items, download LEDES files, and raise disputes without granting system access — reducing inbound billing queries by 50%.
**Implementation**: `generate_client_portal_token(tenant_id, client_id, invoice_id, ttl_hours)` creates a short-lived HMAC-signed opaque token; `get_invoice_by_portal_token(token)` validates signature and TTL, then returns sanitised invoice data (no internal IDs, no tenant metadata) for unauthenticated rendering.
**Competitive reference**: Clio Client Portal, MyCase client access, PracticePanther portal, ActionStep
