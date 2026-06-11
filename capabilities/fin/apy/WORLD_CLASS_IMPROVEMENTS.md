# Accounts Payable — World-Class Improvement Proposals

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>

---

## 1. Intelligent OCR Invoice Ingestion with Line-Item Extraction

**Category:** Automation / AI

**Justification:** Over 70% of AP processing time is consumed by manual data entry. OCR + LLM extraction of vendor name, invoice number, line items, taxes, and totals from PDF/image invoices eliminates this bottleneck and reduces keying errors to near-zero. Every touchless invoice captured directly improves DPO and AP team capacity.

**Implementation:** Integrate an Ollama-hosted vision model (e.g. LLaVA, MiniCPM-V) via `POST /api/generate`. Extract structured JSON with invoice header and line-item fields. Confidence scores gate auto-post (>0.92) vs. human review queue (<0.92). Store raw document hash for duplicate fingerprinting. Expose as `async ingest_invoice_ocr(document_bytes, mime_type, tenant_id)` returning an `APInvoiceCreate`-compatible dict with `ocr_confidence`.

**Competitor Benchmark:** SAP Ariba, Coupa, and Tipalti ship OCR ingestion as table-stakes. Coupa Intelligence achieves >95% touchless processing rates.

---

## 2. ML-Powered Duplicate Invoice Detection

**Category:** Fraud Prevention / AI

**Justification:** Duplicate payments cost organisations 0.1–0.5% of total AP spend annually. Fuzzy matching on invoice number alone misses near-duplicates (slight amount, date, or vendor name variations used to evade rule-based filters). Vector similarity catches these structural duplicates that exact-match rules miss.

**Implementation:** Embed invoice features (vendor, amount, date, invoice_number hash, document hash) via a local embedding model (e.g. `nomic-embed-text`) through Ollama. On submission, compute cosine similarity against the last 90 days of invoices in `pgvector`. Flag pairs above a configurable threshold (default 0.92) for human review. Expose as `async detect_duplicate_invoice(invoice_id, tenant_id)` returning a `DuplicateCheckResult`.

**Competitor Benchmark:** Medius AI, Basware, and JAGGAER ship anomaly-detection engines that catch >99% of duplicates including cross-vendor collusion patterns.

---

## 3. Predictive Cash Flow Forecasting

**Category:** Treasury / Analytics

**Justification:** AP is the primary driver of short-term cash outflows. Accurate 13-week rolling forecasts reduce idle cash (opportunity cost) and prevent overdraft charges. Integrating invoice due dates, payment terms, and historical payment velocity produces forecasts far more accurate than spreadsheet extrapolation.

**Implementation:** Build a weekly bucket model: for each approved, unpaid invoice, place `outstanding_amount` into the week bucket for `due_date`. Layer in historical payment-day distribution (paid N days before/after due) to produce probabilistic bands (P10/P50/P90). Expose as `async forecast_cash_outflows(horizon_weeks, tenant_id)` returning weekly buckets with confidence intervals and a total projected outflow.

**Competitor Benchmark:** Kyriba, HighRadius Treasury, and Cashforce produce ML-driven 13-week AP cash forecasts with actuals-vs-forecast variance tracking and ERP integration.

---

## 4. Automated VAT and Withholding Tax Compliance

**Category:** Compliance / Tax

**Justification:** Kenyan VAT at 16%, withholding tax at 3–6% on services, and reverse-charge rules create significant compliance risk. Manual tax coding is error-prone and causes costly KRA penalties. Automated tax determination at invoice ingestion eliminates this risk and generates audit-ready schedules.

**Implementation:** Maintain a `TaxRule` table keyed by `(supplier_country, line_item_category, tax_type)`. On invoice capture, resolve applicable rates and compute gross/net/tax splits. Generate iTax-compliant VAT schedules for monthly filing. Expose `async compute_invoice_tax(invoice_id, tax_profile)` returning tax breakdown and `async generate_vat_schedule(period, tenant_id)` returning a filing-ready schedule.

**Competitor Benchmark:** Oracle Fusion Tax, SAP Tax Compliance, and Avalara automatically apply jurisdiction-specific tax rules with audit trails for over 200 countries.

---

## 5. Vendor Risk Scoring and Sanctions Screening

**Category:** Risk Management / Compliance

**Justification:** Paying a sanctioned vendor (OFAC, UN, EU lists) carries criminal liability. Financial distress or poor delivery performance by a vendor creates supply-chain and credit risk. Proactive scoring prevents onboarding high-risk counterparties before liabilities materialise.

**Implementation:** Build a `VendorRiskProfile` with fields for payment history score, delivery score, and `sanctions_check_status`. On vendor registration, invoke an Ollama-powered entity resolution check against a locally cached sanctions list. Score 0–100 using weighted factors: late deliveries, disputed invoices, price variance, PEP status. Block payments to `sanctions_status=match` without CISO override. Expose as `async score_vendor_risk(vendor_id, tenant_id)`.

**Competitor Benchmark:** Coupa Risk Assess, SAP Ariba Supplier Risk, and Dun & Bradstreet Supplier Evaluation integrate live sanctions and ESG screening at onboarding and payment time.

---

## 6. Contract-Driven PO Generation and Blanket Order Management

**Category:** Procurement / Automation

**Justification:** Maverick spend (purchases outside negotiated contracts) inflates costs by 10–25%. Linking invoice matching to contract terms (price lists, approved quantities, validity periods) catches off-contract invoices before payment and provides procurement with real leverage at renewal.

**Implementation:** Add a `_contracts` store keyed by `contract_id`. A contract carries `vendor_id`, `line_item_prices`, `max_quantity`, `validity_start/end`, and `blanket_po_amount`. `three_way_match` queries active contracts to validate invoice unit prices against contracted rates. Expose `async register_contract(contract_data, tenant_id)` and `async contract_compliance_report(period, tenant_id)` methods.

**Competitor Benchmark:** SAP SRM, Ivalua, and Jaggaer enforce contract compliance at PO creation and invoice matching with real-time price deviation alerts and spend-against-contract dashboards.

---

## 7. Straight-Through Processing (STP) Orchestration

**Category:** Automation / Workflow

**Justification:** Invoices that pass all automated checks (OCR extraction, duplicate detection, three-way match, tax validation, vendor risk) should progress to payment without human touch. Gating only edge cases at the human queue maximises throughput and compresses cycle time from days to hours.

**Implementation:** Implement `async straight_through_process(invoice_id, tenant_id)` that runs the full checklist pipeline: capture validation → duplicate check → match → tax compute → approve → schedule → run. Each step returns a `StepResult(passed, reason)`. Only when all steps pass does the invoice auto-approve with a system principal (`"auto_stp"`). Failed steps route to the appropriate exception queue. Emit `stp_completed` or `stp_escalated` event.

**Competitor Benchmark:** Tipalti and Stampli advertise 80%+ touchless invoice processing via STP pipelines. Basware claims 96% automation rates for approved supplier networks.

---

## 8. Vendor Self-Service Portal with PEPPOL / UBL Electronic Invoice Submission

**Category:** Supplier Experience / Interoperability

**Justification:** Emailing PDF invoices creates manual rework and delays. PEPPOL/UBL electronic invoice standards allow suppliers to submit structured XML invoices that integrate directly into the AP system without re-keying. Supplier portals reduce invoice-status query calls by over 60% and cut invoice processing time by 40%.

**Implementation:** Add a PEPPOL UBL 2.1 parser that converts `Invoice` XML elements to the internal invoice dict. Extend `supplier_invoice_submission` to accept `format: "UBL" | "PDF" | "JSON"`. The UBL parser extracts header, line items, tax totals, and payment means. Expose a `/supplier-portal` Flask-AppBuilder blueprint with invoice submission, status tracking, and dynamic discounting acceptance UI.

**Competitor Benchmark:** Tradeshift, Basware, and Tungsten Network are built around PEPPOL e-invoicing. The EU mandates B2G PEPPOL compliance in all member states.

---

## 9. Accruals Engine for Period-End Automation

**Category:** Accounting / Close

**Justification:** Unaccrued AP liabilities cause material misstatements at period end. Finance teams spend days manually identifying received-not-invoiced (RNI) items. An automated accruals engine running at period close eliminates this risk, accelerates close, and produces IFRS/IAS 37-compliant accrual journals.

**Implementation:** Add `async compute_accruals(period, tenant_id)` that scans GRNs with no matching approved invoice, and approved POs past delivery date with no GRN, to generate accrual journal entries. Output `[{account, debit, credit, description, period}]`. Persist accruals in `_accruals` store. On actual invoice arrival, reverse the accrual and post the actuals. Integrate with `post_payment_run_to_gl`.

**Competitor Benchmark:** Oracle AP Accruals, SAP FI Accrual Engine, and Workday Accruals automate RNI detection and auto-reverse on invoice receipt.

---

## 10. Configurable Multi-Level Approval Workflow with Delegation

**Category:** Governance / Workflow

**Justification:** Static two-person approval does not scale. Organisations need tiered approval matrices: invoices under 50k auto-approve, 50k–500k require line manager, over 500k require CFO. Delegation handles approver absence without creating bottlenecks that delay payment and damage vendor relationships.

**Implementation:** Add `_approval_matrix: list[ApprovalTier]` where each tier has `min_amount`, `max_amount`, `required_roles: list[str]`, and `quorum`. `approve_invoice` resolves the applicable tier and validates that all required roles have signed off. `async delegate_approval(from_user, to_user, valid_until, tenant_id)` records a delegation chain. Expired or missing delegation raises `ApprovalWorkflowError`. Expose `async configure_approval_matrix(tiers, tenant_id)`.

**Competitor Benchmark:** Coupa Business Spend Management, SAP Ariba Buying, and Medius all ship no-code approval matrix builders with mobile push notifications and escalation timers.

---

## 11. Real-Time Payment Fraud Detection

**Category:** Security / AI

**Justification:** Business Email Compromise (BEC) fraud targeting AP payments cost $1.8B in 2023 (FBI IC3). Detecting anomalous payment instructions — new bank account for known vendor, unusual amount, unusual timing — before funds leave the organisation prevents unrecoverable losses that no insurance fully covers.

**Implementation:** Build `async score_payment_fraud_risk(payment_id, tenant_id)` that evaluates: (a) is the bank account new for this vendor within the last 90 days? (b) does the amount deviate >3σ from the vendor's historical payment distribution? (c) is payment scheduled for a weekend or public holiday? (d) was the bank account changed within 72h of this payment? Score 0–100; block at >75, flag for review at >40. Emit `payment_fraud_risk_flagged` event.

**Competitor Benchmark:** Bottomline Technologies, TrustPair, and Coupa Pay integrate real-time beneficiary validation and anomaly scoring against payment fraud databases.

---

## 12. Touchless PO Flip for Recurring Service Orders

**Category:** Procurement / Automation

**Justification:** For recurring service vendors (facilities management, IT subscriptions, professional services), manually creating invoices against blanket POs is repetitive, error-prone, and slow. PO flip generates draft invoices automatically on schedule, requiring only supplier confirmation to enter the approval flow.

**Implementation:** Add `async flip_po_to_invoice(po_id, billing_period, confirmed_by_supplier, tenant_id)` that creates a draft invoice with line items, amounts, and payment terms copied from the PO. The invoice status is `draft_pending_supplier_confirm`. On supplier portal confirmation, it transitions to `captured` and enters the standard approval flow. Supports proration for partial-period billing via `billing_period_days` parameter.

**Competitor Benchmark:** SAP Ariba PO Flip, Coupa PO Flip, and Ivalua all support supplier-confirmed PO flips as a standard supplier network feature with straight-through routing.

---

## 13. Integrated AP-to-GL Reconciliation with Trial Balance Tie-Out

**Category:** Accounting / Close

**Justification:** Reconciling the AP subledger to the GL control account is a monthly close requirement under IFRS and local GAAP. Discrepancies indicate posting errors, timing differences, or fraud. Manual reconciliation is error-prone and slow; automation compresses close from days to minutes and provides a documented audit trail.

**Implementation:** Add `async reconcile_ap_to_gl(period, tenant_id)` that computes the AP subledger balance (sum of outstanding approved invoices) and compares it against the GL AP control account balance (from `_gl_postings`). Produces a variance report with drill-down to individual journals. Flags unreconciled items for investigation. Integrates with `post_payment_run_to_gl` to maintain a complete GL balance.

**Competitor Benchmark:** BlackLine AP Reconciliation, Trintech Cadency, and FloQast automate subledger-to-GL tie-outs with workflow-managed sign-off and continuous accounting.

---

## 14. Vendor Performance Scorecard with SLA Tracking

**Category:** Procurement / Analytics

**Justification:** AP sits on rich data about vendor reliability: invoice accuracy rates, dispute frequency, response time to RFIs, and credit note issuance. Surfacing this as a performance scorecard enables procurement to renegotiate terms with underperformers, reward reliable suppliers with faster payment, and build a data-driven supplier development programme.

**Implementation:** Add `async compute_vendor_scorecard(vendor_id, period, tenant_id)` that calculates: invoice accuracy rate (% invoices with no exceptions), average match-pass rate, dispute resolution time in days, on-time submission rate, and credit note rate. Persist in `_vendor_scorecards`. Expose rankings in `spend_analytics` and the supplier portal. Weight scores and compute a composite `performance_index` 0–100.

**Competitor Benchmark:** SAP Ariba Supplier Performance, Coupa Supplier Management, and GEP SMART all include quantitative vendor scorecards with drill-down to transaction level and automated SLA alerts.

---

## 15. Embedded AP Natural Language Query Interface

**Category:** User Experience / AI

**Justification:** Finance staff spend significant time navigating menus to query AP status: "What is outstanding for Vendor X?", "Show invoices due this week", "Which invoices are blocked?". An Ollama-backed natural language interface eliminates navigation overhead, surfaces answers in seconds, and is accessible to non-technical stakeholders who would never use raw API calls.

**Implementation:** Add `async nl_query(question: str, tenant_id: str)` that routes the question through a local Ollama LLM (e.g. `llama3.1:8b`) with a system prompt describing available AP data structures and a tool-call interface. The LLM selects and calls one of: `aging_summary`, `ap_kpi_dashboard`, `supplier_payment_status`, `match_exception_queue`, `spend_analytics`. Returns a structured response with both the natural-language answer and the underlying data. Guard with tenant-scoped data access enforcement.

**Competitor Benchmark:** Tipalti Synapse, Stampli Billy, and HighRadius Freeda are dedicated AP AI assistants handling invoice queries, payment status, and exception resolution via conversational chat with ERP context.

---
