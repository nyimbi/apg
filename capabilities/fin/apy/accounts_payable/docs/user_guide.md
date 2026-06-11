# Accounts Payable User Guide

## Daily Workflow

1. Register or confirm the vendor with owner, tax profile, and payment method evidence.
2. Capture the invoice with vendor, invoice number, amount, currency, and document reference.
3. Match PO-backed invoices to receipt evidence.
4. Route variance, duplicate, bank-change, and policy-exception cases for review.
5. Approve invoices with separation of duties.
6. Schedule payments from an approved cash account.
7. Release payment batches after treasury or AP review.
8. Review AP aging and close the period only after exceptions and unposted invoices are cleared.

## Screens

- Dashboard: tenant summary, lifecycle counts, and stream metadata.
- Vendors: vendor onboarding and bank-change review.
- Invoices: invoice capture, duplicate handling, and status review.
- Matching: PO, receipt, and variance review.
- Approvals: approval queue and hold placement.
- Payments: payment scheduling and batch release.
- Expenses: employee reimbursement capture and policy review.
- Aging: open payable exposure.
- Close: period close blockers and close evidence.
- Agents: AP-agent registration and privileged-action review.

## AP Agents

AP agents can assist with review and preparation work. Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Privileged AP agents still require human approval before execution.

---

## New AI and Analytics Features

### Straight-Through Processing (STP)

Invoices that pass all automated checks can be processed without human intervention.

```python
result = await service.straight_through_process(invoice_id, tenant_id)
if result["passed"]:
    print("Invoice auto-approved and ready for payment scheduling")
else:
    # Check which step failed and route to the appropriate queue
    failed_step = next(s for s in result["steps"] if not s["passed"])
    print(f"Escalated at step: {failed_step['step']} — {failed_step['detail']}")
```

STP pipeline steps:
1. Invoice validation — confirms the invoice exists and is in a processable state
2. Duplicate check — exact-match rule-based check against vendor + invoice number + amount
3. PO match — two-way or three-way match depending on whether a GRN is linked
4. System approval — auto-approved with the `auto_stp` principal if all prior steps pass

Failed steps route to the match exception queue or hold the invoice for human review. A `stp_completed` or `stp_escalated` audit event is always emitted.

### ML Duplicate Invoice Detection

Catch near-duplicate invoices that evade exact-match rules.

```python
result = await service.ml_duplicate_invoice_detect(invoice_id, tenant_id)
if result["is_duplicate"]:
    print(f"Possible duplicate of: {result['fuzzy_match_ids']}")
    print(f"Confidence: {result['confidence']}")
```

When `OLLAMA_BASE_URL` is set, invoices are embedded using `nomic-embed-text` and compared via cosine similarity against the last 90 days of invoices. Without Ollama, an exact-match fallback (vendor + invoice number + amount) is used. The `ml_enhanced` field indicates which path ran.

### Cash Flow Forecasting

Project AP cash outflows up to 13 weeks ahead.

```python
forecast = await service.forecast_cash_outflows(tenant_id, horizon_weeks=8)
for bucket in forecast["weekly_buckets"]:
    print(f"Week {bucket['week']} ({bucket['week_start']}): "
          f"P50={bucket['amount_p50']}  "
          f"P10={bucket['amount_p10']}  P90={bucket['amount_p90']}")
```

Outflows are bucketed by invoice due date. P10/P50/P90 confidence bands are derived from the historical spread between scheduled and actual payment dates. Use the forecast to inform weekly treasury positioning and cash reserve decisions.

### VAT and Tax Compliance

Compute invoice tax and generate iTax VAT schedules.

```python
# Compute tax for a single invoice
tax = await service.compute_invoice_tax(invoice_id, tenant_id, tax_profile="standard")
print(f"VAT: {tax['vat_amount']}  WHT: {tax['wht_amount']}  Net payable: {tax['net_payable']}")

# Generate monthly VAT schedule for KRA iTax filing
schedule = await service.generate_vat_schedule(tenant_id, {"start": "2026-05-01", "end": "2026-05-31"})
print(f"Total input VAT: {schedule['total_input_vat']}")
print(f"Total WHT withheld: {schedule['total_wht_withheld']}")
```

Supported tax profiles: `standard` (VAT 16% + WHT 5%), `exempt`, `vat_only`, `wht_only`, `zero_rated`.  Tax metadata is persisted on the invoice for downstream payment scheduling and GL posting.

### Vendor Risk Scoring

Assess vendor risk before approving large payments or onboarding new suppliers.

```python
risk = await service.score_vendor_risk(vendor_id, tenant_id)
print(f"Risk score: {risk['risk_score']}/100 ({risk['risk_tier']})")
print(f"Recommendation: {risk['recommendation']}")
```

Scoring factors: invoice exception rate (25 pts), payment dispute rate (25 pts), average price variance (20 pts), bank account change (15 pts), on-time submission (15 pts). Tiers: `low` (0–25), `moderate` (26–50), `high` (51–75), `critical` (76–100).

### Payment Fraud Detection

Score payments for fraud indicators before release.

```python
fraud = await service.score_payment_fraud_risk(payment_id, tenant_id)
if fraud["risk_tier"] == "high":
    print(f"BLOCKED: {fraud['recommendation']}")
    for f in fraud["fired_factors"]:
        print(f"  - {f['factor']}: {f['detail']}")
```

Evaluated factors: new bank account for vendor, amount > 2x vendor median, weekend payment scheduling, bank account changed within 72h of payment scheduling. Scores above 75 block the payment; scores 40–74 flag for second-pair-of-eyes review.

### Vendor Performance Scorecard

Track vendor reliability across accuracy, matching, and timeliness dimensions.

```python
scorecard = await service.compute_vendor_scorecard(
    vendor_id, tenant_id,
    period={"start": "2026-01-01", "end": "2026-06-30"}
)
print(f"Performance index: {scorecard['performance_index']}/100 ({scorecard['performance_tier']})")
print(f"Invoice accuracy: {scorecard['metrics']['invoice_accuracy_rate']}%")
print(f"On-time submission: {scorecard['metrics']['on_time_submission_rate']}%")
```

The composite performance index (0–100) weights: invoice accuracy 30%, match pass rate 25%, on-time submission 20%, dispute score 15%, credit note score 10%.  Tiers: `preferred` (85+), `good` (70–84), `fair` (50–69), `poor` (<50).

### Accruals Engine

Generate period-end accrual journals automatically.

```python
accruals = await service.compute_accruals(tenant_id, {"start": "2026-05-01", "end": "2026-05-31"})
print(f"Accrual entries generated: {accruals['entry_count']}")
print(f"Total accrual amount: {accruals['total_accrual_amount']}")
for entry in accruals["journal_entries"]:
    print(f"  {entry['accrual_type']}: {entry['amount']} — {entry['description']}")
```

Two accrual types are detected: received-not-invoiced (RNI) from GRNs with no matching invoice, and service accruals from POs past delivery date with no GRN. All entries include auto-reversal dates for the first day of the following period.

### Natural Language AP Query

Ask AP questions in plain English.

```python
answer = await service.nl_query("Which vendors have overdue invoices?", tenant_id)
print(answer["answer"])
# Underlying data is also available for programmatic use
print(answer["data"])
```

When `OLLAMA_BASE_URL` is set, questions are routed through a local LLM (default `llama3.1:8b`) that selects and calls the most relevant AP data method. A keyword-based router handles the fallback when Ollama is unavailable. Set `OLLAMA_AP_MODEL` to override the model.

---

## New Features (2026-Q2)

### Peppol / UBL 2.1 E-Invoice Ingest

Accept supplier invoices in Peppol BIS 3.0 / UBL 2.1 XML format directly — zero manual data entry.

```python
with open("invoice.xml", "rb") as f:
    xml_bytes = f.read()

result = await service.ingest_peppol_invoice(xml_bytes, tenant_id)
if result["peppol_valid"]:
    print(f"Ingested: {result['invoice_number']}  Amount: {result['amount']}  {result['currency']}")
else:
    print(f"Validation errors: {result['parse_errors']}")
    # result["requires_review"] == True — route to AP clerk queue
```

The ingestor uses `lxml` when available and falls back to stdlib `ElementTree`.  Extracted fields: invoice number, issue date, due date, currency, supplier name, supplier tax ID, line totals, and tax totals.  Missing mandatory fields set `requires_review=True`.

### Dual-Control Vendor Bank Account Change

Enforce two-person integrity (TPI) for every bank account change — blocks Business Email Compromise (BEC) fraud.

```python
# Step 1 — AP Clerk proposes the change
proposal = await service.propose_vendor_bank_change(
    vendor["id"], tenant_id,
    new_bank_account="KCB-001122334",
    new_iban="GB29NWBK60161331926819",
    proposed_by="alice",
)
# proposal["status"] == "pending_confirmation"
# proposal["iban_valid"] == True  (ISO 7064 MOD 97-10 check digit verified)

# Step 2 — AP Controller confirms (must be a different user)
confirmed = await service.confirm_vendor_bank_change(
    proposal["change_id"], tenant_id, confirmed_by="bob"
)
# confirmed["status"] == "confirmed"
# Vendor bank_account updated atomically; audit event emitted
```

Any attempt by the same user to both propose and confirm raises `PermissionError: separation of duties violation`.  Payment runs against vendors with pending (unconfirmed) changes are automatically flagged `bank_change_pending=True`.

### Payment Schedule Optimisation

Maximise working capital by ranking early-payment discount opportunities against your cost of capital.

```python
schedule = await service.optimise_payment_schedule(
    tenant_id,
    available_cash=1_000_000.0,
    cost_of_capital_pct=12.0,  # annual %
)
print(f"Invoices scheduled early: {schedule['scheduled_early']}")
print(f"Projected savings: {schedule['total_projected_savings']}")
for item in schedule["schedule"]:
    print(f"  {item['invoice_id']}: ROI {item['annualised_roi']}% p.a. — save {item['discount_pct']}%")
```

The optimiser computes `annualised_roi = discount_pct / days_saved * 365` for each eligible invoice, subtracts cost of capital, and greedily allocates cash to highest-ROI opportunities first.  Invoices where ROI advantage ≤ 0 are deferred to standard terms.

### KRA WHT Certificate Generation (P9A / P9B)

Issue withholding tax certificates to suppliers as required by the Kenya Income Tax Act Cap 470.

```python
# First, ensure tax has been computed
await service.compute_invoice_tax(invoice["id"], tenant_id, "standard")

# Then issue the certificate
cert = await service.generate_wht_certificate(invoice["id"], tenant_id, certificate_type="P9A")
print(f"Certificate: {cert['certificate_number']}")
print(f"WHT Amount: {cert['wht_amount']}  Net Payment: {cert['net_payment']}")
# cert["issued_at"] — timestamp for the 30-day KRA deadline
```

`P9A` is for resident suppliers; `P9B` for non-residents (royalties, dividends, management fees).  Certificate numbers are tenant-scoped sequential (e.g. `P9A-TENANT-202606-0001`).  Requires `compute_invoice_tax` to have run on the invoice first.

### Supplier KYB (Know Your Business) Onboarding

Automated due diligence before activating a new supplier — sanctions screening, registration format, KRA PIN validation.

```python
kyb = await service.initiate_supplier_kyb(
    tenant_id,
    {
        "legal_name": "Savanna Supplies Ltd",
        "registration_number": "PVT/2021/123456",
        "tax_pin": "P051234567A",
        "director_names": ["John Kamau"],
        "beneficial_owners": ["Jane Wanjiku"],
    },
    requested_by="procurement-officer",
)
print(f"KYB score: {kyb['kyb_risk_score']}/100  Decision: {kyb['decision']}")
for check in kyb["checks"]:
    status = "PASS" if check["passed"] else "FAIL"
    print(f"  [{status}] {check['check']}")
```

Auto-approved when score < 30.  Escalated for manual review when score ≥ 70.  Set `SANCTIONS_API_URL` to integrate a live sanctions screening endpoint; falls back to a keyword blocklist.

### Match Exception Triage

Priority-rank open match exceptions so your team resolves the highest-impact issues first.

```python
triage = await service.triage_match_exceptions(tenant_id, top_n=10)
print(f"Total open exceptions: {triage['total_open']}")
print(f"Average age: {triage['avg_age_days']} days")
for exc in triage["triaged"]:
    print(f"  [{exc['priority_score']:.0f}] {exc['invoice_id']} — "
          f"{exc['exception_type']} — Action: {exc['recommended_action']} — "
          f"SLA: {exc['sla_hours']}h")
```

Priority score (0–100) combines: financial weight (outstanding / total AP × 40), age weight (days open / 90 × 40), vendor risk weight (20).  `recommended_action` is derived from `exception_type`; `sla_hours` gives the target resolution window.

### Cash Flow What-If Scenarios

Model the AP impact of different payment strategies before committing to treasury.

```python
scenarios = [
    {"name": "baseline",       "payment_offset_days": 0,   "held_fraction": 0.0, "discount_capture_pct": 0},
    {"name": "pay_early_10d",  "payment_offset_days": -10, "held_fraction": 0.0, "discount_capture_pct": 80},
    {"name": "dispute_20pct",  "payment_offset_days": 0,   "held_fraction": 0.2, "discount_capture_pct": 0},
]
analysis = await service.cash_flow_sensitivity(tenant_id, scenarios)
for row in analysis["comparison"]:
    print(f"  {row['name']:20s}  Cash out: {row['total_cash_out']}  "
          f"Delta: {row['delta_vs_baseline']}  ({row['delta_pct']}%)")
```

Scenarios run concurrently via `asyncio.gather`.  `payment_offset_days` shifts all due dates (negative = earlier payment); `held_fraction` defers a random fraction of invoices; `discount_capture_pct` models how many eligible discounts are captured.

### Dormant Vendor Identification

Clean vendor master records and reduce fraud surface by deactivating suppliers with no recent activity.

```python
result = await service.identify_dormant_vendors(
    tenant_id,
    inactive_days=365,
    auto_deactivate=False,  # set True to deactivate automatically
)
print(f"Dormant vendors: {result['dormant_count']} / {result['total_vendors']}")
for v in result["dormant_vendors"]:
    print(f"  {v['vendor_name']:30s}  Last activity: {v['last_activity_date']}  "
          f"({v['days_since_last_activity']} days)")
```

When `auto_deactivate=True`, vendors are set to `status=inactive` and a `vendor_deactivated` audit event is emitted.  Records are never deleted — the status change is reversible by updating the vendor record.

### AP Compliance Scorecard

Continuous monitoring of 10 AP controls — replaces sample-based periodic audit.

```python
scorecard = await service.compute_compliance_scorecard(
    tenant_id, {"start": "2026-01-01", "end": "2026-06-30"}
)
print(f"Compliance grade: {scorecard['grade']}  Score: {scorecard['composite_score']}/100")
for control in scorecard["controls"]:
    print(f"  {control['control']:40s}  {control['score']:3d}/100")
print("Remediation items:")
for item in scorecard["remediation_items"]:
    print(f"  - {item}")
```

Ten controls evaluated (10 pts each): segregation of duties, PO coverage, three-way match rate, exception aging, WHT certificate issuance, duplicate rate, bank change review compliance, expense receipt coverage, payment fraud indicators, and accounting period completeness.  Grade: A (≥90), B (≥75), C (≥60), D (≥45), F (<45).

---

## Configuring AI Features

```bash
# Enable Ollama-powered features
export OLLAMA_BASE_URL=http://localhost:11434

# Override the NL query model (optional)
export OLLAMA_AP_MODEL=llama3.1:8b

# Sanctions screening API for supplier KYB (optional)
export SANCTIONS_API_URL=http://localhost:8080/screen
```

All AI features are opt-in and fail safe. When `OLLAMA_BASE_URL` is not set, every async method falls back to a deterministic rule-based equivalent. The `ml_enhanced` field in every response indicates whether the ML path executed.
