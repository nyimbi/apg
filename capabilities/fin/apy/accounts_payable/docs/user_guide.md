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

## Configuring AI Features

```bash
# Enable Ollama-powered features
export OLLAMA_BASE_URL=http://localhost:11434

# Override the NL query model (optional)
export OLLAMA_AP_MODEL=llama3.1:8b
```

All AI features are opt-in and fail safe. When `OLLAMA_BASE_URL` is not set, every async method falls back to a deterministic rule-based equivalent. The `ml_enhanced` field in every response indicates whether the ML path executed.
