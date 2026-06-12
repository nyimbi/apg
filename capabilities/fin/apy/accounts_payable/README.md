# Accounts Payable

`apy_accounts_payable` is the APG capability for composing vendor liability, invoice, matching, approval, payment, reimbursement, close, and AP-agent workflows into generated Python applications. It provides an executable service surface, deterministic guardrails, UI metadata, theme metadata, and Bytewax lifecycle-stream declarations.

## What It Provides

- Vendor registration with owner, tax profile, payment method, and bank-change review controls.
- Invoice capture with vendor, invoice number, currency, document evidence, duplicate review, and amount validation.
- Matching workflow for PO-backed invoices, receipt evidence, and variance review.
- Approval workflow with high-value controls and separation of duties.
- Invoice hold and hold-release lifecycle.
- Payment scheduling and payment-batch release.
- Expense report capture with receipt and policy-exception review controls.
- AP aging and period-close guardrails.
- First-class AP agents for Codex, Claude Code, OpenCode, and Pi.
- Deterministic rules for tenant, policy, financial, agent, and stream guardrails.
- Bytewax lifecycle stream metadata.
- UI route and theme metadata for APG composition.

## Quick Start

```python
from capabilities.fin.apy.accounts_payable import AccountsPayableService

service = AccountsPayableService()
vendor = service.register_vendor(
    "vendor-1",
    "tenant-a",
    "Northwind Supplies",
    "ap-operations",
    "vat-profile",
    "ach",
)
invoice = service.record_invoice(
    "invoice-1",
    "tenant-a",
    vendor["id"],
    "INV-001",
    5000,
    "USD",
    "doc-001",
)
service.match_invoice(
    "tenant-a",
    invoice["id"],
    po_backed=True,
    receipt_reference="receipt-001",
)
service.approve_invoice(
    "tenant-a",
    invoice["id"],
    approved_by="controller",
    requested_by="ap-operations",
)
payment = service.schedule_payment(
    "payment-1",
    "tenant-a",
    invoice["id"],
    5000,
    "operating-cash",
    "2026-06-05",
)
service.release_payment_batch("batch-1", "tenant-a", [payment["id"]], "treasury")
summary = service.dashboard_summary("tenant-a")
```

## Contract

Use `get_capability_contract()` to inspect the APG composition surface.

```python
from capabilities.fin.apy.accounts_payable import get_capability_contract

contract = get_capability_contract("tenant-a")
print(contract["provides"])
print(contract["streaming"]["processor"])
```

The contract exposes:

- `configuration`
- `configuration_schema`
- `rule_engine`
- `ui`
- `theme`
- `streaming`

## Guardrails

The rule engine blocks or routes review for:

- Missing tenant context.
- Writes without policy attachment.
- Vendors without owner, tax profile, or payment method.
- Vendor bank changes without independent review.
- Invoices without vendor, invoice number, currency, positive amount, or document reference.
- Potential duplicate invoices without review.
- PO-backed invoices without receipt evidence.
- Matching variance above threshold without review.
- High-value approvals without approval evidence.
- Self-approval by invoice requesters.
- Payments against unapproved or held invoices.
- Payments without positive amount or cash account.
- Payment batches without review.
- Holds without reason and hold releases without approval.
- Expense reports without employee, positive amount, or receipt evidence.
- Expense policy exceptions without review.
- Period close with open exceptions, unposted invoices, or missing aging review.
- Batch and lifecycle events not routed through Bytewax.
- Unsupported AP-agent runtime or role.
- Privileged AP-agent actions without human approval.

## UI And Theme

The capability publishes route metadata for:

- `/apy-accounts-payable/dashboard`
- `/apy-accounts-payable/vendors`
- `/apy-accounts-payable/invoices`
- `/apy-accounts-payable/matching`
- `/apy-accounts-payable/approvals`
- `/apy-accounts-payable/payments`
- `/apy-accounts-payable/expenses`
- `/apy-accounts-payable/aging`
- `/apy-accounts-payable/close`
- `/apy-accounts-payable/agents`
- `/apy-accounts-payable/settings`

The default theme is `apy_accounts_payable_control`. View helpers in `views.py` return dashboard, vendor, invoice, matching, approval, payment, expense, aging, close, and agent workbench models.

## AI Agents

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `vendor_risk_reviewer`
- `invoice_exception_reviewer`
- `matching_reviewer`
- `payment_run_reviewer`
- `cash_flow_reviewer`
- `close_reviewer`

Register an agent with `register_ap_agent()` and validate privileged proposals with `validate_agent_ap_action()`.

## Async AI and Analytics Methods

All async methods are awaitable and degrade gracefully when external services (Ollama, sanctions APIs) are unavailable.

### Original Async Extensions (2026-Q1)

| Method | Category | Description |
|---|---|---|
| `ml_duplicate_invoice_detect(invoice_id, tenant_id)` | Fraud Prevention | Ollama embedding cosine-similarity duplicate detection with exact-match fallback |
| `forecast_cash_outflows(tenant_id, horizon_weeks)` | Treasury | 13-week rolling AP cash outflow forecast with P10/P50/P90 bands |
| `compute_invoice_tax(invoice_record_id, tenant_id, tax_profile)` | Compliance | VAT (16%) and WHT computation with iTax-compatible output |
| `generate_vat_schedule(tenant_id, period)` | Compliance | KRA iTax input VAT schedule aggregated from all period invoices |
| `score_vendor_risk(vendor_record_id, tenant_id)` | Risk | 0–100 composite vendor risk score: exceptions, disputes, price variance, bank-change |
| `straight_through_process(invoice_record_id, tenant_id)` | Automation | Full STP pipeline: validate → duplicate check → PO match → auto-approve |
| `compute_vendor_scorecard(vendor_record_id, tenant_id, period)` | Analytics | Vendor performance index: accuracy, match pass, on-time submission, dispute score |
| `score_payment_fraud_risk(payment_record_id, tenant_id)` | Security | Real-time fraud scoring: new bank account, amount anomaly, weekend payment, proximity |
| `compute_accruals(tenant_id, period)` | Accounting | Period-end RNI and service accrual journal entries from unmatched GRNs/POs |
| `nl_query(question, tenant_id)` | UX / AI | Natural language AP query via Ollama LLM with keyword-router fallback |

### New Async Extensions (2026-Q2)

| Method | Category | Description |
|---|---|---|
| `ingest_peppol_invoice(xml_bytes, tenant_id)` | E-Invoicing | Parse Peppol BIS 3.0 / UBL 2.1 XML invoices; validate structure, extract line items and tax totals |
| `propose_vendor_bank_change(vendor_record_id, tenant_id, new_bank_account, new_iban, proposed_by)` | Fraud Prevention | Initiate dual-control bank account change; validates IBAN check digit (ISO 7064 MOD 97-10) |
| `confirm_vendor_bank_change(change_id, tenant_id, confirmed_by)` | Fraud Prevention | Confirm pending bank change; enforces SoD (confirmed_by != proposed_by), atomically updates vendor |
| `optimise_payment_schedule(tenant_id, available_cash, cost_of_capital_pct)` | Working Capital | Rank invoices by NPV of early-pay discount vs cost of capital; greedy allocation of available cash |
| `generate_wht_certificate(invoice_record_id, tenant_id, certificate_type)` | Tax Compliance | Issue KRA P9A/P9B withholding tax certificate with sequential cert number and audit event |
| `initiate_supplier_kyb(tenant_id, supplier_data, requested_by)` | Supplier Risk | KYB due diligence: registration format, KRA PIN, sanctions screening, beneficial owner check |
| `triage_match_exceptions(tenant_id, top_n)` | AP Operations | Priority-score open exceptions by financial impact, age, and vendor risk; return top-N with recommended action |
| `cash_flow_sensitivity(tenant_id, scenarios)` | Treasury | What-if cash flow modelling: shift payment dates, hold fraction, discount capture — parallel via asyncio.gather |
| `identify_dormant_vendors(tenant_id, inactive_days, auto_deactivate)` | Vendor Governance | Detect and optionally deactivate vendors with no AP activity for N days |
| `compute_compliance_scorecard(tenant_id, period)` | Compliance | 10-control AP compliance scorecard (SoD, PO coverage, WHT certs, exceptions, fraud indicators) with A–F grade |

## Usage Examples

### Core Workflow

```python
import asyncio
from capabilities.fin.apy.accounts_payable import AccountsPayableService

service = AccountsPayableService()
vendor = service.register_vendor("v1", "tenant-a", "Acme Ltd", "ap-ops", "vat-ke", "ach")
invoice = service.record_invoice("inv-1", "tenant-a", vendor["id"], "INV-2026-001", 50000, "KES", "doc-001")

# Compute tax
tax = asyncio.run(service.compute_invoice_tax(invoice["id"], "tenant-a", "standard"))
# -> {"vat_amount": "8000.00", "wht_amount": "2500.00", "net_payable": "55500.00", ...}

# Issue WHT certificate
cert = asyncio.run(service.generate_wht_certificate(invoice["id"], "tenant-a", "P9A"))
# -> {"certificate_number": "P9A-TENANT-202606-0001", "wht_amount": "2500.00", ...}

# STP pipeline
result = asyncio.run(service.straight_through_process(invoice["id"], "tenant-a"))
# -> {"passed": True, "steps": [...], "completed_at": "..."}

# Cash flow forecast
forecast = asyncio.run(service.forecast_cash_outflows("tenant-a", horizon_weeks=4))
# -> {"total_projected_outflow": "50000.00", "weekly_buckets": [...], ...}
```

### Dual-Control Bank Account Change

```python
# Proposer initiates change
proposal = asyncio.run(
    service.propose_vendor_bank_change(vendor["id"], "tenant-a", "KCB-001122", "GB29NWBK60161331926819", "alice")
)
# proposal["status"] == "pending_confirmation"

# A different user confirms (enforces SoD)
confirmed = asyncio.run(
    service.confirm_vendor_bank_change(proposal["change_id"], "tenant-a", "bob")
)
# confirmed["status"] == "confirmed"
```

### Payment Schedule Optimisation

```python
# Maximise early-pay discount capture within available cash
schedule = asyncio.run(
    service.optimise_payment_schedule("tenant-a", available_cash=500000, cost_of_capital_pct=12.0)
)
# -> {"scheduled_early": 5, "total_projected_savings": "3420.00", "schedule": [...], ...}
```

### Compliance Scorecard

```python
scorecard = asyncio.run(
    service.compute_compliance_scorecard("tenant-a", {"start": "2026-01-01", "end": "2026-06-30"})
)
# -> {"composite_score": 82.0, "grade": "B", "controls": [...], "remediation_items": [...], ...}
```

### Cash Flow What-If Scenarios

```python
scenarios = [
    {"name": "baseline", "payment_offset_days": 0, "held_fraction": 0.0, "discount_capture_pct": 0},
    {"name": "pay_early_10d", "payment_offset_days": -10, "held_fraction": 0.0, "discount_capture_pct": 100},
    {"name": "dispute_20pct", "payment_offset_days": 0, "held_fraction": 0.2, "discount_capture_pct": 0},
]
analysis = asyncio.run(service.cash_flow_sensitivity("tenant-a", scenarios))
# -> {"comparison": [{"name": "pay_early_10d", "delta_vs_baseline": "-45000.00", ...}, ...], ...}
```

### Supplier KYB Onboarding

```python
kyb = asyncio.run(service.initiate_supplier_kyb(
    "tenant-a",
    {
        "legal_name": "Savanna Supplies Ltd",
        "registration_number": "PVT/2021/123456",
        "tax_pin": "P051234567A",
        "director_names": ["John Kamau", "Jane Wanjiku"],
    },
    requested_by="procurement-officer",
))
# -> {"kyb_risk_score": 0, "decision": "approved", "checks": [...], ...}
```

### Peppol E-Invoice Ingest

```python
ubl_xml = b"""<?xml version="1.0"?>
<Invoice xmlns:cbc="urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2"
         xmlns:cac="urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2">
  <cbc:ID>INV-2026-0042</cbc:ID>
  <cbc:DocumentCurrencyCode>KES</cbc:DocumentCurrencyCode>
  <cac:LegalMonetaryTotal><cbc:PayableAmount>125000.00</cbc:PayableAmount></cac:LegalMonetaryTotal>
</Invoice>"""
result = asyncio.run(service.ingest_peppol_invoice(ubl_xml, "tenant-a"))
# -> {"invoice_number": "INV-2026-0042", "amount": 125000.0, "peppol_valid": True, ...}
```

### Natural Language Query

```python
answer = asyncio.run(service.nl_query("Which invoices are overdue?", "tenant-a"))
# -> {"answer": "AP aging: 1 open invoice...", "data": {...}, "ml_enhanced": False, ...}
```

## OLLAMA Integration

Set `OLLAMA_BASE_URL` to enable ML-enhanced features:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_AP_MODEL=llama3.1:8b   # default model for nl_query
```

All ML-enhanced methods degrade gracefully to rule-based fallbacks when Ollama is unavailable. The `ml_enhanced` field in every response indicates which execution path was taken.

Optional — sanctions screening API:

```bash
export SANCTIONS_API_URL=http://localhost:8080/screen  # POST {name} -> {match: bool}
```

## Verification

Focused verification for this package:

```bash
./.venv/bin/python -m py_compile \
  capabilities/fin/apy/accounts_payable/__init__.py \
  capabilities/fin/apy/accounts_payable/capability_contract.py \
  capabilities/fin/apy/accounts_payable/service.py \
  capabilities/fin/apy/accounts_payable/api.py \
  capabilities/fin/apy/accounts_payable/views.py \
  capabilities/fin/apy/accounts_payable/app.py \
  capabilities/fin/apy/accounts_payable/tests/test_package_contract.py

./.venv/bin/pytest -q capabilities/fin/apy/accounts_payable/tests/test_package_contract.py
./.venv/bin/python capabilities/fin/apy/accounts_payable/app.py
```

Deferred live-system work includes durable stores, live GL, cash-management, document, audit, notification, and authorization adapters, durable Bytewax deployment, rendered browser UI, and performance testing.

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Peppol / UBL 2.1 E-Invoicing Ingest** [E-Invoicing Compliance]
- **I2. IBAN / Bank Account Dual-Control Verification Workflow** [Fraud Prevention / Controls]
- **I3. SWIFT ISO 20022 Pain.001 Payment File Generation** [Treasury / Bank Integration]
- **I4. AI-Powered OCR Invoice Data Extraction** [Intelligent Automation]
- **I5. Intelligent Payment Terms Optimisation Engine** [Working Capital Analytics]
- **I6. Automated Withholding Tax Certificates (P9A / P9B)** [Tax Compliance]
- **I7. Period-End Accruals Auto-Reversal Scheduling** [Accounting / Close Automation]
- **I8. Supplier Onboarding Due Diligence (KYB) Workflow** [Supplier Risk / Compliance]
- **I9. Multi-Currency FX Revaluation at Period End** [Financial Reporting]
- **I10. Supplier Credit Limit Enforcement** [Risk Controls]
- **I11. Comprehensive Audit Trail with Immutable Event Log** [Compliance / Governance]
- **I12. Intelligent Exception Triage with Priority Scoring** [AP Operations / AI]
- **I13. Cash Flow Sensitivity Analysis (What-If)** [Treasury / Analytics]
- **I14. Dormant Supplier Deactivation Workflow** [Vendor Master Governance]
- **I15. Real-Time AP Compliance Monitoring Dashboard** [Compliance / Controls]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
