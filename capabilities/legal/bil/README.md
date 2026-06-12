# leg_bil — Legal Billing & Time Tracking

Time capture, matter billing, disbursements, invoice approval, client trust accounting.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/bil/health | Health check |
| GET | /api/legal/bil/time-entries | List time entries |
| GET | /api/legal/bil/time-entries/{id} | Get time entry |
| POST | /api/legal/bil/time-entries | Create time entry |
| PUT | /api/legal/bil/time-entries/{id} | Update time entry |
| DELETE | /api/legal/bil/time-entries/{id} | Write off time entry |
| POST | /api/legal/bil/time-entries/{id}/submit | Submit time entry |
| POST | /api/legal/bil/time-entries/{id}/approve | Approve time entry |
| GET | /api/legal/bil/disbursements | List disbursements |
| POST | /api/legal/bil/disbursements | Record disbursement |
| PUT | /api/legal/bil/disbursements/{id} | Update disbursement |
| DELETE | /api/legal/bil/disbursements/{id} | Cancel disbursement |
| GET | /api/legal/bil/invoices | List invoices |
| GET | /api/legal/bil/invoices/{id} | Get invoice |
| POST | /api/legal/bil/invoices | Create invoice |
| PUT | /api/legal/bil/invoices/{id} | Update invoice |
| DELETE | /api/legal/bil/invoices/{id} | Write off invoice |
| POST | /api/legal/bil/invoices/{id}/approve | Approve invoice |
| POST | /api/legal/bil/invoices/{id}/send | Send invoice |
| POST | /api/legal/bil/invoices/{id}/pay | Record payment |
| GET | /api/legal/bil/trust-accounts | List trust accounts |
| POST | /api/legal/bil/trust-accounts | Open trust account |
| POST | /api/legal/bil/trust-accounts/{id}/transactions | Trust transaction |
| GET | /api/legal/bil/trust-accounts/{id}/transactions | List transactions |
| GET | /api/legal/bil/dashboard | Billing dashboard |
| GET | /api/legal/bil/audit | Audit events |

## Service Class

`LegalBillingService` — ABA activity codes, time entry approval workflow, auto-invoice calculation with 16% Kenya VAT, trust account ledger with running balance, attorney rate cards.

## World-Class Enhancements (v2.0)

**I1. AI-Powered Time Narrative Quality Scoring** — Ollama/Mistral 7B scores descriptions 0–100 at entry time and surfaces rewrite suggestions to prevent vague-narrative write-downs. [AI/ML]

**I2. LEDES 98B / LEDES XML 2.1 Electronic Invoice Export** — Serialises invoice lines into LEDES 98B tab-delimited or XML 2.1 format with UTBMS code validation for corporate eBilling portals. [Compliance]

**I3. Matter Budget vs. Actual Burn with Velocity Projection** — Stores phase budgets, emits alert events at 80%/100% burn, and returns projected completion cost and days remaining per matter. [Performance]

**I4. Decimal-Precision Monetary Arithmetic** — Replaces all `float` monetary fields with `Decimal` + `ROUND_HALF_UP`, eliminating rounding drift across trust accounting and KRA audit trails. [Compliance]

**I5. Recurring / Subscription Fee Schedule Automation** — Stores fee schedules and auto-generates draft invoices for all elapsed periods on a daily cron cadence. [Feature]

**I6. Write-Off Approval Workflow with Mandatory Reason Codes** — Enforces a pending → approved/rejected state machine for write-offs; direct write-offs without approval are blocked. [Compliance]

**I7. Invoice Dispute Resolution Sub-Ledger** — Structured raise → respond → negotiate → resolve workflow with optional credit note application and DSO-improving status transitions. [Feature]

**I8. Realization Rate Analytics (Worked → Billed → Collected Cascade)** — Per-attorney and firm-wide realization/collection percentages with quartile banding over configurable date ranges. [Performance]

**I9. Automated Overdue Detection and AR Aging Buckets** — Scans sent invoices past due date, transitions status, emits tiered dunning events, and groups outstanding balances into 0–30/31–60/61–90/90+ buckets. [Feature]

**I10. Multi-Currency Support with Exchange Rate Snapshots** — Stores dated rate snapshots and auto-converts time-entry currency to invoice currency, recording rate and date on each line. [Feature]

**I11. Partial Payment and Payment Plan Instalment Tracking** — Appends payments to an invoice ledger, tracks `outstanding_amount`, and stores instalment schedules with per-instalment due dates and statuses. [Feature]

**I12. Bulk Time Entry Import from CSV with Duplicate Detection** — Validates rows against UTBMS codes and rate cards, deduplicates by content hash, and commits transactionally with per-row error detail. [Integration]

**I13. Real-Time Billing Timer (Start/Stop Time Capture)** — Creates timer records per matter/attorney, computes elapsed time rounded to 6-minute increments on stop, and creates the resulting time entry automatically. [UX]

**I14. KRA eTIMS E-Invoice Compliance Submission** — Serialises invoices to the KRA eTIMS JSON schema and POSTs to the eTIMS endpoint with exponential-backoff retry, storing submission ID and status. [Compliance]

**I15. Client Portal Read-Only Invoice Access via Scoped Tokens** — Generates HMAC-signed, TTL-bounded opaque tokens granting unauthenticated read access to sanitised invoice data for client-facing portals. [UX/Security]

## New Methods

Three high-impact async methods added in v2.0:

### `start_timer` / `stop_timer` — Capture billable time without manual entry

```python
svc = LegalBillingService(tenant_id="acme")

# Start a timer against an open matter
timer = await svc.start_timer(
    tenant_id="acme",
    matter_id="mat_001",
    attorney_id="att_jane",
    activity_code="L110",  # ABA fact investigation
)

# ... time passes ...

# Stop timer; rounds to nearest 0.1 hr and creates a time entry
entry = await svc.stop_timer(
    tenant_id="acme",
    timer_id=timer["id"],
    description="Reviewed discovery documents and drafted outline",
)
# entry["hours"] == 0.6, entry["status"] == "draft"
```

### `realization_report` — Per-attorney revenue intelligence

```python
report = await svc.realization_report(
    tenant_id="acme",
    attorney_id="att_jane",
    period_start="2026-01-01",
    period_end="2026-03-31",
)
# {
#   "worked_value": "450000.00",
#   "billed_amount": "420000.00",
#   "collected_amount": "378000.00",
#   "realization_pct": "93.3",
#   "collection_pct": "90.0",
# }
```

### `submit_invoice_etims` — KRA eTIMS compliance submission

```python
result = await svc.submit_invoice_etims(
    tenant_id="acme",
    invoice_id="inv_2026_0042",
)
# {
#   "etims_submission_id": "KRA-2026-XXXXX",
#   "etims_status": "accepted",
#   "submitted_at": "2026-06-12T08:31:00Z",
# }
# Invoice record is updated in-place with etims_submission_id and etims_status.
```
