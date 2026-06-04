# General Ledger — API Reference

Base URL: `/api/glr`  
Authentication: `X-Tenant-ID` header (required on every request)  
Content-Type: `application/json`

All responses use the envelope:
```json
{"data": <payload>, "error": null, "meta": {}}
```

---

## Health

### GET /api/glr/health
Liveness probe.
**Response 200**
```json
{"data": {"status": "ok", "capability": "glr_general_ledger"}}
```

---

## Dashboard

### GET /api/glr/dashboard
Returns KPIs, work queue counts, and recent audit events.
**Headers**: `X-Tenant-ID`
**Response 200**: `{account_count, period_count, journal_entry_count, posted_journal_count, pending_approvals, open_reconciliations, recent_events}`

---

## Chart of Accounts

### GET /api/glr/accounts
List accounts with pagination.
**Query params**: `page`, `page_size`, `include_inactive`
**Response 200**: `{data: [account...], meta: {total, page, page_size}}`

### GET /api/glr/accounts/hierarchy
Return nested account tree.
**Response 200**: `{data: {tree: [...], account_count}}`

### POST /api/glr/accounts/create
Create a new account.
**Body**:
```json
{
  "tenant_id": "acme",
  "account_code": "1000",
  "account_name": "Cash",
  "account_type": "asset",
  "currency": "USD",
  "allow_posting": true,
  "parent_account_code": null,
  "description": "Main operating cash account",
  "ifrs_mapping": "ifrs-full:CashAndCashEquivalents",
  "tags": ["current_asset"]
}
```
**Response 201**: Account record

### GET /api/glr/accounts/{account_id}
Retrieve a single account.
**Response 200/404**

### PUT /api/glr/accounts/{account_id}
Partial update (name, allow_posting, tags, description).
**Response 200**

### DELETE /api/glr/accounts/{account_id}
Soft delete (sets status=inactive, is_deleted=true).
**Response 200**: `{"id": "...", "status": "deleted"}`

### GET /api/glr/accounts/{account_id}/analysis
Full transaction history and running balance for a period.
**Query params**: `period_code` (required)
**Response 200**: `{account_code, account_name, opening_balance, lines: [...], closing_balance}`

---

## Accounting Periods

### GET /api/glr/periods
List all periods. Optional `fiscal_year` filter.
**Query params**: `fiscal_year`
**Response 200**: `{data: [period...], meta: {total}}`

### POST /api/glr/periods/create
Create a new period.
**Body**:
```json
{
  "tenant_id": "acme",
  "period_code": "2026-01",
  "fiscal_year": 2026,
  "period_number": 1,
  "start_date": "2026-01-01",
  "end_date": "2026-01-31",
  "allows_adjustments": false
}
```
**Response 201**: Period record

### GET /api/glr/periods/{period_code}
Get period by code.
**Response 200/404**

### POST /api/glr/periods/{period_code}/open
Open a period. Body: `{"opened_by": "controller"}`
**Response 200**

### POST /api/glr/periods/{period_code}/close
Close a period. Runs pre-close checks.
Body: `{"closed_by": "controller"}`
**Response 200** / **400** with `period_close_blocked` if checks fail.

### POST /api/glr/periods/{period_code}/lock
Lock a closed period. Body: `{"locked_by": "cfo"}`
**Response 200**

### POST /api/glr/periods/{period_code}/reopen
Reopen a closed period. Requires reason + authoriser.
Body: `{"reason": "...", "authorised_by": "cfo"}`
**Response 200** / **403** if locked.

### GET /api/glr/periods/{period_code}/checklist
Period-end readiness checklist.
**Response 200**: `{items: [...], outstanding_count, ready_to_close}`

---

## Journal Entries

### GET /api/glr/journals
List journal entries.
**Query params**: `page`, `page_size`, `status` (draft|pending_approval|posted|reversed)
**Response 200**: `{data: [journal...], meta: {total, page, page_size}}`

### POST /api/glr/journals/create
Create and post a balanced journal entry.
**Body**:
```json
{
  "tenant_id": "acme",
  "journal_date": "2026-01-15",
  "journal_type": "standard",
  "description": "Sales revenue INV-001",
  "reference": "INV-001",
  "posted_by": "alice",
  "lines": [
    {"account_id": "acct-1000", "debit": "5000", "credit": "0",
     "description": "Cash receipt", "cost_center": "CC01"},
    {"account_id": "acct-4000", "debit": "0", "credit": "5000",
     "description": "Service revenue", "cost_center": "CC01"}
  ]
}
```
**Response 201**: Posting record  
**Response 400**: `journal_not_balanced`, `no_open_period_for_date`, `account_not_found`  
**Response 422**: Pydantic validation error

### GET /api/glr/journals/{journal_id}
Get journal by ID.
**Response 200/404**

### PUT /api/glr/journals/{journal_id}
Update description/reference on draft journals only.
**Response 200** / **400** if already posted.

### DELETE /api/glr/journals/{journal_id}
Cancel a draft journal (cannot cancel posted — use reversal).
**Response 200** / **400** if posted.

### POST /api/glr/journals/{journal_id}/approve
Approve a journal for posting.
Body: `{"approved_by": "manager"}`
**Response 200**

### POST /api/glr/journals/{journal_id}/post
Post an approved journal.
Body: `{"posted_by": "alice", "idempotency_key": "unique-key"}`
**Response 200**

### POST /api/glr/journals/{journal_id}/reverse
Create a mirror reversal entry.
```json
{
  "reversal_date": "2026-02-01",
  "description": "Accrual reversal",
  "reversed_by": "bob"
}
```
**Response 200**

### POST /api/glr/journals/{journal_id}/approval-workflow
Route through approval workflow with amount threshold.
```json
{"amount_threshold": "10000", "approver_id": "manager"}
```
**Response 200**: `{decision: "auto_approved"|"pending"}`

### POST /api/glr/journals/bulk-import
Import journals from CSV. Send CSV as request body or in `{"csv": "..."}`.
CSV columns: `journal_date, description, reference, account_id, debit, credit, posted_by`
**Response 200**: `{posted_count, failed_count, posted: [...], failed: [...]}`

---

## Reconciliation

### GET /api/glr/reconciliations
List reconciliations for the tenant.
**Response 200**: `{data: [...], meta: {total}}`

### POST /api/glr/reconciliations/create
Open a reconciliation for an account/period.
```json
{"tenant_id": "acme", "account_code": "1000", "period_code": "2026-01"}
```
**Response 201**: Reconciliation record with GL balance

### GET /api/glr/reconciliations/{reconciliation_id}
Get reconciliation detail.
**Response 200/404**

### POST /api/glr/reconciliations/{reconciliation_id}/submit
Submit with reconciling items.
```json
{
  "reconciled_by": "controller",
  "reconciling_items": [
    {"description": "Outstanding cheque #1234", "amount": "1500.00", "item_type": "outstanding_cheque"}
  ],
  "balance_per_statement": "98500.00"
}
```
**Response 200**

### POST /api/glr/reconciliations/{reconciliation_id}/approve
Approve a submitted reconciliation.
Body: `{"approved_by": "cfo"}`
**Response 200**

### POST /api/glr/reconciliations/bank
Automated bank feed matching.
```json
{"bank_account_code": "1000", "statement_id": "stmt-jan-2026"}
```
**Response 200**: `{matched_items, unmatched_gl, unmatched_bank, difference}`

### POST /api/glr/reconciliations/subledger
Subledger vs control account check.
```json
{"period_code": "2026-01"}
```
**Response 200**

---

## Budgets

### GET /api/glr/budgets
List budget lines.
**Response 200**

### POST /api/glr/budgets/create
Create a budget line.
```json
{
  "tenant_id": "acme",
  "budget_code": "BUD-2026-01-4000",
  "fiscal_year": 2026,
  "budget_type": "original",
  "account_code": "4000",
  "period_code": "2026-01",
  "amount": "500000",
  "currency": "USD"
}
```
**Response 201**

### PUT /api/glr/budgets/{budget_id}
Update budget amount or type.
**Response 200**

### DELETE /api/glr/budgets/{budget_id}
Soft delete a budget line.
**Response 200**

---

## Currency Rates

### GET /api/glr/currency-rates
List FX rates.
**Response 200**

### POST /api/glr/currency-rates/create
Add an FX rate.
```json
{
  "tenant_id": "acme",
  "from_currency": "USD",
  "to_currency": "KES",
  "rate_type": "spot",
  "effective_date": "2026-01-01",
  "exchange_rate": "130.50"
}
```
**Response 201**

---

## Financial Reports

All report endpoints accept `period_code` as a query parameter.

### GET /api/glr/reports/trial-balance
**Query**: `period_code`, `include_zero_balances`  
**Response**: `{rows: [{account_code, account_name, account_type, opening_balance, period_debit, period_credit, closing_debit, closing_credit}], total_closing_debit, total_closing_credit, balanced}`

### GET /api/glr/reports/balance-sheet
**Query**: `period_code`, `comparative_period`  
**Response**: `{assets, liabilities, equity, total_assets, total_liabilities_and_equity, balanced}`

### GET /api/glr/reports/income-statement
**Query**: `period_code`, `comparative_period`, `segment`  
**Response**: `{revenue, cost_of_goods_sold, gross_profit, operating_expenses, ebit, finance_cost, ebt, tax_expense, pat, comparative}`

### GET /api/glr/reports/cash-flow
**Query**: `period_code`, `method` (indirect|direct)  
**Response**: `{operating_activities, investing_activities, financing_activities, net_change_in_cash}`

### GET /api/glr/reports/budget-vs-actual
**Query**: `period_code`, `budget_version`  
**Response**: `{rows: [{account_code, actual, budget, variance, variance_pct, indicator}], row_count}`

### GET /api/glr/reports/segment
**Query**: `period_code`, `segment_dimension` (cost_center|department|project)  
**Response**: `{segments: [{segment, revenue, expenses, contribution}]}`

### GET /api/glr/reports/statement-of-equity
**Query**: `fiscal_year`  
**Response**: `{opening_equity, profit_for_year, other_comprehensive_income, dividends_paid, closing_equity}`

### GET /api/glr/reports/management-pack
All statements + ratios in one call.  
**Query**: `period_code`

### GET /api/glr/reports/xbrl
XBRL tagging extract.  
**Query**: `period_code`, `framework` (IFRS|GAAP)  
**Response**: `{facts: [{xbrl_concept, account_code, value, period, unit}], fact_count}`

---

## Year-End

### POST /api/glr/year-end/close
Close fiscal year, post closing entries.
```json
{"tenant_id": "acme", "fiscal_year": 2026, "retained_earnings_account": "3100", "executed_by": "cfo"}
```
**Response 200**: Closing entry record

### POST /api/glr/year-end/opening-balances
Carry forward balance sheet accounts to new year.
```json
{"tenant_id": "acme", "new_fiscal_year": 2027}
```
**Response 200**

### POST /api/glr/year-end/prior-year-adjustment
IAS 8 prior-year error correction.
```json
{
  "tenant_id": "acme",
  "account_code": "1000",
  "amount": "5000",
  "adjustment_reason": "Depreciation error in FY2025",
  "executed_by": "cfo"
}
```
**Response 200**

---

## Consolidation & Intercompany

### POST /api/glr/consolidation
IFRS group consolidation.
```json
{
  "tenant_id": "parent",
  "subsidiaries": ["sub-ke", "sub-ug"],
  "group_adjustments": [
    {"description": "Goodwill amortisation", "account_code": "3300", "amount": "-50000", "entity": "parent"}
  ],
  "minority_interest": {"subsidiary": "sub-ke", "percentage": 30}
}
```
**Response 200**: `{consolidated_rows, eliminations, minority_interest, entity_count}`

### POST /api/glr/intercompany
Post matching entries in two entities simultaneously.
```json
{
  "tenant_id": "entity-a",
  "counterpart_entity": "entity-b",
  "amount": "100000",
  "currency": "USD",
  "account_mapping": {
    "entity_account": "acct-ic-receivable",
    "counterpart_account": "acct-ic-payable"
  }
}
```
**Response 200**: `{entity_posting_id, counterpart_posting_id, status}`

### POST /api/glr/intercompany/reconcile
Check intercompany balances for elimination.
**Response 200**: `{difference, status}`

---

## Recurring Templates

### GET /api/glr/recurring-templates
### POST /api/glr/recurring-templates/create
### POST /api/glr/recurring-templates/{template_id}/run
Run a template for a period: `{"period": "2026-01"}`

---

## Audit Trail

### GET /api/glr/audit-events
All audit events for the tenant.
**Response 200**: `{data: [{tenant_id, event_type, record_id, emitted_at, processor}], meta: {total}}`

---

## Error codes

| Code | Meaning |
|---|---|
| `tenant_required` | Missing X-Tenant-ID header |
| `not_found` | Record not found or wrong tenant |
| `validation_error` | Pydantic model validation failed |
| `bad_request` | Business rule violation |
| `journal_not_balanced` | Debits ≠ credits |
| `no_open_period_for_date` | No open period covers journal_date |
| `period_close_blocked` | Outstanding items prevent close |
| `segregation_of_duties_required` | Same user prepared and posted |
| `journal_already_posted` | Double-post attempt |
| `locked_period_cannot_be_reopened` | Period is locked |
