# General Ledger — Troubleshooting Guide

© 2025 Datacraft

---

## Diagnostic checklist

Before filing a bug:

1. `GET /api/glr/health` returns `{"status": "ok"}`
2. `X-Tenant-ID` header is present on every request
3. At least one period is in `open` status
4. Journal lines balance: `sum(debit) == sum(credit)`
5. Database schema is up to date

---

## Common errors

### `journal_not_balanced`
**Cause**: `sum(debit_lines) != sum(credit_lines)` or total is zero.  
**Fix**: Recalculate lines. Use `POST /api/glr/journals/{id}/approval-workflow` to debug the total before posting.  
**Code**: `domain/rules.py::assert_journal_balanced`

---

### `no_open_period_for_date`
**Cause**: No accounting period with status=`open` covers the journal date.  
**Fix**:
```bash
# Check current period status
curl -H "X-Tenant-ID: acme" /api/glr/periods

# Open a period
curl -X POST -H "X-Tenant-ID: acme" \
  -d '{"opened_by": "controller"}' \
  /api/glr/periods/2026-01/open
```

---

### `period_close_blocked`
**Cause**: Outstanding items prevent period close. The error message lists each blocker.  
**Fix**: Check the period checklist:
```bash
curl -H "X-Tenant-ID: acme" /api/glr/periods/2026-01/checklist
```
Resolve each `outstanding` item: post all draft journals, approve all reconciliations.

---

### `segregation_of_duties_required`
**Cause**: The same user attempted to both create/approve and post a journal.  
**Fix**: A different user must post. Use `posted_by` with a different user ID.  
**Note**: In test/dev environments, use `prepared_by="alice"` and `posted_by="bob"`.

---

### `journal_approval_required`
**Cause**: Journal was not approved before posting via the legacy two-step path.  
**Fix**: Use `post_journal_v2` (single-step) or call `/journals/{id}/approve` before `/journals/{id}/post`.

---

### `locked_period_cannot_be_reopened`
**Cause**: A locked period is permanently immutable.  
**Fix**: If correction is needed, post a prior-year adjustment journal:
```bash
curl -X POST -H "X-Tenant-ID: acme" \
  -d '{"account_code": "1000", "amount": "5000", "adjustment_reason": "Error correction"}' \
  /api/glr/year-end/prior-year-adjustment
```

---

### `account_not_found:{code}`
**Cause**: The account_id in a journal line doesn't exist or belongs to a different tenant.  
**Fix**: Verify account ID:
```bash
curl -H "X-Tenant-ID: acme" /api/glr/accounts/{account_id}
```
Ensure the account's `tenant_id` matches the header.

---

### `account_disallows_posting`
**Cause**: The account has `allow_posting=false` (it's a header/summary account).  
**Fix**: Post to a leaf account in the hierarchy, or update `allow_posting`:
```bash
curl -X PUT -H "X-Tenant-ID: acme" \
  -d '{"allow_posting": true}' \
  /api/glr/accounts/{account_id}
```

---

### `retained_earnings_account_not_found`
**Cause**: Year-end close can't find the retained earnings account code.  
**Fix**: Create an equity-type account and tag it:
```python
svc.accounts[acct_id]["tags"] = ["retained_earnings"]
```
Or ensure the account_code passed to `year_end_close` matches an existing equity account.

---

### `tenant_context_required`
**Cause**: `tenant_id` is empty or missing.  
**Fix**: Always pass `X-Tenant-ID` header or `tenant_id` in the JSON body.

---

### `cross_tenant_access_denied`
**Cause**: The actor's tenant does not match the resource's tenant.  
**Fix**: Ensure you are requesting data for your own tenant only.

---

### Trial balance not balanced
**Symptom**: `trial_balance.balanced == false`  
**Causes**:
- Postings created outside `post_journal_v2` with unbalanced lines
- Direct manipulation of `postings` store in tests
- Intercompany journal partially failed (compensating reversal may have been posted)

**Fix**:
1. Run `GET /api/glr/reports/trial-balance?period_code=...&include_zero_balances=true`
2. Sum `closing_debit` and `closing_credit` columns manually
3. Identify the account with an unexpected balance
4. Post a correcting journal entry or prior-year adjustment

---

### Period end checklist stuck
**Symptom**: Checklist shows items outstanding even after taking action.  
**Fix**: The checklist is computed live from service state. After posting the journal or approving the reconciliation, re-fetch the checklist — no cache to clear.

---

## Performance issues

### Slow trial balance
**Cause**: `trial_balance` iterates all postings in memory. In production with millions of rows this will be slow.  
**Fix**: Add a PostgreSQL materialized view on `gl_posting` grouped by `(tenant_id, period_code, account_id)`. Refresh it on each `post_journal_v2` call.

### Slow `management_accounts_pack`
**Cause**: Calls 5 reports sequentially.  
**Fix**: Use `asyncio.gather` to parallelize:
```python
tb, bs, inc, cfs, bva = await asyncio.gather(
    svc.trial_balance(tenant, period),
    svc.balance_sheet(tenant, period),
    svc.income_statement(tenant, period),
    svc.cash_flow_statement(tenant, period),
    svc.budget_vs_actual(tenant, period),
)
```

---

## Backup and recovery

### In-memory service (development)
All state is in Python dicts — lost on restart. Use only for testing.

### PostgreSQL (production)
Standard PostgreSQL backup:
```bash
pg_dump -U postgres apg_db > glr_backup_$(date +%Y%m%d).sql
```

Point-in-time recovery: enable WAL archiving in `postgresql.conf`.

---

## APG observability

Audit events are accessible at:
```bash
curl -H "X-Tenant-ID: acme" /api/glr/audit-events
```

Every state-changing operation emits a structured event to the `apg.fin.glr.lifecycle` Bytewax stream with `event_type`, `record_id`, `record_type`, `status`, and `emitted_at`.

For APG-level monitoring, configure Bytewax to forward events to your observability platform (Prometheus, Grafana, Datadog).

---

## Getting help

- Source: `capabilities/fin/glr/general_ledger/`
- Tests: `pytest -vxs capabilities/fin/glr/general_ledger/tests/`
- Contract: `GET /contract` on the running service
- Capability spec: `capabilities/fin/glr/general_ledger/cap_spec.md`
