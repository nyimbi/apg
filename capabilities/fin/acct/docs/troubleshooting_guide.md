# Bank Account Management — Troubleshooting Guide

## Common Errors

### `tenant_context_required` / `missing_tenant` (401)
**Cause**: `X-Tenant-ID` header missing or empty.
**Fix**: Include `X-Tenant-ID: your-tenant` on every request.

### `insufficient_funds`
**Cause**: `available_balance < requested_amount`.
**Check**:
```http
GET /api/fin/acct/accounts/{id}/balance
```
Note: `available_balance = book_balance - locked_balance + overdraft_available`. Locked funds reduce available but not book balance.

### `currency_mismatch`
**Cause**: The currency in the credit/debit request doesn't match the account's base currency.
**Fix**: Use the account's currency as returned by `GET /accounts/{id}`.

### `account_frozen_debits_blocked`
**Cause**: Account is frozen. Credits are allowed; debits are not.
**Fix**: Unfreeze via `POST /accounts/{id}/unfreeze` with appropriate authority.

### `cannot_close_non_zero_balance`
**Cause**: Book balance is non-zero. Zero the balance before closing (debit remaining funds to another account via transfer).

### `cannot_close_account_with_active_locks`
**Cause**: Active fund locks exist. Release all locks first via `POST /accounts/{id}/locks/release`.

### `account_not_found:{id}` (404)
**Cause**: Wrong account ID or wrong tenant. Verify `X-Tenant-ID` matches the account's tenant.

### `duplicate_account_number`
**Cause**: You specified an `account_number` that already exists.
**Fix**: Omit `account_number` to have one auto-generated, or use a unique value.

### `product_not_found`
**Cause**: Unknown `product_code`. Available codes: `CURR001`, `SVGS001`, `USD001`.

## GL Circuit Breaker Open

If GL posting is failing, the circuit breaker opens after 5 consecutive failures. Transactions still proceed but the `gl_journal_id` will be `null` and a retry event is emitted.

**Check**: `GET /api/fin/acct/health` — inspect `gl_circuit_state`.

**Resolution**: Fix the GL service connectivity, then restart the service to reset the circuit.

## Balance Inconsistency

If `book_balance - locked_balance != available_balance - overdraft_available`, a bug in a concurrent update path has occurred.

**Recovery**:
1. Recalculate: `available = book - locked + max(0, overdraft_limit - overdraft_used)`
2. Patch via the service: `svc.accounts[account_id]["available_balance"] = str(recalculated)`

## Performance: Slow Transaction Listing

The in-memory store does a full scan. For accounts with >10,000 transactions, replace `self.transactions` with a PostgreSQL-backed repository with a B-tree index on `(account_id, posted_at)`.

## Tests Failing After Code Change

```bash
python -m pytest capabilities/fin/acct/tests/ -v --tb=short
```

Common fix: if `posted_at` is stored as a datetime object (not string), ensure the comparison in `get_transactions` uses `isinstance(posted_raw, datetime)` branching (already implemented in service.py).
