# APG Digital Payments — Troubleshooting Guide

## Common Issues

### M-Pesa STK Push not received by customer

**Symptoms**: Request returns `201` but customer never sees PIN prompt.

**Causes / Fixes**:
1. Phone not registered on Safaricom — verify MSISDN is active
2. Short code not whitelisted for STK — check Daraja portal
3. Phone in offline/no-signal area — retry after delay
4. Callback URL unreachable — ensure HTTPS endpoint is publicly accessible

**Check**: `GET /transactions/<id>` — if `status=initiated` after 30s, assume timeout and retry with same `idempotency_key`

---

### "mpesa_invalid_phone" error

**Cause**: Phone not in E.164 format `254XXXXXXXXX`.

**Fix**: Normalize before calling API:
```python
# 0712345678 → 254712345678
# +254712345678 → 254712345678
phone = re.sub(r"^(\+254|0)", "254", phone.strip())
```

---

### "duplicate_payment_detected" (422)

**Cause**: Same `idempotency_key` used for different payment.

**Fix**: The existing payment is returned — check `data.id` in response. If intentional retry, use same key; if new payment, use new key.

---

### "mpesa_insufficient_float" (422)

**Cause**: Agent/merchant float balance too low for B2C payout.

**Fix**: Top up M-Pesa agent float via Safaricom portal before retrying. Monitor float balance via `GET /merchants/<id>/report`.

---

### "raw_pan_storage_forbidden" (422/403)

**Cause**: `card_token` looks like a raw 16-digit PAN.

**Fix**: Tokenise card before calling this API. Integrate with a PCI-DSS vault (e.g. Stripe, Basis Theory, AWS Payment Cryptography). Never store or transmit raw card numbers.

---

### "3ds_required_for_high_value" (422)

**Cause**: Card payment >KES 10,000 without 3D Secure result.

**Fix**: Complete 3DS challenge with your card processor and pass `three_ds_result` in the authorise request.

---

### "reversal_window_expired" (422)

**Cause**: Wrong-number reversal attempted >24h after transaction.

**Fix**: Raise a dispute instead: `POST /transactions/<id>/dispute` with reason `wrong_number`. Dispute allows longer resolution window.

---

### "kyc_daily_limit_exceeded" (422)

**Cause**: Customer has exceeded their CBK daily transaction limit.

**Fix**:
1. Inform customer of limit
2. Advise KYC upgrade path (Basic → Standard → Full KYC)
3. Customer can transact again tomorrow

---

### Settlement variance

**Symptoms**: `reconcile_settlement` returns `status: variance`

**Diagnosis**:
```bash
GET /settlement/<id>/reconcile
# Check variance_amount and reconciliation_records
```

**Causes**:
- Network timeout caused duplicate — check `payment_idempotency_keys`
- FX rate applied at different times — use settlement FX snapshot
- Bank holiday float delay — mark as `partial_settlement` and process next cycle

---

### Performance: slow transaction queries

**Diagnosis**: Check if query is hitting the correct partition:
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM payment_transactions
WHERE tenant_id = 'acme' AND created_at > '2025-12-01';
```

**Fix**: Ensure `created_at` range filter is always included. Add composite index if querying by `status + method` frequently:
```sql
CREATE INDEX CONCURRENTLY idx_txn_status_method
ON payment_transactions (tenant_id, status, method, created_at DESC);
```

---

### Webhook not firing

**Symptoms**: `fire_webhook` returns success but endpoint not receiving.

**Check**:
1. Endpoint URL uses HTTPS (HTTP rejected at registration)
2. Endpoint responds with 2xx within 30s
3. HMAC-SHA256 signature verification not rejecting
4. Check `payment_notifications` table for `sent=FALSE` records

**Signature verification**:
```python
import hmac, hashlib
expected = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
assert hmac.compare_digest(expected, received_signature)
```

---

### "cross_tenant_access_denied" (403)

**Cause**: `X-Tenant-ID` header doesn't match resource's `tenant_id`.

**Fix**: Ensure `X-Tenant-ID` header matches the tenant that created the resource. Multi-tenant applications must scope all requests to the correct tenant.

---

### Import error: "attempted relative import with no known parent package"

**Cause**: Running tests directly from the payments directory instead of repo root.

**Fix**: Run from repo root:
```bash
# Correct
python -m pytest capabilities/fintech/payments/tests/ -v

# Wrong (will fail)
cd capabilities/fintech/payments && python -m pytest tests/
```

---

## Monitoring

Key metrics to watch:
- `payment.failed` event rate > 5% → investigate provider connectivity
- `dispute.opened` rate > 0.1% → review fraud rules
- Settlement variance > 10 bps → escalate to finance team
- `mpesa_insufficient_float` errors → trigger float top-up alert
- Batch `failed` count > 0 → review individual failures in `validation_errors` field

## Log patterns

```
txn=<id> status=completed amount=1000 KES      ← successful payment
txn=<id> status=failed amount=500 KES          ← failed payment
batch=<id> count=100 total=500000              ← batch summary
```
