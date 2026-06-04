# APG Point of Sale — Troubleshooting Guide
© 2025 Datacraft | www.datacraft.co.ke

## Common Issues

### "terminal already has an open session"
Cause: A session is still open on the terminal (possibly from a crash).
Fix:
1. `GET /sessions?terminal_id=<id>&status=open` to find it.
2. Force-close: `POST /sessions/<id>/close {"closing_cash": 0}`.
3. If the session is corrupt, use `POST /sessions/<id>/suspend` first, then close.

### "session must be open" on transaction creation
Cause: Session is suspended, closed, or reconciled.
Fix: Resume with `POST /sessions/<id>/resume`, or open a new session.

### "underpaid / insufficient tender"
Cause: Sum of payments < grand_total.
Fix: Add the remaining balance as an additional payment in the `payments` array.

### "insufficient loyalty points"
Cause: Customer's loyalty balance < points requested.
Fix: `GET /loyalty/<customer_id>` to check balance. Reduce redemption amount.

### "loyalty_redemption_limit_exceeded"
Cause: Points value > 50% of transaction total.
Fix: Loyalty points can cover at most 50% of any transaction. Collect remainder by other method.

### "supervisor_required_for_price_override"
Cause: Price override attempted without supervisor_id.
Fix: Obtain supervisor PIN/ID and include `"supervisor_id"` in the override payload.

### "self_approval_denied"
Cause: cashier_id == supervisor_id in override request.
Fix: Use a different supervisor for approval.

### "eod_already_run"
Cause: EOD report already generated for this store/date.
Fix: Retrieve existing report: `GET /reports/eod/<store_id>/<date>`.

### "sync_sequence_not_monotone"
Cause: Offline sync batch has same or lower sequence number as a prior batch.
Fix: Increment `sync_sequence` by 1 for each new batch per terminal. Do not replay old batches.

### "cross_tenant_access_denied"
Cause: Request's X-Tenant-ID doesn't match the resource's tenant_id.
Fix: Ensure X-Tenant-ID header matches the tenant that owns the resource.

### Receipt rendered_content is None
Cause: Transaction not in COMPLETED status when receipt was generated.
Fix: Complete the transaction first (`POST /transactions/<id>/complete`), then generate receipt.

## Performance

- Use PostgreSQL for production — the in-memory store loses state on restart.
- Add tenant-specific partitions for tenants with >1M transactions/month.
- Index `session_id` on queries that list transactions by session.
- For high-volume stores, run EOD generation asynchronously (offload to a background worker).

## Monitoring

- `GET /health` returns `{"status": "ok"}` — use for liveness probes.
- `GET /contract` returns full capability metadata — use for dependency health checks.
- Log lines are structured: `pos | op=<op> tenant=<tenant> entity=<id>`.
- Cash variance > KES 50 triggers a `RuleViolation("cash_variance_exceeds_tolerance")` — log and alert.

## Offline Sync Checklist

1. Terminal must have `offline_capable: true`.
2. Each sync batch must increment `sync_sequence` by exactly 1.
3. Submit batches in order — gaps cause rejection.
4. Check `rejected` list in sync response for individual transaction failures.
5. Inventory deductions for offline sales are applied on sync, not at transaction creation.
