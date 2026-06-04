# Tax Administration — Troubleshooting Guide

© 2025 Datacraft | Author: Nyimbi Odero

---

## Common Issues

### `PermissionError: duplicate_pin_denied`
A taxpayer with the same PIN already exists in this tenant. Use `GET /api/v1/tax/taxpayers/{tin}/verify` to confirm existence before re-registering.

### `PermissionError: objection_deadline_passed`
The objection was filed more than 30 days after the assessment date. Check `assessment_date` on the assessment record. In legacy interface calls, ensure `within_deadline=True`.

### `PermissionError: demand_notice_required`
A demand notice must be issued (`POST /api/v1/tax/debts/demand-notice`) before initiating any collection action. The `demand_notice_reference` field must be non-empty.

### `AssertionError: taxpayer not found: {tin}`
The PIN does not exist in the tenant. Check `X-Tenant-ID` header matches the tenant the taxpayer was registered in. PINs are tenant-scoped.

### `AssertionError: return not found: {id}`
The return ID is not in the current tenant's store. Check tenant header.

### `AssertionError: objection only valid after dismissed or partially_upheld objection`
Appeals require an objection in status `dismissed` or `partially_upheld`. Check the objection status before filing an appeal.

### Clearance Certificate returns `status: rejected`
The taxpayer has outstanding debt. Use `GET /api/v1/tax/debts?status=outstanding` to list and resolve debts before requesting a clearance certificate.

### `validate_batch` raises `PermissionError`
Only the `bytewax` event stream is supported. Pass `event_stream="bytewax"` (or omit, it defaults to `bytewax`).

---

## Test Failures

### `sys.modules` pollution between tests
The `conftest.py` pre-loads `capability_contract` and `models` before any test runs. If running tests in isolation, ensure `conftest.py` is on the path.

### `292 passed` expected but fewer seen
Run `python -m pytest tests/ -v` and check for import errors. Ensure `uuid6` is installed: `pip install uuid6`.

---

## Performance

- In-process `_Store` is O(n) for tenant scans. For >10k records per tenant, replace with a DB-backed store using `SELECT WHERE tenant_id = $1` queries.
- `delinquency_report` and `compliance_rate_report` scan all records. Materialize results into `tax.compliance_risk_profiles` table on a schedule for production use.

---

## Multi-Tenancy

Every API operation reads `X-Tenant-ID` from the request header. If the header is missing, it defaults to `"default"`. Ensure all production requests include the correct tenant header — there is no server-side session-based tenant resolution.
