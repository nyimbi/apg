# Tax Administration — API Reference

© 2025 Datacraft | Author: Nyimbi Odero

Base URL: `/api/v1/tax`

All requests require `X-Tenant-ID` header. All responses use the envelope `{"data": ..., "meta": {...}}`.

## Authentication Headers

| Header | Required | Default |
|--------|----------|---------|
| `X-Tenant-ID` | Yes | `"default"` |
| `X-Actor-ID` | No | `"system"` |

## Error Codes

| HTTP | Meaning |
|------|---------|
| 400 | Bad request / missing field |
| 403 | Policy denial (PermissionError) |
| 404 | Resource not found |
| 422 | Validation failure (AssertionError) |
| 500 | Internal error |

---

## Taxpayers

| Method | Path | Description |
|--------|------|-------------|
| GET | `/taxpayers` | List taxpayers. Params: `q`, `search_type`, `limit`, `offset` |
| POST | `/taxpayers` | Register taxpayer |
| GET | `/taxpayers/{tin}` | Get by PIN |
| PUT | `/taxpayers/{tin}` | Update fields |
| DELETE | `/taxpayers/{tin}` | Deregister (soft delete) |
| GET | `/taxpayers/{tin}/verify` | Verify TIN format + existence |
| GET | `/taxpayers/{tin}/compliance-risk` | Compliance risk profile |

## Returns

| Method | Path | Description |
|--------|------|-------------|
| GET | `/returns` | List returns. Params: `tin`, `status`, `limit`, `offset` |
| POST | `/returns` | File return |
| POST | `/returns/nil` | File nil return |
| GET | `/returns/{id}` | Get return |
| PUT | `/returns/{id}` | Amend return |
| POST | `/returns/{id}/validate` | Validate consistency |
| GET | `/returns/status` | Check filing status. Params: `tin`, `tax_type`, `period` |

## Assessments

| Method | Path | Description |
|--------|------|-------------|
| GET | `/assessments` | List assessments |
| POST | `/assessments` | Create assessment |
| GET | `/assessments/{id}` | Get assessment |
| PUT | `/assessments/{id}` | Update assessment |
| POST | `/assessments/{id}/penalty-interest` | Calculate penalty + interest |

## Payments

| Method | Path | Description |
|--------|------|-------------|
| GET | `/payments` | List payments |
| POST | `/payments` | Record payment |
| GET | `/payments/{id}` | Get payment |
| POST | `/payments/{id}/allocate` | Allocate to debts (FIFO) |

## Debts

| Method | Path | Description |
|--------|------|-------------|
| GET | `/debts` | List debts |
| GET | `/debts/{id}` | Get debt |
| POST | `/debts/demand-notice` | Issue demand notice |
| POST | `/debts/collection-action` | Initiate collection |

## Audits

| Method | Path | Description |
|--------|------|-------------|
| GET | `/audits` | List audits |
| POST | `/audits` | Open audit case |
| GET | `/audits/{id}` | Get audit |
| PUT | `/audits/{id}` | Update audit |
| POST | `/audits/{id}/findings` | Record findings |
| POST | `/audits/{id}/close` | Close audit |

## Objections

| Method | Path | Description |
|--------|------|-------------|
| GET | `/objections` | List objections |
| POST | `/objections` | File objection |
| GET | `/objections/{id}` | Get objection |
| POST | `/objections/{id}/determine` | Determine (upheld/dismissed) |

## Appeals

| Method | Path | Description |
|--------|------|-------------|
| GET | `/appeals` | List appeals |
| POST | `/appeals` | File appeal |
| GET | `/appeals/{id}` | Get appeal |

## Refunds

| Method | Path | Description |
|--------|------|-------------|
| GET | `/refunds` | List refunds |
| POST | `/refunds` | Apply for refund |
| GET | `/refunds/{id}` | Get refund |
| POST | `/refunds/{id}/review` | Assign reviewer |
| POST | `/refunds/{id}/approve` | Approve refund |

## Clearance Certificates

| Method | Path | Description |
|--------|------|-------------|
| GET | `/clearances` | List certificates |
| POST | `/clearances` | Request certificate |
| GET | `/clearances/{id}` | Get certificate |
| GET | `/clearances/verify/{cert_number}` | Verify certificate validity |

## Exchange of Information

| Method | Path | Description |
|--------|------|-------------|
| POST | `/eoi` | Submit EOI request |
| GET | `/eoi` | List EOI requests |

## Reports

| Method | Path | Description |
|--------|------|-------------|
| GET | `/reports/dashboard` | KPI dashboard |
| GET | `/reports/revenue` | Revenue collection. Param: `period` |
| GET | `/reports/compliance` | Compliance rate. Param: `period`, `sector` |
| GET | `/reports/delinquency` | Debt aging. Param: `as_of` |
| GET | `/reports/audits` | Audit analytics. Param: `period` |
| GET | `/reports/refunds` | Refund analytics. Param: `period` |

## Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Capability health check |
