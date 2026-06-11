# Bank Account Management — API Reference

Base URL: `/api/fin/acct`
Auth: `X-Tenant-ID: <tenant>` header required on all endpoints.
Envelope: `{"data": ..., "error": null, "meta": null}`

## Accounts

| Method | Path | Description |
|--------|------|-------------|
| POST | `/accounts` | Open new account |
| GET | `/accounts` | List accounts (`?customer_id=&status=&account_type=`) |
| GET | `/accounts/{id}` | Get account by ID |
| GET | `/accounts/by-number/{number}` | Get account by account number |
| POST | `/accounts/{id}/close` | Close account (zero balance required) |
| POST | `/accounts/{id}/freeze` | Freeze account |
| POST | `/accounts/{id}/unfreeze` | Unfreeze account |
| POST | `/accounts/{id}/dormant` | Mark dormant |
| POST | `/accounts/{id}/reactivate` | Reactivate dormant account |

## Balances

| Method | Path | Description |
|--------|------|-------------|
| GET | `/accounts/{id}/balance` | Full balance breakdown |
| GET | `/accounts/{id}/balance/check?amount=X` | Check if sufficient funds |

## Transactions

| Method | Path | Description |
|--------|------|-------------|
| POST | `/accounts/{id}/credit` | Credit account |
| POST | `/accounts/{id}/debit` | Debit account |
| POST | `/accounts/{id}/transfer` | Internal transfer |
| GET | `/accounts/{id}/transactions` | List transactions (pagination: `?limit=50&page=1&from_date=&to_date=`) |
| GET | `/transactions/{id}` | Get transaction by ID |
| POST | `/accounts/{id}/statement` | Generate statement |

## Fund Locks

| Method | Path | Description |
|--------|------|-------------|
| POST | `/accounts/{id}/locks` | Lock funds |
| POST | `/accounts/{id}/locks/release` | Release lock by reference |

## Overdraft

| Method | Path | Description |
|--------|------|-------------|
| PUT | `/accounts/{id}/overdraft` | Set overdraft limit |

## Product

| Method | Path | Description |
|--------|------|-------------|
| GET | `/accounts/{id}/product` | Get account product |
| PUT | `/accounts/{id}/product` | Change product |

## Dormancy

| Method | Path | Description |
|--------|------|-------------|
| GET | `/dormancy-candidates?days_inactive=180` | List dormancy candidates |

## Bulk & Sweep

| Method | Path | Description |
|--------|------|-------------|
| POST | `/bulk-credit` | Bulk payroll credit |
| POST | `/accounts/{id}/sweep` | Sweep to linked savings |

## Signatories

| Method | Path | Description |
|--------|------|-------------|
| POST | `/accounts/{id}/signatories` | Add joint holder |
| GET | `/accounts/{id}/signatories` | List signatories |

## Audit & Reporting

| Method | Path | Description |
|--------|------|-------------|
| GET | `/accounts/{id}/history` | Full lifecycle audit |
| GET | `/accounts/{id}/summary/{period}` | Transaction summary (period: `YYYY-MM`) |
| GET | `/stats/{customer_id}` | Account stats for customer |

## Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health |

## Error Codes

| Code | HTTP | Meaning |
|------|------|---------|
| `missing_tenant` | 401 | X-Tenant-ID header missing |
| `not_found` | 404 | Account/transaction not found |
| `bad_request` | 400 | Validation failure (insufficient funds, wrong currency, etc.) |
| `internal_error` | 500 | Unexpected server error |
