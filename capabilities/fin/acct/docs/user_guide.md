# Bank Account Management — User Guide

© 2025 Datacraft | `fin.acct` v1.0.0

## Overview

The Bank Account Management capability (`fin.acct`) manages the full regulatory lifecycle of bank accounts: opening, closing, freezing, dormancy management, credits, debits, internal transfers, fund locks, overdraft facilities, bulk payroll disbursement, and statement generation.

It is distinct from digital wallets — every account carries a generated IBAN, full audit trail, GL integration, and compliance controls.

---

## Quick Start

### Open an Account

```http
POST /api/fin/acct/accounts
X-Tenant-ID: your-tenant
Content-Type: application/json

{
  "customer_id": "cust-001",
  "product_code": "CURR001",
  "currency": "KES",
  "opening_deposit": "5000.00"
}
```

Response:
```json
{
  "data": {
    "id": "0191f3a2-...",
    "account_number": "ACME0000000001",
    "iban": "KE000000000000000001",
    "status": "active",
    "book_balance": "5000.00",
    "available_balance": "5000.00",
    ...
  },
  "error": null
}
```

---

## Account Products

| Code | Name | Type | Currency | Overdraft |
|------|------|------|----------|-----------|
| `CURR001` | Standard Current Account | current | KES | Up to 50,000 |
| `SVGS001` | Standard Savings Account | savings | KES | No |
| `USD001` | USD Current Account | current | USD | No |

---

## Account Lifecycle

```
PENDING → ACTIVE → FROZEN → ACTIVE
                 → DORMANT → ACTIVE
                 → CLOSED
```

### Status Rules

| Status | Credits | Debits | Notes |
|--------|---------|--------|-------|
| active | ✓ | ✓ | Normal operation |
| frozen | ✓ | ✗ | Debits blocked; credits allowed |
| dormant | ✓ | ✓ | Reactivated on first transaction |
| closed | ✗ | ✗ | Terminal state |

---

## Key Operations

### Credit Account
```http
POST /api/fin/acct/accounts/{account_id}/credit
{ "amount": "1000.00", "currency": "KES", "reference": "SAL-001", "description": "Salary" }
```

### Debit Account
```http
POST /api/fin/acct/accounts/{account_id}/debit
{ "amount": "500.00", "currency": "KES", "reference": "WDW-001", "description": "ATM withdrawal" }
```

### Internal Transfer
```http
POST /api/fin/acct/accounts/{from_id}/transfer
{ "to_account_id": "...", "amount": "2000.00", "reference": "TXF-001", "description": "Savings sweep" }
```

### Lock Funds (for pending payments, guarantees)
```http
POST /api/fin/acct/accounts/{account_id}/locks
{ "amount": "500.00", "lock_reference": "GUAR-001", "reason": "Bank guarantee" }
```

### Release Lock
```http
POST /api/fin/acct/accounts/{account_id}/locks/release
{ "lock_reference": "GUAR-001" }
```

### Set Overdraft Limit
```http
PUT /api/fin/acct/accounts/{account_id}/overdraft
{ "limit": "25000.00", "approved_by": "credit-manager-id" }
```

### Bulk Credit (Payroll)
```http
POST /api/fin/acct/bulk-credit
{
  "credits": [
    { "account_id": "...", "amount": "45000.00", "reference": "PAY-2026-06", "description": "June salary" },
    ...
  ]
}
```

### Generate Statement
```http
POST /api/fin/acct/accounts/{account_id}/statement
{ "from_date": "2026-01-01", "to_date": "2026-06-30", "format": "json" }
```

---

## Balance Components

| Field | Meaning |
|-------|---------|
| `book_balance` | Ledger balance including locked funds |
| `available_balance` | Funds available for new debits: `book - locked + overdraft_available` |
| `locked_balance` | Funds reserved by active locks |
| `overdraft_used` | How much of the overdraft facility is drawn |

---

## Dormancy Management

Accounts with no activity for 180 days (configurable) are candidates for dormancy.

```http
GET /api/fin/acct/dormancy-candidates?days_inactive=180
```

Mark dormant:
```http
POST /api/fin/acct/accounts/{account_id}/dormant
```

Reactivate:
```http
POST /api/fin/acct/accounts/{account_id}/reactivate
```

---

## Joint Accounts / Signatories

```http
POST /api/fin/acct/accounts/{account_id}/signatories
{ "customer_id": "cust-002", "signing_authority": "joint_any" }
```

Signing authority options: `single`, `joint_any`, `joint_all`

---

## APG Platform Integration

- **Auth**: Pass `X-Tenant-ID` header on every request. Tenant isolation is strictly enforced.
- **Audit**: All state changes are recorded in `get_account_history`.
- **GL**: Every credit/debit posts a journal entry to the General Ledger (`fin.glr`) via the ACCT_EVENT_STREAM.
- **Events**: All state changes emit NATS events on `apg.fin.acct.lifecycle` for downstream consumers.
