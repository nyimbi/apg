# Mobile Banking USSD — User Guide

## Overview

The Mobile Banking USSD capability (`fintech_ussd_mob`) delivers a full-featured mobile banking experience via USSD, enabling customers to perform banking operations from any mobile phone without requiring internet access or a smartphone.

## Use Cases

1. **Account Balance Enquiry** — Check current and available balance via PIN-secured USSD session.
2. **Mini-Statement** — View last 5 transactions without visiting a branch.
3. **Fund Transfer** — Send money to another account within daily limits (KES 500,000/day, KES 150,000/transaction).
4. **Standing Orders** — Set up recurring payments (daily/weekly/monthly/quarterly) that execute automatically.
5. **PIN Management** — Change PIN, reset via OTP, and recover locked accounts.

## USSD Flow

Dial `*123#` to access the main menu:

```
Welcome to MobBank
1. Account Balance
2. Mini Statement
3. Fund Transfer
4. Standing Orders
5. Change PIN
0. Exit
```

Customers navigate by entering the menu number and pressing send. Sessions are stateful and identified by `session_id` from the telco gateway.

## API Reference

### Create Account

```
POST /api/fintech/ussd/mob/accounts
{
  "phone_number": "0712345678",
  "account_number": "1234567890",
  "account_type": "savings",
  "customer_name": "Jane Doe",
  "national_id": "12345678",
  "pin": "1234",
  "currency": "KES",
  "tenant_id": "default"
}
```

### Balance Enquiry

```
POST /api/fintech/ussd/mob/accounts/1234567890/balance
{
  "pin": "1234",
  "tenant_id": "default"
}
```

### Fund Transfer

```
POST /api/fintech/ussd/mob/transfers
{
  "from_account": "1234567890",
  "to_account": "0987654321",
  "amount": "5000.00",
  "pin": "1234",
  "narration": "School fees",
  "currency": "KES",
  "tenant_id": "default"
}
```

### Create Standing Order

```
POST /api/fintech/ussd/mob/standing-orders
{
  "from_account": "1234567890",
  "to_account": "0987654321",
  "amount": "3000.00",
  "frequency": "monthly",
  "start_date": "2026-07-01",
  "pin": "1234",
  "narration": "Rent",
  "tenant_id": "default"
}
```

### USSD Gateway Integration

```
POST /api/fintech/ussd/mob/ussd
Content-Type: application/x-www-form-urlencoded

sessionId=SESSION123&phoneNumber=0712345678&serviceCode=*123%23&text=1
```

Returns plain text response (`CON ...` for continuing sessions, `END ...` for terminal responses).

### PIN Change

```
POST /api/fintech/ussd/mob/pin/change
{
  "account_number": "1234567890",
  "old_pin": "1234",
  "new_pin": "5678",
  "confirm_pin": "5678",
  "tenant_id": "default"
}
```

## Security

- All sensitive operations require PIN verification
- PIN is stored as SHA-256 hash — never in plain text
- 3 failed PIN attempts lock the account; admin unlock required
- OTP-based PIN reset expires in 5 minutes
- Daily transfer limits enforced per account, reset at midnight UTC

## Error Codes

| Code | Meaning |
|------|---------|
| `invalid_pin` | PIN did not match; N attempts remaining shown |
| `account_locked_too_many_pin_attempts` | Account locked after 3 failures |
| `insufficient_funds` | Available balance below transfer amount |
| `daily_transfer_limit_exceeded` | Daily KES 500,000 limit reached |
| `exceeds_single_transfer_limit_150000` | Single transfer above KES 150,000 |
| `otp_expired` | OTP older than 5 minutes |
| `otp_already_used` | OTP has already been consumed |
| `account_number_already_exists` | Duplicate account number for tenant |
