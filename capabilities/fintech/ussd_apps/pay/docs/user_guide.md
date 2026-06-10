# Payment USSD App — User Guide

## Overview

The Payment USSD App capability (`fintech_ussd_pay`) delivers five payment verticals over USSD: bill payments, merchant payments, airtime top-ups, utility payments, and person-to-person send money. All operations work on any mobile phone without internet access.

## Use Cases

1. **Bill Payment** — Pay KPLC electricity, NHIF, KRA, NSSF, DStv, school fees via paybill numbers.
2. **Merchant Payment** — Pay at shops, supermarkets, fuel stations via till numbers.
3. **Airtime Top-up** — Buy airtime for yourself or another number (Safaricom, Airtel, Telkom, Faiba).
4. **Utility Payment** — Pay water and electricity bills; prepaid electricity generates a meter token.
5. **Send Money** — Transfer money to another phone; amounts >= KES 10,000 require confirmation.

## USSD Flow

Dial `*144#` to access the payment menu:

```
Welcome to PayUSSD
1. Pay Bill
2. Pay Merchant
3. Buy Airtime
4. Pay Utility
5. Send Money
0. Exit
```

### Pay Bill Flow

```
*144*1# → Enter Paybill number: 888880
*144*1*888880# → Enter account reference: 12345678
*144*1*888880*12345678# → Enter amount: 2000
*144*1*888880*12345678*2000# → Enter PIN: ****
→ END Kenya Power Prepaid payment of KES 2,000 complete. Receipt: AB1234567X
```

### Send Money with Confirmation

Amounts >= KES 10,000 trigger a confirmation step:

```
*144*5*0712345678*15000*1234# →
CON Confirm send KES 15000 to 254712345678
1. Confirm
2. Cancel

→ *144*5*0712345678*15000*1234*1# → END Sent KES 15,000. Receipt: XY9876543Z
```

## API Reference

### Pay Bill

```
POST /api/fintech/ussd/pay/bills
{
  "phone_number": "0712345678",
  "biller_code": "KPLC_PRE",
  "account_reference": "12345678",
  "amount": "2000.00",
  "pin": "1234",
  "narration": "Monthly electricity",
  "tenant_id": "default"
}
```

Response:
```json
{
  "id": "pay-bill-...",
  "biller_name": "Kenya Power Prepaid",
  "receipt_number": "AB1234567X",
  "status": "completed"
}
```

### Pay Merchant

```
POST /api/fintech/ussd/pay/merchants
{
  "phone_number": "0712345678",
  "merchant_till": "174379",
  "amount": "1500.00",
  "pin": "1234",
  "narration": "Groceries"
}
```

### Buy Airtime

```
POST /api/fintech/ussd/pay/airtime
{
  "phone_number": "0712345678",
  "recipient_phone": "0722000000",
  "amount": "100.00",
  "telco": "safaricom",
  "pin": "1234"
}
```

### Pay Utility (with token generation)

```
POST /api/fintech/ussd/pay/utilities
{
  "phone_number": "0712345678",
  "utility_code": "kplc_prepaid",
  "meter_number": "12345678901",
  "amount": "1000.00",
  "pin": "1234"
}
```

Response includes `token` and `units_purchased` for prepaid electricity.

### Initiate Send Money

```
POST /api/fintech/ussd/pay/send-money
{
  "from_phone": "0712345678",
  "to_phone": "0722000000",
  "amount": "15000.00",
  "pin": "1234",
  "narration": "Rent payment"
}
```

If `requires_confirmation: true`, confirm with:

```
POST /api/fintech/ussd/pay/send-money/{id}/confirm
{
  "pin": "1234",
  "tenant_id": "default"
}
```

### Payment History

```
GET /api/fintech/ussd/pay/history?phone_number=0712345678&tenant_id=default
```

Returns consolidated history across all payment types with totals.

### Daily Volume Report

```
GET /api/fintech/ussd/pay/volume/daily?date=2026-06-10&tenant_id=default
```

### Search Payments

```
GET /api/fintech/ussd/pay/search?phone_number=0712345678&payment_type=bill&date_from=2026-06-01&date_to=2026-06-10
```

Supported `payment_type` values: `bill`, `merchant`, `airtime`, `utility`, `send_money`.

## Built-in Billers

| Code | Name | Paybill |
|------|------|---------|
| KPLC_PRE | Kenya Power Prepaid | 888880 |
| KPLC_POST | Kenya Power Postpaid | 888882 |
| NWC | Nairobi Water | 888861 |
| KRA | Kenya Revenue Authority | 572572 |
| NHIF | National Hospital Insurance Fund | 200222 |
| NSSF | National Social Security Fund | 333200 |
| DStv | DStv Africa | 444200 |
| ZUKU | Zuku Internet | 100600 |
| SAFARICOM_POSTPAID | Safaricom Postpaid | 100200 |

## Error Codes

| Code | Meaning |
|------|---------|
| `biller_not_found` | Biller code not in registry |
| `amount_outside_biller_limits` | Amount below min or above max for biller |
| `unsupported_telco` | Telco not in: safaricom, airtel, telkom, faiba |
| `unsupported_utility_code` | Utility code not supported |
| `transaction_not_pending_confirmation` | Confirm called on non-pending transaction |
| `cannot_send_money_to_self` | from_phone and to_phone are identical |
| `phone_locked_too_many_pin_attempts` | 3 failed PIN attempts |
| `invalid_pin` | PIN mismatch; remaining attempts shown |
