# Payment USSD App (fintech_ussd_pay)

USSD payments: bill pay, merchant payment, airtime top-up, utility pay, send money confirmation.

## Overview

Provides a comprehensive USSD-based payment interface covering East African payment use cases. Supports 9 pre-loaded billers (KPLC, Nairobi Water, KRA, NHIF, NSSF, DStv, Zuku, Safaricom Postpaid) and populates with tenant-specific billers.

Large send-money transactions (>= KES 10,000) require a second-factor confirmation step, matching telco USSD UX conventions.

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/fintech/ussd/pay/health` | Service health check |
| GET | `/api/fintech/ussd/pay/describe` | Capability metadata |
| GET | `/api/fintech/ussd/pay/billers` | List billers |
| POST | `/api/fintech/ussd/pay/billers` | Register biller |
| GET | `/api/fintech/ussd/pay/billers/<id>` | Get biller |
| PUT | `/api/fintech/ussd/pay/billers/<id>` | Update biller |
| DELETE | `/api/fintech/ussd/pay/billers/<id>` | Deactivate biller |
| GET | `/api/fintech/ussd/pay/bills` | List bill payments |
| POST | `/api/fintech/ussd/pay/bills` | Pay a bill |
| GET | `/api/fintech/ussd/pay/bills/<id>` | Get bill payment |
| POST | `/api/fintech/ussd/pay/bills/<id>/reverse` | Reverse bill payment |
| GET | `/api/fintech/ussd/pay/merchants` | List merchant payments |
| POST | `/api/fintech/ussd/pay/merchants` | Pay merchant |
| GET | `/api/fintech/ussd/pay/merchants/<id>` | Get merchant payment |
| GET | `/api/fintech/ussd/pay/airtime` | List airtime top-ups |
| POST | `/api/fintech/ussd/pay/airtime` | Buy airtime |
| GET | `/api/fintech/ussd/pay/airtime/<id>` | Get airtime top-up |
| GET | `/api/fintech/ussd/pay/utilities` | List utility payments |
| POST | `/api/fintech/ussd/pay/utilities` | Pay utility |
| GET | `/api/fintech/ussd/pay/utilities/<id>` | Get utility payment |
| GET | `/api/fintech/ussd/pay/send-money` | List send money transactions |
| POST | `/api/fintech/ussd/pay/send-money` | Initiate send money |
| GET | `/api/fintech/ussd/pay/send-money/<id>` | Get transaction |
| POST | `/api/fintech/ussd/pay/send-money/<id>/confirm` | Confirm transaction |
| POST | `/api/fintech/ussd/pay/send-money/<id>/cancel` | Cancel transaction |
| POST | `/api/fintech/ussd/pay/ussd` | Process USSD request |
| GET | `/api/fintech/ussd/pay/history` | Payment history for phone |
| GET | `/api/fintech/ussd/pay/statistics` | Tenant statistics |
| GET | `/api/fintech/ussd/pay/volume/daily` | Daily volume report |
| GET | `/api/fintech/ussd/pay/search` | Search payments |
| GET | `/api/fintech/ussd/pay/audit-events` | Audit event log |

## Business Rules

- Send money >= KES 10,000 requires explicit confirmation step
- Airtime: KES 5 minimum, KES 10,000 maximum per transaction
- KPLC Prepaid payments generate an electricity token (5-group numeric)
- Bill amounts validated against per-biller min/max limits
- Telcos: safaricom, airtel, telkom, faiba
- Utility codes: kplc_prepaid, kplc_postpaid, nairobi_water, mombasa_water, kisumu_water, nwsc
