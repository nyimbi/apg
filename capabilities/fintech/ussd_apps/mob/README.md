# Mobile Banking USSD (fintech_ussd_mob)

USSD mobile banking: account balance, mini-statement, fund transfer, standing orders, PIN management.

## Overview

Provides a complete USSD-driven mobile banking interface for East African telco networks. Handles ATK (Africa's Talking) and similar USSD gateway protocols with PIN-secured operations and daily transfer limits.

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/fintech/ussd/mob/health` | Service health check |
| GET | `/api/fintech/ussd/mob/describe` | Capability metadata |
| GET | `/api/fintech/ussd/mob/accounts` | List accounts |
| POST | `/api/fintech/ussd/mob/accounts` | Create account |
| GET | `/api/fintech/ussd/mob/accounts/<id>` | Get account by ID |
| PUT | `/api/fintech/ussd/mob/accounts/<id>` | Update account |
| DELETE | `/api/fintech/ussd/mob/accounts/<id>` | Close account |
| POST | `/api/fintech/ussd/mob/accounts/<no>/balance` | Balance enquiry (PIN required) |
| POST | `/api/fintech/ussd/mob/accounts/<no>/deposit` | Credit account |
| POST | `/api/fintech/ussd/mob/accounts/<no>/statement` | Mini-statement (PIN required) |
| GET | `/api/fintech/ussd/mob/transfers` | List transfers |
| POST | `/api/fintech/ussd/mob/transfers` | Create fund transfer |
| GET | `/api/fintech/ussd/mob/transfers/<id>` | Get transfer |
| POST | `/api/fintech/ussd/mob/transfers/<id>/reverse` | Reverse transfer |
| GET | `/api/fintech/ussd/mob/standing-orders` | List standing orders |
| POST | `/api/fintech/ussd/mob/standing-orders` | Create standing order |
| GET | `/api/fintech/ussd/mob/standing-orders/<id>` | Get standing order |
| PUT | `/api/fintech/ussd/mob/standing-orders/<id>` | Update standing order |
| DELETE | `/api/fintech/ussd/mob/standing-orders/<id>` | Cancel standing order |
| POST | `/api/fintech/ussd/mob/pin/change` | Change PIN |
| POST | `/api/fintech/ussd/mob/pin/reset/otp` | Request PIN reset OTP |
| POST | `/api/fintech/ussd/mob/pin/reset` | Reset PIN via OTP |
| POST | `/api/fintech/ussd/mob/ussd` | Process USSD request |
| GET | `/api/fintech/ussd/mob/audit-events` | Audit event log |
| GET | `/api/fintech/ussd/mob/statistics` | Tenant statistics |

## Limits

- Single transfer: KES 150,000
- Daily transfer: KES 500,000
- Max failed PIN attempts: 3 (then account locks)
- Standing order frequencies: daily, weekly, monthly, quarterly
- Supported currencies: KES, USD, EUR, GBP, UGX, TZS, RWF
