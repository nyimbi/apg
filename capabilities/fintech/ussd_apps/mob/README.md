# Mobile Banking USSD (fintech_ussd_mob)

USSD mobile banking: account balance, mini-statement, fund transfer, standing orders, PIN management, cross-border FX, beneficiaries, fraud scoring, spending analytics, audit chain verification, and personalised menus.

## Overview

Provides a complete USSD-driven mobile banking interface for East African telco networks. Handles Africa's Talking and similar USSD gateway protocols with PIN-secured operations, daily transfer limits, idempotent transfer processing, and multi-tenant service code routing.

## New in v2.0

| Feature | Description |
|---------|-------------|
| Beneficiary management | Save/list/remove named payees (12-char alias, 1-keystroke transfer) |
| Fraud velocity scoring | 0–100 risk score; auto-challenges or holds high-risk transfers |
| Idempotent transfers | 24-hour deduplication key prevents double-debit on gateway retries |
| Cross-border FX transfers | KES→UGX/TZS/RWF at configurable spread; rate fed from NATS |
| Spending analytics | Category breakdown (utilities/food/transport/…) with USSD-safe summary |
| Service code multi-tenancy | One APG instance hosts N bank brands via distinct `*NNN#` codes |
| Session token integrity | HMAC-SHA256 tokens bound to MSISDN + session_id, 5 min TTL |
| Audit chain verification | Merkle-chain hashing on every audit event; `/verify-chain` endpoint |
| Statement export | JSON, CSV, or summary format — QuickBooks/Xero compatible |
| Balance threshold alerts | `mob.balance.alert` NATS event when balance drops below threshold |
| Personalised USSD menu | Top-2 most-used items promoted; ~40% fewer keystrokes for repeat users |

## API Reference

### Core Accounts

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/fintech/ussd/mob/health` | Service health |
| GET | `/api/fintech/ussd/mob/describe` | Capability metadata |
| GET | `/api/fintech/ussd/mob/accounts` | List accounts |
| POST | `/api/fintech/ussd/mob/accounts` | Create account |
| GET | `/api/fintech/ussd/mob/accounts/<id>` | Get account by ID |
| PUT | `/api/fintech/ussd/mob/accounts/<id>` | Update account |
| DELETE | `/api/fintech/ussd/mob/accounts/<id>` | Close account |
| POST | `/api/fintech/ussd/mob/accounts/<no>/balance` | Balance enquiry (PIN) |
| POST | `/api/fintech/ussd/mob/accounts/<no>/deposit` | Credit account |
| POST | `/api/fintech/ussd/mob/accounts/<no>/withdraw` | Debit account (PIN) |
| POST | `/api/fintech/ussd/mob/accounts/<no>/statement` | Mini-statement (PIN) |
| POST | `/api/fintech/ussd/mob/accounts/<no>/full-statement` | Full statement with date range |
| GET | `/api/fintech/ussd/mob/accounts/<no>/summary` | Account summary (PIN) |
| POST | `/api/fintech/ussd/mob/accounts/<no>/lock` | Lock account |
| POST | `/api/fintech/ussd/mob/accounts/<no>/unlock` | Unlock account |
| PUT | `/api/fintech/ussd/mob/accounts/<no>/daily-limit` | Update daily limit |
| POST | `/api/fintech/ussd/mob/accounts/<no>/alert-threshold` | Set balance alert threshold |
| GET | `/api/fintech/ussd/mob/accounts/<no>/alert-check` | Check balance alert |

### Transfers

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/fintech/ussd/mob/transfers` | List transfers |
| POST | `/api/fintech/ussd/mob/transfers` | Create fund transfer |
| POST | `/api/fintech/ussd/mob/transfers/idempotent` | Idempotent transfer |
| POST | `/api/fintech/ussd/mob/transfers/fx` | Cross-border FX transfer |
| GET | `/api/fintech/ussd/mob/transfers/<id>` | Get transfer |
| POST | `/api/fintech/ussd/mob/transfers/<id>/reverse` | Reverse transfer |

### Standing Orders

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/fintech/ussd/mob/standing-orders` | List standing orders |
| POST | `/api/fintech/ussd/mob/standing-orders` | Create standing order |
| GET | `/api/fintech/ussd/mob/standing-orders/<id>` | Get standing order |
| PUT | `/api/fintech/ussd/mob/standing-orders/<id>` | Update standing order |
| DELETE | `/api/fintech/ussd/mob/standing-orders/<id>` | Cancel standing order |
| POST | `/api/fintech/ussd/mob/standing-orders/<id>/execute` | Execute (scheduler) |

### Beneficiaries

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/fintech/ussd/mob/accounts/<no>/beneficiaries` | List beneficiaries (PIN) |
| POST | `/api/fintech/ussd/mob/accounts/<no>/beneficiaries` | Add beneficiary (PIN) |
| DELETE | `/api/fintech/ussd/mob/accounts/<no>/beneficiaries/<alias>` | Remove beneficiary (PIN) |

### PIN Management

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/fintech/ussd/mob/pin/change` | Change PIN |
| POST | `/api/fintech/ussd/mob/pin/reset/otp` | Request PIN reset OTP |
| POST | `/api/fintech/ussd/mob/pin/reset` | Reset PIN via OTP |

### USSD Gateway

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/fintech/ussd/mob/ussd` | Process USSD request |
| GET | `/api/fintech/ussd/mob/ussd/sessions` | List sessions |
| GET | `/api/fintech/ussd/mob/ussd/sessions/<id>` | Get session |
| GET | `/api/fintech/ussd/mob/ussd/menu` | Personalised menu |
| POST | `/api/fintech/ussd/mob/service-codes` | Register service code |
| GET | `/api/fintech/ussd/mob/service-codes` | List service codes |

### Analytics & Compliance

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/fintech/ussd/mob/accounts/<no>/insights` | Spending insights (PIN) |
| POST | `/api/fintech/ussd/mob/accounts/<no>/statement/export` | Export statement (json/csv/summary) |
| GET | `/api/fintech/ussd/mob/audit-events` | Audit event log |
| GET | `/api/fintech/ussd/mob/audit-events/verify-chain` | Verify Merkle audit chain |
| GET | `/api/fintech/ussd/mob/statistics` | Tenant statistics |
| POST | `/api/fintech/ussd/mob/fraud/score` | Score transfer fraud risk |
| PUT | `/api/fintech/ussd/mob/fx/rates` | Update FX rate |

## Limits

| Limit | Value |
|-------|-------|
| Single transfer | KES 150,000 |
| Daily transfer | KES 500,000 |
| Max failed PIN attempts | 3 (then account locks) |
| High-value transfer threshold | KES 50,000 (TOTP challenge) |
| Fraud score — hold transfer | ≥ 75 |
| Fraud score — TOTP challenge | 50–74 |
| Max beneficiaries per account | 20 |
| Beneficiary alias max length | 12 chars |
| Standing order frequencies | daily, weekly, monthly, quarterly |
| Supported currencies | KES, USD, EUR, GBP, UGX, TZS, RWF |
| Idempotency cache TTL | 24 hours |
| Session token TTL | 300 seconds |
| Transaction history per account | 200 entries |

## Event Bus (NATS)

All material operations publish to NATS subjects when a `NATSPublisher` is injected:

| Subject | Trigger |
|---------|---------|
| `mob.transfer.completed` | Successful fund transfer |
| `mob.transfer.reversed` | Transfer reversal |
| `mob.fx_transfer.completed` | Cross-border FX transfer |
| `mob.balance.alert` | Balance drops below threshold |
| `mob.standing_order.executed` | Scheduled order executed |
| `mob.standing_order.failed` | Order failed (insufficient funds) |
| `mob.fraud.score_computed` | Fraud risk scored |
| `mob.pin.changed` | PIN changed |
| `mob.account.locked` | Account locked |

Downstream processing uses **bytewax** pipelines subscribing to `mob.*`.

## Composability

The service exposes a `describe()` coroutine returning a machine-readable capability manifest. It composes with:

- `capabilities/intel/alerts` — route `mob.balance.alert` to SMS/push
- `capabilities/intel/correlation` — join fraud events with transaction streams
- `capabilities/intel/prediction` — predict churn from balance velocity
- `capabilities/fintech/terminal/terramoni` — display account summaries in terminal UI
