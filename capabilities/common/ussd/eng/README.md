# ussd_eng — USSD Engine

USSD session state machine, Africa's Talking + Safaricom gateway integration, menu DSL, session persistence.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ussd/eng/health` | Service health |
| GET | `/api/ussd/eng/gateways` | List gateways |
| POST | `/api/ussd/eng/gateways` | Register gateway |
| GET | `/api/ussd/eng/gateways/<id>` | Get gateway |
| PUT | `/api/ussd/eng/gateways/<id>` | Update gateway |
| DELETE | `/api/ussd/eng/gateways/<id>` | Delete gateway |
| GET | `/api/ussd/eng/menus` | List menus |
| POST | `/api/ussd/eng/menus` | Create menu |
| GET | `/api/ussd/eng/menus/<id>` | Get menu |
| PUT | `/api/ussd/eng/menus/<id>` | Update menu |
| DELETE | `/api/ussd/eng/menus/<id>` | Delete menu |
| GET | `/api/ussd/eng/sessions` | List sessions |
| POST | `/api/ussd/eng/sessions` | Create session |
| GET | `/api/ussd/eng/sessions/<id>` | Get session |
| PUT | `/api/ussd/eng/sessions/<id>` | Update session |
| DELETE | `/api/ussd/eng/sessions/<id>` | Delete session |
| POST | `/api/ussd/eng/callback` | USSD gateway callback |
| GET | `/api/ussd/eng/analytics` | Session analytics |
| GET | `/api/ussd/eng/analytics/menus` | Menu visit analytics |
| GET | `/api/ussd/eng/analytics/dropoff` | Drop-off analysis |
| GET | `/api/ussd/eng/dashboard` | Summary dashboard |
| GET | `/api/ussd/eng/audit` | Audit event log |
