# ussd_eng User Guide

## Overview

`ussd_eng` is the APG USSD Engine capability. It manages the full USSD session lifecycle: gateway registration (Africa's Talking and Safaricom), menu DSL definition, session state machine, multi-hop navigation, session variable persistence, and analytics.

## Use Cases

- Mobile money menu flows (check balance, send money, buy airtime)
- Customer self-service portals via USSD shortcodes
- Survey and data-collection flows on feature phones
- Multi-language USSD deployments
- A/B-testable menu trees (used via ussd_flo)

## Concepts

### Gateway
A registered gateway connects a shortcode (e.g. `*384#`) to a gateway provider (Africa's Talking or Safaricom). Each gateway tracks session counts and environment (sandbox / production).

### Menu DSL
Menus are defined as JSON objects with a `title`, `body`, and `items` list. Each item has an `action`:
- `navigate` — go to another menu (`target` = menu_id)
- `back` — return to the previous menu
- `execute` — call a registered Python handler
- `end` — end the session
- `input` — capture free-text input into a session variable (`target` = variable name)

Menu bodies support `{variable_name}` substitution from session variables.

### Session
A session is created on the first callback from the gateway. It tracks:
- Current menu position
- Full navigation history (breadcrumb)
- All user inputs
- Session variables (key-value store)
- Hop count (max 30 per session)
- Timeout state

### Handlers
Register Python callables that execute business logic during menu navigation:
```python
async def check_balance(session, item):
    return {"variables": {"balance": "KES 1,234.56"}, "next_menu": "balance_menu"}

await svc.register_handler("check_balance", check_balance)
```

## API Reference

### Callback Integration

**Africa's Talking** — POST to `/api/ussd/eng/callback?gateway=africastalking`

Expected fields: `sessionId`, `serviceCode`, `phoneNumber`, `text`

Response: plain text starting with `CON ` (continue) or `END ` (end session).

**Safaricom** — POST to `/api/ussd/eng/callback?gateway=safaricom`

Expected fields: `sessionId`, `serviceCode`, `msisdn`, `input`

Response: JSON `{"sessionID": ..., "responseType": ..., "responseMsg": ...}`

### Session Timeout
Call `POST /api/ussd/eng/sessions/expire` or invoke `expire_timed_out_sessions()` periodically to mark stale sessions as `timeout`.

### Analytics
- `/api/ussd/eng/analytics` — aggregate session stats (completion rate, avg hops)
- `/api/ussd/eng/analytics/menus` — per-menu visit counts
- `/api/ussd/eng/analytics/dropoff` — menus with highest session drop-off
