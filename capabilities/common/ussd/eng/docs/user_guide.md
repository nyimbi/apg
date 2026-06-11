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

---

## v1.1 Feature Reference

### Session Resumption

USSD sessions on 2G/2.5G drop frequently. `resume_session` re-activates the most recent timed-out session for a phone number if it fell within a grace window, preserving all session variables and menu position.

```python
try:
    session = await svc.resume_session(
        phone_number="+254712345678",
        service_code="*384#",
        grace_seconds=90,   # default; M-Pesa uses 90 s
    )
    # hop_count, variables, and current_menu all preserved
except KeyError:
    session = await svc.create_session(phone_number="+254712345678", service_code="*384#")
```

The audit log records a `session_resumed` event with `hop_count_at_resume` for billing reconciliation.

---

### Idempotent Handler Execution

GSM DTAP retransmission can deliver the same USSD callback twice within milliseconds. `execute_idempotent` computes a SHA-256 key over `session_id:hop_count:handler_name` and returns the cached result for duplicates. All monetary fields (`amount`, `balance`, `total`, `fee`, `charge`) are coerced to `Decimal` in the returned result.

```python
result = await svc.execute_idempotent(
    session_id=session["id"],
    hop_count=session["hop_count"],
    handler_name="send_money",
    payload={"recipient": "+254799000001", "amount": "1000"},
)
# result["idempotent"] == True means a duplicate was detected
# result["result"]["amount"] is always Decimal
```

---

### Rate Limiting

`check_rate_limit` uses a per-process sliding-window counter to cap sessions per phone+service. Call it before `create_session`.

```python
rl = await svc.check_rate_limit(
    phone_number="+254712345678",
    service_code="*384#",
    window_seconds=3600,
    max_sessions=10,
)
# {"allowed": True, "remaining": 9, "count": 1, "reset_at": "2026-06-11T10:00:00Z"}
```

When `allowed` is `False` the engine emits a `rate_limit_exceeded` audit event and no session timestamp is added to the bucket.

---

### Input Validation Schema

Attach a schema to any `input` menu item and call `validate_input_against_schema` from the handler before processing. Supported types: `str`, `int`, `decimal`/`amount`, `phone`, `pin`.

```python
schema = {
    "type": "decimal",
    "min_value": "10",
    "max_value": "70000",
    "required": True,
}
v = await svc.validate_input_against_schema(user_input, schema)
if not v["valid"]:
    # Render error back to user on the same menu
    return {"variables": {"error": v["error_message"]}, "next_menu": "amount_input"}
amount = v["coerced"]  # Decimal("1500.00")
```

---

### Menu Versioning and Rollback

Take a snapshot before any menu update. Roll back instantly if a bad deploy reaches production.

```python
# Before update
snap = await svc.create_menu_version("loan_menu", "*123#")
# snap == {"menu_id": "loan_menu", "version": 3, "snapshotted_at": "..."}

# Deploy fails — rollback
await svc.rollback_menu("loan_menu", "*123#", version=3)
```

Up to 20 versions are retained per menu. Rollback is atomic: the in-memory live entry is replaced in one assignment.

---

### Dead-Letter Queue

When a handler raises an exception, call `queue_dead_letter` immediately to preserve the full context:

```python
try:
    result = await svc.execute_idempotent(...)
except Exception as exc:
    await svc.queue_dead_letter(
        session_id=session["id"],
        handler_name="debit_account",
        payload=payload,
        error=str(exc),
    )
    # Return graceful error to the user
```

Query the queue for ops dashboards:

```python
pending = await svc.get_dead_letters(handler_name="debit_account", status="pending", limit=20)
```

Each entry contains `session_snapshot` (menu position, variables, hop count) for deterministic replay.

---

### Paginated Session Queries

For dashboards at scale, use `list_sessions_paginated` instead of `list_sessions`:

```python
page = await svc.list_sessions_paginated(
    page=1,
    page_size=100,
    service_code="*384#",
    session_state="active",
    sort_by="created_at",
    sort_dir="desc",
)
# {"items": [...], "total": 4200, "page": 1, "pages": 42, "page_size": 100}
```

---

### Session Replay

Replay a completed session step-by-step against the current menu tree:

```python
trace = await svc.replay_session(session_id="sess-abc123")
for step in trace:
    print(f"Hop {step['hop']:2d}  input={step['input']!r:10s}  menu={step['menu']!r:20s}  {step['response_type']}")
```

The replay runs in a shadow session that is cleaned up automatically — no live state is mutated. Pass `stop_at_hop=N` to replay only the first N hops.
