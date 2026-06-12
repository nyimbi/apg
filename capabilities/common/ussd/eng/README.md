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

## New Features (v1.1)

| Feature | Method | Description |
|---------|--------|-------------|
| Session resumption | `resume_session` | Re-activate timed-out sessions within a configurable grace window (default 90 s) |
| Idempotent execution | `execute_idempotent` | One-time handler execution per session+hop+handler triple; prevents duplicate debits from GSM retransmits |
| Rate limiting | `check_rate_limit` | Sliding-window per-phone session cap; returns `allowed`, `remaining`, `reset_at` |
| Input validation | `validate_input_against_schema` | Schema-driven validation (type, regex, range) for free-text USSD inputs; Decimal for numeric fields |
| Menu versioning | `create_menu_version` / `rollback_menu` | Snapshot and atomic rollback of menu definitions; keeps last 20 versions |
| Dead-letter queue | `queue_dead_letter` / `get_dead_letters` | Captures failed handler executions with full session snapshot for replay and alerting |
| Paginated sessions | `list_sessions_paginated` | Cursor-free offset pagination with filtering and sort direction |
| Session replay | `replay_session` | Step-by-step re-execution of a completed session's input history against the current menu tree |

## Usage Examples

### Resume a dropped session

```python
svc = UssdEngService(tenant_id="acme")
try:
    session = await svc.resume_session(
        phone_number="+254712345678",
        service_code="*384#",
        grace_seconds=90,
    )
    # session["current_menu"] and session["variables"] are preserved
except KeyError:
    # No resumable session — start fresh
    session = await svc.create_session(...)
```

### Idempotent payment execution

```python
result = await svc.execute_idempotent(
    session_id="sess-abc123",
    hop_count=3,
    handler_name="debit_account",
    payload={"amount": "500"},
)
if result["idempotent"]:
    print("Duplicate request — returning cached result")
# result["result"]["amount"] is a Decimal
```

### Rate-limit before session creation

```python
rl = await svc.check_rate_limit(
    phone_number="+254712345678",
    service_code="*384#",
    window_seconds=3600,
    max_sessions=10,
)
if not rl["allowed"]:
    return "END Too many requests. Try again at " + rl["reset_at"]
session = await svc.create_session(...)
```

### Validate a money transfer amount

```python
result = await svc.validate_input_against_schema(
    value=user_input,
    schema={"type": "decimal", "min_value": "10", "max_value": "150000", "required": True},
)
if not result["valid"]:
    return "CON " + result["error_message"] + "\n0. Back"
amount = result["coerced"]  # Decimal
```

### Menu versioning and rollback

```python
# Snapshot before a deploy
snap = await svc.create_menu_version("main_menu", "*384#")
# ... deploy new menu items ...
# Bad deploy — roll back instantly
await svc.rollback_menu("main_menu", "*384#", version=snap["version"])
```

### Replay a failed session for debugging

```python
trace = await svc.replay_session(session_id="sess-xyz", stop_at_hop=5)
for step in trace:
    print(f"Hop {step['hop']} input={step['input']!r} menu={step['menu']} {step['response_type']}")
```

---

## World-Class Enhancements (v2.0)

- **I1.** USSD Engine — World-Class Improvement Proposals
- **I2.** Multi-Language Menu Fallback Chain
- **I3.** Session Resumption After Timeout
- **I4.** Idempotent Transaction Execution
- **I5.** Rate Limiting Per Phone Number
- **I6.** Input Validation Schema per Menu Item
- **I7.** Menu Versioning and Rollback
- **I8.** Bulk Session Import from Gateway
- **I9.** Conditional Menu Item Weighting and A/B Testing
- **I10.** Real-Time Session Broadcast / Webhook Delivery
- **I11.** Session Encryption for PII at Rest
- **I12.** Dead-Letter Queue for Failed Handler Executions
- **I13.** Paginated Session and Audit Log Queries
- **I14.** Phone Number Masking and Anonymization
- **I15.** Menu Import/Export (JSON Schema)

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
