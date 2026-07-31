# Webhooks

The generated app fires HMAC-signed outbound webhooks on create, update, and delete events.

## Enable webhooks

```bash
export APG_WEBHOOK_URL="https://your-service.example.com/webhook"
export APG_WEBHOOK_SECRET="$(openssl rand -hex 32)"
```

Multiple targets (comma-separated):

```bash
export APG_WEBHOOK_URL="https://svc-a.example.com/hook,https://svc-b.example.com/hook"
```

## Event payload

```json
{
  "event": "record.created",
  "entity": "Contact",
  "record_id": "01923abc-...",
  "timestamp": "2025-07-31T14:23:01Z",
  "data": {
    "id": "01923abc-...",
    "name": "Alice",
    "email": "alice@example.com",
    "created_at": "2025-07-31T14:23:01Z"
  }
}
```

## Event types

| Event | Trigger |
|-------|---------|
| `record.created` | Successful POST to `/records` |
| `record.updated` | Successful PUT or PATCH |
| `record.deleted` | Successful DELETE |

## HMAC signature

Each request includes an `X-APG-Signature` header:

```
X-APG-Signature: sha256=<hex_digest>
```

Verify in Python:

```python
import hashlib, hmac

def verify(payload: bytes, secret: str, header: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header)
```

## Delivery guarantees

- Webhooks are delivered **at-least-once** with up to 3 retry attempts (exponential back-off).
- Delivery failures are logged to `APG_AUDIT_LOG_FILE` but do not affect the API response.
- Each delivery attempt has a 10-second timeout.

## Inspecting delivery history

```bash
curl http://localhost:8080/entities/Contact/webhooks
```

Returns the last 100 delivery attempts for all Contact events (timestamp, target URL, HTTP status, duration).

## Disabling webhooks per entity

Webhooks fire for all entities by default when `APG_WEBHOOK_URL` is set. To disable for a specific entity, remove its events from the webhook handler by overriding the generated `_should_fire_webhook` function in a post-compile hook (advanced use).
