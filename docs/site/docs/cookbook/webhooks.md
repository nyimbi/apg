# Adding Webhooks

Wire outbound webhooks to your APG app to integrate with Slack, Zapier, n8n, or any HTTP endpoint.

## Quick setup

```bash
export APG_WEBHOOK_URL="https://hooks.slack.com/services/T.../B.../..."
export APG_WEBHOOK_SECRET="$(openssl rand -hex 32)"
python out/app.py
```

Every create, update, and delete on any entity now fires a signed POST to that URL.

## Multiple targets

```bash
export APG_WEBHOOK_URL="https://hook-a.example.com/in,https://hook-b.example.com/in"
```

Both endpoints receive every event independently. A failure on one does not block the other.

## Payload structure

```json
{
  "event":     "record.created",
  "entity":    "Order",
  "record_id": "01923abc-...",
  "timestamp": "2025-07-31T10:05:22Z",
  "data": {
    "id":           "01923abc-...",
    "order_number": "ORD-1001",
    "status":       "draft",
    "total":        "0.00",
    "created_at":   "2025-07-31T10:05:22Z"
  }
}
```

## Verifying the signature

```python
import hashlib, hmac

def verify_apg_webhook(payload: bytes, secret: str, header: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header)
```

Pass the raw request body as `payload` and the `X-APG-Signature` header value as `header`.

## Receiving webhooks — Flask example

```python
from flask import Flask, request, abort
import hashlib, hmac

app = Flask(__name__)
WEBHOOK_SECRET = "your-secret"

@app.post("/webhook")
def handle():
    sig = request.headers.get("X-APG-Signature", "")
    body = request.get_data()
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, sig):
        abort(401)
    event = request.json
    print(event["event"], event["entity"], event["record_id"])
    return {"ok": True}
```

## Receiving webhooks — n8n / Zapier

1. Create a **Webhook** trigger node in n8n (or Zapier).
2. Copy the generated webhook URL.
3. Set `APG_WEBHOOK_URL` to that URL.
4. Restart the app.
5. Create a record in your APG app — the workflow fires immediately.

## Retry behaviour

| Attempt | Delay |
|---------|-------|
| 1 | Immediate |
| 2 | 5 seconds |
| 3 | 30 seconds |

After 3 failures the delivery is marked failed and logged to the audit log. The app continues normally.

## Delivery history

```bash
curl http://localhost:8080/entities/Order/webhooks
```

Returns the last 100 delivery records:

```json
[
  {
    "event":      "record.created",
    "target_url": "https://hook.example.com/in",
    "status":     200,
    "duration_ms": 142,
    "timestamp":  "2025-07-31T10:05:22Z"
  }
]
```

## Local testing with ngrok

```bash
ngrok http 8080
export APG_WEBHOOK_URL="https://<ngrok-id>.ngrok-free.app/webhook"
python out/app.py
```

Now create a record and watch the event arrive in your tunnel.
