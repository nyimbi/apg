# In-process Job Queue

APG ships a lightweight in-process job queue (Wave R). It runs worker threads inside the same Python process as the Flask app, persists job state to an `apg_jobs` table (SQLite by default, PostgreSQL with Wave X.1), and requires no external broker — no Celery, no Redis, no RabbitMQ.

## What it is (and what it is not)

| Property | APG job queue |
|---|---|
| Transport | In-process thread pool |
| Persistence | `apg_jobs` table (same DB as the app) |
| Broker required | No |
| Survives process restart | Yes — jobs in `pending`/`failed` state are picked up on next startup |
| Max sustainable throughput | ~100 rps (single process; see [When to reach for a real queue](#when-to-reach-for-a-real-queue)) |
| Cross-process workers | No |

The queue is **app-level**, not per-entity. Any entity's mutation can enqueue a job; so can your custom code.

## Minimal APG source

The job queue is always enabled when the generated app starts — no special DSL syntax required. The `apg_jobs` table is created automatically on first run.

```apg
module myapp version 1.0.0 {
    description: "App with background jobs";
}

table Order {
    customer_email: str;
    item_sku:       str;
    quantity:       int = 1;
    status:         str = "pending";
}

app MyApp {
    routes: ["/orders"];
}
```

## Compile and run

```bash
apg compile myapp.apg -o out/ --verify

export APG_WORKER_THREADS=4      # default: 2
export APG_JOB_MAX_RETRIES=5    # default: 3
export APG_WEBHOOK_SECRET=hmac-secret   # optional — HMAC-signs apg.webhook payloads

python out/app.py
# [INFO] Job queue started — 4 worker threads, max retries 3
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `APG_WORKER_THREADS` | `2` | Number of threads polling for pending jobs |
| `APG_JOB_MAX_RETRIES` | `3` | Attempts before a job is marked `failed` |
| `APG_WEBHOOK_SECRET` | — | If set, HMAC-SHA256 signs `apg.webhook` payloads |
| `APG_SMTP_HOST` | — | SMTP server for `apg.email` jobs |
| `APG_SMTP_PORT` | `587` | SMTP port |
| `APG_SMTP_USER` | — | SMTP username |
| `APG_SMTP_PASSWORD` | — | SMTP password |
| `APG_SMTP_FROM` | — | Envelope From address |

Retry backoff is exponential: attempt `n` waits `2^n` seconds before re-queuing (1 s, 2 s, 4 s, …).

## Enqueue a job

```bash
curl -s -X POST http://localhost:5000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "type": "apg.webhook",
    "payload": {
      "url": "https://hooks.example.com/order-created",
      "body": {"order_id": "01926abc", "status": "pending"}
    }
  }'
# → {"id": "01928def", "type": "apg.webhook", "status": "pending", "attempt": 0, "created_at": "..."}
```

```bash
curl -s -X POST http://localhost:5000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "type": "apg.email",
    "payload": {
      "to": "customer@example.com",
      "subject": "Your order was received",
      "body": "We are processing order #01926abc."
    }
  }'
# → {"id": "01929fed", "type": "apg.email", "status": "pending", "attempt": 0, "created_at": "..."}
```

## Poll for job status

```bash
curl -s http://localhost:5000/jobs/01928def
# → {"id": "01928def", "type": "apg.webhook", "status": "done", "attempt": 1, "completed_at": "..."}
```

Possible `status` values: `pending`, `running`, `done`, `failed`.

## List jobs

```bash
# Pending
curl -s "http://localhost:5000/jobs?status=pending"

# Failed
curl -s "http://localhost:5000/jobs?status=failed"

# All jobs (paginated, newest first)
curl -s "http://localhost:5000/jobs?limit=50&offset=0"
```

## Retry a failed job

```bash
curl -s -X POST http://localhost:5000/jobs/01928def/retry
# → {"id": "01928def", "status": "pending", "attempt": 0}
```

Manual retry resets the attempt counter to 0 so the job gets the full `APG_JOB_MAX_RETRIES` budget again.

## Built-in job handlers

### `apg.webhook`

Sends an HTTP POST to `payload.url` with `payload.body` as the JSON body. If `APG_WEBHOOK_SECRET` is set, the request includes an `X-APG-Signature` header (HMAC-SHA256 over the serialised body) — the same signing scheme used by entity-mutation webhooks (Wave P), so webhook consumers need only one verification helper.

```json
{
  "type": "apg.webhook",
  "payload": {
    "url": "https://hooks.example.com/path",
    "body": { "any": "json" },
    "headers": { "X-Custom": "value" }
  }
}
```

### `apg.email`

Sends a plain-text email via SMTP. Requires `APG_SMTP_HOST` and `APG_SMTP_FROM` to be set.

```json
{
  "type": "apg.email",
  "payload": {
    "to": "user@example.com",
    "subject": "Hello",
    "body": "Plain-text body here."
  }
}
```

For HTML email, add `"html": "<p>Body</p>"` to the payload; the handler sends a `multipart/alternative` message.

## Retry semantics

```
attempt 0  → immediate execution
attempt 1  → wait 2 s  (on first failure)
attempt 2  → wait 4 s
attempt 3  → wait 8 s
…
attempt N  → status = "failed" (N = APG_JOB_MAX_RETRIES)
```

Failed jobs remain in the `apg_jobs` table indefinitely and are visible via `GET /jobs?status=failed`. Use `POST /jobs/{id}/retry` to re-queue them, or query the table directly for bulk operations.

## Extending with a custom handler

Custom handlers are registered in the generated app's `_APG_JOB_HANDLERS` dict before the server starts:

```python
# out/app.py  (after apg compile)
from apg_runtime.jobs import _APG_JOB_HANDLERS

def handle_send_sms(payload: dict) -> None:
    # call your SMS provider here
    phone   = payload["to"]
    message = payload["text"]
    sms_client.send(phone, message)

_APG_JOB_HANDLERS["myapp.sms"] = handle_send_sms
```

Then enqueue as normal:

```bash
curl -s -X POST http://localhost:5000/jobs \
  -H "Content-Type: application/json" \
  -d '{"type": "myapp.sms", "payload": {"to": "+254700000000", "text": "Your code is 1234"}}'
```

!!! note "Re-generate awareness"
    If you re-run `apg compile` after adding a custom handler, the generated `app.py` will be overwritten and your handler registration will be lost. Keep custom handler wiring in a thin wrapper that imports and patches `app.py`, or add your handler to the source before compiling. Native DSL-level custom handlers are planned for a future wave.

## When to reach for a real queue

The in-process queue is the right default for most apps. Switch to Celery, RQ, or a managed queue service when:

- **Sustained throughput > 100 rps** — the thread pool becomes the bottleneck.
- **Cross-process workers** — you want to scale workers independently of web workers.
- **Persistence across catastrophic crashes is critical** — SQLite WAL survives clean shutdowns but a hard kill can lose in-flight jobs; PostgreSQL with `synchronous_commit = on` is safer, and a dedicated broker (Redis Streams, RabbitMQ) is safer still.
- **Fan-out / pub-sub** — multiple consumers need the same event; the APG queue is point-to-point.

For those cases, generate the app with APG, then swap the job-enqueue calls for your broker's client. The rest of the generated CRUD surface stays untouched.
