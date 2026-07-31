# Quick Start

From zero to a running REST API in under 5 minutes.

## 1. Install APG

```bash
git clone https://github.com/nyimbi/apg
cd apg
uv venv .venv && uv pip install -e ".[dev]"
```

## 2. Write your first APG file

Create `hello.apg`:

```apg
module hello version 1.0.0 {
    description: "Hello World APG app";
}

table Contact {
    name:    str;
    email:   str;
    phone:   str?;
    active:  bool = true;
}

app Hello {
    description: "My first APG app";
    routes: ["/contacts"];
}
```

## 3. Compile

```bash
apg compile hello.apg --output out/ --verify
```

The `--verify` flag runs a smoke test on the generated app immediately. Output:

```
✓ Compiled hello.apg → out/app.py
✓ Smoke test passed (3 endpoints)
```

## 4. Run the generated app

```bash
python out/app.py --host 127.0.0.1 --port 8080
```

## 5. Call the API

```bash
# Create a contact
curl -s -X POST http://127.0.0.1:8080/entities/Contact/records \
  -H "Content-Type: application/json" \
  -d '{"record": {"name": "Alice", "email": "alice@example.com", "active": true}}'

# List all contacts
curl -s http://127.0.0.1:8080/entities/Contact/records | python -m json.tool

# Fetch one (replace <id> with the returned id)
curl -s http://127.0.0.1:8080/entities/Contact/records/<id>

# Partial update
curl -s -X PATCH http://127.0.0.1:8080/entities/Contact/records/<id> \
  -H "Content-Type: application/json" \
  -d '{"record": {"active": false}}'

# Delete (soft)
curl -s -X DELETE http://127.0.0.1:8080/entities/Contact/records/<id>
```

## 6. Explore the generated UI

Open [http://127.0.0.1:8080/ui](http://127.0.0.1:8080/ui) in your browser for the generated management shell.

## 7. Check the OpenAPI schema

```bash
curl -s http://127.0.0.1:8080/openapi.json | python -m json.tool
```

## What was generated?

The single `out/app.py` includes:

- Full CRUD REST API for every `table` entity
- Session + API-key authentication (opt-in via env vars)
- Audit log, rate limiting, CSP headers
- Prometheus metrics at `/metrics`
- Liveness/readiness probes at `/livez` `/readyz`
- OpenAPI 3.1 schema at `/openapi.json`
- Dark-mode HTML management UI at `/ui`
- Auto-migration on first run

## Next steps

- [Your First App](first-app.md) — relationships, enums, validation
- [Entities & Fields](../reference/entities.md) — complete field reference
- [Configuration Reference](../generated/configuration.md) — all env vars
