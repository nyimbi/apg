# CRUD App in 5 Minutes

Build a fully functional contacts manager with REST API, management UI, and OpenAPI docs.

## Prerequisites

```bash
git clone https://github.com/nyimbi/apg && cd apg
uv venv .venv && uv pip install -e ".[dev]"
```

## Step 1 — Write the schema (1 min)

Create `contacts.apg`:

```apg
module contacts version 1.0.0 {
    description: "Simple contacts manager";
}

table Contact {
    first_name:   str;
    last_name:    str;
    email:        str;
    phone:        str?;
    company:      str | None;
    title:        str?;
    notes:        text?;
    is_active:    bool = true;
    tags:         List[str];
}

app Contacts {
    description: "Contact manager";
    routes: ["/contacts"];
}
```

## Step 2 — Compile (30 sec)

```bash
apg compile contacts.apg --output out/ --verify
```

Expected output:

```
✓ Parsed contacts.apg
✓ Generated out/app.py  (1 file)
✓ Smoke test passed
```

## Step 3 — Run (30 sec)

```bash
python out/app.py --host 127.0.0.1 --port 8080
```

## Step 4 — Use the API (3 min)

### Create contacts

```bash
curl -s -X POST http://localhost:8080/entities/Contact/records \
  -H "Content-Type: application/json" \
  -d '{
    "record": {
      "first_name": "Alice",
      "last_name": "Kamau",
      "email": "alice@example.com",
      "company": "Datacraft",
      "tags": ["vip", "customer"]
    }
  }' | python -m json.tool
```

### List with pagination

```bash
curl "http://localhost:8080/entities/Contact/records?page=1&per_page=10&sort=last_name"
```

### Filter

```bash
curl "http://localhost:8080/entities/Contact/records?filter_company=Datacraft"
```

### Search

```bash
curl "http://localhost:8080/entities/Contact/search?q=alice"
```

### Update one field

```bash
curl -s -X PATCH http://localhost:8080/entities/Contact/records/<id> \
  -H "Content-Type: application/json" \
  -d '{"record": {"is_active": false}}'
```

### Export to CSV

```bash
curl http://localhost:8080/entities/Contact/records/export.csv > contacts.csv
```

## Step 5 — Browse the UI

Open [http://localhost:8080/ui](http://localhost:8080/ui) — a generated dark-mode management shell with sortable tables, inline editing, and density controls.

## Step 6 — Explore OpenAPI

```bash
curl http://localhost:8080/openapi.json | python -m json.tool
```

Or paste the URL into [Swagger Editor](https://editor.swagger.io).

## Adding authentication

```bash
export APG_API_KEY="$(openssl rand -hex 32)"
python out/app.py --host 127.0.0.1 --port 8080
```

```bash
curl -H "Authorization: Bearer $APG_API_KEY" \
  http://localhost:8080/entities/Contact/records
```

## Connecting to PostgreSQL

```bash
export APG_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/contacts"
python out/app.py
```

Auto-migration runs on startup and creates the `contact` table.

## What you built

In 5 minutes you have:

- Full CRUD REST API for `Contact`
- Pagination, filtering, sorting, CSV export
- Full-text search
- OpenAPI 3.1 schema
- Dark-mode HTML management UI
- Prometheus metrics + liveness/readiness probes
- Soft-delete with audit trail
- Optional API key auth
- Optional PostgreSQL backend
