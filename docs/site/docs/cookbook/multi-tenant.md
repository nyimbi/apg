# Multi-tenant SaaS

APG provides first-class multi-tenancy through row-level tenant isolation. Every record is scoped to a tenant identified by an HTTP request header — no separate schema or database per tenant required (though that is a valid alternative for compliance-hard cases).

## How isolation works

A `tenant_id` column on an entity table is the isolation primitive:

- On **write** (POST / PUT / PATCH), APG reads the tenant value from the configured header and sets `tenant_id` automatically. Clients cannot override it.
- On **read** (GET / LIST), APG appends `WHERE tenant_id = '<value>'` to every query. Cross-tenant reads return 0 rows, not an error.
- Entities **without** a `tenant_id` field are global and not scoped (useful for shared lookup tables, plans, etc.).

## Minimal APG source

```apg
module saas version 1.0.0 {
    description: "Multi-tenant SaaS walkthrough";
}

// Global — not scoped to any tenant
table Plan {
    name:        str;
    price_cents: int = 0;
    is_active:   bool = true;
}

// Tenant-scoped — carries tenant_id
table Project {
    tenant_id:   str;          // presence of this field triggers scoping
    name:        str;
    description: str?;
    is_archived: bool = false;
}

table Task {
    tenant_id:  str;
    project_id: str;
    title:      str;
    status:     str = "todo";
}

app SaaS {
    routes: ["/plans", "/projects", "/tasks"];
}
```

## Compile and run

```bash
apg compile saas.apg -o out/ --verify

export APG_MULTI_TENANT=1
export APG_PRODUCTION=1          # missing header → 400 in production
export APG_ADMIN_KEY=supersecret

python out/app.py
```

## curl walkthroughs

### POST as tenant `acme`

```bash
curl -s -X POST http://localhost:5000/projects \
  -H "Content-Type: application/json" \
  -H "X-APG-Tenant: acme" \
  -d '{"name": "Acme Landing Page", "description": "Q3 relaunch"}'
# → {"id": "01926...", "tenant_id": "acme", "name": "Acme Landing Page", ...}
```

### POST as tenant `globex`

```bash
curl -s -X POST http://localhost:5000/projects \
  -H "Content-Type: application/json" \
  -H "X-APG-Tenant: globex" \
  -d '{"name": "Globex Internal Wiki"}'
# → {"id": "01927...", "tenant_id": "globex", "name": "Globex Internal Wiki", ...}
```

### GET as `acme` — proves isolation

```bash
curl -s http://localhost:5000/projects \
  -H "X-APG-Tenant: acme"
# → [{"id": "01926...", "tenant_id": "acme", "name": "Acme Landing Page", ...}]
# globex row is not visible
```

### Missing header in production → 400

```bash
curl -s http://localhost:5000/projects
# HTTP 400
# {"error": "Missing required tenant header: X-APG-Tenant"}
```

When `APG_PRODUCTION=0` (default in dev mode) a missing header is treated as `tenant_id = ""` and returns an empty list rather than an error — useful for local iteration without always setting the header.

### Admin override — sees all tenants

```bash
curl -s http://localhost:5000/projects \
  -H "X-APG-Tenant: acme" \
  -H "X-APG-Admin: 1" \
  -H "X-APG-API-Key: $APG_ADMIN_KEY"
# → [{acme row}, {globex row}, ...]
```

The `X-APG-Admin: 1` + `X-APG-API-Key` combination bypasses the `WHERE tenant_id = ?` filter entirely. The `X-APG-Tenant` header must still be present (production mode requires it) but its value is ignored for filtering.

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `APG_MULTI_TENANT` | `0` | Enable tenant isolation |
| `APG_TENANT_HEADER` | `X-APG-Tenant` | HTTP header carrying the tenant identifier |
| `APG_PRODUCTION` | `0` | Reject requests with missing tenant header |
| `APG_ADMIN_KEY` | — | Secret enabling admin cross-tenant reads |

### Renaming the header

```bash
export APG_TENANT_HEADER=X-Workspace-ID
```

All clients must then send `X-Workspace-ID: <tenant>` instead.

## Combining with column ACL and row ownership

`APG_FIELD_ACL` and `APG_ROW_OWNERSHIP` compose cleanly with multi-tenancy — they add independent WHERE clauses / column masks on top of the tenant filter:

```bash
export APG_MULTI_TENANT=1
export APG_ROW_OWNERSHIP=1       # each row also tracks created_by; users see only their own rows
export APG_FIELD_ACL='{
  "Project": {"description": ["admin"]}
}'                               # non-admin clients never see the description column
```

Evaluation order: tenant filter → row-ownership filter → field ACL mask.

## PostgreSQL notes

When `APG_DATABASE_URL` points to PostgreSQL (Wave X.1), the same behaviour applies. APG creates covering indexes automatically:

```sql
CREATE INDEX ix_project_tenant_id ON project (tenant_id);
CREATE INDEX ix_task_tenant_id    ON task    (tenant_id);
```

This keeps per-tenant LIST queries O(log n) regardless of total row count across tenants.

## Pitfalls

- **Never leak tenant IDs in error messages.** A 404 that says `"project 01926... not found for tenant globex"` reveals cross-tenant existence. Return a plain 404 with no tenant context.
- **One DB per tenant for compliance-hard cases.** Row-level isolation is sufficient for most SaaS products. When a customer's data-residency contract requires physical separation, provision a separate `APG_DATABASE_URL` per tenant and route at the reverse-proxy layer. APG's generated app is stateless — spinning up N processes each pointing at a different DB is straightforward.
- **Tenant ID ≠ authentication.** The header is an isolation hint, not a proof of identity. Put authentication (Wave M JWT / Wave N API keys) in front and derive the tenant from the authenticated principal, not from a freely-settable header, in production.
