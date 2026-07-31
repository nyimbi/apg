# Multi-tenant SaaS

APG provides first-class multi-tenancy through tenant isolation scaffolding. Every record is scoped to a tenant identified by an HTTP header.

## Enable multi-tenancy

```bash
export APG_MULTI_TENANT=1
export APG_TENANT_HEADER=X-Tenant-ID   # default
```

## Schema

Add a `tenant_id` field to every entity you want tenant-scoped:

```apg
module saas version 1.0.0 {
    description: "Multi-tenant SaaS";
}

table Organisation {
    slug:       str;
    name:       str;
    plan:       str = "free";
    is_active:  bool = true;
}

table Project {
    tenant_id:   str;          // isolates this entity by tenant
    name:        str;
    description: str?;
    owner_id:    str;
    is_archived: bool = false;
}

table Task {
    tenant_id:   str;
    project_id:  str;
    title:       str;
    status:      str = "todo";
    assignee_id: str?;
}

app SaaS {
    routes: ["/organisations", "/projects", "/tasks"];
}
```

## How isolation works

When `APG_MULTI_TENANT=1` and a request arrives with `X-Tenant-ID: acme`, the generated app:

1. Reads the tenant value from the configured header.
2. Applies `WHERE tenant_id = 'acme'` to all queries on tenant-scoped entities.
3. Sets `tenant_id = 'acme'` on all new records.
4. Rejects requests that try to set a different `tenant_id`.

Entities **without** a `tenant_id` field are not scoped (e.g. `Organisation` is global).

## Request flow

```bash
# Requests for tenant "acme"
curl -H "X-Tenant-ID: acme" \
  http://localhost:8080/entities/Project/records

# Requests for tenant "beta"
curl -H "X-Tenant-ID: beta" \
  http://localhost:8080/entities/Project/records
```

Both return only the records belonging to their respective tenant.

## Admin cross-tenant access

The admin key bypasses tenant scoping:

```bash
curl -H "Authorization: Bearer $APG_ADMIN_KEY" \
  http://localhost:8080/entities/Project/records
# Returns all projects across all tenants
```

## Tenant header customisation

```bash
export APG_TENANT_HEADER=X-Workspace-ID
```

Now the isolation key is read from `X-Workspace-ID`.

## Per-tenant configuration

For per-tenant theming, locale, or feature flags, store preferences in an `Organisation` or `TenantConfig` entity and read it in your capability layer:

```apg
table TenantConfig {
    tenant_id: str;
    locale:    str = "en";
    theme:     str = "default";
    features:  Dict[str, Any];
}
```

## Compile and run

```bash
apg compile saas.apg -o out/ --verify
export APG_MULTI_TENANT=1
python out/app.py
```

## Database schema

When `APG_DATABASE_URL` points to PostgreSQL, tenant-scoped entities get a `tenant_id` index:

```sql
CREATE INDEX ix_project_tenant_id ON project (tenant_id);
CREATE INDEX ix_task_tenant_id ON task (tenant_id);
```

This keeps tenant queries O(log n) regardless of total record count.
