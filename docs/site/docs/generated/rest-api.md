# REST API

Every `table` entity in your APG source file gets a complete CRUD REST API with pagination, filtering, sorting, bulk operations, and CSV export — automatically.

## Endpoint structure

Given a `table Foo`, the following endpoints are generated:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/entities/Foo/records` | List records (paginated) |
| `POST` | `/entities/Foo/records` | Create a record |
| `GET` | `/entities/Foo/records/<id>` | Fetch one record |
| `PUT` | `/entities/Foo/records/<id>` | Replace a record |
| `PATCH` | `/entities/Foo/records/<id>` | Partial update |
| `DELETE` | `/entities/Foo/records/<id>` | Soft-delete |
| `GET` | `/entities/Foo/records/export.csv` | Export all records as CSV |
| `GET` | `/entities/Foo/search?q=term` | Full-text search |
| `POST` | `/entities/Foo/records/bulk` | Bulk create/update |

## List endpoint

```
GET /entities/Contact/records?page=1&per_page=25&sort=name&order=asc&filter_email=alice@example.com
```

Query parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `page` | `1` | Page number (1-based) |
| `per_page` | `25` | Records per page (max 1000) |
| `sort` | `id` | Field to sort by |
| `order` | `asc` | `asc` or `desc` |
| `filter_<field>` | — | Exact match filter on any field |
| `q` | — | Full-text search query |
| `include_deleted` | `false` | Include soft-deleted records |

Response envelope:

```json
{
  "records": [...],
  "total": 142,
  "page": 1,
  "per_page": 25,
  "pages": 6
}
```

## Create

```bash
curl -X POST http://localhost:8080/entities/Contact/records \
  -H "Content-Type: application/json" \
  -d '{"record": {"name": "Alice", "email": "alice@example.com"}}'
```

Response: `201 Created` with the full record including auto-generated `id`, `created_at`, `updated_at`.

## Fetch one

```bash
curl http://localhost:8080/entities/Contact/records/01923abc-...
```

Response: `200 OK` or `404 Not Found`.

## Update (replace)

```bash
curl -X PUT http://localhost:8080/entities/Contact/records/01923abc-... \
  -H "Content-Type: application/json" \
  -d '{"record": {"name": "Alice B.", "email": "alice@example.com"}}'
```

## Partial update

```bash
curl -X PATCH http://localhost:8080/entities/Contact/records/01923abc-... \
  -H "Content-Type: application/json" \
  -d '{"record": {"active": false}}'
```

Only the supplied fields are changed. All other fields are left unchanged.

## Delete (soft)

```bash
curl -X DELETE http://localhost:8080/entities/Contact/records/01923abc-...
```

Sets `deleted_at` to the current timestamp. The record is excluded from all list queries unless `include_deleted=true` is set.

## Bulk operations

```bash
curl -X POST http://localhost:8080/entities/Contact/records/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"name": "Bob", "email": "bob@example.com"},
      {"name": "Carol", "email": "carol@example.com"}
    ]
  }'
```

Returns a list of created/updated records and any per-record errors.

## CSV export

```bash
curl http://localhost:8080/entities/Contact/records/export.csv > contacts.csv
```

Applies the same filter parameters as the list endpoint.

## Full-text search

```bash
curl "http://localhost:8080/entities/Contact/search?q=alice"
```

Uses FTS5 SQLite or PostgreSQL full-text search. Returns matching records ranked by relevance.

## System endpoints

| Path | Description |
|------|-------------|
| `GET /livez` | Liveness probe — returns `{"ok": true}` |
| `GET /readyz` | Readiness probe — checks DB connection |
| `GET /metrics` | Prometheus metrics (text format) |
| `GET /openapi.json` | OpenAPI 3.1 schema |
| `GET /ui` | Generated management UI |

## Content-Type enforcement

All write endpoints require `Content-Type: application/json`. Requests with other content types receive `415 Unsupported Media Type`.

## Error responses

```json
{"error": "not_found", "status": 404}
{"error": "validation_error", "field": "email", "message": "Invalid format", "status": 422}
{"error": "rate_limit_exceeded", "status": 429}
{"error": "internal_error", "status": 500}
```
