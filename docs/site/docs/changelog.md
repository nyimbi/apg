# Changelog

## v0.1.0

### Wave U — Packaging
- `pyproject.toml` with editable install and PyPI build scaffold
- `Makefile` targets: `compile`, `test`, `lint`, `build`, `clean`
- `CHANGELOG.md` and `VERSION` file
- `setup.py` shim for legacy tooling compatibility

### Wave T — CLI Excellence
- `apg init` scaffolds a new project directory with sample schema, `.env.example`, `Makefile`, `.gitignore`
- `apg doctor` validates the local environment (Python version, grammar, ANTLR runtime, smoke test)
- `apg watch` recompiles on file save with sub-200 ms latency
- `apg serve` compiles and immediately starts the generated Flask server
- `apg export --format docker` writes `Dockerfile` and `docker-compose.yml`

### Wave S — Language Server Protocol
- LSP server for completion, hover, diagnostics, and go-to-definition
- VS Code extension wires to the LSP via `apg language-server`
- Code actions: extract entity, add field, rename symbol

### Wave R — Job Queue & Computed Fields
- In-process job queue for background tasks
- Computed fields: server-side expressions derived at read time
- Computed fields backed by the job queue for long-running calculations

### Wave Q — Multi-tenancy & i18n
- Tenant isolation scaffolding via `tenant_id` field + `APG_MULTI_TENANT` flag
- Configurable tenant header (`APG_TENANT_HEADER`)
- i18n scaffolding: locale detection, `.po`/`.mo` file loading, JSON locale overrides

### Wave P — VS Code Extension
- Syntax highlighting for `.apg` files
- Snippet library for common patterns (table, app, enum, capability)
- Language configuration (bracket matching, comment tokens)

### Wave O — File Uploads & Email
- `file` field type generates multipart upload endpoints
- File storage in `APG_UPLOAD_DIR` with per-record subdirectories
- SMTP-backed email notifications (`APG_SMTP_*` vars)
- Alert and notification email targets

### Wave N — Enums & Validation
- Named `enum` declarations with strongly-typed fields
- Field validators: `email`, `min_length`, `max_length`, `min`, `max`, `pattern`, `unique`, `values`
- `422 Unprocessable Entity` responses with per-field error details

### Wave M — Relationships
- `has_many`, `belongs_to`, `has_one` relationship declarations
- `through` for many-to-many join tables
- Nested REST endpoints: `GET /entities/Parent/<id>/children`

### Wave L — Data Governance
- Column ACL filtering (`APG_FIELD_ACL`) restricts fields per role
- Row ownership: non-admin users read/write only their own records
- Field-diff audit logging records before/after on every update

### Wave K — Full-text Search & Test Scaffolding
- FTS5-backed full-text search endpoint: `GET /entities/Foo/search?q=term`
- Generated `smoke_test.py` with pytest-compatible assertions
- `apg compile --verify` runs the smoke test immediately after compilation

### Wave J — Webhooks
- Outbound HMAC-signed webhooks on create/update/delete
- `APG_WEBHOOK_URL` (comma-separated multi-target) and `APG_WEBHOOK_SECRET`
- At-least-once delivery with 3-attempt exponential back-off
- Delivery history endpoint: `GET /entities/Foo/webhooks`

### Wave I — Database Lifecycle
- Auto-generated `created_at`, `updated_at` timestamps on every entity
- Soft-delete via `deleted_at` field; `include_deleted=true` query param
- Auto-migration on startup (`APG_AUTO_MIGRATE=1`)
- Bulk create/update: `POST /entities/Foo/records/bulk`

### Wave H — Compiler Diagnostics
- Line and column numbers in all error messages
- "Did you mean?" suggestions for unknown field types and entity names
- Duplicate field and entity validation
- Cleaner error output with Rich formatting

### Wave G — UI Polish
- Dark mode with system-preference detection and manual toggle
- Mobile-responsive layout for the management UI
- Print CSS for record export

### Wave F — Production Hardening
- Startup validation: aborts with a clear message when `APG_SECRET_KEY` is absent in production
- Audit log: append-only JSONL file recording all write operations
- Rate limiting: 100 requests/min per IP, `429` on excess
- JSON content-type guard: `415` for non-JSON write requests

### Wave E — API Completeness
- OpenAPI 3.1 schema at `/openapi.json` and `/api-docs`
- Pagination (`page`, `per_page`) on all list endpoints
- Filtering (`filter_<field>=value`) on all list endpoints
- Sorting (`sort`, `order`) on all list endpoints
- CSV export: `GET /entities/Foo/records/export.csv`

### Wave D — Accessibility & UI
- Table accessibility: `scope`, `aria-sort`, keyboard navigation
- Density toggle (compact / comfortable / spacious)
- CSP nonce on all inline scripts
- ARIA live region for status messages

### Wave C — HTTP Efficiency
- Gzip compression for responses > 1 KB
- `Cache-Control` and `ETag` headers for list and record endpoints
- Conditional GET support (`If-None-Match`)

### Wave B — Operations Baseline
- Structured JSON logging with `X-Request-ID` propagation
- Liveness probe: `GET /livez`
- Readiness probe: `GET /readyz` (checks DB connection)
- Prometheus metrics: `GET /metrics`

### Wave A — Security Hardening
- bcrypt password hashing for local users
- Login throttle: 5 failed attempts → 60-second lockout
- Session fixation defense: new session ID after login
- Request size limits
- Secure API-key comparison (timing-safe)
- Graceful `404` and `500` error handlers
- Security headers: CSP, HSTS, X-Frame-Options, X-Content-Type-Options
