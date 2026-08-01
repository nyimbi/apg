# Configuration Reference

All runtime behaviour of generated APG apps is controlled through environment variables. No config files are needed.

## Core

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_HOST` | `127.0.0.1` | Bind address for the Flask server |
| `APG_PORT` | `8080` | Listen port |
| `APG_ENV` | `development` | `production` activates HSTS, secure cookies |
| `APG_PRODUCTION` | — | Set to `1` to enable production mode (overrides `APG_ENV`) |
| `APG_DEBUG` | — | Set to `1` for Flask debug mode (never in prod) |
| `APG_APP_NAME` | module name | Display name shown in the UI header and OpenAPI title |
| `APG_APP_DESCRIPTION` | — | One-line description in the UI and OpenAPI info block |
| `APG_APP_VERSION` | `1.0.0` | Semantic version string reported by `/livez` and OpenAPI |

## Security

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_SECRET_KEY` | auto-generated | Flask session signing key. **Required in production.** |
| `APG_SESSION_SECRET` | — | Alias for `APG_SECRET_KEY` |
| `APG_API_KEY` | — | Bearer token for API access. Unset = open access. |
| `APG_ADMIN_KEY` | — | Elevated-privilege API key for admin routes |
| `APG_JWT_SECRET` | — | HS256 JWT shared secret |
| `APG_JWT_PUBLIC_KEY` | — | RS256 public key PEM for JWT verification |
| `APG_SESSION_COOKIE_NAME` | `apg_session` | Session cookie name |
| `APG_SESSION_COOKIE_SAMESITE` | `Lax` | Cookie SameSite: `Lax`, `Strict`, or `None` |
| `APG_SESSION_COOKIE_SECURE` | `0` | Set to `1` to mark session cookie Secure (HTTPS only) |
| `APG_SECURITY_HEADERS` | `1` | Set to `0` to disable default security headers (CSP, HSTS, X-Frame-Options) |
| `APG_MAX_BODY_BYTES` | `10485760` | Maximum request body size in bytes (10 MB default) |
| `APG_MAX_PASSWORD_BYTES` | `72` | bcrypt/scrypt input truncation limit |
| `APG_SCRYPT_N` | `16384` | scrypt CPU/memory cost factor |
| `APG_SCRYPT_R` | `8` | scrypt block size |
| `APG_SCRYPT_P` | `1` | scrypt parallelisation factor |
| `APG_SCRYPT_MAXMEM` | `33554432` | scrypt maximum memory in bytes (32 MB) |
| `APG_FIELD_ACL` | `{}` | JSON map of `{role: {Entity: [field, ...]}}` |
| `APG_ROW_OWNERSHIP` | `0` | Set to `1` to restrict record mutations to the creating user |

## Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_AUTH_USERS` | — | JSON array of user objects `[{username, password_hash, role}]` |
| `APG_AUTH_USERNAME` | `admin` | Single-user username (simpler alternative to `APG_AUTH_USERS`) |
| `APG_AUTH_PASSWORD` | `admin` | Single-user plaintext password |
| `APG_AUTH_PASSWORD_HASH` | — | bcrypt hash; takes precedence over `APG_AUTH_PASSWORD` |
| `APG_AUTH_DISPLAY_NAME` | username | Display name shown in the UI header |
| `APG_AUTH_EMAIL` | — | Email shown in the user profile |
| `APG_API_KEY_OWNER` | `api` | Username associated with API key requests |
| `APG_AUTH_REQUIRED` | `1` | Set to `0` to allow unauthenticated access to all routes |

## Database

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_DATABASE_URL` | — | Full SQLAlchemy connection URL (takes precedence) |
| `DATABASE_URL` | — | Alias for `APG_DATABASE_URL` (Heroku-compatible) |
| `APG_PG_URL` | — | PostgreSQL connection URL alias |
| `APG_SQLITE_PATH` | — | Path to SQLite database file |
| `APG_DB_PATH` | — | Alias for `APG_SQLITE_PATH` |
| `APG_DATA_FILE` | — | Data seed file path (JSON) |
| `APG_DATA_PATH` | — | Alias for `APG_DATA_FILE` |
| `APG_AUTO_MIGRATE` | `1` | Set to `0` to disable automatic schema migration on startup |
| `APG_DB_DIALECT` | `sqlite` | SQL dialect: `sqlite` or `postgresql` |
| `APG_DB_POOL_SIZE` | `5` | Connection pool size for PostgreSQL |
| `APG_DB_POOL_SEMAPHORE` | `10` | Maximum concurrent DB operations (semaphore limit) |

## Email / SMTP

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_SMTP_HOST` | — | SMTP server hostname. Leave unset to disable email. |
| `APG_SMTP_FROM` | — | Sender address (falls back to `APG_SMTP_USER`) |
| `APG_SMTP_USER` | — | SMTP username |
| `APG_SMTP_PASSWORD` | — | SMTP password |
| `APG_ALERT_EMAIL` | — | Address for system alert emails |
| `APG_NOTIFY_EMAIL` | — | Address for record-event notification emails |
| `APG_EMAIL_ON_LOGIN` | `0` | Set to `1` to send an email notification on each successful login |
| `APG_EMAIL_THREADS` | `2` | Thread-pool size for outbound email sending |

## Webhooks

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_WEBHOOK_URL` | — | Comma-separated outbound webhook target URLs |
| `APG_WEBHOOK_SECRET` | — | HMAC signing secret for webhook payloads |

## File uploads

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_UPLOAD_DIR` | `./uploads` | Directory for uploaded files |
| `APG_UPLOAD_MAX_BYTES` | `10485760` | Maximum allowed upload size in bytes (10 MB default) |
| `APG_UPLOAD_ALLOWED_TYPES` | `*` | Comma-separated MIME types; `*` permits all |

## Multi-tenancy

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_MULTI_TENANT` | — | Set to `1` to enable tenant isolation |
| `APG_MULTI_TENANT_ENABLED` | — | Alias for `APG_MULTI_TENANT` |
| `APG_TENANT_HEADER` | `X-Tenant-ID` | HTTP header used to identify the tenant |
| `APG_TENANT_HEADER_DEFAULT` | — | Fallback tenant ID when the header is absent |
| `APG_TENANT_DEFAULT` | `default` | Tenant ID used for single-tenant deployments |

## Rate limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_RATE_LIMIT_ANON` | `60` | Requests per minute allowed for unauthenticated callers |
| `APG_RATE_LIMIT_AUTH` | `300` | Requests per minute allowed for authenticated callers |
| `APG_RATE_BUCKETS` | `1024` | Number of sliding-window buckets (one per remote IP) |
| `APG_RATE_EXEMPT_PATHS` | `/livez,/readyz,/metrics` | Comma-separated paths exempt from rate limiting |

## Records

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_EXPOSE_TIMESTAMPS` | `0` | Set to `1` to include `created_at` and `updated_at` in GET responses. Omitted by default to keep response shapes minimal. `deleted_at` is never exposed (soft-delete implementation detail). |

## Internationalisation

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_LOCALE` | `en` | Default locale code |
| `APG_DEFAULT_LANGUAGE` | `en` | Alias for `APG_LOCALE` used in generated i18n scaffold |
| `APG_FALLBACK_LANGUAGE` | `en` | Language used when a translation key is missing |
| `APG_SUPPORTED_LANGUAGES` | `en` | Comma-separated list of enabled locale codes |
| `APG_EXPORT_LOCALE` | — | Locale used when exporting CSV/XLSX data |
| `APG_LOCALE_DIR` | — | Directory containing `.po`/`.mo` translation files |
| `APG_LOCALE_FILE` | — | Single JSON locale override file |

## Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_AUDIT_LOG_FILE` | — | Path to append-only JSONL audit log |
| `APG_METRICS_TOKEN` | — | Bearer token required to read `/metrics` |
| `APG_METRICS_ENABLED` | `1` | Set to `0` to disable Prometheus `/metrics` endpoint |

## Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_DISABLE_GZIP` | `0` | Set to `1` to disable gzip response compression |
| `APG_GZIP_MIN_BYTES` | `1400` | Minimum response size in bytes before gzip is applied |

## Job queue

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_WORKER_THREADS` | `2` | Thread-pool size for the in-process background job queue |

## UI

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_LANDING_STYLE` | `default` | Landing page style variant |
| `APG_SWAGGER_UI` | `1` | Set to `0` to disable the built-in Swagger UI at `/docs` |

## Agent / AI

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_AGENT_PROVIDER_COMMAND` | — | Shell command to invoke the AI agent |
| `APG_AGENT_RUNTIME_TIMEOUT` | `30` | Seconds before agent call times out |
| `APG_AGENT_WORKDIR` | `.` | Working directory for agent subprocess |

## Minimal production `.env`

```bash
APG_SECRET_KEY=<64-hex-chars>
APG_PRODUCTION=1
APG_AUTH_USERS=[{"username":"admin","password_hash":"$2b$12$...","role":"admin"}]
APG_DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/myapp
APG_AUDIT_LOG_FILE=/var/log/apg/audit.jsonl
APG_HOST=0.0.0.0
APG_PORT=8080
```
