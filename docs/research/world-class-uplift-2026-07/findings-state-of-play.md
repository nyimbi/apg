# State-of-Play Audit — APG Generated App & Platform (2026-07-10)

Evidence-backed map of what exists vs what's missing, compiled from the gap-analysis
docs in `docs/research/` and a targeted sweep of `compiler/code_generator.py` (11.7k lines).
Line numbers reference the file as of commit `5f53f346`.

## 1. Gap-doc inventory

### composition-systems-gap-analysis.md (2026-06-15)
Benchmarked against Temporal, Camunda, MuleSoft, Step Functions, Zapier/Make, Apache Camel,
Conductor/Orkes, Prefect/Dagster, n8n, Boomi. **All 12 gaps remain open:**

| # | Gap | Severity |
|---|-----|----------|
| 1 | No durable execution / event-sourced state — JSON write is not crash-safe | Critical |
| 2 | No non-HTTP connectors (MQTT, AMQP, SAP RFC/IDoc, SWIFT, FIX, FTP, JMS) | Critical |
| 3 | No streaming / CDC integration (Debezium-style triggers) | High |
| 4 | No connector OAuth / token lifecycle (PKCE, refresh, rotation, per-tenant) | High |
| 5 | No visual workflow debugger event store (flow-debugger UI exists, no durable events) | High |
| 6 | No versioning of in-flight workflows | High |
| 7 | No cross-workflow saga coordination | Medium |
| 8 | No connector pagination / rate-limiting | Medium |
| 9 | No EDI / B2B / mainframe connectivity (X12, EDIFACT, AS2) | Medium |
| 10 | No multi-tenant connector credential isolation | Medium |
| 11 | No marketplace / versioned connector registry | Low-Med |
| 12 | No BPMN / visual design tool | Low |

### enterprise-ui-gap-analysis.md (2026-06-15) vs generated-ui-workspaces (2026-07-06)
Of 20 UI gaps: **12 DONE** (inline edit, command palette Cmd+K, activity feed, KPI cards,
saved views, breadcrumbs, related lists, recent items, stepper, notification inbox, dark
mode, keyboard nav in kanban/palette), **4 PARTIAL** (record highlights panel, skeleton
screens — `apg-skeleton` not emitted, column pinning, empty states), **4 OPEN** (density
toggle, pivot/group-by, Gantt/timeline, field widget override).

### ui_integration_gaps_2026.md (+v1)
Open: zero cross-capability NATS subscriptions in generated apps; `requires:[]` is
metadata-only; 42 zero-contract capabilities (agriculture ×12, insurance ×8, legal ×8,
hospitality ×8, NGO ×6, and manufacturing gaps).

## 2. Generated Flask app — dimension-by-dimension

### Authentication
**Done:**
- Credential loading from `APG_AUTH_USERS` env JSON, multi-user, roles/permissions (line 2457)
- Timing-safe compare `hmac.compare_digest` (line 2494)
- Session cookie hardening: HttpOnly, SameSite (configurable, default Lax), Secure in prod (lines 8802–8813)
- Session secret: `APG_SESSION_SECRET` or ephemeral `secrets.token_urlsafe(48)` (committed 5f53f346)
- Logout CSRF (line 8919)

**Missing:**
- **Password hashing** — plaintext compare; no pbkdf2/bcrypt/argon2 anywhere
- **Login rate limiting/lockout** — "lockout" text at line 2641 is informational UI copy only
- **Session fixation defense** — `_issue_login_session` (2504) assigns without `session.clear()`

### Security
**Done:** CSP (default-src 'self', object-src 'none', frame-ancestors 'none', form-action 'self', …),
X-Frame-Options DENY, nosniff, Referrer-Policy, Permissions-Policy, COOP, CORP (lines 8816–8837);
CSRF token per session + hidden input in every mutating form + header acceptance (2521–2605);
API key / JWT auth for JSON mutations (2689–2708).

**Partial/missing:**
- CSP carries `'unsafe-inline'` for script-src and style-src (8825–8826) — needs nonces
- HSTS only when `request.is_secure` (8845)
- `_authorized()` returns True when `APG_API_KEY` unset (2708) — mutations open by default
- API key compare `supplied_key == required_key` (2707) — not timing-safe
- No `@errorhandler(404/500)` — Werkzeug defaults leak
- No `MAX_CONTENT_LENGTH`

### Ops
**Done:** `/health` (8066), `/validate` (8077), `/self-test` (8152), `/metrics` JSON (8150),
`/capabilities/health` + per-capability (8183–8194), `validate_application()` CLI + endpoint.

**Missing:** structured logging (all `print()`), request IDs, Prometheus exposition format,
liveness/readiness split, OpenTelemetry.

### Performance
**Done:** PWA service worker + offline banner + install prompt (628–632, 4662); static assets
via Flask static handler; SSE with `Cache-Control: no-cache` (9005).

**Missing:** gzip/deflate compression, ETag/conditional GET on generated responses,
asset fingerprinting.

### Accessibility / i18n
**Done:** `<html lang dir>` incl. RTL (4636, 4652); ARIA roles/labels throughout; `.apg-sr-only`;
`aria-current="page"`; `aria-describedby` on form inputs; focus management in command palette;
locale detection cookie → Accept-Language → default; Swahili/Arabic/French catalogs.

**Partial/missing:** no `scope="col"`/`<caption>` on tables; label `for` linkage unconfirmed
in most forms; density toggle not emitted; no `?` shortcut-help modal.

## 3. Test coverage map (tests/, generated-app scope)

| File | Tests | Covers |
|------|-------|--------|
| test_generated_ui_auth.py | 7 | security headers, session secret/cookie, CSRF, login/logout flow |
| test_generated_ui_assets.py | 2 | CSS class coverage, no external CDN URLs |
| test_generated_ui_dashboard.py | 12 | shell XSS treatment, charts, saved views, kanban, record detail, forms, wizard, consoles, catalog |
| test_generated_ui_i18n.py | 2 | language switcher, locale format helpers |
| test_generated_ui_pwa.py | 4 | PWA assets/hooks, shell power-user controls, size budgets |
| test_generated_ui_sse.py | 2 | SSE stream, agent output sanitizer |
| test_generated_ui_templates.py | 1 | template route resolution |
| test_generated_workflow_runtime.py | 3 | step sequencing, run persistence/resume, guards |
| test_generated_app_hardening.py | 10 (new, wave A) | password hashing, throttling, fixation, body limit, branded errors |
| tests/ui/test_generated_ui_playwright.py | 1 | browser smoke test |

Last full green run: **1484 passed, 1 skipped** (pre-FAB-5 upgrade).
