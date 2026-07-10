# World-Class Uplift Mission — July 2026

**Date started:** 2026-07-10
**Goal:** Systematically raise APG above world-class in every dimension — security, ops, performance, accessibility, competitive feature set — driven by deep research, implemented in verified waves.

## Mission structure

1. **State-of-play audit** (this folder, `findings-state-of-play.md`) — evidence-backed map of what the generated app already has vs lacks.
2. **Deep research** (sibling folders, produced in parallel):
   - `../generated-app-runtime-baseline/` — OWASP ASVS 5.0, password storage, login hardening, ops baseline, framework-default competitive bar (Django/Rails/Laravel/Phoenix).
   - `../app-generator-competitive-landscape-2026/` — Amplication, JHipster, Refine, Retool, Appsmith et al.; differentiation opportunities.
   - `../generated-ui-excellence-2026/` — WCAG 2.2 AA, Core Web Vitals for no-build server-rendered UI, resilience UX.
3. **Implementation waves** (each: tests first → implement → full suite → commit → push):

| Wave | Scope | Status |
|------|-------|--------|
| A | Security-critical: password hashing (PBKDF2), login rate limiting/lockout, session-fixation rotation, request body limit, branded 404/500, timing-safe API-key compare, production-mode secure-by-default mutations | Tests written (`tests/test_generated_app_hardening.py`), implementation pending baseline-green |
| B | Ops: structured JSON logging, X-Request-ID generation/propagation, Prometheus exposition on /metrics, /livez + /readyz split | Planned |
| C | Performance: gzip compression, ETag/conditional GET | Planned |
| D | Accessibility/UI: table `scope="col"` + `<caption>`, density toggle emission, CSP nonce migration off `unsafe-inline` | Planned |

## Key findings so far

See `findings-state-of-play.md` for the full audit. Headlines:

- **Passwords in `APG_AUTH_USERS` are compared in plaintext** (`compiler/code_generator.py:2494` uses `hmac.compare_digest` for timing safety only — no hashing at rest).
- **No login throttling**: the "lockout" string in `_login_auth_intelligence()` is informational UI text; nothing is enforced.
- **No session-fixation defense**: `_issue_login_session()` (line 2504) assigns into the existing session without `session.clear()`; Flask does not rotate session identity on assignment.
- **API mutations are open by default**: `_authorized()` returns `True` when `APG_API_KEY` is unset (line 2708); the key comparison itself (`==`, line 2707) is not timing-safe.
- **No `MAX_CONTENT_LENGTH`**, no registered `@errorhandler(404/500)` — Werkzeug defaults leak through.
- Ops: all logging is `print()`; `/metrics` is custom JSON (not Prometheus exposition); no request IDs; no liveness/readiness split.
- Performance: no compression, no ETags on generated responses.
- Already strong (prior waves): 7-header security suite incl. CSP, CSRF on all mutating session forms, hardened session cookies, timing-safe compares, i18n + RTL, ARIA coverage, PWA/service worker, command palette, health/validate/self-test endpoints. 1484 tests passing at last green run.

## Environment repairs performed (2026-07-10)

Recorded in `findings-environment.md`. Summary: stale editable `flask-appbuilder` install pointed at a deleted checkout (`~/src/pjs/fab-ext/flask_appbuilder`); replaced with PyPI `flask-appbuilder 5.2.2`. Pre-existing pyOpenSSL/cryptography version skew (cryptography 3.4.8 from 2021, with known CVEs, shadowing via the hermes venv) repaired by upgrading to cryptography 49.0.0 + pyOpenSSL 26.3.0 in both homebrew site-packages and the venv.

## Rationale

See `rationale.md`. Thinking log in `thinking.md`.
