# Rationale — World-Class Uplift Mission

## Why start with the generated-app runtime (and not the platform gaps)?

The composition-systems gap list (durable execution, non-HTTP connectors, CDC) is real but
each item is a multi-week subsystem. The generated-app runtime gaps are (a) security-critical —
plaintext password comparison ships in every generated app today, (b) small enough to land
verified waves in hours, and (c) multiplicative: one generator fix uplifts every app APG has
ever generated and will generate. Highest leverage per token, lowest risk.

## Why implement inline rather than delegate to Codex?

The global workflow prefers Codex for mechanical implementation. Rejected here because:
`compiler/code_generator.py` is an 11.7k-line file whose generated app is one giant f-string
template (all braces doubled `{{ }}`); edits are delicate and context-heavy. There is no
`codex-companion.mjs` runtime in this repo (checked; only the bare `codex` CLI). Concurrent
Codex edits to the same file as tests regenerate examples via subprocess would race. The
judgment-to-mechanics ratio of security code in a template DSL is high — this is the
"Claude decides" category.

## Wave A design decisions

- **scrypt default, PBKDF2-SHA256 fallback** (per `../generated-app-runtime-baseline/`):
  scrypt is memory-hard (128 MiB at N=2^17, r=8, p=1) — GPU-resistant where PBKDF2 is not.
  Both live in `hashlib` (stdlib-only constraint preserved). Verification accepts both formats
  so operators can choose; `hash_password()` emits scrypt.
- **Keep plaintext `password` field working** — dev ergonomics; `password_hash` wins when both
  present. Generated apps must boot with zero config. Production mode is where enforcement
  tightens.
- **Sliding-window login throttle keyed by username+IP, in-memory** — ASVS 6.3.1 L1. In-memory
  is per-worker; a generated comment tells operators to swap Redis for multi-worker. Chose
  bounded lockout with `Retry-After` over progressive delay: simpler to test, no thread sleeps
  in request handlers.
- **Session fixation**: `session.clear()` + repopulate on login. In Flask's signed-cookie model
  this is equivalent to session-ID rotation (the cookie value is the session). Verified
  observable: CSRF token differs pre/post login.
- **`MAX_CONTENT_LENGTH` 16 MiB default**, `APG_MAX_BODY_BYTES` override — Werkzeug has no
  default limit; unbounded bodies are a memory-DoS vector.
- **Branded 404/500 with content negotiation** — JSON for `/api/*` or `Accept: application/json`,
  branded HTML otherwise; never leak tracebacks (ASVS V16).
- **Timing-safe API-key compare** (`hmac.compare_digest`) — `==` at line 2707 was a timing oracle.
- **Production-mode secure-by-default mutations**: `_authorized()` returning True with no
  configured key is fine for dev, but `APG_PRODUCTION=1` with no API key, no JWT, and no session
  user must deny JSON mutations. Session users (CSRF-verified) remain authorized — the UI keeps
  working.
- **HSTS stays conditional on `request.is_secure`** — research confirmed emitting HSTS on HTTP
  permanently poisons dev environments (browser-persisted). Existing behavior is correct; not a gap.

## Environment repair decisions

- **Replaced stale editable flask-appbuilder with PyPI 5.2.2** rather than restoring the fork:
  `git ls-files` in `fab-ext` shows `flask_appbuilder/` is no longer tracked — the fork was
  restructured into `pgappforge_*` packages. The editable finder pointed at a deleted path.
- **Upgraded cryptography 3.4.8 → 49.0.0 + pyOpenSSL → 26.3.0 in both homebrew and the hermes
  venv**: the pair must move together (`utils.deprecated(name=)` / `GEN_EMAIL` skew breaks either
  direction); 3.4.8 is 2021-era with known CVEs. Risk to hermes-agent accepted — modern
  cryptography is API-compatible for mainstream usage and strictly safer.

## Test strategy

TDD per project standard: `tests/test_generated_app_hardening.py` written before generator
changes; each wave runs the full suite (1484-test baseline) before commit; tests compile a
real APG source and exercise the generated app via Flask test client — no mocks.
