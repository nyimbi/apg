# Thinking Log — World-Class Uplift Mission (raw, stream-of-consciousness)

## 2026-07-10 — session start

- Prompt: "make everything better than world class, research first, save everything."
  Scope is unbounded → must self-scope. Signals for where to aim: recent commit series
  (a646364d, 7375c3b8, d58f3636) is all generated-app security hardening; staged work in
  tree continued it (session cookies). The momentum and the multiplier (one generator fix →
  every generated app) both point at the generated-app runtime. Platform-level gaps
  (durable execution, connectors) are real but each is a subsystem, not a wave.
- Found staged-but-uncommitted session-cookie hardening → verified (7/7) → committed 5f53f346.
  Never let verified work sit unpushed.

## Baseline hunt

- First full-suite run: 3 collection errors, all `flask_appbuilder` missing. Surprise: FAB is
  mandated by CLAUDE.local.md for capability UIs, yet not importable — how did 1484 tests pass
  before? Hypothesis: previous runs used an env where the editable install still resolved.
  The fab-ext checkout moved under it.
- pip said FAB installed; python said no module. Classic editable-install rot: the `.pth`
  finder mapped to a deleted directory. Editable installs are invisible dependencies on
  sibling checkouts — noted as a systemic risk for this machine.
- Uninstall refused ("outside environment") — hermes venv + homebrew hybrid. Had to drive
  homebrew python directly with --break-system-packages. Then the SECOND layer surfaced:
  pyOpenSSL/cryptography skew, which had been masked by the earlier collection abort.
  Lesson: collection errors mask each other; fix → rerun → expect new ones.
- cryptography 3.4.8 in a 2026 environment is itself a finding (CVEs). Upgrading was not just
  test-fixing but a real security repair of the dev machine.
- Wrong first attempt: `pip install --python` placed after subcommand → error. Flag order matters.

## Wave A design churn

- Initial test design assumed PBKDF2-SHA256 (600k iters, OWASP number). Research agent came
  back recommending scrypt-first (memory-hard, GPU-resistant; PBKDF2 falls to GPU rigs even at
  600k). Both are stdlib `hashlib`. Decision: `hash_password()` emits scrypt; verifier accepts
  both `scrypt$...` and `pbkdf2_sha256$...` so operators can bring either. Tests updated
  rather than defending the first draft — the point of researching before building.
- Considered progressive delay instead of lockout (lockout can be weaponized to DoS a user).
  Kept bounded sliding-window lockout keyed username+IP: testable without sleeps, Retry-After
  is honest UX, and per-IP keying blunts the DoS-a-victim vector. Documented Redis caveat for
  multi-worker.
- Session fixation in Flask signed-cookie model: there is no server-side session ID to rotate;
  clearing + repopulating the session changes the signed cookie value entirely — equivalent
  defense. Observable for tests: CSRF token (stored in session) must differ pre/post login.
- `_authorized()` returning True when no API key configured: dev-friendly, prod-dangerous.
  Rather than break dev (every example app would 401), gate the deny on `_production_mode()`
  which the cookie-secure work already introduced. Secure-by-default where it matters,
  zero-config where it doesn't. Session users must stay authorized or the UI breaks —
  CSRF already authenticates them.
- HSTS: my instinct said "emit always in prod mode"; research says conditional-on-HTTPS is
  CORRECT because HSTS is browser-persisted and poisons dev origins. Existing code stands.
  (A gap list that survives contact with research is shorter than the one you started with.)

## Competitive research surprises

- Wasp's own post-mortem: the custom DSL was their fatal friction. APG's DSL strategy needs
  LLM-assisted authoring so users rarely hand-write it. Filed for a later wave.
- Amplication domain now redirects to overcut.ai — market is consolidating.
- Lovable CVE (170 apps, RLS off) validates "secure-by-default generation" as a moat, which is
  exactly the wave being built. The WebAIM number (95.9% of the web fails WCAG) means the a11y
  wave (D) lands APG in the top few percent by default.
- "ARIA present → MORE errors on average" — partial ARIA is worse than none. Generator must
  emit complete widget patterns or nothing. Constraint adopted for Wave D.

## Process notes

- Held generator edits while the full suite runs: some tests boot generated apps via
  subprocess; editing the template mid-run poisons the baseline signal for the FAB 5.2.2
  upgrade. Test-file writes are safe (collection already done).
- Codex delegation rejected for this file (giant f-string template, doubled braces, no
  codex-companion runtime present) — recorded in rationale.md.
