# APG Platform Code Review — Auth Hub, Core Banking GL, SACCO System

**Date:** 2026-06-11
**Commits reviewed:** `c1abeb3f` (Auth Hub) · `ffa3eaeb` (Core Banking GL + SACCO) · preceding SACCO commits
**Reviewer:** 6 parallel specialist agents (Architecture, Security, Code Quality, Performance, Testing, Documentation)

---

## 📊 Executive Summary

Three major additions reviewed: Auth Hub (interchangeable auth/authz adapter, 5 providers), Core Banking General Ledger (double-entry engine), and SACCO system (FOSA, check-off, guarantor, group lending, SASRA reporting, SACCO GL).

**The three additions share strong design vocabulary** — Protocol adapters, BoundedCache, Decimal arithmetic, guard_* validators — and the SACCO GL is the standout with genuine double-entry enforcement and a comprehensive test suite. However, **three security bugs can each independently compromise the entire platform** without an external attacker: the factory defaults to the null provider when env vars are missing, JWT tokens are signed with a hardcoded fallback secret, and the token cache key uses only the first 32 bytes of the token (identical for all JWTs issued with the same algorithm). Additionally, `fin/gl` GLService has zero tests despite being a production accounting engine, and `close_year()` only zeroes one income account rather than all P&L accounts.

---

## Quality Scorecard

```
┌─────────────────┬───────┬──────────────────────────────────────────────────┐
│ Aspect          │ Score │ Notes                                            │
├─────────────────┼───────┼──────────────────────────────────────────────────┤
│ Architecture    │ 6/10  │ Adapter pattern correct; FOSA/GL split broken;   │
│                 │       │ middleware event loop issue critical             │
│ Code Quality    │ 7/10  │ Strong in SACCO GL; circuit breaker inconsistent │
│                 │       │ across providers; 4 divergent guard patterns     │
│ Security        │ 4/10  │ 3 critical bugs: null default, hardcoded secret, │
│                 │       │ token prefix collision                           │
│ Performance     │ 5/10  │ O(n×m) GL scans unacceptable; no connection pool;│
│                 │       │ CB lock on every request                         │
│ Testing         │ 5/10  │ SACCO GL: excellent; fin/gl: zero; middleware:   │
│                 │       │ untested; null→production bypass: untested       │
│ Documentation   │ 6/10  │ Provider matrix excellent; no migration guide;   │
│                 │       │ 4 API endpoints return wrong HTTP status codes   │
└─────────────────┴───────┴──────────────────────────────────────────────────┘
```

**Issue totals:** 6 Critical · 15 High · 10 Medium

---

## 🔴 Critical Issues (Fix Immediately)

### C1. 🔒 Factory Defaults to Null Provider — Silent Auth Bypass on Misconfiguration
**File:** `capabilities/common/auth_hub/factory.py:59,89`

`os.environ.get("APG_AUTH_PROVIDER", "null")` — a missing, misspelled, or overwritten env var silently
starts accepting all tokens and approving all permissions. An attacker who can delete or overwrite
the env var achieves full platform access.

```python
# Add to factory._create_auth_provider():
def _assert_not_null_in_production(name: str, var: str) -> None:
    env = os.environ.get("APG_ENV", os.environ.get("FLASK_ENV", "development")).lower()
    if env in ("production", "prod", "staging") and name in ("null", "dev", "test", ""):
        raise RuntimeError(
            f"SECURITY: {var}={name!r} is dev-only and must not be used in APG_ENV={env!r}. "
            f"Valid providers: keycloak, clerk, betterauth, fab"
        )
```

Also add `reset_providers(*, _testing_only: bool = False)` guard to prevent accidental production resets.

---

### C2. 🔒 JWT Secret Falls Back to Hardcoded `"apg-dev-secret"` — Token Forgery
**File:** `capabilities/common/auth_hub/providers/fab_provider.py:60`

```python
os.environ.get("APG_JWT_SECRET", os.environ.get("SECRET_KEY", "apg-dev-secret"))
```
Anyone with codebase access can mint valid tokens. Replace with:
```python
def _get_jwt_secret() -> str:
    secret = os.environ.get("APG_JWT_SECRET") or os.environ.get("SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "APG_JWT_SECRET must be set. "
            "Generate: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    if len(secret) < 32:
        raise RuntimeError(f"APG_JWT_SECRET is only {len(secret)} chars — minimum 32 required")
    return secret
```

---

### C3. 🔒 Token Cache Key = `token[:32]` — Cache Collision Enables Cross-User Data Leakage
**Files:** `keycloak_provider.py:150,185` · `clerk_provider.py:114,146` · `betterauth_provider.py:131,165` · `fab_provider.py:144,157`

JWT tokens share the same header prefix `eyJhbGciOiJ...` — the first 32 bytes are identical for all
tokens issued with the same algorithm. Two distinct users can collide on the cache key, returning
the wrong user's `TokenPayload`. A revoked token with a crafted matching prefix can poison the cache.

```python
import hashlib

def _cache_key(token: str) -> str:
    return hashlib.blake2b(token.encode(), digest_size=32).hexdigest()

# Replace all token[:32] with _cache_key(token) across all 4 providers
# In logout: self._token_cache.delete(_cache_key(token))
```

---

### C4. 🔒 BetterAuth Authz Falls Through to NullAuthzProvider Silently
**File:** `capabilities/common/auth_hub/factory.py:108-113`

`APG_AUTH_PROVIDER=betterauth APG_AUTHZ_PROVIDER=betterauth` → null authorization with a WARNING log
that may be filtered. Silent privilege escalation. Fix:
```python
if name in ("betterauth",):
    env = os.environ.get("APG_ENV", "development").lower()
    if env in ("production", "prod", "staging"):
        raise ValueError(
            "BetterAuth has no authz provider. Set APG_AUTHZ_PROVIDER=spicedb for production."
        )
```

---

### C5. 🏗️ `sync_wrapper` in Middleware Spawns New Event Loop Per Request
**File:** `capabilities/common/auth_hub/middleware.py:82-88`

`asyncio.new_event_loop()` inside a running async context (Quart, ASGI) raises `RuntimeError`.
The new loop doesn't share the process-level provider cache or circuit breaker state. `flask.g`
state written inside the async wrapper won't be visible in the sync route handler.

```python
def require_auth(fn: Callable) -> Callable:
    if not asyncio.iscoroutinefunction(fn):
        raise TypeError(
            f"@require_auth requires 'async def' view function: {fn.__name__!r}. "
            "Install flask[async] and use async views."
        )
    # async_wrapper only — remove sync_wrapper entirely
```

---

### C6. 🧪 `fin/gl` GLService Has Zero Tests — Financial Accounting Engine Entirely Unverified
**File:** `capabilities/fin/gl/tests/` (empty — only scaffolded subdirectories, no `.py` files)

The double-entry invariant, `GLImbalanceError`, `PostingToClosedPeriodError`, trial balance, P&L,
balance sheet, year-close, and SASRA capital adequacy report are all untested. This is a production
accounting engine consumed by multiple other capabilities.

Minimum viable tests to add immediately:
```python
# capabilities/fin/gl/tests/test_service.py

async def test_imbalanced_entry_raises(svc):
    with pytest.raises(GLImbalanceError) as exc:
        await svc.post_journal_entry(entries=[
            {"account_code": "1010", "debit_amount": 10000, "credit_amount": 0},
            {"account_code": "2100", "debit_amount": 0, "credit_amount": 9999},  # off by 1
        ], description="bad", reference="BAD-001", posting_date=TODAY, period_id="2025-01")
    assert exc.value.debits == Decimal("10000.00")
    assert exc.value.credits == Decimal("9999.00")

async def test_closed_period_rejects_posting(svc):
    await svc.close_period("2024-12")
    with pytest.raises(PostingToClosedPeriodError):
        await svc.post_journal_entry(..., period_id="2024-12")

async def test_trial_balance_balanced_after_posting(svc):
    await svc.post_journal_entry(entries=[
        {"account_code": "1010", "debit_amount": 5000, "credit_amount": 0},
        {"account_code": "2100", "debit_amount": 0, "credit_amount": 5000},
    ], ...)
    tb = await svc.get_trial_balance()
    total = next(r for r in tb if r["code"] == "TOTAL")
    assert total["balanced"] is True
```

---

## 🟠 High Priority Issues

### H1. 🏗️ `@require_permission` Double-Wraps `@require_auth` — `validate_token` Called Twice Per Request
**File:** `capabilities/common/auth_hub/middleware.py:106`

Every `@require_permission` route calls `validate_token()` twice. For Keycloak with HTTP introspection,
this doubles auth latency on every protected endpoint. Inline the auth check directly in `require_permission`'s
wrapper rather than delegating to `require_auth`.

---

### H2. ⚡ GL Balance Computation: O(accounts × entries) Per Report — Production Catastrophic
**File:** `capabilities/fin/gl/service.py:451-481`

`_compute_balance()` iterates all journal entries on every call. `get_trial_balance()` calls it
for every account. For 50 accounts × 1.825M entries (10 years of 500 txns/day) = **456M iterations
per trial balance ≈ 4.5 seconds blocking the event loop.** `_balance_cache.clear()` on every post
means zero effective cache hit rate in a write-heavy system.

Fix — maintain running balances on write:
```python
def _apply_entry_to_balance(self, je: dict) -> None:
    """O(lines_in_entry) — called once per post, replaces O(all_entries) per read."""
    for line in je["lines"]:
        code = line["account_code"]
        acc = self._accounts.get(code)
        if acc is None:
            continue
        dr = Decimal(str(line["debit_amount"]))
        cr = Decimal(str(line["credit_amount"]))
        delta = (dr - cr) if acc["normal_balance"] == "DEBIT" else (cr - dr)
        self._running_balances[code] = self._running_balances.get(code, Decimal("0")) + delta

# In post_journal_entry: replace _balance_cache.clear() with self._apply_entry_to_balance(je)
# In _compute_balance: O(1) return self._running_balances.get(account_code, Decimal("0"))
```

---

### H3. 💥 `close_year()` Only Closes Account `4100` — All Other P&L Accounts Left Open
**File:** `capabilities/fin/gl/service.py:812`

Year-end close posts entries only against "Interest Income - Loans" (`4100`). Fee income (`4300`),
expense accounts (`5100-5600`), and all other income accounts remain open with non-zero balances.
Retained earnings will be wrong. SASRA capital adequacy ratios on year-end data will be wrong.

Fix: iterate `get_profit_and_loss()` result and zero every P&L account:
```python
async def close_year(self, year: int, closed_by: str = "system") -> dict[str, Any]:
    pnl = await self.get_profit_and_loss(f"{year}-01-01", f"{year}-12-31")
    lines = []
    for item in pnl["income"]:      # zero income accounts (debit them)
        bal = Decimal(item["amount"])
        if bal != 0:
            lines.append({"account_code": item["code"], "debit_amount": bal,
                          "credit_amount": Decimal("0"), "narrative": f"Year {year} close"})
    for item in pnl["expenses"]:    # zero expense accounts (credit them)
        bal = Decimal(item["amount"])
        if bal != 0:
            lines.append({"account_code": item["code"], "debit_amount": Decimal("0"),
                          "credit_amount": bal, "narrative": f"Year {year} close"})
    net = Decimal(pnl["net_surplus"])
    if net != 0:
        lines.append({"account_code": "3300",
                      "credit_amount": net if net > 0 else Decimal("0"),
                      "debit_amount": abs(net) if net < 0 else Decimal("0"),
                      "narrative": "Transfer net surplus to retained earnings"})
    if lines:
        await self.post_journal_entry(entries=lines, description=f"Year {year} close",
            reference=f"YEARCLOSE-{year}", posting_date=f"{year}-12-31",
            period_id=f"{year}-12", posted_by=closed_by)
    return {"year": year, "net_surplus": str(net), "closed_by": closed_by}
```

---

### H4. 🔒 SSRF via `APG_BETTERAUTH_URL` — No URL Validation
**File:** `capabilities/common/auth_hub/providers/betterauth_provider.py:57`

Raw env var with no validation. `APG_BETTERAUTH_URL=http://169.254.169.254/latest/meta-data/` turns
every auth call into an AWS/GCP/Azure IMDS probe, exfiltrating instance credentials. Add scheme +
host validation blocking metadata service IPs and private RFC-1918 ranges in production. Same
validation needed for `APG_KEYCLOAK_URL` and `APG_SPICEDB_URL`.

---

### H5. 🔒 Keycloak: Revoked Tokens Valid for Up to 5 Minutes After Logout
**File:** `capabilities/common/auth_hub/providers/keycloak_provider.py:217-228`

`logout()` deletes the cache entry but does NOT call Keycloak's `/revoke` endpoint for the access
token. A compromised credential remains valid until the cache TTL expires. For a financial platform,
a 5-minute post-revocation window is unacceptable. Add explicit token revocation on logout and a
process-local deny-list.

---

### H6. 🏗️ `FOSAService` Maintains a Separate In-Memory GL — Reconciliation Structurally Broken
**File:** `capabilities/fintech/sacco/fosa/service.py:51,114-141`

`FOSAService` stores GL entries in `self.gl_entries` (a plain list), completely separate from
`SACCOGLService`. `reconcile_subsidiary_ledgers` in `SACCOGLService` will always show a difference
after FOSA transactions. FOSA also hardcodes account codes `1002, 1003, 1004, 2101` that don't
exist in `_STANDARD_SACCO_COA` (which uses `1010`, `2100`, etc.).

Fix: inject `SACCOGLService` into `FOSAService` and call its existing `post_member_deposit`,
`post_withdrawal` methods. Remove `_post_gl` entirely.

---

### H7. 🏗️ Factory Singleton Not Concurrent-Safe — Double Initialization Race
**File:** `capabilities/common/auth_hub/factory.py:35-48`

Two concurrent coroutines both see `_auth_provider is None`, both create providers. Two Keycloak
instances = two independent caches, circuit breakers, admin token state.
```python
_init_lock = threading.Lock()

def get_auth_provider() -> Any:
    if _auth_provider is not None:
        return _auth_provider
    with _init_lock:
        if _auth_provider is None:
            _auth_provider = _create_auth_provider()
    return _auth_provider
```

---

### H8. 💥 `@idempotent` Key Function Has Wrong Signature in `post_batch_entries`
**File:** `capabilities/fin/gl/service.py:427`

Lambda expects `tenant_id` as second positional arg, but method signature has `entries_batch` there
and no `tenant_id` parameter. Every call raises `TypeError` at runtime.
```python
# Current (broken):
@idempotent(key_fn=lambda self, tenant_id, batch_id, **_: f"gl_batch:{tenant_id}:{batch_id}")

# Fix:
@idempotent(key_fn=lambda self, entries_batch, batch_id, **_: f"gl_batch:{self._tenant_id}:{batch_id}")
```

---

### H9. 🔒 Keycloak Admin Token Refresh Has No Lock — Concurrent Thundering Herd
**File:** `capabilities/common/auth_hub/providers/keycloak_provider.py:73-91`

If the admin token expires and 50 concurrent requests all see `expires_at < now`, all 50 fire
refresh requests to Keycloak simultaneously. Add `asyncio.Lock` with double-checked locking inside:
```python
self._admin_token_lock = asyncio.Lock()

async def _get_admin_token(self) -> str:
    if time.monotonic() < self._admin_token_expires - 30:
        return self._admin_token           # fast path, no lock
    async with self._admin_token_lock:
        if time.monotonic() < self._admin_token_expires - 30:  # double-check
            return self._admin_token
        # exactly one refresh happens
        ...
```

---

### H10. 🧪 No Production Safety Guard Test — Null Provider Deployable Without Failure
Critical test missing:
```python
def test_null_provider_blocked_in_production(monkeypatch):
    monkeypatch.setenv("APG_AUTH_PROVIDER", "null")
    monkeypatch.setenv("APG_ENV", "production")
    from capabilities.common.auth_hub.factory import reset_providers, _create_auth_provider
    reset_providers(_testing_only=True)
    with pytest.raises(RuntimeError, match="not permitted in APG_ENV"):
        _create_auth_provider()
```

---

### H11. ⚡ `httpx.AsyncClient` Created Per-Request in 3 Providers — No Connection Pool
**Files:** `keycloak_provider.py:70` · `clerk_provider.py:60` · `betterauth_provider.py:68`

TCP setup adds 1-50ms per auth request. At 1,000 req/s this is 1-50 seconds of aggregate wait per
second — the service collapses under load. Store a shared persistent client:
```python
self._client = httpx.AsyncClient(
    timeout=10.0,
    limits=httpx.Limits(
        max_connections=50,
        max_keepalive_connections=20,
        keepalive_expiry=30.0,
    ),
)
```

---

### H12. ⚡ Circuit Breaker `_before_call()` Acquires asyncio.Lock on Every Request
**File:** `capabilities/common/reliability/circuit_breaker.py:124-136`

Under 10,000 req/s × 3 CB checks per request = 30,000 lock acquisitions/sec. Add a fast path that
skips the lock in the common CLOSED state:
```python
async def _before_call(self) -> None:
    if self._state == CircuitState.CLOSED:
        return   # fast path — no lock needed
    async with self._lock:
        if self._state == CircuitState.OPEN:
            ...  # existing logic
```

---

### H13. 🏗️ Two `GLImbalanceError`/`AccountNotFoundError` Definitions — Type Identity Mismatch
**Files:** `fin/gl/models.py:45,66` and `fin/gl/service.py:40,57`

Two classes with identical names in different modules. Code catching `GLImbalanceError` from
`models` won't catch the one raised in `service.py`. Remove from `service.py`, import from `models`.

---

### H14. 📝 No Migration Guide from `common/auth` → `auth_hub`
Both capabilities coexist with no upgrade path. Minimum addition to `auth_hub/README.md`:
- Class mapping table (`AuthService` → `AuthHubService`)
- Exception type changes (`ValueError` → `AuthenticationError`)
- User ID format differences (FAB integer strings → provider-specific)
- Env var configuration steps

---

### H15. 💥 Float Leakage in Check-Off Service — Silent Amount Corruption
**File:** `capabilities/fintech/sacco/ckf/service.py:464,497,698`
```python
rem["amount_expected"] = float(grand_total)   # silently corrupts Decimal precision
rem["amount_received"] = float(total_received)
```
Python `float` has 15 significant digits. KES amounts above ~1B corrupt silently. Remove all
`float()` wrappers — `Decimal` serialises fine as a dict value.

---

## 🟡 Medium Priority Issues

### M1. 🔒 Keycloak OAuth URL Built with Raw f-String — Parameter Injection Risk
**File:** `keycloak_provider.py:369-375` — A `redirect_uri` containing `&scope=admin&other=` injects
extra query parameters. Use `urllib.parse.urlencode` for all OAuth URL construction.

### M2. ⚡ `timedelta.seconds` Instead of `total_seconds()` — Wrong Token Cache TTL
**File:** `keycloak_provider.py:184` — `timedelta(days=2).seconds == 0` means long-lived tokens
never expire from cache. Use `int((expires_at - datetime.now(tz)).total_seconds())`.

### M3. 🏗️ Four SACCO Services Have Four Different `guard_tenant_id` Patterns
FOSA uses `_tenant()`, guarantor uses `_guard()`, reg has a `try/except ImportError` variant with
different exception type, check-off calls it directly without `self.tenant_id` fallback. Standardise
all four to one pattern: `def _t(self, tenant_id: str | None) -> str`.

### M4. 🧪 Expired Token, Middleware 403, and Provider-Switch Paths Completely Untested
No test exercises `is_expired=True`, the 403 branch in `@require_permission`, or the factory reset
→ new provider path. Add Flask test client tests for all three.

### M5. 🏗️ `SACCOGLService._post_lines` Mutates `GLAccount.balance` — Breaks `as_of_date` Queries
**File:** `fintech/sacco/gl/service.py:154-160` — Running mutable balance and historical recomputed
balance diverge after any backdated posting. Remove balance mutation from `_post_lines`; compute
balance exclusively from journal entries (the `as_of_date` path already does this correctly).

### M6. ⚡ `get_profit_and_loss()` Has Independent O(income_accounts × entries) + O(expense_accounts × entries) Scans
**File:** `fin/gl/service.py:586-629` — Does not call `_compute_balance()`; has its own full-table
scans. Same fix as H2: use the running balance dict.

### M7. 📝 4 API Endpoints Return 400 Instead of 501 for `ProviderNotImplementedError`
**File:** `capabilities/common/auth_hub/api.py:197,217,249,282`
`reset_password`, `verify_magic_link`, `oauth_callback`, `verify_mfa` catch
`(AuthenticationError, ProviderNotImplementedError)` together and return 400. Callers cannot
distinguish "bad input" from "feature not available on this provider". Fix: separate except blocks.

### M8. 📝 `KeycloakAuthzProvider` Docstring Claims Fine-Grained Policies — Implementation Is Simple RBAC
**File:** `keycloak_provider.py:455,483-494` — Class docstring says "Keycloak Authorization
Services with fine-grained policies", but `check_permission` is `permission in roles or "admin" in roles`.
Document actual behavior to prevent production misconfiguration.

### M9. 🔒 `FABAuthzProvider.check_permission` Silently Returns `False` on Infrastructure Errors
**File:** `fab_provider.py:317-335` — A misconfigured FAB security manager causes every permission
check to silently return `False` (deny all). No error surfaced. Add to `AuthzProvider.check_permission`
docstring: providers may return `False` on infrastructure errors; check `health_check()` on unexpected denials.

### M10. 🔒 SpiceDB `bulk_check_permissions` Has No Concurrency Limit — Unbounded Fan-Out
**File:** `spicedb_provider.py:230-253` — 500 permission checks → 500 concurrent HTTP requests to
SpiceDB. Add `asyncio.Semaphore(20)` to bound concurrency.

---

## ✨ Strengths to Preserve

- **Double-entry enforcement in GLService** — `GLImbalanceError` raised before any write, with SHA-256
  tamper-evident hash chain. The invariant is at exactly the right boundary.
- **SACCO GL test suite** — Real objects, no mocks, covers tenant isolation, period close, reconciliation,
  and all standard transaction types. This is the pattern all services should follow.
- **Circuit breaker + timeout on every external call** — Keycloak and BetterAuth providers consistently
  apply `_cb._before_call()/_on_success()/_on_failure()`. Correct pattern.
- **`@runtime_checkable` Protocol-based adapters** — Structural typing is the correct choice over ABC;
  enables testing without import-time coupling to any specific provider.
- **IdempotencyRegistry per-key locking** — Correctly handles concurrent in-flight deduplication with
  per-key asyncio.Lock, preventing the TOCTOU race that most naive implementations miss.
- **SACCO guarantor double-check at acceptance time** — Re-validates eligibility at acceptance (not just
  at request), correctly handling TOCTOU where savings change between request and consent.
- **SASRA regulatory user_guide** — Documents actual regulation references, DPD provision matrix,
  traffic light thresholds, and legal consequences. Publication-quality domain documentation.

---

## 🚀 Top Proactive Improvements

### 1. Shared `_cache_key()` utility and `_http()` persistent client pool in a provider base class

```python
# capabilities/common/auth_hub/providers/_base.py
import hashlib, httpx

class BaseExternalAuthProvider:
    def _cache_key(self, token: str) -> str:
        return hashlib.blake2b(token.encode(), digest_size=32).hexdigest()

    @classmethod
    def _make_client(cls) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=10.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
```

One base class eliminates the security vulnerability (cache collision) and the performance
anti-pattern (new client per request) across all 4 providers simultaneously.

### 2. Eager provider initialization at app startup (eliminates singleton race + fails fast)

```python
# In Flask app factory or lifespan hook — call once before accepting requests
async def startup():
    from capabilities.common.auth_hub.factory import get_auth_provider, get_authz_provider
    auth_health = await get_auth_provider().health_check()
    authz_health = await get_authz_provider().health_check()
    if auth_health["status"] != "ok":
        raise RuntimeError(f"Auth provider unhealthy at startup: {auth_health}")
```

### 3. Materialized running balance in GLService (O(n×m) → O(1))

The single highest-ROI fix in the review. Maintaining `_running_balances` on write:
- Reduces trial balance from minutes to milliseconds at production volume
- Eliminates `_balance_cache.clear()` on every post
- Makes `get_balance_sheet()` and `get_profit_and_loss()` proportional to accounts, not entries

### 4. `CapabilityBase` mixin to standardise tenant guard and datetime patterns

```python
class CapabilityBase:
    def __init__(self, tenant_id: str = "default") -> None:
        from capabilities.common.reliability import guard_tenant_id
        guard_tenant_id(tenant_id)
        self._tenant_id = tenant_id

    def _t(self, tenant_id: str | None = None) -> str:
        from capabilities.common.reliability import guard_tenant_id
        t = tenant_id or self._tenant_id
        guard_tenant_id(t)
        return t

    def _now(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
```

Eliminates the four divergent guard patterns across SACCO services and the `datetime.utcnow()`
deprecation warnings.

---

## 📊 Issue Distribution

| Dimension    | Critical | High | Medium |
|--------------|---------|------|--------|
| Security     | 4       | 3    | 2      |
| Architecture | 1       | 3    | 3      |
| Performance  | 0       | 2    | 3      |
| Testing      | 1       | 2    | 2      |
| Documentation| 0       | 1    | 3      |
| Code Quality | 0       | 4    | 2      |
| **Total**    | **6**   | **15**| **15** |

---

## ⚠️ Systemic Issues (Patterns Requiring Team-Level Attention)

### 1. `token[:32]` prefix as cache key — 4 providers, same bug
One `_cache_key()` utility in a base class eliminates a security vulnerability across all 4 files.
→ **Estimated impact: eliminates cross-user data leakage under concurrent load**

### 2. New `httpx.AsyncClient` per request — 3 providers, same anti-pattern
Will cause connection exhaustion under production load. Shared persistent client with connection
pool is a one-class fix.
→ **Estimated impact: -1-50ms per auth request; prevents connection pool exhaustion**

### 3. Circuit breaker applied to 2-3 methods per provider, not all
Inconsistent fail-fast protection. `refresh_token`, `update_user`, `list_users` bypass the CB
and hang until timeout when the upstream is down.
→ **Fix: `_call(coro)` helper in base provider class, inherited by all providers**

### 4. Factory defaults to null for both `APG_AUTH_PROVIDER` and `APG_AUTHZ_PROVIDER`
One `_assert_not_null_in_production()` function + one test protects the entire platform from
the highest-risk deployment misconfiguration.
→ **Fix: ~10 lines, eliminates the most dangerous single point of failure**

### 5. Four divergent tenant guard patterns across SACCO services
The `guard_tenant_id` / `_tenant()` / `_guard()` / `_t()` / `try/except ImportError` fragmentation
across FOSA, guarantor, reg, and check-off services will cause composition breakage.
→ **Fix: `CapabilityBase` mixin with standardised `_t()` method**

---

## Prioritised Fix Order

| Priority | Fix | Effort | Risk if deferred |
|----------|-----|--------|-----------------|
| P0 | C1 + C4: null provider production guard + test | 30 min | Full platform auth bypass |
| P0 | C2: hardcoded JWT secret | 20 min | Token forgery |
| P0 | C3: token cache key collision | 30 min | Cross-user data leakage |
| P0 | C6: fin/gl tests (minimum viable) | 2 hrs | Undetected accounting errors |
| P1 | H3: close_year closes all P&L accounts | 1 hr | Wrong financials / SASRA non-compliance |
| P1 | H6: FOSA GL integration | 4 hrs | Reconciliation permanently broken |
| P1 | H8: @idempotent key_fn fix in post_batch_entries | 5 min | TypeError on every batch GL post |
| P1 | C5: remove sync_wrapper from middleware | 30 min | Runtime crashes on async Flask |
| P2 | H2 + M6: materialized GL balances | 2 hrs | Production unscalable |
| P2 | H11: persistent httpx client pool | 1 hr | Connection exhaustion under load |
| P2 | M3: standardise SACCO tenant guards | 1 hr | Tenant isolation bugs in composition |
| P3 | H14: migration guide | 2 hrs | Developer confusion / wrong usage |
| P3 | M7: HTTP status codes 400→501 | 30 min | API clients can't distinguish errors |

---

*Generated by 6 parallel specialist code review agents: Architecture · Security · Code Quality · Performance · Testing · Documentation*
*APG Platform · © 2025 Datacraft · www.datacraft.co.ke*
