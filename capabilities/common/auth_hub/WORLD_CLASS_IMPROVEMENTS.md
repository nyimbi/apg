# Auth Hub — World-Class Improvements

Fifteen targeted improvements to lift `auth_hub` from a solid provider facade to
production-grade infrastructure. Each entry specifies why the gap matters, how to
close it, and which real-world system demonstrates the pattern.

---

## 1. Token Introspection Cache with Negative-TTL

**Category**: Performance / Reliability

**Justification**: `validate_token` is invoked on every API request. Without an
in-process cache keyed on the token's SHA-256 fingerprint, every hot path hits the
auth provider over the network. Keycloak's own benchmarks show token introspection
at 2–8 ms remote vs. ~0.02 ms in-process. At p99 latency budgets of 50 ms the
difference is the entire latency floor. A *negative* TTL (caching "invalid" results
for a short window) prevents thundering-herd replay attacks on revoked tokens.

**Implementation**:
```python
import hashlib, time
from dataclasses import dataclass, field

@dataclass
class _CacheEntry:
	payload: TokenPayload | None  # None = invalid
	valid: bool
	expires: float  # monotonic

class TokenValidationCache:
	def __init__(self, positive_ttl: float = 60.0, negative_ttl: float = 10.0):
		self._store: dict[str, _CacheEntry] = {}
		self._positive_ttl = positive_ttl
		self._negative_ttl = negative_ttl

	def _key(self, token: str) -> str:
		return hashlib.sha256(token.encode()).hexdigest()

	def get(self, token: str) -> _CacheEntry | None:
		entry = self._store.get(self._key(token))
		if entry and time.monotonic() < entry.expires:
			return entry
		return None

	def set(self, token: str, payload: TokenPayload | None, valid: bool) -> None:
		ttl = self._positive_ttl if valid else self._negative_ttl
		self._store[self._key(token)] = _CacheEntry(
			payload=payload, valid=valid, expires=time.monotonic() + ttl
		)
```

Integrate into `AuthHubService.validate_token` with a background eviction sweep
called once every `positive_ttl` seconds via `asyncio.create_task`.

**Competitor Reference**: AWS Cognito's [local JWT validation](https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-with-identity-providers.html)
avoids network round-trips entirely for JWTs. Okta's SDK caches introspection
responses with configurable `cache_ttl`.

---

## 2. Structured Audit Log Sink

**Category**: Compliance / Observability

**Justification**: PCI-DSS 10.2, SOC 2 CC6.8, and ISO 27001 A.12.4 all require
tamper-evident audit trails for auth events. `_log.info(...)` to a text file is not
a compliant audit log — it lacks event schema, actor identity, and forwarding to a
SIEM. Every `authenticate`, `assign_role`, `delete_user`, and `check_permission`
call must emit a structured event with at minimum: timestamp, actor, action,
resource, outcome, tenant, source IP, and trace ID.

**Implementation**: Add an `AuditSink` protocol and a default `LogAuditSink` writing
JSON-L. Plug in Kafka, CloudWatch, or Datadog sinks via DI. Every mutating method
calls `await self._audit.emit(AuditEvent(...))` after the provider call.

**Competitor Reference**: Auth0's [log streams](https://auth0.com/docs/customize/log-streams)
and Okta's System Log API both model structured auth events as first-class objects
shipped to SIEM targets. Clerk exposes a `/audit` endpoint returning typed event
records.

---

## 3. Adaptive Rate Limiting per Identity

**Category**: Security

**Justification**: Credential stuffing attacks target `authenticate` with 10k+
attempts/minute per IP or email. A single shared rate limit is too blunt; per-identity
limits (email + IP sliding window) with exponential backoff after 5 failures are
standard. NIST 800-63B Section 5.2.2 recommends throttling after repeated failed
attempts and locking accounts after a configurable threshold.

**Implementation**:
```python
class _RateLimitState:
	failures: int = 0
	locked_until: float = 0.0

async def _check_rate_limit(self, identity: str) -> None:
	state = self._rate_limit_store.get(identity, _RateLimitState())
	if time.monotonic() < state.locked_until:
		raise AuthenticationError("Account temporarily locked", code="rate_limited")
	# record failure on AuthenticationError; clear on success

async def authenticate(self, credentials):
	identity = credentials.get("email") or credentials.get("username", "")
	await self._check_rate_limit(identity)
	try:
		result = await self._auth.authenticate(credentials)
		self._rate_limit_store.pop(identity, None)
		return result
	except AuthenticationError:
		await self._record_failure(identity)
		raise
```

Use Redis sorted-set sliding windows for multi-process deployments.

**Competitor Reference**: Cloudflare Access and Okta both implement per-identity
exponential backoff. BetterAuth has a built-in `rateLimit` plugin. Auth0's anomaly
detection uses a similar per-user brute-force shield.

---

## 4. Permission Decision Caching (AuthzCache)

**Category**: Performance

**Justification**: In SpiceDB-backed deployments, `check_permission` involves a gRPC
call with consistency semantics. Latency is 5–20 ms per call. Multi-tenant SaaS
dashboards call it 20–50 times per page render. A time-bounded per-user permission
cache with cache-key `(user_id, permission, tenant_id, resource_type, resource_id)`
cuts this to one warm hit per cache window. Invalidation triggers on `assign_role`,
`revoke_role`, `write_relationship`, and `delete_relationship`.

**Implementation**: Add `AuthzCache` to `__init__` (default TTL 30 s). Cache positive
and negative results. Tag cache entries by `user_id` for bulk invalidation.
Thread-safe using `asyncio.Lock` per user.

**Competitor Reference**: Open Policy Agent (OPA) ships a decision cache out of the
box. Casbin has a caching layer in `casbin-server`. Google Zanzibar's design paper
describes hedge caching and zookie-consistent lookups to balance freshness and
latency.

---

## 5. Synchronous Context-Propagation Helpers (Tenant Guard)

**Category**: Developer Experience / Multi-tenancy

**Justification**: Every method accepts an optional `tenant_id` and falls back to
`self._tenant_id`. There is no enforcement that callers actually set the tenant when
operating in a multi-tenant context. Production incidents where `tenant_id="default"`
leaks across tenant boundaries are common. A `tenant_context()` async context manager
that temporarily overrides `_tenant_id` and asserts it is not the bare default in
non-dev environments closes this class of bug at the call site.

**Implementation**:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def tenant_context(self, tenant_id: str):
	guard_tenant_id(tenant_id)
	prev = self._tenant_id
	self._tenant_id = tenant_id
	try:
		yield self
	finally:
		self._tenant_id = prev
```

**Competitor Reference**: Clerk's `auth().orgId` and Supabase's `supabase.auth.setSession`
both scope sessions to organizational units. WorkOS's `withOrganization` wrapper
provides similar request-scoped tenant isolation.

---

## 6. Passkey / WebAuthn Support

**Category**: Feature / Security

**Justification**: FIDO2/WebAuthn is now the dominant phishing-resistant second factor
and is rapidly replacing TOTP as first factor. NIST 800-63B Authenticator Assurance
Level 2 and 3 both name passkeys as preferred. Clerk, Auth0, and Hanko all support
WebAuthn natively. Adding `register_passkey`, `verify_passkey_assertion`, and
`list_passkeys` to the protocol (with a `ProviderNotImplementedError` default on
providers that don't support it) future-proofs the facade without breaking existing
providers.

**Implementation**: Extend `AuthProvider` protocol with three new methods. Implement
on `KeycloakAuthProvider` (WebAuthn policy in Keycloak 22+), `ClerkAuthProvider`
(Clerk PassKeys API), and stub `ProviderNotImplementedError` on others.

**Competitor Reference**: Hanko is built around passkeys as a first-class primitive.
Clerk added PassKeys (FaceID, TouchID, Windows Hello) in 2024. Auth0 supports
WebAuthn via their `passkeys` connection.

---

## 7. Delegated Token Exchange (Token Impersonation)

**Category**: Feature / Service Mesh

**Justification**: Service-to-service calls where a backend service needs to act on
behalf of a user require RFC 8693 Token Exchange. Without this, services either share
a service account (privilege escalation risk) or pass the user's token to downstream
services (token leakage). Keycloak 20+ and Auth0 both implement RFC 8693 natively.
An `exchange_token` method enables the APG microservice mesh to propagate user context
without credential sharing.

**Implementation**:
```python
async def exchange_token(
	self,
	subject_token: str,
	target_service: str,
	requested_token_type: str = "urn:ietf:params:oauth:token-type:access_token",
	scope: list[str] | None = None,
) -> TokenPair:
	guard_non_empty_string(subject_token, "subject_token")
	guard_non_empty_string(target_service, "target_service")
	return await self._auth.exchange_token(subject_token, target_service, requested_token_type, scope)
```

**Competitor Reference**: Keycloak's [token exchange](https://www.keycloak.org/docs/latest/securing_apps/#_token-exchange)
implements RFC 8693. AWS STS `AssumeRoleWithWebIdentity` solves the same problem at
the cloud layer.

---

## 8. Continuous Session Risk Scoring

**Category**: Security / Zero-Trust

**Justification**: Traditional auth validates identity at login time only. Zero-trust
requires continuous validation: if a session's risk score spikes (new country,
impossible travel, unusual resource access pattern) mid-session, it must be
challenged or revoked. CISA's Zero Trust Maturity Model Level 3 requires continuous
monitoring of session context. This feeds directly into `check_permission` context
enrichment.

**Implementation**:
```python
async def score_session_risk(
	self,
	user_id: str,
	session_id: str,
	event_context: dict[str, Any],
) -> dict[str, Any]:
	"""Return {score: float[0,1], factors: list[str], action: allow|challenge|revoke}"""
```

Integrate with `require_auth` middleware to inject risk context on each request.

**Competitor Reference**: Auth0's [Adaptive MFA](https://auth0.com/docs/secure/multi-factor-authentication/adaptive-mfa)
uses risk signals. Okta ThreatInsight scores every authentication event. Microsoft
Entra Conditional Access uses real-time risk evaluation.

---

## 9. Cross-Tenant Federation (Identity Bridging)

**Category**: Feature / Enterprise

**Justification**: Enterprise customers often have their own IdP (SAML, OIDC) and
need to federate into the APG tenant. Without `federate_user` and `get_federated_identity`
methods, onboarding enterprise SSO requires provider-specific provisioning scripts.
A federation layer abstracts SAML assertions and external OIDC tokens into `AuthUser`
objects with tenant-scoped roles.

**Implementation**:
```python
async def federate_user(
	self,
	external_token: str,
	external_provider: str,  # "saml" | "oidc" | "google-workspace" | "azure-ad"
	tenant_id: str | None = None,
	auto_provision: bool = True,
) -> AuthResult:
	guard_non_empty_string(external_token, "external_token")
	guard_tenant_id(tenant_id or self._tenant_id)
	return await self._auth.federate_user(external_token, external_provider, tenant_id, auto_provision)
```

**Competitor Reference**: WorkOS is built around enterprise federation (SAML, OIDC,
Azure AD, Google Workspace). Clerk's SAML SSO and Keycloak's Identity Brokering
solve this at the provider level.

---

## 10. Policy-as-Code Integration (OPA / Cedar)

**Category**: Authorization / Compliance

**Justification**: RBAC with static role→permission mappings cannot express time-bound
permissions, geographic restrictions, data classification policies, or contextual
constraints (e.g., "VP only during business hours"). ABAC policy engines like OPA
(Rego) or AWS Cedar can evaluate these in <1 ms when colocated. Adding an optional
`policy_engine` field to `AuthHubService` that falls back to the authz provider
enables gradual policy migration without a flag day.

**Implementation**: Add `PolicyEngine` protocol with `evaluate(input: dict) -> bool`.
Implement `OPAEngine` (HTTP REST API) and `CedarEngine` (Python bindings). Wire
into `check_permission` as pre-authz hook.

**Competitor Reference**: Netflix's Conductor uses OPA for workflow authorization.
AWS IAM now exposes Cedar as the policy language. Styra's DAS is built on OPA.

---

## 11. Secrets-Aware Configuration (No Env Vars in Prod)

**Category**: Security / Operations

**Justification**: `factory.py` reads credentials directly from environment variables.
In Kubernetes this means secrets appear in pod specs or ConfigMaps, which are
base64-encoded (not encrypted) and visible to anyone with `kubectl get configmap`.
The factory should support a `SecretBackend` protocol resolving secrets at init time
from Vault, AWS Secrets Manager, or GCP Secret Manager, with the env-var backend as
the default (dev only).

**Implementation**:
```python
class SecretBackend(Protocol):
	async def get(self, key: str) -> str: ...

class EnvSecretBackend:
	async def get(self, key: str) -> str:
		return os.environ[key]

class VaultSecretBackend:
	async def get(self, key: str) -> str:
		# hvac client call
		...
```

Inject `SecretBackend` into factory; default to `EnvSecretBackend`.

**Competitor Reference**: HashiCorp Vault's dynamic secrets. Doppler and Infisical
as lightweight alternatives. AWS EKS IRSA + Secrets Store CSI Driver is the
Kubernetes-native pattern.

---

## 12. Graceful Provider Failover (Circuit Breaker)

**Category**: Reliability

**Justification**: The current implementation propagates provider exceptions directly.
A flap in the Keycloak cluster or a SpiceDB network partition returns 503 to all
authenticated users. A circuit breaker (`CLOSED → OPEN → HALF_OPEN`) absorbs
transient faults. In `OPEN` state, fall back to cached token validation and deny-safe
permission defaults (fail closed on write, fail open on read with warning log). This
matches the SRE principle of graceful degradation.

**Implementation**: Wrap each provider call in `breaker.call(coro)`. Use
`circuitbreaker` PyPI package or a hand-rolled 20-line state machine. Expose circuit
state in `health_check()`.

**Competitor Reference**: Hystrix (Netflix), resilience4j (Java ecosystem), and
`circuitbreaker` (Python) all implement this. Kong's auth plugin has built-in
circuit breaking for OIDC upstream calls.

---

## 13. Tenant-Scoped Permission Inheritance Hierarchy

**Category**: Authorization / Multi-tenancy

**Justification**: Multi-tenant SaaS products commonly need permissions that cascade:
`platform:admin` inherits all `tenant:admin` permissions, which inherit all `user`
permissions. Without a hierarchy, every new resource type requires manual role
explosion. Adding `create_permission_hierarchy`, `get_inherited_permissions`, and
`check_permission_with_inheritance` enables role pyramids without migrating to
SpiceDB.

**Implementation**:
```python
async def create_permission_hierarchy(
	self,
	parent_permission: str,
	child_permissions: list[str],
	tenant_id: str | None = None,
) -> None: ...

async def get_inherited_permissions(
	self,
	permission: str,
	tenant_id: str | None = None,
) -> list[str]: ...
```

Store hierarchy in PostgreSQL adjacency list; evaluate at check time with recursive
CTE.

**Competitor Reference**: Oso's role hierarchy. Permit.io's hierarchical RBAC.
AWS IAM policy inheritance via permission boundaries.

---

## 14. Developer Experience: Typed Capability Contract

**Category**: Developer Experience

**Justification**: Callers import `AuthHubService` and discover its API through IDE
autocomplete, but the method signatures use `dict[str, Any]` for credentials and
`dict[str, Any]` for permission context. This makes static analysis impossible and
the API hard to get right without reading docs. Replacing loose dicts with typed
`Credentials`, `PermissionContext`, and `ResourceCheck` Pydantic models enables
schema validation at the call site and generates accurate OpenAPI specs.

**Implementation**: Define typed input models in `models.py`. Add overloaded
`authenticate` that accepts either `Credentials` or raw dict (for backward compat).
Generate JSON Schema from models for API documentation.

**Competitor Reference**: Clerk's TypeScript SDK uses discriminated unions for every
auth operation. Auth0's Python SDK uses typed request/response objects. Pydantic
v2's `model_validate` accepts both dict and model instances.

---

## 15. Automated Key Rotation and JWKS Refresh

**Category**: Security / Operations

**Justification**: JWTs are validated against public keys published at the provider's
JWKS endpoint. If the signing key rotates (every 90 days in CIS Benchmark
recommendations) and the JWKS cache is stale, all token validations fail until the
cache is manually flushed. An async background task polling the JWKS endpoint every
`jwks_refresh_interval` seconds with a forced refresh on `kid` miss closes this
operational gap. Without it, key rotation becomes a manual, error-prone deployment
event.

**Implementation**:
```python
class JWKSCache:
	def __init__(self, url: str, refresh_interval: float = 3600.0):
		self._url = url
		self._keys: dict[str, Any] = {}
		self._last_refresh = 0.0
		self._refresh_interval = refresh_interval
		self._lock = asyncio.Lock()

	async def get_key(self, kid: str) -> dict[str, Any]:
		if kid not in self._keys or time.monotonic() - self._last_refresh > self._refresh_interval:
			await self._refresh()
		if kid not in self._keys:
			await self._refresh(force=True)  # kid miss → immediate refresh
		return self._keys[kid]

	async def _refresh(self, force: bool = False) -> None:
		async with self._lock:
			# aiohttp fetch JWKS; update self._keys; set self._last_refresh
			...
```

Expose `jwks_last_refreshed_at` in `health_check()` output.

**Competitor Reference**: Okta's SDK auto-rotates JWKS. AWS Cognito's Python helper
library includes a JWKS cache with kid-miss retry. Auth0 recommends this pattern
in their JWT validation guide.
