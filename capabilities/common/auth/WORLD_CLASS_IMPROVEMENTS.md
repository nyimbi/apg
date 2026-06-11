# AUTH - World Class Improvement Opportunities

**Capability**: Authentication & RBAC (`auth`) | **Domain**: `common`
**Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Persistent Token Refresh with Sliding Expiry Windows

**Category**: Security / Session Lifecycle

**Justification**: The current JWT implementation issues tokens with a fixed expiry and no refresh path. Production systems require short-lived access tokens (5-15 min) paired with long-lived refresh tokens that rotate on use. Without this, operators must choose between short sessions (user friction) or long-lived tokens (attack surface). The current model conflates both into a single token with no rotation semantics.

**Implementation**:
```python
async def oauth2_token_exchange(
    self,
    tenant_id: str,
    grant_type: str,      # "authorization_code" | "refresh_token" | "client_credentials"
    code: str | None = None,
    refresh_token: str | None = None,
    code_verifier: str | None = None,
) -> dict[str, Any]:
    # Validate grant, issue access_token (15min) + refresh_token (7d)
    # Store refresh_token hash; rotate on each use (refresh token rotation)
    # Emit oauth2_token_issued audit event with grant_type
    ...
```

**Competitor Reference**: Auth0's token rotation policy, AWS Cognito's `ALLOW_REFRESH_TOKEN_AUTH` flow, Okta's token lifecycle management.

---

## 2. Attribute-Based Access Control (ABAC) Policy Engine

**Category**: Authorization

**Justification**: Pure RBAC cannot express fine-grained conditions like "allow read on records where record.owner == user.id AND user.department == resource.department AND time is business_hours". The `models.py` already defines `PolicyCondition`, `PolicyCreate`, and `ABACDecisionRequest` Pydantic models, but `service.py` has no corresponding evaluation engine. This is a half-implemented feature that blocks real-world multi-tenant deployments.

**Implementation**:
```python
async def evaluate_abac_policy(
    self,
    tenant_id: str,
    subject_id: str,
    resource: str,
    action: str,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Load active policies sorted by priority
    # Evaluate subject_conditions, resource_conditions, action_conditions, environment_conditions
    # First matching policy wins (deny-overrides variant)
    # Return ABACDecisionResponse-compatible dict with matched_policy + reasons
    ...
```

**Competitor Reference**: AWS IAM condition operators, OPA Rego policy language, Casbin's ABAC model, Google Zanzibar's check API.

---

## 3. OpenID Connect Discovery and JWKS Endpoint Support

**Category**: Federation / Identity Provider Integration

**Justification**: The SAML stub in `saml_response_verify` does structural text matching rather than real cryptographic verification. More critically, there is no OIDC support at all, which is the dominant federation protocol in modern SaaS. Generated applications that compose AUTH with external IdPs (Google, Entra ID, Keycloak) need a standards-compliant token verification path including JWKS URI fetching, `kid`-based key selection, and `nonce` validation.

**Implementation**:
```python
async def oidc_verify_id_token(
    self,
    tenant_id: str,
    id_token: str,
    issuer: str,
    client_id: str,
    nonce: str | None = None,
    jwks_uri: str | None = None,
) -> dict[str, Any]:
    # Decode header to extract kid
    # Fetch JWKS (cached per issuer, 5min TTL)
    # Verify RS256/ES256 signature using matching JWK
    # Validate iss, aud, exp, iat, nonce claims
    # Map sub -> tenant identity via federated_mesh adapter
    ...
```

**Competitor Reference**: Auth0 OIDC support, Google Identity Platform, Microsoft MSAL, Keycloak's OIDC broker.

---

## 4. Step-Up Authentication with Challenge Orchestration

**Category**: Adaptive Authentication

**Justification**: `start_session` accepts `step_up_completed: bool` as a flat parameter with no mechanism to trigger or orchestrate the step-up challenge. High-value operations (fund transfers, admin role assignment, data export) require a structured challenge-response cycle: detect that the current session trust is insufficient, issue a step-up challenge (TOTP, push notification, hardware key), verify the response, and elevate the session trust score. Without this, `step_up_completed=True` is unenforceable.

**Implementation**:
```python
async def issue_step_up_challenge(
    self,
    tenant_id: str,
    session_id: str,
    required_assurance_level: str,   # "mfa" | "hardware_key" | "biometric"
    operation_context: str,
) -> dict[str, Any]:
    # Validate session exists and is active
    # Determine eligible challenge methods from identity's MFA registrations
    # Issue challenge record with nonce + expiry (5 min)
    # Return challenge_id, methods[], expires_at
    ...

async def verify_step_up_challenge(
    self,
    tenant_id: str,
    challenge_id: str,
    response_token: str,
) -> dict[str, Any]:
    # Validate response against challenge nonce
    # On success: elevate session trust, set step_up_completed=True
    # Emit step_up_completed audit event
    ...
```

**Competitor Reference**: Okta Step-Up Authentication, Auth0 Acr Values, Duo Security's secondary auth, FIDO2 transaction confirmation.

---

## 5. Delegated Authorization (Impersonation + Constrained Delegation)

**Category**: Authorization / Governance

**Justification**: `models.py` defines `DelegationCreate` and `DelegationResponse` but `service.py` has no delegation lifecycle. Support teams, operations engineers, and automated pipelines regularly need time-bounded, permission-constrained, audit-logged impersonation. Without it, operators grant permanent elevated roles (dangerous) or share credentials (catastrophic). Constrained delegation - where Alice delegates only a subset of her permissions to Bob for a limited time - is a SOC2 and ISO 27001 compliance requirement in multi-tenant SaaS.

**Implementation**:
```python
async def grant_delegation(
    self,
    tenant_id: str,
    delegator_id: str,
    delegate_id: str,
    permission_ids: list[str],
    expires_at: str,          # ISO8601
    justification: str,
    requires_mfa: bool = True,
) -> dict[str, Any]:
    # Validate delegator has all permissions in permission_ids
    # Validate delegate exists in tenant
    # Store delegation record with full audit trail
    # Emit delegation_granted event
    ...
```

**Competitor Reference**: AWS IAM role assumption with `sts:AssumeRole`, Azure AD delegation scopes, Okta's `on_behalf_of` token flow, Google Workspace domain-wide delegation.

---

## 6. Real-Time Threat Intelligence Feed Integration

**Category**: Risk / Security Intelligence

**Justification**: `contextual_risk.py` hardcodes lists and uses static scores. A production auth layer needs to consume a threat intelligence feed (STIX/TAXII, commercial feeds, or an internal SIEM) and update risk scores dynamically. IP reputation, ASN reputation, and TOR exit node lists change hourly. Stale data means high-risk sessions get scored as low-risk.

**Implementation**:
```python
async def ingest_threat_feed(
    self,
    tenant_id: str,
    feed_url: str,
    feed_type: str,         # "stix", "iplist", "asn", "custom_json"
    api_key: str | None = None,
    max_entries: int = 10_000,
) -> dict[str, Any]:
    # Fetch feed (async HTTP with timeout, retry, circuit breaker)
    # Parse entries and upsert into BoundedCache with TTL
    # Return ingestion summary: added, updated, expired, errors
    # Emit threat_feed_ingested audit event
    ...
```

**Competitor Reference**: Cloudflare's threat intelligence API, Palo Alto AutoFocus, CrowdStrike Falcon Intelligence, Recorded Future.

---

## 7. WebAuthn / Passkey Registration and Verification

**Category**: Authentication Assurance

**Justification**: `mfa_integration` supports `passkey` as a string type but provides no real WebAuthn ceremony implementation. Passkeys are now the dominant phishing-resistant authentication method, mandated by NIST SP 800-63B AAL3 for high-assurance contexts. The stub implementation misleads integrators into thinking passkeys are supported when they are not.

**Implementation**:
```python
async def webauthn_register_begin(
    self,
    tenant_id: str,
    user_id: str,
    rp_id: str,
    rp_name: str,
    attestation: str = "none",
) -> dict[str, Any]:
    # Generate challenge (32 random bytes, base64url)
    # Build PublicKeyCredentialCreationOptions per WebAuthn L2 spec
    # Store challenge with user_id binding and 5min expiry
    # Return options for browser navigator.credentials.create()
    ...

async def webauthn_register_complete(
    self,
    tenant_id: str,
    user_id: str,
    credential_response: dict[str, Any],
) -> dict[str, Any]:
    # Verify attestation, extract public key + credential_id
    # Store credential metadata (COSE algorithm, transport hints, aaguid)
    # Emit webauthn_credential_registered audit event
    ...
```

**Competitor Reference**: Okta's native passkey support, Auth0 Passkeys, Google Identity passkey APIs, Hanko.io.

---

## 8. Session Binding and Anomaly Detection

**Category**: Security / Anti-Session-Hijacking

**Justification**: Sessions are currently stored as flat records with no binding to the network or device context that created them. A stolen session token is indistinguishable from a legitimate one. Session binding to IP subnet, user agent hash, and TLS channel binding token prevents token replay attacks. Sudden changes mid-session (IP country hop, user-agent change, impossible travel) should trigger automatic step-up or revocation.

**Implementation**:
```python
async def bind_session_context(
    self,
    tenant_id: str,
    session_id: str,
    ip_address: str,
    user_agent: str,
    tls_channel_binding: str | None = None,
) -> dict[str, Any]:
    # Hash and store binding context at session creation
    ...

async def validate_session_binding(
    self,
    tenant_id: str,
    session_id: str,
    current_ip: str,
    current_user_agent: str,
) -> dict[str, Any]:
    # Compare current context against stored binding
    # Score deviation (country change = 0.8 risk, UA change = 0.3, subnet change = 0.2)
    # If combined deviation > threshold: revoke session, emit session_anomaly_detected
    ...
```

**Competitor Reference**: Google's continuous session validation, Microsoft Conditional Access, Cloudflare Access session binding, BeyondCorp context-aware access.

---

## 9. Tenant-Scoped Password Policy Engine

**Category**: Identity / Credential Management

**Justification**: `models.py` defines `PasswordPolicyCreate` and `PasswordPolicyResponse` with breach check, history, lockout, and complexity fields, but `service.py` has no password policy evaluation. Every enterprise compliance framework (SOC2, ISO 27001, PCI-DSS) requires configurable password complexity, age limits, history enforcement, and account lockout. Without it, password policy is security theater.

**Implementation**:
```python
async def validate_password_policy(
    self,
    tenant_id: str,
    user_id: str,
    candidate_password: str,
    policy_id: str | None = None,
) -> dict[str, Any]:
    # Load tenant's active policy (or default)
    # Check: length, uppercase, lowercase, digits, special chars
    # Check: password history (compare pbkdf2 hashes)
    # Check: breach database (k-anonymity prefix call to password_breach_check)
    # Return: passed: bool, violations: list[str], strength_score: float (0-1)
    ...

async def enforce_account_lockout(
    self,
    tenant_id: str,
    user_id: str,
    failed_attempt: bool,
) -> dict[str, Any]:
    # Track failed attempts per user with rolling window
    # Lock account after max_failed_attempts within lockout_window
    # Return: locked: bool, attempts_remaining: int, unlock_at: str | None
    ...
```

**Competitor Reference**: Azure AD Password Protection, Okta password policies, Auth0 Attack Protection, NIST SP 800-63B credential service provider requirements.

---

## 10. Cross-Tenant Identity Federation and Trust Anchors

**Category**: Multi-Tenancy / Federation

**Justification**: `federated_mesh.py` exists but is decoupled from the `AuthService` lifecycle. Multi-tenant SaaS requires explicit trust anchor management: Tenant A can allow Tenant B's users to access specific resources with scoped permissions. This requires a trust registry, per-trust-relationship permission scopes, and a cross-tenant token exchange path. Without it, operators implement this ad-hoc and bypass the governance layer.

**Implementation**:
```python
async def register_trust_anchor(
    self,
    source_tenant_id: str,
    target_tenant_id: str,
    trust_level: str,           # "full" | "scoped" | "read_only"
    permitted_scopes: list[str],
    valid_until: str,
    registered_by: str,
    justification: str,
) -> dict[str, Any]:
    # Validate both tenants exist
    # Require approval flow for "full" trust level
    # Store trust anchor with explicit scope list and expiry
    # Emit trust_anchor_registered audit event
    ...

async def evaluate_cross_tenant_access(
    self,
    source_tenant_id: str,
    target_tenant_id: str,
    user_id: str,
    permission: str,
) -> dict[str, Any]:
    # Look up trust anchor between tenants
    # Check permission is in permitted_scopes
    # Check trust anchor not expired
    # Return cross_tenant_decision: allow | deny | require_review
    ...
```

**Competitor Reference**: AWS Organizations Service Control Policies, Google Workspace domain federation, Okta Org2Org, Azure B2B cross-tenant access.

---

## 11. Service Account Key Rotation Automation

**Category**: Credential Lifecycle Management

**Justification**: `models.py` defines `ServiceAccountResponse` with `key_rotation_days` and `next_rotation_at` but there is no rotation lifecycle in `service.py`. Service account credentials are a primary attack vector in supply chain compromises. Automated rotation with zero-downtime overlap windows (issue new key, grace period, revoke old key) is a CIS Control 5.4 and PCI-DSS 8.6 requirement.

**Implementation**:
```python
async def rotate_service_account_key(
    self,
    tenant_id: str,
    service_account_id: str,
    overlap_seconds: int = 300,     # 5-minute overlap for zero-downtime rotation
    rotated_by: str = "system",
) -> dict[str, Any]:
    # Issue new API key for service account
    # Keep old key active for overlap_seconds
    # Schedule revocation of old key
    # Update next_rotation_at = now + key_rotation_days
    # Emit service_account_key_rotated audit event
    # Return: new_key_id, old_key_expires_at, next_rotation_at
    ...
```

**Competitor Reference**: HashiCorp Vault dynamic secrets, AWS Secrets Manager automatic rotation, GCP Secret Manager rotation schedules, CyberArk Privileged Access Manager.

---

## 12. Differential Privacy Budget Metering with Epsilon Accounting

**Category**: Privacy Engineering

**Justification**: The current `run_privacy_query` deducts epsilon as a flat float and allows budget to hit zero. Real differential privacy systems require per-query epsilon accounting with composition theorems (basic composition, advanced composition, Renyi DP), budget recharge schedules, and query type sensitivity classification. Using `float` for epsilon introduces floating-point accumulation errors; `Decimal` with precise rounding is required for audit-grade accounting.

**Implementation**:
```python
async def compute_privacy_composition(
    self,
    tenant_id: str,
    user_id: str,
    proposed_epsilon: Decimal,
    proposed_delta: Decimal,
    composition_method: str = "advanced",   # "basic" | "advanced" | "renyi"
) -> dict[str, Any]:
    # Load historical epsilon expenditure for user
    # Apply composition theorem to compute total epsilon
    # Compare against budget ceiling
    # Return: feasible: bool, total_epsilon: Decimal, remaining_budget: Decimal, method_used
    ...

async def recharge_privacy_budget(
    self,
    tenant_id: str,
    user_id: str,
    amount: Decimal,
    recharge_reason: str,
    authorized_by: str,
) -> dict[str, Any]:
    # Validate amount <= max_recharge_per_period
    # Require approval for large recharges
    # Emit privacy_budget_recharged audit event
    ...
```

**Competitor Reference**: Google's differential privacy library, Apple's privacy budget (WWDC 2020), IBM Diffprivlib, OpenDP library.

---

## 13. Distributed Rate Limiting and Brute Force Protection

**Category**: Security / Attack Protection

**Justification**: `risk_score_login` scores individual logins but has no state for cumulative attack detection. Brute force, credential stuffing, and password spraying attacks operate across multiple requests over time. Without a sliding window rate limiter keyed on (ip, user_id, tenant_id), AUTH cannot enforce lockout policies or alert on distributed attacks. `models.py` tracks `LoginAttemptOutcome.BLOCKED_LOCKOUT` and `BRUTE_FORCE_DETECTED` audit event types that currently have no path to reach those states.

**Implementation**:
```python
async def check_rate_limit(
    self,
    tenant_id: str,
    key: str,               # e.g. "login:ip:203.0.113.1" or "login:user:alice"
    window_seconds: int = 300,
    max_attempts: int = 5,
) -> dict[str, Any]:
    # Sliding window counter using in-memory BoundedCache with TTL
    # Track attempts with timestamps
    # Return: allowed: bool, attempts: int, window_resets_at: str, blocked_until: str | None
    ...

async def record_login_attempt(
    self,
    tenant_id: str,
    user_id: str | None,
    email: str,
    ip_address: str,
    outcome: str,           # "success" | "failed_credentials" | "failed_mfa"
    user_agent: str = "",
    risk_score: float = 0.0,
) -> dict[str, Any]:
    # Record LoginAttemptResponse
    # Update rate limit counters for ip + user_id
    # Trigger lockout if threshold exceeded
    # Emit brute_force_detected if pattern matches credential stuffing signature
    ...
```

**Competitor Reference**: Auth0 Attack Protection (breached password detection + brute force), Cloudflare Bot Management, Akamai Account Protector, OWASP ASVS V11.

---

## 14. Audit Log Streaming and SIEM Integration

**Category**: Observability / Compliance

**Justification**: Audit events are stored in an in-memory dict keyed by `(tenant_id, event_id)`. For SOC2 Type II, ISO 27001, and PCI-DSS compliance, audit logs must be immutable, tamper-evident, and forwarded to an external SIEM (Splunk, Elastic, Datadog) in near real-time. The Bytewax stream integration for batch mutations points at the right architecture but audit events are not routed through it.

**Implementation**:
```python
async def stream_audit_events(
    self,
    tenant_id: str,
    since_event_id: str | None = None,
    event_types: list[str] | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    # Return cursor-paginated audit events for SIEM ingestion
    # Include HMAC chain: each event includes hash of previous event for tamper detection
    # Support filtering by event_type, actor, decision, time range
    # Return: events[], next_cursor, chain_valid: bool
    ...

async def export_audit_report(
    self,
    tenant_id: str,
    from_iso: str,
    to_iso: str,
    report_format: str = "jsonl",   # "jsonl" | "csv" | "cef"
) -> dict[str, Any]:
    # Collect events in time range
    # Format for SIEM ingestion (CEF = Common Event Format for Splunk/ArcSight)
    # Sign export payload with tenant key for integrity verification
    ...
```

**Competitor Reference**: Auth0 Log Streaming (to Splunk/Datadog/Sumo Logic), Okta System Log API, AWS CloudTrail, Duo Security's Splunk integration.

---

## 15. Identity Lifecycle Automation with Provisioning and Deprovisioning

**Category**: Identity Governance and Administration (IGA)

**Justification**: `register_identity` creates a static identity record with no lifecycle automation. Enterprise environments require SCIM 2.0-compatible provisioning (automatic identity creation from HR systems), joiner/mover/leaver workflows, automated deprovisioning (revoke all sessions, revoke all role assignments, disable API keys on termination), and dormant account detection. Without this, stale accounts with active sessions are the leading cause of insider threat and audit findings.

**Implementation**:
```python
async def deprovision_identity(
    self,
    tenant_id: str,
    user_id: str,
    reason: str,                    # "termination" | "transfer" | "suspension"
    deprovisioned_by: str,
    revoke_sessions: bool = True,
    revoke_assignments: bool = True,
    revoke_api_keys: bool = True,
    grace_period_seconds: int = 0,
) -> dict[str, Any]:
    # Atomically: lock identity, revoke all active sessions, deactivate all role assignments
    # Revoke all API keys for the identity
    # Emit identity_deprovisioned audit event with full inventory of revoked objects
    # Return: user_id, sessions_revoked, assignments_revoked, keys_revoked, deprovisioned_at
    ...

async def detect_dormant_accounts(
    self,
    tenant_id: str,
    inactivity_days: int = 90,
) -> dict[str, Any]:
    # Scan identities with no session activity in the last inactivity_days
    # Return: dormant_user_ids[], last_activity_by_user, recommended_action
    ...
```

**Competitor Reference**: Okta Lifecycle Management, Azure AD Access Reviews, SailPoint IdentityNow, Saviynt IGA, SCIM 2.0 RFC 7644.

---

*© 2026 Datacraft — www.datacraft.co.ke*
