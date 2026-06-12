# Identity Federation (IDFD)

IDFD is APG's tenant-scoped identity federation capability. It provides a
deterministic, dependency-light surface for SAML, OIDC, LDAP, SCIM, claim
mapping, federated sessions, certificate rotation, operational health,
federation governance agents, Bytewax lifecycle batch validation, audit,
and governance review.

A generated APG app can import the contract, register the capability, call
service/runtime helpers, render route/view-model payloads, and publish package
evidence without requiring Flask, Flask-AppBuilder, a database session, a live
identity provider, or a model server.

## What The Capability Provides

- Provider lifecycle for SAML, OIDC, LDAP, and SCIM federation providers.
- Protocol guardrails: SAML assertion encryption, signed responses, OIDC
  redirect allowlists, PKCE, LDAP TLS, and SCIM external IDs.
- Claim mapping governance with source/target requirements and review gates
  for sensitive mappings.
- Federated session issuance and revocation with MFA, risk, duration, and
  tenant-isolation controls.
- OIDC client registration with redirect URI and PKCE enforcement.
- Token exchange (RFC 8693) from active sessions to audience-scoped tokens.
- SAML SP metadata generation (structured dict) for provider registration.
- SCIM-style user provisioning and deprovisioning with cascade session revocation.
- Group sync from IdP into the tenant directory.
- Cross-domain SSO assertion establishment from active sessions.
- Attribute release consent recording with policy reference enforcement.
- Trust revocation: suspend a provider and cascade-revoke all active sessions.
- Certificate registration, expiry monitoring, and rotation evidence.
- Structured federation audit reports with event-type breakdowns.
- Session and provider search with protocol/status filtering.
- Federation analytics aggregation for dashboards.
- Health reporting: stale metadata, active sessions, expiring certificates.
- Provider-neutral federation governance agents (Codex, Claude Code, opencode,
  Pi, and future runtimes) through adapter contracts.
- Bytewax-first lifecycle batch validation for all mutation types.
- Deterministic rule engine, UI route manifest, visual theme tokens, Bytewax
  streaming adapter evidence, and package metadata.

## Quick Start

```python
from capabilities.common.idfd.service import IdfdService, expires_in_days

service = IdfdService()
tenant_id = "tenant-sso"

# Register a provider
provider = service.register_provider(
    provider_id="corp-oidc",
    tenant_id=tenant_id,
    name="Corporate OIDC",
    protocol="oidc",
    owner_id="identity",
    signing_key_id="key-1",
    metadata_url="https://idp.example.test/.well-known/openid-configuration",
    redirect_allowlist=["https://app.example.test/callback"],
)

# Map a claim, issue a session, register a certificate
service.add_claim_mapping("map-email", tenant_id, provider["id"], "mail", "email")
session = service.issue_session("session-1", tenant_id, provider["id"], "user-1")
service.register_certificate("cert-1", tenant_id, provider["id"], "key-1", expires_in_days(30))
summary = service.dashboard_summary(tenant_id)
```

## Agent Composition And Lifecycle Batches

IDFD treats AI agents as governed composition records. Register an agent that
reviews federation evidence while the actual runtime client stays behind an
AICR adapter.

```python
agent = service.register_federation_agent(
    agent_id="agent-federation-review",
    tenant_id=tenant_id,
    name="Federation Review Agent",
    runtime="codex",
    role="provider_reviewer",
    scope="provider metadata and claim mappings",
    owner="identity-governance",
    purpose="review federation rollout evidence",
)

batch = service.validate_idfd_lifecycle_batch(
    tenant_id=tenant_id,
    event_stream="bytewax",
    mutation_count=2,
    operation="federation_agent_batch",
)

assert agent["status"] == "active"
assert batch["status"] == "accepted"
```

Privileged roles (`session_risk_reviewer`, `certificate_rotation_reviewer`,
`scim_reviewer`, `privacy_reviewer`, `lifecycle_batch_reviewer`,
`federation_steward`) are `pending_review` until human approval evidence is
recorded. Non-Bytewax lifecycle batches are denied by the rule engine.

## Composition Contract

```python
from capabilities.common.idfd.capability_contract import get_capability_contract

contract = get_capability_contract("tenant-sso")
routes  = contract["ui"]["routes"]
rules   = contract["rule_engine"]["rules"]
adapters = contract["configuration"]["adapters"]
```

Key adapter evidence:

| Adapter key | Value |
|---|---|
| `generated_app_runtime` | `service.IdfdService` |
| `event_stream` | `bytewax` |
| `authentication` | `auth` |
| `mfa_provider` | `mfau` |
| `encryption` | `encr` |
| `audit_sink` | `audl` |
| `key_management` | `keym` |
| `agent_adapter` | `aicr_provider_neutral_identity_federation_agent_adapter` |

## API Reference

### Core (synchronous)

| Method | Description |
|---|---|
| `register_provider(...)` | Register a SAML/OIDC/LDAP/SCIM provider with full protocol guardrails |
| `refresh_provider_metadata(provider_id, tenant_id)` | Mark metadata as refreshed; clears STALE status |
| `add_claim_mapping(mapping_id, tenant_id, provider_id, source, target)` | Map IdP claim to local claim with optional transform |
| `issue_session(session_id, tenant_id, provider_id, subject_id)` | Issue a federated session with MFA and risk controls |
| `revoke_session(session_id, tenant_id, reason)` | Revoke a session with a mandatory reason |
| `register_certificate(certificate_id, ...)` | Register a signing certificate with expiry tracking |
| `health_report(report_id, tenant_id)` | Generate a health summary: stale providers, active sessions, expiring certs |
| `register_federation_agent(agent_id, ...)` | Register a governed AI federation agent |
| `validate_idfd_lifecycle_batch(tenant_id, ...)` | Validate a Bytewax lifecycle mutation batch |
| `dashboard_summary(tenant_id)` | Aggregate counts for the dashboard screen |
| `list_providers / list_sessions / list_certificates / ...` | Standard tenant-scoped list accessors |
| `evaluate(context)` | Run the deterministic rule engine against an arbitrary context dict |
| `describe(tenant_id)` | Return the full capability contract |

### New async methods (v2.0)

| Method | Description |
|---|---|
| `idp_register(...)` | Async wrapper for `register_provider` |
| `idp_test(tenant_id, provider_id)` | Connectivity and metadata-freshness test |
| `saml_sp_metadata(tenant_id, provider_id, sp_entity_id, acs_url)` | Generate SAML SP metadata (structured dict) |
| `oidc_client_register(tenant_id, provider_id, client_id, ...)` | Register an OIDC client app with redirect URI enforcement |
| `token_exchange(tenant_id, session_id, target_audience)` | RFC 8693 token exchange from an active session |
| `claim_map(...)` | Async alias for `add_claim_mapping` |
| `group_sync(tenant_id, provider_id, groups, actor)` | Sync group memberships from IdP |
| `user_provision(tenant_id, provider_id, subject_id, attributes, actor)` | SCIM-style user provisioning |
| `user_deprovision(tenant_id, provider_id, subject_id, actor)` | Deprovision user and cascade-revoke all their active sessions |
| `federation_session(session_id, ...)` | Async wrapper for `issue_session` |
| `cross_domain_sso(tenant_id, source_session_id, target_domain, actor)` | Establish a cross-domain SSO assertion |
| `attribute_release(tenant_id, provider_id, subject_id, ...)` | Record attribute release consent with policy reference |
| `trust_revoke(tenant_id, provider_id, reason, actor)` | Suspend provider and cascade-revoke all active sessions |
| `federation_audit(tenant_id, period_start, period_end)` | Structured audit summary with per-event-type breakdown |
| `session_search(tenant_id, subject_id, provider_id, status_filter)` | Filter sessions by subject, provider, and/or status |
| `certificate_expiry_check(tenant_id, warn_days)` | Return certificates expiring within warn_days |
| `provider_search(tenant_id, protocol, status_filter)` | Filter providers by protocol and/or status |
| `federation_analytics(tenant_id)` | Aggregate federation metrics for dashboards |

## New Methods — Usage Examples

### trust_revoke — Suspend a provider and cascade-revoke sessions

```python
result = await service.trust_revoke(
    tenant_id="tenant-sso",
    provider_id="corp-oidc",
    reason="security_incident",
    actor="security-team",
)
# result["revoked_session_count"] tells you how many sessions were invalidated
```

### user_deprovision — Offboard a user atomically

```python
result = await service.user_deprovision(
    tenant_id="tenant-sso",
    provider_id="corp-oidc",
    subject_id="alice@corp.example",
    actor="hr-system",
    reason="employment_terminated",
)
# All active sessions for this subject are revoked atomically
assert result["revoked_session_count"] >= 0
```

### token_exchange — RFC 8693 downstream token issuance

```python
record = await service.token_exchange(
    tenant_id="tenant-sso",
    session_id="session-1",
    target_audience="https://api.internal.example/payments",
)
# record["issued_token_ref"] is a stable reference for downstream validation
```

### federation_audit — Compliance reporting

```python
report = await service.federation_audit(
    tenant_id="tenant-sso",
    period_start="2026-01-01T00:00:00Z",
    period_end="2026-06-30T23:59:59Z",
)
# report["events_by_type"] maps event_type -> count
# report["trust_revocation_count"] is a pre-computed compliance KPI
```

### session_search + certificate_expiry_check — Operational hygiene

```python
# Find all active sessions for a specific user
sessions = await service.session_search(
    tenant_id="tenant-sso",
    subject_id="alice@corp.example",
    status_filter="active",
)

# Warn on certificates expiring in the next 14 days
expiring = await service.certificate_expiry_check(tenant_id="tenant-sso", warn_days=14)
```

## Guardrails

IDFD enforces deterministic rules for: tenant context, provider ownership,
signing keys, metadata review, SAML encryption, OIDC redirect allowlists, PKCE,
LDAP TLS, SCIM deprovisioning, claim mapping review, privileged MFA, high-risk
reauthentication, session duration, certificate rotation review, federation
agent registration, Bytewax lifecycle batch validation, tenant isolation, and
required audit evidence.

```python
from capabilities.common.idfd.capability_contract import evaluate_capability_rules

result = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation": "batch_federation_mutation",
    "event_stream": "legacy_queue",
})
assert result["decision"] == "deny"
```

## World-Class Enhancements (v2.0)

The following 15 improvements are targeted for production-grade hardening. Each
is tracked against the priority listed below.

| # | Title | Priority | Summary |
|---|---|---|---|
| 1 | Persistent Storage Backend Abstraction | Critical | `FederationRepository` ABC with PostgreSQL and in-memory implementations; wired at construction time. Enables rolling restarts and cross-replica session consistency. |
| 2 | Async-Native I/O Throughout | High | Convert all store operations to true async repository calls; use `asyncio.gather` in `health_report` and `federation_analytics`. Keeps event-loop time under 5 ms with external DB latency. |
| 3 | Structured Federation Policy Engine with Hot-Reload | High | Rule DAG in versioned JSON/YAML with `inotify`/`watchfiles` hot-reload. `policy_version` on every `evaluate()` response. Enables canary rollouts and instant incident response. |
| 4 | JWKS Endpoint Caching and Key Rotation Automation | High | `jwks_refresh(tenant_id, provider_id)` fetches the IdP JWKS URI via `httpx`, validates JWKs, caches with TTL, and triggers `certificate_expiry_check` pre-emptively. |
| 5 | SAML Assertion Signature Verification | Critical | `saml_assertion_verify(...)` using `xmlsec1`/`pysaml2` to verify XML-DSIG, `NotBefore`/`NotOnOrAfter` conditions, and ACS URL. Prevents assertion replay and IdP impersonation. |
| 6 | OIDC Token Introspection and Revocation | High | `oidc_token_introspect` (RFC 7662) and `oidc_token_revoke` (RFC 7009) with provider-side HTTP calls. Revocation cascades from `trust_revoke`. |
| 7 | SCIM 2.0 Bulk Operations and Patch Support | High | `scim_bulk_op(tenant_id, provider_id, operations, actor)` implementing RFC 7644 §3.7 bulk semantics plus `PATCH add/remove/replace`. RFC-compliant bulk response with per-operation status codes. |
| 8 | Risk-Based Adaptive Authentication Steps | High | `evaluate_session_risk(tenant_id, session_id, signals)` returning `allow\|step_up\|block` with `challenge_type`. Integrated into `issue_session` so high-risk scores require `reauth_completed=True`. |
| 9 | Federation Event Streaming to External Sinks | High | Pluggable `FederationEventSink` protocol with Kafka, Webhook, and Null implementations. `_audit()` becomes async and publishes to the configured sink. Enables SIEM integration. |
| 10 | Multi-Tenant Provider Discovery and Metadata Caching | Medium-High | `discover_provider(tenant_id, issuer_url, protocol)` fetches `/.well-known/openid-configuration` or SAML metadata, auto-populates fields, and caches with configurable TTL. |
| 11 | Session Binding to Device Fingerprints | Medium | Optional `device_fingerprint` and `client_ip` on `FederatedSession`. Subsequent validation can assert fingerprint continuity, preventing session token theft across devices. |
| 12 | Comprehensive Observability with OpenTelemetry | Medium | OTel spans on every public method carrying `tenant_id`, `provider_id`, and outcome. Counters: `federation.session.issued`, `federation.provider.stale`, `federation.trust.revoked`. |
| 13 | Claim Transformation Pipeline with DSL | Medium | Typed `ClaimTransform` DSL (`copy`, `regex_extract`, `jmespath`, `static_value`, `join`, `split`) with `apply_claim_pipeline(tenant_id, provider_id, raw_claims)`. |
| 14 | Certificate Chain Validation and Pinning | High | `certificate_chain_validate(tenant_id, certificate_id, pem_chain)` verifying issuer chain, OCSP/CRL revocation, key usage extensions, and minimum RSA-2048/ECDSA P-256 key size. |
| 15 | Tenant-Scoped Rate Limiting and Abuse Detection | Medium | `FederationRateLimiter` with sliding-window counters (Redis or in-process). Per-tenant quotas via capability contract. `rate_limit_status(tenant_id)` returns current consumption. Exceeding quota raises `PermissionError("rate_limit_exceeded")`. |

## Screens

The contract exposes route metadata for: dashboard, providers, protocols,
mappings, sessions, certificates, SCIM directory, risk console, reviews,
agents, lifecycle, audit, and settings.

The view helpers in `views.py` return dependency-light payloads for those
screens and include the theme component names required by generated UIs.

## Verification

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/idfd/__init__.py \
  capabilities/common/idfd/capability_contract.py \
  capabilities/common/idfd/federation_runtime.py \
  capabilities/common/idfd/models.py \
  capabilities/common/idfd/service.py \
  capabilities/common/idfd/api.py \
  capabilities/common/idfd/views.py \
  capabilities/common/idfd/app.py

./.venv/bin/pytest -q \
  capabilities/common/idfd/test_capability_contract.py \
  capabilities/common/idfd/tests/test_package_contract.py

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/idfd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/idfd --json
```
