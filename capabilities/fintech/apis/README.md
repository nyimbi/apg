# Banking APIs

## Overview
Banking APIs is the Open Banking and API-as-a-product layer for the APG fintech platform. It governs the full lifecycle of API products, developer onboarding, application registration, customer consent grants, API client credential issuance, endpoint policy publishing, webhook subscriptions, call auditing, rate limiting, and SLA incident management. It implements Open Banking-style consent flows where scopes must be explicitly granted before client credentials can be issued.

The capability enforces a strict chain of trust: **product → developer → application → consent → client**. Breaking any link in that chain produces a deterministic deny. All API call records and gateway events stream to `apg.fintech.apis.lifecycle` via Bytewax for real-time monitoring and anomaly detection.

## Capability ID
`fintech_apis`  Version: 2.0.0

## Quick Start

```python
from capabilities.fintech.apis.service import BankingAPIsService

svc = BankingAPIsService()

# Register a product
svc.register_api_product("prod-1", "acme", "Payments API", "owner-1", "payments", "sandbox", ["payments"])

# Onboard a developer
svc.onboard_developer("dev-1", "acme", "Acme Corp", "kyb-ref", "sec-ref", "risk-ref")

# Register an app
svc.register_application("app-1", "acme", "dev-1", "My App", "sandbox", "https://acme.com/cb", "terms-ref")

# Create consent
svc.create_consent_grant("consent-1", "acme", "app-1", "cust-1", ["payments"], "2026-12-31")

# Issue client
svc.issue_api_client("client-1", "acme", "app-1", "authorization_code", "key-ref", ["payments"])

# Create API key (async)
key = await svc.create_api_key("app-1", ["payments"], rate_limit=1000, tenant_id="acme")
```

## Provides
| Service | Description |
|---------|-------------|
| banking_api_product_governance | Register and version API products with environment and scope controls |
| developer_onboarding_workflow | Onboard developer organizations with KYB, security review, and risk clearance |
| developer_application_workflow | Register applications with redirect URIs and terms acceptance |
| banking_consent_workflow | Issue and manage scoped customer consent grants with expiry |
| api_client_credential_workflow | Issue OAuth2/mTLS clients bound to consented scopes |
| api_endpoint_policy_workflow | Publish endpoint policies with throttle and risk policy attachments |
| webhook_subscription_workflow | Subscribe applications to platform events with signed-secret verification |
| api_call_audit_workflow | Record and audit every API call with risk reference |
| api_rate_limit_workflow | Manage per-client rate limit buckets |
| api_sla_incident_workflow | Open and track SLA incidents with severity-gated approvals |
| banking_api_agent_workflow | Register AI agents for API operations review roles |
| api_key_lifecycle | Generate, rotate, and revoke API keys with rate-limit binding |
| oauth2_token_issuance | Issue and introspect OAuth 2.0 Bearer tokens |
| sandbox_testing | Deterministic sandbox transactions for integration testing |
| open_banking_aisp | Account information retrieval under PSD2 consent |
| open_banking_pisp | Payment initiation and funds confirmation under PSD2 consent |
| psd2_compliance | Runtime PSD2/Open Finance compliance checks |
| developer_portal_analytics | Portal usage stats, monetization metrics, developer tier management |
| iso20022_validation | ISO 20022 message structure validation (pain.001, pacs.008, etc.) |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Platform authentication |
| audl | Audit trail |
| ntfy | Incident and developer notifications |
| nlpc | NLP for incident narrative |
| keym | Key management for client credentials |
| fintech_payments | Payments API product backing |
| fintech_wallets | Wallets API product backing |
| fintech_cards | Cards API product backing |
| fintech_kyc | Customer identity for consent |
| fintech_aml | AML checks on high-risk API access |
| fintech_fraud | Fraud screening for call patterns |
| fintech_neobanking | Accounts and statements products |
| fintech_lending | Loans API product backing |
| fintech_bnpl | BNPL API product backing |
| fintech_agency | Agency API product backing |
| fintech_mobile | Mobile channel API access |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| rate_limits.default_limit | number | 1000 | Default calls per window |
| rate_limits.burst_limit | number | 5000 | Burst capacity |
| rate_limits.window_seconds | number | 60 | Rate limit rolling window |
| calls.high_volume_threshold | number | 10000 | Call batch count requiring review |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-apis/dashboard | GET | fintech_apis:view | Overview |
| products | /fintech-apis/products | GET/POST | fintech_apis:products | Products |
| developers | /fintech-apis/developers | GET/POST | fintech_apis:developers | Developers |
| applications | /fintech-apis/applications | GET/POST | fintech_apis:applications | Developers |
| consents | /fintech-apis/consents | GET/POST | fintech_apis:consents | Consent |
| clients | /fintech-apis/clients | GET/POST | fintech_apis:clients | Security |
| endpoints | /fintech-apis/endpoints | GET/POST | fintech_apis:endpoints | Gateway |
| webhooks | /fintech-apis/webhooks | GET/POST | fintech_apis:webhooks | Gateway |
| calls | /fintech-apis/calls | GET | fintech_apis:calls | Operations |
| rate_limits | /fintech-apis/rate-limits | GET/POST | fintech_apis:rate_limits | Operations |
| incidents | /fintech-apis/incidents | GET/POST | fintech_apis:incidents | Operations |
| agents | /fintech-apis/agents | GET/POST | fintech_apis:admin | Automation |
| settings | /fintech-apis/settings | GET/POST | fintech_apis:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| developer_kyb_required | Developer without KYB evidence | deny |
| developer_security_required | Developer without security review | deny |
| client_scopes_allowed_by_consent | Client scopes exceed active consent | deny |
| api_call_rate_limit_allowed | Call exceeds rate limit | deny |
| high_volume_api_call_requires_review | Batch > 10,000 calls without review | require_review |
| critical_incident_requires_approval | Critical severity incident without approval | require_review |
| webhook_signing_secret_required | Webhook without signing secret | deny |
| endpoint_throttle_required | Endpoint without throttle policy | deny |
| endpoint_risk_required | Endpoint without risk policy | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| APIProduct | id, name, owner_id, product_type, environment, scopes, status |
| DeveloperOrganization | id, name, kyb_reference, security_review_reference, risk_clearance_reference, status |
| DeveloperApplication | id, developer_id, name, environment, redirect_uri, terms_reference, status |
| ConsentGrant | id, application_id, customer_reference, scopes, expiry_date |
| APIClient | id, application_id, auth_flow, key_reference, scopes |
| EndpointPolicy | id, product_id, route, scope, throttle_policy, risk_policy |
| WebhookSubscription | id, application_id, event_type, endpoint, signing_secret |
| APICallRecord | id, client_id, product_id, endpoint_id, risk_reference, status_code |
| RateLimitBucket | id, client_id, limit, burst_limit, window_seconds |
| SLAIncident | id, severity, owner_id, evidence_references, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| api_product_registered | New API product published |
| developer_onboarded | Developer passes KYB/security/risk checks |
| developer_application_registered | Application registered |
| consent_grant_created | Customer consent recorded |
| api_client_issued | OAuth/mTLS client credentials issued |
| endpoint_policy_published | Endpoint throttle/risk policy activated |
| webhook_subscribed | Webhook subscription confirmed |
| api_call_recorded | Individual API call audited |
| rate_limit_updated | Rate limit bucket modified |
| sla_incident_opened | SLA breach incident created |
| api_agent_registered | AI agent registered |
| api_key_created | API key generated |
| api_key_revoked | API key invalidated |
| oauth2_token_issued | Bearer token issued |
| webhook_event_delivered | Webhook delivery attempt completed |
| psd2_compliance_check_failed | PSD2 check failed |

---

## World-Class Enhancements (v2.0)

1. **Dynamic Consent Scope Narrowing** — `narrow_consent_scopes()` removes specific scopes from an active consent without full revocation; non-compliant clients auto-suspended. Satisfies PSD2 Art. 67/68 partial-withdrawal requirement.

2. **mTLS Certificate Lifecycle Management** — `rotate_mtls_certificate()` / `check_certificate_expiry()` track fingerprints, expiry dates, and emit `certificate_expiry_warning` 30 days pre-expiry. Eliminates silent 401 spikes from expired certs.

3. **Adaptive Rate Limiting with Burst Detection** — `adaptive_rate_limit_update()` analyzes rolling call histograms, detects P99 bursts, and auto-adjusts burst cap. Integrates with `fintech_fraud` signals to tighten limits under elevated fraud scores.

4. **Consent Journey Analytics** — `consent_funnel_analytics()` tracks initiation→scope selection→confirmation→active grant with timestamps. Returns conversion rates, median time-to-consent, and scope popularity rankings for AISP regulatory reporting.

5. **Webhook Delivery Retry with Exponential Backoff** — `webhook_retry_delivery()` with configurable max attempts and backoff. Emits `webhook_delivery_failed_permanent` after exhaustion. Critical for payment notifications where missed events cause reconciliation failures.

6. **API Product Deprecation Workflow** — `deprecate_api_product()` sets a sunset date, injects `Deprecation`/`Sunset` HTTP headers (RFC 8594), and triggers developer notifications at 90/30/7-day intervals. Hard deletion blocked until active client count reaches zero.

7. **Fine-Grained Audit Trail with Diff Capture** — `_audit()` upgraded to capture `{"before": {...}, "after": {...}, "changed_fields": [...]}`. `get_audit_trail()` supports filters by entity type, tenant, time range, and actor. SOC 2 Type II / PSD2 compliant.

8. **SCA Challenge Orchestration** — `initiate_sca_challenge()` / `verify_sca_challenge()` implement a full OTP/push-token challenge cycle: time-bound tokens, 3-attempt lockout, result bound to a specific payment or consent operation. PSD2 RTS compliant.

9. **Cross-Tenant Federated API Product Catalog** — `publish_to_catalog()` / `discover_catalog_products()` expose products to a shared marketplace with public/private/allowlisted visibility. Enables the API-as-a-product monetization model for embedded banking.

10. **Real-Time Call Anomaly Detection** — `detect_call_anomalies()` maintains rolling statistical baselines per client/endpoint and flags volume spikes (>3σ), off-hours bursts, impossible geo-sequences, and error rate degradation. Returns anomaly score 0–100.

11. **Token Introspection and Revocation (RFC 7662 / RFC 7009)** — `introspect_token()` returns active/inactive status with scope, expiry, and client metadata. `revoke_token()` provides immediate invalidation with audit trail. Required for FAPI / Open Banking UK / Berlin Group interoperability.

12. **API Dependency Graph and Impact Analysis** — `build_dependency_graph()` traverses product→endpoint→client→application→developer. `impact_analysis()` returns affected entities ranked by call volume; exports as JSON or Mermaid. Enables CAB-compliant blast-radius assessment before changes.

13. **Payment Initiation Status Polling and Webhooks** — `get_payment_status()` for synchronous polling; `subscribe_payment_events()` for webhook-driven updates. Tracks state machine: `pending → authorized → submitted → settled | failed | cancelled`. Required for PISP reconciliation.

14. **Developer Onboarding Self-Service Portal Data Layer** — `get_onboarding_status()` returns per-step checklist (KYB/security review/risk clearance) with statuses. `update_onboarding_step()` supports document attachment. Reduces time-to-first-API-call from days to hours.

15. **Tiered SLA Tracking with Business-Hours SLA** — `sla_tier_config()` sets per-severity response/resolution targets with optional business-hours-only windows. `api_sla_report()` computes time-to-acknowledge and time-to-resolve against tier targets, flags breaches, and generates SLA credit calculations.

---

## New Methods

### `create_api_key` — API Key Generation
Generates a hashed key pair (public `key_id` + one-time `secret`) and binds an initial rate-limit bucket. The raw secret is returned only once at creation.

```python
key = await svc.create_api_key(
    app_id="app-1",
    scopes=["payments", "account_information"],
    rate_limit=500,
    tenant_id="acme",
    environment="sandbox",
)
# key["secret"] — use and discard; only the hash is stored
# key["key_id"] — reference for subsequent operations
```

### `oauth2_token` — Bearer Token Issuance
Validates client credentials against the stored key hash, then issues a signed Bearer token with a 1-hour TTL. Supports all `SUPPORTED_AUTH_FLOWS`.

```python
token = await svc.oauth2_token(
    client_id="client-1",
    client_secret="<raw-secret-from-create_api_key>",
    scope="payments",
    grant_type="client_credentials",
    tenant_id="acme",
)
# token["access_token"] — Bearer value for downstream requests
# token["expires_in"]  — 3600 seconds
```

### `open_banking_payment_initiation` — PSD2 Payment Initiation
Validates an active consent with `payment_initiation` scope, verifies payment fields, and returns a payment instruction reference with `pending` status.

```python
payment = await svc.open_banking_payment_initiation(
    payment_data={
        "amount": 5000.00,
        "currency": "KES",
        "creditor_account": "KE1234567890",
    },
    consent_id="consent-1",
    tenant_id="acme",
)
# payment["payment_id"] — track via get_payment_status()
# payment["status"]    — "pending" initially
```

### `psd2_compliance_check` — Runtime PSD2 Gate
Checks SCA presence, consent validity, AISP/PISP license reference, and TLS version in a single call. Returns per-check results and a `compliant` boolean.

```python
result = await svc.psd2_compliance_check(
    request={
        "sca_reference": "sca-token-xyz",
        "consent_id": "consent-1",
        "aisp_license": "AISP-KE-001",
        "tls_version": "1.3",
        "risk_reference": "risk-ref-1",
    },
    tenant_id="acme",
)
# result["compliant"]      — True if all checks pass
# result["failed_checks"]  — list of failing check names
```

### `api_usage_analytics` — Per-App Usage Metrics
Aggregates call volume, error rate, top endpoints by call count, and remaining rate-limit headroom for a given application and period.

```python
stats = await svc.api_usage_analytics(
    app_id="app-1",
    period="2026-06",
    tenant_id="acme",
)
# stats["total_calls"]        — aggregate across all app clients
# stats["error_rate"]         — fraction of 4xx/5xx responses
# stats["top_endpoints"]      — list of {endpoint_id, calls}
# stats["rate_limit_remaining"] — remaining quota or None
```

---

## Edge Cases Handled
- Client scopes are validated against active consent at issuance time — a client cannot be issued with broader scopes than what the customer explicitly granted, even if the product definition allows them
- Webhook endpoints require a signing secret; unsigned webhook subscriptions are denied to prevent data exfiltration via misconfigured endpoints
- Rate limit enforcement fires at the call-record level, not just at the gateway — audit completeness is guaranteed even if a gateway allows a call through
- API call endpoint must belong to the selected product — cross-product authorization using a mismatched endpoint is denied
- `device_code` auth flow is supported for IoT/embedded scenarios where a browser redirect is not available
- Sandbox transactions are environment-gated — only `environment=sandbox` API keys may call `sandbox_transaction`
- OAuth2 token issuance validates the client secret against a stored hash; raw secrets are never persisted

## Composability
- **Upstream**: Developer KYB from `fintech_kyc`; fraud screening for call patterns from `fintech_fraud`; AML for high-risk access from `fintech_aml`
- **Downstream**: `fintech_embedded` consumes Banking APIs to surface product placements in partner applications; `fintech_mobile` uses the API layer for device-bound client credentials
- **Peer**: Deployed alongside `fintech_gateway` (provider routing) and `fintech_payments` (the most commonly exposed API product)

## Development Notes
- The five-step chain (product → developer → application → consent → client) has separate deny rules at each step; missing a prerequisite at any level blocks the next step
- `SUPPORTED_ENVIRONMENTS` (sandbox, pilot, production) controls which lifecycle stage a product or application operates in; environment mismatch is denied
- Webhook signing uses HMAC; the signing secret must be stored in `keym` and referenced by ID, not stored as plaintext
- Both batch operations and individual high-volume calls require Bytewax routing
- All new async methods assert preconditions at entry; `PermissionError` is raised for tenant boundary violations, `ValueError` for missing or invalid inputs

---

*© 2025 Datacraft | Author: Nyimbi Odero*
