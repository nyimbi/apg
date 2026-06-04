# Banking APIs

## Overview
Banking APIs is the Open Banking and API-as-a-product layer for the APG fintech platform. It governs the full lifecycle of API products, developer onboarding, application registration, customer consent grants, API client credential issuance, endpoint policy publishing, webhook subscriptions, call auditing, rate limiting, and SLA incident management. It implements Open Banking-style consent flows where scopes must be explicitly granted before client credentials can be issued.

The capability enforces a strict chain of trust: product → developer → application → consent → client. Breaking any link in that chain produces a deterministic deny. All API call records and gateway events stream to `apg.fintech.apis.lifecycle` via Bytewax for real-time monitoring and anomaly detection.

## Capability ID
`fintech_apis`  Version: 1.1.0

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

## Edge Cases Handled
- Client scopes are validated against active consent at issuance time — a client cannot be issued with broader scopes than what the customer explicitly granted, even if the product definition allows them
- Webhook endpoints require a signing secret; unsigned webhook subscriptions are denied to prevent data exfiltration via misconfigured endpoints
- Rate limit enforcement fires at the call-record level, not just at the gateway — audit completeness is guaranteed even if a gateway allows a call through
- API call endpoint must belong to the selected product — cross-product authorization using a mismatched endpoint is denied
- `device_code` auth flow is supported for IoT/embedded scenarios where a browser redirect is not available

## Composability
- **Upstream**: Developer KYB from `fintech_kyc`; fraud screening for call patterns from `fintech_fraud`; AML for high-risk access from `fintech_aml`
- **Downstream**: `fintech_embedded` consumes Banking APIs to surface product placements in partner applications; `fintech_mobile` uses the API layer for device-bound client credentials
- **Peer**: Deployed alongside `fintech_gateway` (provider routing) and `fintech_payments` (the most commonly exposed API product)

## Development Notes
- The five-step chain (product → developer → application → consent → client) has separate deny rules at each step; missing a prerequisite at any level blocks the next step
- `SUPPORTED_ENVIRONMENTS` (sandbox, pilot, production) controls which lifecycle stage a product or application operates in; environment mismatch is denied
- Webhook signing uses HMAC; the signing secret must be stored in `keym` and referenced by ID, not stored as plaintext
- Both batch operations and individual high-volume calls require Bytewax routing
