# APG Banking APIs Capability Specification

## Purpose

The Banking APIs capability makes open-banking and embedded-finance API
composition a first-class APG fintech package. It lets generated applications
publish API products, onboard developers, register applications, capture consent,
issue client credentials, manage endpoint policies, route API calls, enforce
rate limits, publish webhooks, track SLA incidents, and use provider-neutral AI
agents for API governance.

The executable core runs locally without live gateways. Live API gateways,
OAuth authorization servers, consent registries, developer portals, webhook
delivery networks, regulator reporting, and durable Bytewax workers remain
behind adapters.

## Functional Scope

The package provides:

- API product governance;
- developer organization onboarding;
- developer application registration;
- consent grant lifecycle;
- API client and key issuance;
- endpoint policy publishing;
- webhook subscription management;
- API call audit and rate-limit controls;
- SLA incident tracking;
- provider-neutral AI-agent composition for API operations, consent, developer
  risk, rate-limit, webhook, incident, and compliance review.

## Composition Contract

Capability id: `fintech_apis`

Provides:

- `banking_api_product_governance`
- `developer_onboarding_workflow`
- `developer_application_workflow`
- `banking_consent_workflow`
- `api_client_credential_workflow`
- `api_endpoint_policy_workflow`
- `webhook_subscription_workflow`
- `api_call_audit_workflow`
- `api_rate_limit_workflow`
- `api_sla_incident_workflow`
- `banking_api_agent_workflow`

Requires:

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_cards`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_neobanking`
- `fintech_lending`
- `fintech_bnpl`
- `fintech_agency`
- `fintech_mobile`

## Supported Domains

Supported API products: accounts, balances, transactions, payments, cards,
wallets, loans, BNPL, agency, customer identity, statements, webhooks.

Supported environments: sandbox, pilot, production.

Supported auth flows: OAuth 2 authorization code, client credentials, mTLS,
signed request, device code.

Supported regions: KE, UG, TZ, RW, GH, NG, ZA, GB, US, AE, EU.

Supported webhook events: account updated, transaction posted, payment status,
card event, wallet event, loan event, BNPL event, agency event, fraud alert,
consent revoked.

Supported AI-agent runtimes: Codex, Claude Code, OpenCode, Pi.

## Rule Engine

The deterministic rule engine runs before every state change. It validates
tenant context, write policy, product ownership, supported product and
environment, developer KYB and security evidence, application redirect URI and
terms, consent scope and expiry, client key evidence, endpoint route and
throttle policy, webhook endpoint and signing secret, API call client/product/
endpoint/rate-limit/risk evidence, SLA incident severity and owner, Bytewax
lifecycle processing, and privileged AI-agent approval.

Rules may return `allow`, `deny`, or `require_review`. Service methods raise
`PermissionError` for denied writes and unresolved required reviews.

## UI And Theming

The package publishes APG Python UI metadata for dashboard, API products,
developers, applications, consents, clients, endpoint policies, webhooks, API
calls, rate limits, SLA incidents, agents, and settings. Theme metadata uses
semantic tokens for dense developer-operations consoles with tenant overrides.

## Streaming

Lifecycle metadata uses Bytewax:

- processor: `bytewax`;
- stream: `apg.fintech.apis.lifecycle`;
- key: `tenant_id`.

No alternate broker configuration is part of the contract.

## Non-Goals

This slice does not implement live gateway deployment, live OAuth consent
screens, mTLS certificate validation, webhook delivery retries, developer portal
hosting, regulator filing, rendered UI checks, durable worker deployment, or
performance/load testing.
