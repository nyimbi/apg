# Banking APIs

**Capability ID**: `fintech_apis` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Banking APIs is the Open Banking and API-as-a-product layer for the APG fintech platform. It governs the full lifecycle of API products, developer onboarding, application registration, customer consent grants, API client credential issuance, endpoint policy publishing, webhook subscriptions, call auditing, rate limiting, and SLA incident management. It implements Open Banking-style consent flows where scopes must be explicitly granted before client credentials can be issued.

## Installation

```bash
pip install apg-fintech-apis
```

## Provides

- `banking_api_product_governance`
- `developer_onboarding_workflow`
- `developer_application_workflow`
- `banking_consent_workflow`
- `api_client_credential_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-apis/dashboard` | `fintech_apis:view` | Overview |
| `/fintech-apis/products` | `fintech_apis:products` | Products |
| `/fintech-apis/developers` | `fintech_apis:developers` | Developers |
| `/fintech-apis/applications` | `fintech_apis:applications` | Developers |
| `/fintech-apis/consents` | `fintech_apis:consents` | Consent |
| `/fintech-apis/clients` | `fintech_apis:clients` | Security |
| `/fintech-apis/endpoints` | `fintech_apis:endpoints` | Gateway |
| `/fintech-apis/webhooks` | `fintech_apis:webhooks` | Gateway |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_api_product()`
- `onboard_developer()`
- `register_application()`
- `create_consent_grant()`
- `issue_api_client()`
- `publish_endpoint_policy()`
- `subscribe_webhook()`
- `record_api_call()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_apis` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_apis;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_APIS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
