# APG Banking APIs

`fintech_apis` is an executable APG capability for open banking and
embedded-finance API operations: API products, developer onboarding, developer
applications, consent grants, API clients, endpoint policies, webhooks, API call
audit, rate limits, SLA incidents, and provider-neutral API governance agents.

## What It Provides

- API product governance for accounts, balances, transactions, payments, cards,
  wallets, loans, BNPL, agency, identity, statements, and webhooks.
- Developer organization onboarding with KYB, security, and risk evidence.
- Developer application registration with environment, redirect URI, and terms
  evidence.
- Consent grant lifecycle with scopes, expiry, and customer evidence.
- Client credential issuance with key references and supported auth flows.
- Endpoint policy publishing with route, scope, throttle, and risk policies.
- Webhook subscription management with endpoint and signing-secret evidence.
- API call audit and rate-limit enforcement.
- SLA incident tracking.
- First-class AI-agent composition for Codex, Claude Code, OpenCode, and Pi.

## How To Use It

Inspect the contract:

```bash
./.venv/bin/apg capabilities inspect fintech_apis --json
```

Run the package self-test:

```bash
./.venv/bin/python capabilities/fintech/apis/app.py
```

Use the service directly:

```python
from capabilities.fintech.apis import BankingAPIService

service = BankingAPIService()
product = service.register_api_product(
	"product-1",
	"tenant-1",
	"Accounts API",
	"api-ops",
	"accounts",
	"sandbox",
	["accounts.read", "balances.read"],
)
developer = service.onboard_developer(
	"developer-1",
	"tenant-1",
	"Fintech Builder",
	"kyb-1",
	"security-review-1",
	"risk-clear-1",
)
application = service.register_application(
	"app-1",
	"tenant-1",
	developer["id"],
	"Personal Finance App",
	"sandbox",
	"https://example.test/callback",
	"terms-1",
)
consent = service.create_consent_grant(
	"consent-1",
	"tenant-1",
	application["id"],
	"customer-1",
	["accounts.read"],
	"2026-12-31",
)
client = service.issue_api_client(
	"client-1",
	"tenant-1",
	application["id"],
	"oauth2_auth_code",
	"key-ref-1",
	["accounts.read"],
)
```

## Composition Surfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- View models: `views.py`
- Semantic model: `semantic_model.json`
- Package manifest: `package_manifest.json`

## Guardrails

The deterministic rule engine checks tenant context, write policy, product
ownership, product type, environment, scopes, developer KYB/security/risk
evidence, application redirect URI and terms, consent scope and expiry, client
auth flow and key evidence, endpoint policy route/scope/throttle/risk evidence,
webhook endpoint and signing secret, API call client/product/endpoint/rate-limit
and risk evidence, SLA incident owner/evidence, Bytewax lifecycle processing,
and privileged AI-agent approvals.

## Streaming

Lifecycle metadata uses Bytewax:

- stream: `apg.fintech.apis.lifecycle`;
- processor: `bytewax`;
- key: `tenant_id`.

The package intentionally does not publish alternate broker settings.
