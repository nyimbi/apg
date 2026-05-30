# APG Fintech Gateway Capability

`fintech_gateway` is the APG payment orchestration capability for merchant onboarding, provider connections, payment method tokenization, payment intents, routing, fraud-risk review, authorization, capture, refunds, webhooks, settlements, disputes, and gateway-focused AI agent review.

The package keeps live provider integrations behind adapter boundaries. Importing `capabilities.fintech.gateway` does not require Flask, databases, payment SDKs, provider credentials, or external services.

## Capability ID

- ID: `fintech_gateway`
- Display name: `Fintech Gateway`
- Version: `2.1.0`
- Event stream: `apg.fintech.gateway.lifecycle`
- Stream processor: `bytewax`
- Primary package files: `capability_contract.py`, `service.py`, `api.py`, `views.py`, `app.py`

## What It Provides

- Merchant onboarding lifecycle.
- Provider connection lifecycle for card, bank, mobile money, wallet, settlement, and fraud providers.
- Payment method tokenization workflow.
- Payment intent lifecycle.
- Payment routing workflow.
- Fraud-risk review workflow.
- Authorization and capture workflow.
- Refund lifecycle.
- Webhook ingestion workflow with idempotency.
- Settlement reconciliation workflow.
- Payment dispute workflow.
- Gateway agent registration and privileged-action approval rules.
- UI routes, theme metadata, semantic app metadata, and APG publish evidence.

## Required Capabilities

The contract declares composition dependencies on:

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `keym`
- `encr`
- `cash_management`
- `accounts_receivable`
- `customer_relationship_management`
- `business_intelligence`

The current runtime exposes adapter boundaries for these dependencies and does not require live providers during focused package checks.

## Quick Use

```python
from capabilities.fintech.gateway import FintechGatewayService

svc = FintechGatewayService()
merchant = svc.onboard_merchant("merchant-1", "tenant-1", "MERCH-001", "Merchant One", "KE")
provider = svc.connect_provider("provider-1", "tenant-1", "mpesa", "mobile_money", "vault://mpesa")
method = svc.tokenize_payment_method("method-1", "tenant-1", merchant["id"], "customer-1", "mobile_money", "tok-1")
intent = svc.create_payment_intent("intent-1", "tenant-1", merchant["id"], method["id"], 1000, "KES")
svc.assess_payment_risk("risk-1", "tenant-1", intent["id"], "medium", 0.35)
authorization = svc.authorize_payment("auth-1", "tenant-1", intent["id"], provider["id"])
svc.capture_payment("capture-1", "tenant-1", authorization["id"], 1000)
print(svc.dashboard_summary("tenant-1"))
```

## Rule And Guardrail Coverage

The deterministic rule engine enforces:

- tenant context for operations;
- policy attachment for writes;
- required merchant code, legal name, country, and review for high-risk merchants;
- supported provider names, provider types, and credential references;
- merchant, customer reference, supported method type, and token reference for payment methods;
- merchant, positive amount, supported currency, and payment method for payment intents;
- payment risk parent intent, high-risk payment review, and blocked-risk denial;
- payment intent, provider, and approval gates for authorization;
- authorized payment, positive amount, and overcapture blocking for capture;
- captured payment, positive amount, overrefund blocking, and large-refund review;
- provider, event ID, signature, and idempotency key for webhooks;
- provider, settlement reference, nonnegative amount, and variance review for settlements;
- payment, supported reason, owner, and reviewed resolution for disputes;
- Bytewax routing for gateway batches and events;
- gateway agent runtime, role, and privileged-action approval controls.

## UI Surface

The contract and `views.py` expose these screens:

- Dashboard
- Merchants
- Providers
- Payment Methods
- Payments
- Routing
- Risk
- Webhooks
- Settlements
- Disputes
- Agents
- Settings

The theme is `fintech_gateway_control` and uses compact operational layouts for provider status, routing decisions, risk queues, event inboxes, settlement variance, dispute cases, and agent review lanes.

## AI Agent Composition

Gateway agents are first-class records. Supported runtimes are:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles are:

- `merchant_underwriter`
- `routing_reviewer`
- `fraud_reviewer`
- `settlement_reviewer`
- `dispute_reviewer`
- `provider_operations_reviewer`

Agents may prepare and review gateway operations. Privileged actions require recorded human approval.

## Focused Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/fintech/gateway/__init__.py capabilities/fintech/gateway/capability_contract.py capabilities/fintech/gateway/service.py capabilities/fintech/gateway/api.py capabilities/fintech/gateway/views.py capabilities/fintech/gateway/app.py capabilities/fintech/gateway/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/gateway/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/gateway/app.py
./.venv/bin/apg capabilities inspect fintech_gateway --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/gateway --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/gateway --json
git diff --check -- capabilities/fintech/gateway
```

## Next Extensions

- Wire durable payment, provider, webhook, settlement, and dispute stores.
- Connect live provider adapters for Stripe, Adyen, MPESA, DPO, Flutterwave, Pesapal, PayPal, and regional networks.
- Add durable Bytewax topology deployment and replay checks.
- Add rendered UI validation for the APG shell.
- Add provider failover and settlement reconciliation tests against provider sandboxes.
