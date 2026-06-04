# Fintech Gateway

## Overview
Fintech Gateway is the payment orchestration capability responsible for merchant onboarding, payment provider connections, payment method tokenization, payment intent lifecycle, routing decisions, fraud risk review, authorization and capture, refunds, webhook ingestion, settlement reconciliation, and dispute management. It is the operational hub that connects the APG payment layer to external payment processors (Stripe, Adyen, MPESA, Flutterwave, Pesapal, DPO, PayPal, and others) while enforcing routing, risk, and governance rules on every payment.

Blocked risk levels produce hard denies on authorization. Overcapture and overrefund are blocked by the rule engine. Settlement variance requires review. Webhook ingestion requires idempotency keys and signature verification. All gateway events stream to `apg.fintech.gateway.lifecycle` via Bytewax.

## Capability ID
`fintech_gateway`  Version: 2.1.0

## Provides
| Service | Description |
|---------|-------------|
| merchant_onboarding_lifecycle | Onboard merchants with code, legal name, country, KYC, and high-risk review gates |
| provider_connection_lifecycle | Connect payment providers with type, credential reference, and supported methods |
| payment_method_tokenization_workflow | Tokenize payment methods for card, bank, mobile money, wallet, and cash voucher |
| payment_intent_lifecycle | Create and track payment intents with merchant, amount, currency, and method |
| payment_routing_workflow | Route payment intents to providers based on risk and routing decisions |
| fraud_risk_review_workflow | Score and review payment risk; blocked risk denies authorization |
| authorization_capture_workflow | Authorize and capture payments with overcapture protection |
| refund_lifecycle | Process refunds with overrefund protection and large-refund review |
| webhook_ingestion_workflow | Ingest signed provider webhooks with idempotency key enforcement |
| settlement_reconciliation_workflow | Record settlements with variance review |
| payment_dispute_workflow | Open and resolve payment disputes with owner assignment and resolution review |
| gateway_agents | Register AI agents for merchant underwriting, routing, fraud, and settlement review |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Merchant and operations notifications |
| composition_events | Composition event bus |
| composition_config | Configuration management |
| keym | Key management |
| encr | Encryption for payment credentials |
| cash_management | Cash management integration |
| accounts_receivable | Accounts receivable integration |
| customer_relationship_management | CRM integration |
| business_intelligence | Analytics |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| merchants.supported_providers | list | stripe, adyen, mpesa, dpo, flutterwave, pesapal, paypal, manual | Supported payment providers |
| merchants.supported_provider_types | list | card, bank, mobile_money, wallet, settlement, fraud | Provider categories |
| payment_intents.supported_currencies | list | USD, EUR, GBP, KES, ZAR, NGN, GHS, UGX, TZS | Supported currencies |
| disputes.supported_reasons | list | fraud, duplicate, product_not_received, service_not_provided, authorization, processing_error, other | Dispute reason codes |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-gateway/dashboard | GET | fintech_gateway:view | Overview |
| merchants | /fintech-gateway/merchants | GET/POST | fintech_gateway:manage_merchants | Merchants |
| providers | /fintech-gateway/providers | GET/POST | fintech_gateway:manage_providers | Providers |
| payment_methods | /fintech-gateway/payment-methods | GET/POST | fintech_gateway:manage_payment_methods | Payments |
| payments | /fintech-gateway/payments | GET/POST | fintech_gateway:process | Payments |
| routing | /fintech-gateway/routing | GET/POST | fintech_gateway:route | Operations |
| risk | /fintech-gateway/risk | GET/POST | fintech_gateway:risk | Risk |
| webhooks | /fintech-gateway/webhooks | GET/POST | fintech_gateway:webhooks | Operations |
| settlements | /fintech-gateway/settlements | GET/POST | fintech_gateway:settle | Finance |
| disputes | /fintech-gateway/disputes | GET/POST | fintech_gateway:disputes | Risk |
| agents | /fintech-gateway/agents | GET/POST | fintech_gateway:admin | Automation |
| settings | /fintech-gateway/settings | GET/POST | fintech_gateway:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| high_risk_merchant_requires_review | Merchant with `risk_level: high` without review | require_review |
| provider_requires_credentials | Provider connection without credential reference | deny |
| payment_intent_amount_positive | Payment intent amount <= 0 | deny |
| high_risk_payment_requires_review | Payment risk `high` without review | require_review |
| blocked_risk_denies_authorization | Payment risk `blocked` | deny |
| high_value_authorization_requires_approval | High-value authorization without approval | require_review |
| capture_blocks_overcapture | Capture amount exceeds authorized amount | deny |
| refund_blocks_overrefund | Refund amount exceeds captured balance | deny |
| large_refund_requires_review | Large refund without review | require_review |
| webhook_requires_signature | Webhook without signature | deny |
| webhook_requires_idempotency | Webhook without idempotency key | deny |
| settlement_variance_requires_review | Settlement variance without review | require_review |
| dispute_resolution_requires_review | Dispute resolution without review | deny |
| gateway_batch_requires_bytewax | Batch without Bytewax | deny |
| gateway_event_requires_bytewax | Event without Bytewax | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| Merchant | id, merchant_code, legal_name, country, kyc_reference, risk_level, status |
| ProviderConnection | id, provider, provider_type, credential_reference, status |
| PaymentMethod | id, merchant_id, customer_reference, method_type, token_reference, status |
| PaymentIntent | id, merchant_id, payment_method_id, amount, currency, risk_level, status |
| Authorization | id, payment_intent_id, provider_id, authorized_amount, status |
| Capture | id, authorization_id, capture_amount, status |
| Refund | id, capture_id, refund_amount, status |
| WebhookEvent | id, provider_id, event_id, signature, idempotency_key, payload |
| Settlement | id, provider_id, settlement_reference, amount, variance, status |
| PaymentDispute | id, payment_id, reason, owner_id, resolution_review_reference, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| merchant_onboarded | Merchant passes onboarding |
| provider_connected | Provider connection established |
| payment_method_tokenized | Payment method tokenized |
| payment_intent_created | Payment intent created |
| payment_risk_assessed | Risk assessment recorded |
| payment_authorized | Authorization approved |
| payment_captured | Payment captured |
| payment_refunded | Refund processed |
| webhook_ingested | Provider webhook ingested |
| settlement_recorded | Settlement reconciled |
| payment_dispute_opened | Dispute opened |
| payment_dispute_resolved | Dispute resolved |
| gateway_agent_registered | AI agent registered |

## Edge Cases Handled
- `blocked` risk level produces a hard deny on authorization regardless of all other factors — no override path exists for blocked risk
- Capture amount is validated against the authorized amount; partial captures are allowed but over-captures are blocked; the `overcapture: True` flag in context triggers the deny
- Refund amount is validated against the remaining captured balance; partial refunds are allowed; over-refunds are blocked at the rule engine level
- Webhook idempotency keys prevent duplicate event processing; a webhook without an idempotency key is rejected even if the signature is valid
- Dispute resolution requires a recorded review — disputes cannot be auto-resolved; every resolution must have a review reference
- Settlement amounts use the `_lt` condition suffix (amount < 0) rather than `_lte`; zero-amount settlements are valid reconciliation entries

## Composability
- **Upstream**: `fintech_payments` routes payment orders to the gateway; `fintech_wallets` provides wallet-based payment methods; `fintech_kyc` provides merchant KYC evidence
- **Downstream**: `cash_management` and `accounts_receivable` receive settlement and reconciliation data; `business_intelligence` consumes gateway event data
- **Peer**: Deployed alongside `fintech_payments` (application-facing payment layer) and `fintech_fraud` (risk scoring before authorization)

## Development Notes
- Version is 2.1.0 (higher than other capabilities at 1.1.0) — this reflects the gateway being a more mature capability with additional provider types and routing features
- `composition_events` and `composition_config` are declared as required dependencies (unlike the pattern in other capabilities which use `auth`, `audl`, etc.); these are platform-level event bus and config dependencies
- The `_lte` and `_lt` condition key suffixes enable numeric comparisons: `amount_lte: 0` fires when amount <= 0, `settlement_amount_lt: 0` fires when amount < 0
- `max_autonomous_scope` for gateway agents is set to `recommend_validate_and_prepare` — agents cannot execute payments autonomously; they can only prepare and validate
