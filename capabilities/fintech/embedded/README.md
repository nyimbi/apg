# Embedded Finance

Embedded Finance is an executable APG capability for placing financial products
inside partner applications without making the host application own banking
complexity. It manages partner programs, host applications, product placements,
 customer consent, embedded accounts, payments, card offers, lending offers,
settlement, revenue share, and provider-neutral AI agent review.

The package is intentionally dependency-light. Generated applications can import
the Python service directly, while production deployments can bind the adapter
keys in the capability contract to live APG services.

## Use

```python
from capabilities.fintech.embedded import EmbeddedFinanceService

service = EmbeddedFinanceService()
program = service.register_partner_program(
    "program-1", "tenant-1", "Merchant App", "kyb-1", "contract-1", "risk-1"
)
app = service.register_host_application(
    "app-1", "tenant-1", program["id"], "Merchant Checkout", "production",
    "merchant.example", "terms-1"
)
placement = service.publish_product_placement(
    "placement-1", "tenant-1", app["id"], "wallet", "checkout",
    ["wallet.read", "payments.write"], "risk-policy-1"
)
consent = service.capture_customer_consent(
    "consent-1", "tenant-1", app["id"], "customer-1",
    ["wallet.read", "payments.write"], "2026-12-31"
)
payment = service.initiate_embedded_payment(
    "payment-1", "tenant-1", app["id"], placement["id"], consent["id"],
    "wallet-1", "merchant-1", 1250, "USD", "risk-2"
)
```

## Capability Surfaces

- Partner-program onboarding and risk approval.
- Host application registration with domain, terms, and environment controls.
- Product placement for accounts, wallets, payments, cards, loans, BNPL,
  remittance, insurance, and marketplace finance.
- Consent-scoped embedded journeys.
- Embedded account, payment, card, and lending lifecycle records.
- Settlement batch and revenue-share controls.
- Dashboard, console, settings, and AI agent view models.
- Deterministic rule engine and Bytewax lifecycle stream metadata.

## Integration

The contract declares adapters for `auth`, `audl`, `ntfy`, `nlpc`, `keym`,
`fintech_apis`, `fintech_payments`, `fintech_wallets`, `fintech_cards`,
`fintech_lending`, `fintech_bnpl`, `fintech_kyc`, `fintech_aml`,
`fintech_fraud`, and `bytewax`.

Live gateways, partner portals, OAuth consent screens, ledger posting, card
issuance, loan booking, settlement rails, and durable Bytewax workers stay
behind adapter boundaries.
