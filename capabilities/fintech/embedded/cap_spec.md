# APG Capability Spec: Embedded Finance

- Capability id: `fintech_embedded`
- Display name: `Embedded Finance`
- Version: `1.1.0`
- Target: `python`
- Runtime profile: package-backed capability
- Stream processor: `bytewax`
- Stream: `apg.fintech.embedded.lifecycle`

## Provides

`partner_program_workflow`, `host_application_workflow`,
`embedded_product_placement_workflow`, `embedded_customer_consent_workflow`,
`embedded_account_workflow`, `embedded_payment_workflow`,
`embedded_card_workflow`, `embedded_lending_workflow`,
`embedded_settlement_workflow`, `embedded_revenue_share_workflow`, and
`embedded_finance_agent_workflow`.

## Composition

The package composes with APG identity, audit, notification, language,
key-management, banking API, payment, wallet, card, lending, BNPL, KYC, AML,
fraud, mobile, and Bytewax capabilities.
