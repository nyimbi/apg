# APG Capability Spec: Wealth Management

- Capability id: `fintech_wealth`
- Display name: `Wealth Management`
- Version: `1.1.0`
- Target: `python`
- Runtime profile: package-backed capability
- Stream processor: `bytewax`
- Stream: `apg.fintech.wealth.lifecycle`

## Provides

`wealth_client_profile_workflow`, `suitability_profile_workflow`,
`portfolio_management_workflow`, `advisory_mandate_workflow`,
`portfolio_rebalance_workflow`, `wealth_order_workflow`,
`performance_reporting_workflow`, `wealth_fee_workflow`, and
`wealth_agent_workflow`.

## Composition

The package composes with APG identity, audit, notification, language,
key-management, KYC, AML, fraud, payments, wallets, analytics, reporting, and
Bytewax capabilities.
