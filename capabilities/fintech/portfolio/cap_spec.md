# APG Capability Spec: Portfolio Management

- Capability id: `fintech_portfolio`
- Display name: `Portfolio Management`
- Version: `1.1.0`
- Target: `python`
- Runtime profile: package-backed capability
- Stream processor: `bytewax`
- Stream: `apg.fintech.portfolio.lifecycle`

## Provides

`portfolio_book_workflow`, `portfolio_holding_workflow`,
`portfolio_allocation_policy_workflow`, `portfolio_valuation_workflow`,
`portfolio_benchmark_workflow`, `portfolio_risk_workflow`,
`portfolio_attribution_workflow`, `portfolio_cash_workflow`,
`portfolio_corporate_action_workflow`, `portfolio_compliance_workflow`,
`portfolio_review_workflow`, and `portfolio_agent_workflow`.

## Composition

The package composes with APG identity, audit, notification, language,
key-management, Wealth Management, Robo Advisory, payments, wallets, KYC, AML,
fraud, analytics, reporting, market-data, and Bytewax capabilities.
