# APG Capability Spec: Robo Advisory

- Capability id: `fintech_robo`
- Display name: `Robo Advisory`
- Version: `1.1.0`
- Target: `python`
- Runtime profile: package-backed capability
- Stream processor: `bytewax`
- Stream: `apg.fintech.robo.lifecycle`

## Provides

`robo_investor_profile_workflow`, `robo_goal_plan_workflow`,
`robo_model_portfolio_workflow`, `robo_recommendation_workflow`,
`robo_automation_workflow`, `robo_drift_workflow`,
`robo_tax_loss_workflow`, `robo_review_workflow`, and
`robo_agent_workflow`.

## Composition

The package composes with APG identity, audit, notification, language,
key-management, Wealth Management, KYC, AML, fraud, analytics, reporting,
market data, and Bytewax capabilities.
