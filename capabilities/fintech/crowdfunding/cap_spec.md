# APG Capability Spec: Crowdfunding Platform

- Capability id: `fintech_crowdfunding`
- Display name: `Crowdfunding Platform`
- Version: `1.1.0`
- Target: `python`
- Runtime profile: package-backed capability
- Stream processor: `bytewax`
- Stream: `apg.fintech.crowdfunding.lifecycle`

## Provides

`crowdfunding_issuer_workflow`, `crowdfunding_campaign_workflow`,
`crowdfunding_disclosure_workflow`, `crowdfunding_commitment_workflow`,
`crowdfunding_escrow_workflow`, `crowdfunding_milestone_workflow`,
`crowdfunding_payout_workflow`, `crowdfunding_investor_update_workflow`,
`crowdfunding_compliance_workflow`, `crowdfunding_review_workflow`, and
`crowdfunding_agent_workflow`.

## Composition

The package composes with APG identity, audit, notification, language,
key-management, payments, wallets, KYC, AML, fraud, portfolio, wealth,
analytics, reporting, and Bytewax capabilities.
