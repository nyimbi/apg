# APG Capability Spec: InsurTech

- Capability id: `fintech_insurance`
- Display name: `InsurTech`
- Version: `1.1.0`
- Target: `python`
- Runtime profile: package-backed capability
- Stream processor: `bytewax`
- Stream: `apg.fintech.insurance.lifecycle`

## Provides

`insurance_policyholder_workflow`, `insurance_product_workflow`,
`insurance_quote_workflow`, `insurance_policy_workflow`,
`insurance_premium_workflow`, `insurance_claim_workflow`,
`insurance_document_workflow`, `insurance_risk_workflow`,
`insurance_reinsurance_workflow`, `insurance_compliance_workflow`,
`insurance_review_workflow`, and `insurance_agent_workflow`.

## Composition

The package composes with APG identity, audit, notification, language,
key-management, payments, wallets, KYC, AML, fraud, analytics, reporting, and
Bytewax capabilities.
