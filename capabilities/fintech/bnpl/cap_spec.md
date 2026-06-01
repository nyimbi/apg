# Capability Spec: fintech_bnpl

## Identity

- Name: Buy Now Pay Later
- Capability id: `fintech_bnpl`
- Version: `1.1.0`
- Target runtime: Python
- Lifecycle processor: Bytewax

## Contract Summary

`fintech_bnpl` gives APG applications a composable BNPL domain package covering
merchant program governance, consumer onboarding, merchant profiles, checkout
sessions, affordability decisions, plan creation, installment scheduling,
merchant settlement, disputes, and BNPL AI agents.

## Main Entities

- MerchantProgram
- BNPLConsumer
- MerchantProfile
- CheckoutSession
- AffordabilityDecision
- BNPLPlan
- InstallmentSchedule
- MerchantSettlement
- BNPLDispute
- BNPLevidence

## Main Commands

- Register merchant program.
- Onboard consumer.
- Register merchant profile.
- Create checkout session.
- Record affordability decision.
- Create BNPL plan.
- Schedule installment.
- Record merchant settlement.
- Open BNPL dispute.
- Register BNPL AI agent.
- Validate Bytewax batch.

## UI Screens

- Dashboard
- Programs
- Consumers
- Merchants
- Checkouts
- Affordability
- Plans
- Installments
- Settlements
- Disputes
- Agents
- Settings

## Release Evidence

This package publishes `semantic_model.json`, `package_manifest.json`, and
`release_report.json` for APG compiler/runtime tooling.
