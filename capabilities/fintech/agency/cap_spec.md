# Capability Spec: fintech_agency

## Identity

- Name: Agency Banking
- Capability id: `fintech_agency`
- Version: `1.1.0`
- Target runtime: Python
- Lifecycle processor: Bytewax

## Contract Summary

`fintech_agency` provides executable APG surfaces for agent-network banking:
programs, outlets, agents, float accounts, customers, transactions, cash
movements, commission settlement, disputes, supervision, and agency AI agents.

## Main Entities

- AgencyProgram
- AgencyOutlet
- AccreditedAgent
- FloatAccount
- AgencyCustomer
- AgencyTransaction
- CashMovement
- CommissionSettlement
- AgencyDispute
- SupervisionVisit
- AgencyEvidence

## Main Commands

- Register agency program.
- Onboard outlet.
- Accredit agent.
- Open float account.
- Onboard agency customer.
- Record agency transaction.
- Record cash movement.
- Settle commission.
- Open agency dispute.
- Record supervision visit.
- Register agency AI agent.
- Validate Bytewax batch.

## UI Screens

- Dashboard
- Programs
- Outlets
- Agents
- Float Accounts
- Customers
- Transactions
- Cash Movements
- Commissions
- Disputes
- Supervision
- AI Agents
- Settings

## Release Evidence

This package publishes `semantic_model.json`, `package_manifest.json`, and
`release_report.json` for APG compiler/runtime tooling.
