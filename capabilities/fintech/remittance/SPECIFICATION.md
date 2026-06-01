# Cross-Border Remittance Capability Specification

## Purpose

`fintech_remittance` makes cross-border money movement a first-class APG
capability. It coordinates quotes, corridors, sender and beneficiary evidence,
AML/Fraud decisions, payout execution, settlement proof, exceptions, and
provider-neutral AI-agent review.

The capability is intentionally executable but adapter-bounded. It can run
locally without live money movement providers while publishing the contracts,
rules, UI metadata, and evidence needed for generated applications to compose it
with payments, wallets, KYC, AML, and fraud.

## Functional Scope

- Corridor eligibility for source/destination countries and currencies.
- FX quote lifecycle with explicit rate, fee, expiry, and quote lock evidence.
- Remittance transfer creation with sender, beneficiary, KYC, AML, fraud,
  funding, payout method, purpose, and source-of-funds evidence.
- Payout release with settlement reference and provider receipt.
- Refund/return workflow for failed, cancelled, or compliance-blocked transfers.
- Bytewax lifecycle stream validation for remittance batches and events.
- Provider-neutral AI-agent registration for operations, compliance, payout,
  treasury, and customer-support review.
- Framework-neutral UI/view metadata for dashboards, corridors, quotes,
  transfers, payouts, refunds, agents, and settings.

## Required Compositions

- `auth` for tenant/user context and challenge evidence.
- `audl` for durable event evidence.
- `ntfy` for sender/beneficiary/customer-service notifications.
- `nlpc` for purpose, narrative, and support classification.
- `keym` for tokenized provider credentials and settlement references.
- `fintech_payments` for funding and provider authorization.
- `fintech_wallets` for stored-value funding and payout rails.
- `fintech_kyc` for sender and beneficiary identity evidence.
- `fintech_aml` for sanctions, watchlist, and typology screening.
- `fintech_fraud` for transaction-risk and account-takeover decisions.

## Guardrail Requirements

- Tenant context is mandatory.
- Writes require remittance policy evidence.
- Corridors, currencies, payout methods, purpose codes, and agent roles must be
  supported.
- Quotes require positive amount, positive FX rate, non-negative fee, expiry,
  and supported corridor/currency evidence.
- Transfers require quote lock, sender and beneficiary references, sender and
  beneficiary KYC, funding reference, payout method, purpose code,
  source-of-funds evidence, AML screen, and fraud decision.
- Sanctions hits and blocked fraud decisions deny before state changes.
- High-value transfers, AML review results, and fraud review/hold decisions
  require human approval evidence.
- Payout release requires settlement reference and provider receipt.
- Refunds require transfer reference, reason, and reviewer.
- Remittance batch/event lifecycles must use Bytewax, not Kafka.
- Privileged AI-agent actions require human approval.

## Non-Goals

- Live FX liquidity routing.
- Live payment, wallet, bank, mobile-money, or card-network execution.
- Live sanctions/PEP/adverse-media providers.
- Durable Bytewax topology deployment.
- Production treasury reconciliation and regulatory filing adapters.
