# Digital Cards Capability Specification

## Purpose

`fintech_cards` makes card issuing and card operations a first-class APG
capability. It coordinates card programs, cardholder evidence, virtual/physical
card issuance, token lifecycle, authorization controls, limits, disputes,
Bytewax lifecycle events, visual theming, and provider-neutral AI-agent review.

The capability is executable without live card-network access. It exposes the
domain contract, deterministic rules, local service behavior, API helpers, view
models, and release evidence that generated applications need while keeping
issuer processors, card networks, token service providers, embossers, and 3DS
providers behind adapters.

## Functional Scope

- Card program registration with owner, BIN range, supported currency, and
  settlement configuration.
- Cardholder onboarding linked to KYC and supported countries.
- Virtual or physical card issuance linked to a program, holder, wallet, and
  funding account.
- Token provisioning and suspension for wallet, device, merchant, and network
  tokens.
- Authorization policy checks for amount, currency, merchant category, fraud
  result, AML result, and limit overrides.
- Dispute filing with transaction, reason, evidence, and reviewer assignment.
- Bytewax lifecycle stream validation for card batches.
- Provider-neutral card-agent registration for Codex, Claude Code, OpenCode,
  and Pi.
- Framework-neutral UI/view metadata for programs, cardholders, cards, tokens,
  authorizations, disputes, agents, and settings.

## Required Compositions

- `auth` for tenant/user context and step-up challenge evidence.
- `audl` for card lifecycle and authorization audit events.
- `ntfy` for cardholder and operations notifications.
- `nlpc` for dispute narrative and support classification.
- `keym` and `encr` for tokenized PAN, cryptographic domains, and credential
  protection.
- `fintech_payments` for authorization and settlement handoff.
- `fintech_wallets` for wallet funding, holds, and tokenized wallet cards.
- `fintech_kyc` for cardholder identity evidence.
- `fintech_aml` for sanctions and restricted-party screening.
- `fintech_fraud` for authorization and account-takeover decisions.

## Guardrail Requirements

- Tenant context is mandatory.
- Writes require card policy evidence.
- Programs require owner, BIN range, supported currency, and settlement account.
- Card issuance requires program, cardholder, cardholder KYC, wallet, funding
  account, supported card type, supported product, and consent evidence.
- Physical cards require shipping address evidence.
- Token provisioning requires an existing card, supported token type, token
  reference, device or merchant reference, and key-domain evidence.
- Authorization requires an active card, positive amount, supported currency,
  merchant category, fraud decision, AML result, and human approval for limit
  overrides, high risk, or restricted merchant categories.
- Blocked fraud or AML outcomes deny authorization.
- Disputes require transaction reference, reason, evidence, and reviewer.
- Card batch lifecycle processing must use Bytewax.
- Privileged AI-agent actions require human approval.

## Non-Goals

- Live card-network certification.
- Live issuer-processor, token-service-provider, 3DS, or embossing adapters.
- Production PCI DSS scope implementation.
- Durable Bytewax deployment.
- Network chargeback submission and clearing-file reconciliation.
