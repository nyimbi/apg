# FinTech Risk Management Specification

## Purpose

FinTech Risk Management gives APG applications a first-class risk operating
surface. It turns risk appetite, customer and counterparty scoring, exposure
monitoring, limit breaches, controls, stress testing, events, reviews, and
AI-agent support into composable application capabilities.

## Functional Scope

- Register tenant-specific risk appetite by supported risk domain.
- Create risk profiles for customers, merchants, wallets, accounts,
  portfolios, loans, agents, and counterparties.
- Record exposures with source evidence, supported currency, positive amount,
  positive limit, and explicit approval for limit overrides.
- Evaluate controls with owner evidence and effectiveness scoring.
- Run stress scenarios with impact, probability, and mitigation evidence.
- Record limit breaches and risk events with severity and evidence.
- Record reviews for any risk object.
- Register provider-neutral AI agents with supported runtimes and roles.
- Publish UI routes, theme metadata, and Bytewax lifecycle metadata.

## Guardrails

- Every write requires tenant context and policy evidence.
- Appetite records require supported domain, positive threshold, owner, and
  evidence.
- Profiles require supported subject type, KYC evidence, valid risk score,
  supported currency, and source evidence.
- Exposures require an existing profile, supported exposure type, positive
  amount, supported currency, positive limit, source evidence, and human
  approval when amount exceeds limit.
- Controls require profile, supported type, owner, evidence, and effectiveness
  score between 0 and 100.
- Stress scenarios require profile, supported scenario, positive impact,
  probability between 0 and 10000 basis points, and mitigation evidence.
- Breaches and events require linked records, supported severity/type, and
  evidence.
- Reviews require supported status, reviewer, and evidence.
- Batch lifecycle events require Bytewax routing.
- Privileged AI-agent actions require human approval.

## Non-Goals

- No live payment capture, ledger posting, market-data subscription,
  regulator filing, model training, or durable worker topology is embedded in
  this package.
- External systems remain behind APG adapter contracts.
