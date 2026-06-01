# Robo Advisory Specification

## Purpose

Robo Advisory makes automated investment guidance composable inside APG while
preserving regulatory controls. It must make suitability, model governance,
recommendation evidence, approval status, and audit trails explicit before
automation affects a client portfolio.

## Functional Scope

- Create investor profiles linked to client, KYC, suitability, and risk
  evidence.
- Define goal plans with objective type, target amount, currency, and horizon.
- Publish model portfolios with risk profile, allocation, and policy evidence.
- Generate recommendation packets bound to a profile, goal, model, and analysis.
- Approve recommendation packets before activation.
- Configure automated investment plans with funding source and cadence.
- Record portfolio drift and propose rebalance advice.
- Record tax-loss harvesting candidates.
- Register human review outcomes.
- Register provider-neutral AI agents for suitability, model, recommendation,
  drift, tax, and compliance review.

## Rules

The deterministic rule engine rejects missing tenant context, missing write
policy evidence, incomplete profile/goal/model/recommendation evidence,
unsupported risk/currency/cadence/status values, allocations that do not total
100 percent, automation without approved recommendation evidence, drift without
analysis, tax-loss candidates without tax-lot evidence, unsupported agent
runtimes/roles, non-Bytewax batch streams, and privileged AI-agent actions
without human approval.

## UI And Theming

Generated applications expose dashboard, profiles, goals, models,
recommendations, automation, drift, tax, reviews, agents, and settings routes.
The UI shell is `apg_python`, theming is required, and the package publishes a
compact advisory-control theme with 8px radius.

## Non-Goals

This package does not directly trade securities, ingest live quotes, calculate
official tax lots, hold custody assets, collect fees, render statements, submit
regulatory filings, or run durable Bytewax workers.
