# Embedded Finance Specification

## Purpose

Embedded Finance lets APG users compose regulated financial services into
non-financial applications. The capability must be safe by default: every write
requires tenant context and policy evidence, every customer action requires
consent, and every privileged automation path is reviewable.

## Functional Scope

- Register partner programs with KYB, contract, and risk evidence.
- Register host applications under partner programs.
- Publish product placements by product type, channel, scope, and risk policy.
- Capture and enforce customer consent grants.
- Open embedded accounts and wallet links.
- Initiate embedded payments with amount, currency, placement, consent, and
  risk references.
- Issue embedded card offers with limit and risk evidence.
- Create lending offers with affordability and underwriting evidence.
- Close settlement batches only when reconciled.
- Record revenue-share agreements with bounded percentages.
- Register provider-neutral AI agents for partner-risk, placement, consent,
  settlement, and compliance review.

## Rules

The deterministic rule engine must reject missing tenant context, missing write
policy evidence, incomplete partner/app/placement/consent evidence, payment
attempts without matching placement consent, settlement batches without
reconciliation evidence, invalid revenue-share percentages, unsupported agent
runtimes/roles, non-Bytewax batch streams, and privileged AI-agent actions
without human approval.

## UI And Theming

Generated applications expose dashboard, programs, apps, placements, consents,
accounts, payments, cards, lending, settlements, revenue share, agents, and
settings routes. UI surfaces use `apg_python`, require theming, and publish a
compact operations theme with 8px radius.

## Non-Goals

This package does not directly operate live banking rails, hosted consent pages,
partner developer portals, card processors, loan cores, settlement networks, or
durable Bytewax workers. Those are adapter responsibilities.
