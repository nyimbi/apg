# APG Digital Neobanking Implementation Plan

## Packet 1: Contract And Guardrails

- Define `fintech_neobanking` contract metadata, configuration, dependencies,
  deterministic rules, UI routes, theme tokens, and Bytewax lifecycle stream.
- Publish provider-neutral agent runtimes and roles.
- Keep live banking, card, payment, wallet, support, audit, key, and Bytewax
  systems behind adapter boundaries.

## Packet 2: Executable Runtime

- Add dependency-light models for programs, customers, deposit accounts, rail
  links, transactions, savings pots, statements, service cases, and evidence.
- Add `neobanking_runtime.py` helpers for normalization, account-number
  derivation, date stamping, amount handling, and transaction direction.
- Implement `NeobankingService` methods that enforce deterministic rules before
  local state changes and emit tenant-scoped audit events.

## Packet 3: Composition Surfaces

- Add API helpers for generated APG applications.
- Add view models for dashboards, consoles, rules, routes, theme, and account
  operations screens.
- Add `app.py` self-test, component manifest, and semantic model.

## Packet 4: Evidence And Review

- Add package manifest, release report, semantic model, focused tests,
  specification, runtime spec, and README.
- Update fintech capability metadata, catalog README counts, and progress log.
- Run focused verification: compile checks, package tests, app self-test,
  inspect, publish-plan, implementation audit, lifecycle audit, strict package
  audit, stale-marker scan, disallowed messaging scan, and whitespace check.

## Code Review Checklist

- Every write path has tenant and policy context.
- Program, customer, account, rail, transaction, savings, statement, case,
  batch, and agent guardrails execute through the rule engine.
- High-impact transactions fail closed until approval evidence exists.
- Provider-specific banking systems stay behind adapters.
- UI routes and theme tokens are sufficient for APG application composition.
- Tests cover happy path, guardrail failures, API helpers, view models, and app
  publishability.
