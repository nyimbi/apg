# Portfolio Management Specification

## Purpose

Portfolio Management gives generated APG applications a first-class investment
operations component. It owns portfolio books, holdings, allocation policy,
valuation, benchmark, risk, attribution, cash, corporate-action, compliance,
review, and AI-agent guardrails without requiring live custody or broker
credentials.

## Actors

- Portfolio operations teams create books, holdings, cash, valuation, and
  corporate-action records.
- Advisors and investment committees activate allocation policies and review
  exceptions.
- Risk and compliance teams record exposures, breaches, evidence, and outcomes.
- AI agents assist with review preparation, anomaly triage, attribution
  explanation, breach summarization, and policy checks.

## Functional Requirements

1. Create tenant-scoped portfolio books with owner, name, portfolio type, base
   currency, and policy reference.
2. Record holdings only for existing tenant portfolios, with non-empty
   instruments, positive quantity, and positive cost.
3. Activate allocation policies only when allocations total 100 percent and a
   policy reference is attached.
4. Record valuations only for existing tenant portfolios with positive market
   value, supported currency, valuation date, and source reference.
5. Assign benchmarks to existing portfolios with index and policy evidence.
6. Record risk exposures with metric, value, as-of date, source, and limit
   evidence.
7. Record performance attribution with portfolio, period, benchmark, source,
   and contribution values.
8. Record cash movements with existing portfolio, positive amount, supported
   currency, and reference.
9. Record corporate actions using supported action types and evidence.
10. Record compliance breaches with supported severity and evidence.
11. Record review decisions using supported statuses, reviewer, and evidence.
12. Register provider-neutral AI agents with supported runtimes and roles.
13. Deny privileged AI-agent actions unless human approval is recorded.
14. Publish APG UI routes, theme tokens, deterministic rules, Bytewax lifecycle
    metadata, semantic model, package manifest, release report, and tests.

## Rule Engine

Rules are deterministic. Service methods build a context, evaluate the
capability rule engine, and raise `PermissionError` before mutating state when
a deny rule matches. Tenant context and write-policy evidence are universal
mutation guardrails.

## Non-Goals

This package does not directly execute custody instructions, route broker
orders, value securities from live feeds, calculate official tax lots, render
statements, collect billing, file regulatory reports, or run durable stream
workers. Those concerns remain adapter-backed integration boundaries.
