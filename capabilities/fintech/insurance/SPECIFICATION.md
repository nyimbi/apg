# InsurTech Specification

## Purpose

InsurTech gives generated APG applications a first-class insurance operations
component. It manages policyholders, products, quotes, policies, premiums,
claims, documents, risk assessments, reinsurance, compliance alerts, reviews,
and AI-agent guardrails without requiring live underwriting, claims, or payment
infrastructure.

## Actors

- Policy operations teams onboard policyholders, publish products, bind
  policies, and record premiums.
- Underwriters create quotes, assess risks, and review reinsurance attachments.
- Claims teams open claims and attach evidence documents.
- Risk and compliance teams review alerts, claim exceptions, and policy
  evidence.
- AI agents assist with underwriting review, claim triage, fraud review,
  reinsurance review, and compliance summaries.

## Functional Requirements

1. Onboard tenant-scoped policyholders with KYC, contact, and risk-profile
   evidence.
2. Publish insurance products only for supported product lines with coverage
   terms and pricing references.
3. Generate quotes only for existing policyholders and products with positive
   premium and underwriting evidence.
4. Bind policies only from existing quotes with effective date and payment
   reference.
5. Record premiums only for existing policies with positive amount and payment
   reference.
6. Open claims only for existing policies with supported claim type, positive
   amount, loss date, and evidence.
7. Record documents with supported document type and evidence.
8. Record risk assessments with policyholder, score, and source evidence.
9. Record reinsurance attachments with policy, treaty reference, and share.
10. Record compliance alerts with supported severity and evidence.
11. Record review decisions with supported status, reviewer, and evidence.
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

This package does not directly capture live payments, call external
underwriting engines, schedule repair or medical networks, file regulator
reports, calculate actuarial reserves, issue signed legal documents, or run
durable stream workers. Those concerns remain adapter-backed integration
boundaries.
