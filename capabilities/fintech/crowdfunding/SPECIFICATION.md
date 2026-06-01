# Crowdfunding Platform Specification

## Purpose

Crowdfunding Platform gives generated APG applications a first-class regulated
fundraising component. It manages issuer onboarding, campaign publication,
disclosures, investor commitments, escrow funding, milestones, payout
authorization, investor updates, compliance alerts, review decisions, and
AI-agent guardrails without requiring live payment or securities infrastructure.

## Actors

- Issuers create fundraising campaigns and supply disclosure evidence.
- Investors pledge and fund commitments after KYC and risk acknowledgement.
- Platform operations manage escrow, milestones, payouts, and investor updates.
- Risk and compliance teams review disclosures, alerts, and release approvals.
- AI agents assist with due diligence, disclosure review, commitment checks,
  escrow release preparation, investor updates, and compliance summaries.

## Functional Requirements

1. Onboard tenant-scoped issuers with KYC, beneficial-owner, and risk-rating
   evidence.
2. Publish campaigns only for existing issuers, supported campaign type,
   supported currency, positive target, and disclosure reference.
3. Record disclosures only for existing campaigns, supported disclosure type,
   and evidence.
4. Record investor commitments only for existing campaigns, investor KYC,
   positive amount, supported currency, and risk acknowledgement.
5. Record escrow funding only for existing commitments, wallet reference, and
   positive amount.
6. Record milestones only for existing campaigns with milestone evidence.
7. Authorize payouts only for existing campaign and milestone, positive amount,
   and approval evidence.
8. Publish investor updates only for existing campaigns with disclosure
   reference and recipient scope.
9. Record compliance alerts with supported severity and evidence.
10. Record review decisions with supported status, reviewer, and evidence.
11. Register provider-neutral AI agents with supported runtimes and roles.
12. Deny privileged AI-agent actions unless human approval is recorded.
13. Publish APG UI routes, theme tokens, deterministic rules, Bytewax lifecycle
    metadata, semantic model, package manifest, release report, and tests.

## Rule Engine

Rules are deterministic. Service methods build a context, evaluate the
capability rule engine, and raise `PermissionError` before mutating state when
a deny rule matches. Tenant context and write-policy evidence are universal
mutation guardrails.

## Non-Goals

This package does not directly capture live payments, settle wallets, register
securities, verify investor accreditation, sign legal documents, file regulator
reports, calculate tax forms, run secondary trading, or run durable stream
workers. Those concerns remain adapter-backed integration boundaries.
