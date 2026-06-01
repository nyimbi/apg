# Know Your Customer Capability Specification

## Purpose

Know Your Customer gives generated APG applications an executable onboarding
and customer-due-diligence layer for regulated fintech products. It must be
locally runnable, tenant-scoped, deterministic, themeable, and ready to compose
with Digital Payments, Digital Wallets, AUTH, AUDL, CONS, CVSN, BIOP, NLPC,
NTFY, and KEYM.

## Scope

The capability owns:

- customer identity profile lifecycle;
- consent-backed onboarding evidence;
- identity, address, tax, and business-document registration;
- document verification confidence decisions;
- sanctions, PEP, adverse-media, and watchlist screening;
- KYC risk scoring and due-diligence decisions;
- enhanced due diligence review evidence;
- provider-neutral KYC agent registration;
- deterministic rule evaluation;
- APG Python UI route/view-model metadata;
- Bytewax lifecycle stream metadata.

Live document vendors, biometric providers, sanctions data feeds, government
registries, payment providers, wallets, audit sinks, notifications, identity,
and key-management systems are adapters, not hard runtime dependencies.

## Functional Requirements

1. Every write must include tenant context and policy evidence.
2. Profiles require subject references, legal names, supported customer types,
   country codes, and consent evidence.
3. Documents require an existing tenant-local profile, supported document type,
   tokenized storage reference, extracted subject, and verification confidence.
4. Documents below the configured minimum confidence are denied.
5. Sanctions, PEP, watchlist, and adverse-media hits require review.
6. High risk scores require enhanced due diligence review.
7. Verification decisions require identity, address, screening, risk, and
   consent evidence.
8. Lifecycle batches must use Bytewax.
9. AI KYC agents must use supported runtimes and roles.
10. Privileged agent actions require human approval evidence.

## UI And Theming

The APG Python UI surface exposes dashboard, profiles, documents, screening,
risk, reviews, agents, and settings screens. Theme tokens use compact density,
8px radius, and distinct review, verified, warning, and denied status colors.

## Acceptance Evidence

The package is acceptable when focused py_compile, pytest, app self-test,
inspect, publish-plan, implementation-audit, lifecycle-audit, stale-marker scan,
global strict package-artifact audit, and diff checks pass for
`capabilities/fintech/kyc`.
