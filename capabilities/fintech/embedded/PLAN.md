# Embedded Finance Plan

## Build Order

1. Define the capability contract, configuration, deterministic rules, UI
   routes, theme, Bytewax lifecycle stream, provides, and dependencies.
2. Add dependency-light models for partner programs, host applications, product
   placements, consents, accounts, payments, cards, lending offers, settlements,
   revenue share, and evidence.
3. Implement a service layer that evaluates rules before state mutation.
4. Add API helper functions and view models for generated applications.
5. Add a publishable app entrypoint, semantic model, manifest, and release
   evidence.
6. Add focused tests that validate contract shape, guardrails, lifecycle
   execution, API/view behavior, and publishability.
7. Run focused py_compile, pytest, app self-test, and APG capability audits.
8. Record progress evidence and code review findings.

## Review Focus

- Consent scopes must cover embedded payment actions.
- Product placements must belong to the selected host application.
- Settlement and revenue-share actions must be bounded and evidenced.
- Provider-neutral AI agents must not bypass human approval for privileged work.
- Live financial rails must remain behind adapters.
